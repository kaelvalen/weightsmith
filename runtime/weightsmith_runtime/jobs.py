"""Job manager: one training process at a time per daemon (one GPU), a FIFO queue,
sequence-numbered events and the start/pause/resume/stop/checkpoint protocol.

Each job is a subprocess running the generated train.py with --events, so the
daemon survives OOMs and crashes, and what runs here is exactly what an export runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from weightsmith_codegen import project as codegen
from weightsmith_codegen.templates import data as data_template
from weightsmith_core import ir
from weightsmith_codegen.emit import compile_graph

from .store import Store, run_id

LIVE = ("queued", "preparing", "running", "paused")


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # NixOS keeps the CUDA driver outside the default loader path.
    if Path("/run/opengl-driver/lib").exists() and "/run/opengl-driver/lib" not in env.get("LD_LIBRARY_PATH", ""):
        env["LD_LIBRARY_PATH"] = ":".join(p for p in ("/run/opengl-driver/lib", env.get("LD_LIBRARY_PATH")) if p)
    return env


@dataclass
class Live:
    proc: asyncio.subprocess.Process
    tasks: list[asyncio.Task] = field(default_factory=list)
    last_status: str = "preparing"
    stderr_tail: deque = field(default_factory=lambda: deque(maxlen=40))


class JobManager:
    def __init__(self, store: Store, python_bridge=None):
        self.store = store
        self.python = python_bridge
        self.queue: deque[tuple[str, str | None]] = deque()  # (run id, checkpoint to resume from)
        self.live: dict[str, Live] = {}
        self.seq: dict[str, int] = {}
        self.subs: dict[str, set[asyncio.Queue]] = {}
        self.status_subs: set[asyncio.Queue] = set()
        self._kick = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._files: dict[str, Any] = {}

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self.store.mark_interrupted()
        self._loop_task = asyncio.create_task(self._scheduler())

    async def shutdown(self) -> None:
        for rid, live in list(self.live.items()):
            with contextlib.suppress(Exception):
                await self._send(rid, "stop")
        for rid, live in list(self.live.items()):
            with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                await asyncio.wait_for(live.proc.wait(), 15)
            with contextlib.suppress(ProcessLookupError):
                live.proc.kill()
        if self._loop_task:
            self._loop_task.cancel()
        for f in self._files.values():
            f.close()

    # ------------------------------------------------------------------ submit

    async def submit(self, job: dict[str, Any], project: str | None = None, variant: str | None = None,
                     name: str | None = None, parent: str | None = None, ablation: Any = None,
                     rerun: bool = False) -> dict[str, Any]:
        job = codegen.normalize_job(job)
        hashes = codegen.job_hashes(job)
        hashes["data"] = await asyncio.to_thread(data_template.fingerprint, {"data": job["data"]})
        rid = run_id(hashes["graph"], hashes["config"], hashes["data"], hashes["seed"])

        existing = self.store.get_run(rid)
        if existing and not rerun:
            if existing["status"] in LIVE or existing["status"] == "completed":
                return {**existing, "existing": True}
            if existing["status"] in ("stopped", "interrupted") and self.store.latest_checkpoint(rid):
                await self.control(rid, "resume")
                return {**self.store.get_run(rid), "existing": True}  # type: ignore[dict-item]
        if existing:
            if existing["status"] in LIVE:
                await self.control(rid, "stop")
                raise RuntimeError("run is live; stop it before re-running")
            self.store.delete_run(rid)

        out = self.store.run_dir(rid)
        compiled = await asyncio.to_thread(codegen.write_run, job, out, rid, self.python)
        smoke = await asyncio.to_thread(codegen.smoke_test, compiled.model_py, compiled, job["graph"], out / ".smoke")
        (out / "job.json").write_text(json.dumps({**job, "hashes": hashes, "run_id": rid}, indent=2))
        rec = self.store.create_run(rid, hashes, project, variant, name or job["graph"].get("name"), parent, ablation)
        totals = compiled.analysis.totals
        self.store.update_run(rid, summary={"params": smoke.get("total", totals.total), "trainable": smoke.get("trainable", totals.trainable)})
        self.seq[rid] = 0
        self._event(rid, {"type": "status", "state": "queued"})
        if not smoke["ok"]:
            self._event(rid, {"type": "error", "class": "shape", "message": smoke["issues"][0]["message"], "node": smoke["issues"][0]["node"]})
            self.store.update_run(rid, status="failed")
            return self.store.get_run(rid)  # type: ignore[return-value]
        self.queue.append((rid, None))
        self._kick.set()
        return self.store.get_run(rid) or rec

    # ----------------------------------------------------------------- control

    async def control(self, rid: str, cmd: str) -> dict[str, Any]:
        rec = self.store.get_run(rid)
        if rec is None:
            raise KeyError(rid)
        if cmd not in ("pause", "resume", "stop", "checkpoint"):
            raise ValueError(f"unknown command {cmd!r}")
        if rid in self.live and rec["status"] in ("stopped", "failed", "completed"):
            # the process announced its end but has not been reaped yet
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.live[rid].proc.wait(), 30)
            while rid in self.live:
                await asyncio.sleep(0.02)
            rec = self.store.get_run(rid) or rec
        if rid in self.live:
            await self._send(rid, cmd)
        elif cmd == "stop" and any(q[0] == rid for q in self.queue):
            self.queue = deque(q for q in self.queue if q[0] != rid)
            self._event(rid, {"type": "status", "state": "stopped"})
        elif cmd == "resume" and rec["status"] in ("stopped", "interrupted", "paused", "failed"):
            ckpt = self.store.latest_checkpoint(rid)
            if ckpt is None:
                raise ValueError("no checkpoint to resume from")
            self.queue.append((rid, str(ckpt)))
            self._event(rid, {"type": "status", "state": "queued", "resume_from": ckpt.name})
            self._kick.set()
        else:
            raise ValueError(f"cannot {cmd} a run that is {rec['status']}")
        return self.store.get_run(rid)  # type: ignore[return-value]

    async def _send(self, rid: str, cmd: str) -> None:
        live = self.live[rid]
        assert live.proc.stdin
        try:
            live.proc.stdin.write((json.dumps({"cmd": cmd}) + "\n").encode())
            await live.proc.stdin.drain()
        except (ConnectionResetError, BrokenPipeError):
            raise ValueError("the training process has already exited")

    # --------------------------------------------------------------- scheduler

    async def _scheduler(self) -> None:
        while True:
            await self._kick.wait()
            self._kick.clear()
            while self.queue and not self.live:
                rid, ckpt = self.queue.popleft()
                try:
                    await self._launch(rid, ckpt)
                except Exception as e:  # never let one job kill the scheduler
                    self._event(rid, {"type": "error", "class": "runtime", "message": f"could not start: {e}"})
                    self.store.update_run(rid, status="failed")

    async def _launch(self, rid: str, ckpt: str | None) -> None:
        d = self.store.run_dir(rid)
        args = [sys.executable, "train.py", "--events"]
        if ckpt:
            args += ["--resume", ckpt]
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=d, env=child_env(), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=2**22,
        )
        live = Live(proc)
        self.live[rid] = live
        self.store.update_run(rid, error=None)
        self._event(rid, {"type": "status", "state": "preparing", "pid": proc.pid})
        live.tasks = [asyncio.create_task(self._pump(rid, proc.stdout, False)), asyncio.create_task(self._pump(rid, proc.stderr, True))]
        asyncio.create_task(self._reap(rid))

    async def _pump(self, rid: str, stream, is_err: bool) -> None:
        live = self.live[rid]
        async for raw in stream:
            line = raw.decode(errors="replace").rstrip("\n")
            if not line:
                continue
            if not is_err and line.startswith("@@ "):
                try:
                    ev = json.loads(line[3:])
                except json.JSONDecodeError:
                    ev = {"type": "log", "message": line}
            else:
                if is_err:
                    live.stderr_tail.append(line)
                ev = {"type": "log", "message": line, "stream": "stderr" if is_err else "stdout"}
            self._event(rid, ev)

    async def _reap(self, rid: str) -> None:
        live = self.live[rid]
        code = await live.proc.wait()
        await asyncio.gather(*live.tasks, return_exceptions=True)
        rec = self.store.get_run(rid)
        if rec and rec["status"] in LIVE:
            state = "failed" if code else "completed"
            if code and not rec.get("error"):
                tail = "\n".join(live.stderr_tail)
                self._event(rid, {"type": "error", "class": "crash", "message": f"process exited with code {code}", "traceback": tail})
            self._event(rid, {"type": "status", "state": state, "exit_code": code})
        del self.live[rid]
        self._kick.set()

    # ------------------------------------------------------------------ events

    def _next_seq(self, rid: str) -> int:
        if rid not in self.seq:
            evs = self.store.events(rid)
            self.seq[rid] = evs[-1]["seq"] if evs else 0
        self.seq[rid] += 1
        return self.seq[rid]

    def _event(self, rid: str, ev: dict[str, Any]) -> dict[str, Any]:
        ev = {"seq": self._next_seq(rid), "ts": time.time(), "run": rid, **ev}
        f = self._files.get(rid)
        if f is None or f.closed:
            self.store.run_dir(rid).mkdir(parents=True, exist_ok=True)
            f = self._files[rid] = open(self.store.run_dir(rid) / "events.jsonl", "a", buffering=1)
        f.write(json.dumps(ev, default=float) + "\n")
        self._summarize(rid, ev)
        for q in list(self.subs.get(rid, ())):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(ev)
        if ev["type"] in ("status", "error", "eval"):
            light = {k: v for k, v in ev.items() if k != "traceback"}
            for q in list(self.status_subs):
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(light)
        return ev

    def _summarize(self, rid: str, ev: dict[str, Any]) -> None:
        t = ev["type"]
        if t == "status":
            summary = {}
            for k in ("total_steps", "params", "trainable", "system"):
                if k in ev:
                    summary[k] = ev[k]
            if ev["state"] == "running" and "started" not in (self.store.get_run(rid) or {}).get("summary", {}):
                summary["started"] = ev["ts"]
            if ev["state"] in ("completed", "stopped", "failed"):
                summary["finished"] = ev["ts"]
            self.store.update_run(rid, status=ev["state"], summary=summary or None)
        elif t == "metric":
            keep = {k: ev[k] for k in ("step", "loss", "lr", "grad_norm", "throughput", "gpu_mem_peak_mb") if k in ev}
            self.store.update_run(rid, summary={"last": keep})
        elif t == "eval":
            rec = self.store.get_run(rid)
            s = {"eval": {k: v for k, v in ev.items() if k not in ("seq", "ts", "run", "type")}}
            best = (rec or {}).get("summary", {}).get("best_val_loss")
            if "val_loss" in ev and (best is None or ev["val_loss"] < best):
                s["best_val_loss"] = ev["val_loss"]
            self.store.update_run(rid, summary=s)
        elif t == "error":
            self.store.update_run(rid, error={k: v for k, v in ev.items() if k not in ("seq", "ts", "run", "type")})
            if ev.get("class") in ("nan", "oom", "shape", "runtime", "crash"):
                self.store.update_run(rid, status="failed")

    def subscribe(self, rid: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self.subs.setdefault(rid, set()).add(q)
        return q

    def unsubscribe(self, rid: str, q: asyncio.Queue) -> None:
        self.subs.get(rid, set()).discard(q)

    # ---------------------------------------------------------------- ablation

    async def ablate(self, rid: str, nodes: list[str], adapt: list[dict[str, Any]]) -> dict[str, Any]:
        """Re-run with nodes/modifiers disabled, same config, data and seed; linked to the original."""
        d = self.store.run_dir(rid)
        job = json.loads((d / "job.json").read_text())
        graph = copy.deepcopy(job["graph"])

        def find(ref: str) -> dict[str, Any]:
            scope, _, nid = ref.rpartition("/")
            body = graph if not scope else graph["subgraphs"][scope]
            for n in body["nodes"]:
                if n["id"] == nid:
                    return n
            raise KeyError(ref)

        for ref in nodes:
            find(ref)["disabled"] = True
        for a in adapt:
            find(a["node"])["adapt"][a["index"]]["disabled"] = True
        rec = self.store.get_run(rid) or {}
        label = ", ".join([*nodes, *(f"{a['node']}.adapt[{a['index']}]" for a in adapt)])
        return await self.submit(
            {"graph": graph, "data": job["data"], "train": job["train"]},
            project=rec.get("project"), variant=rec.get("variant"),
            name=f"{rec.get('name') or 'run'} − {label}", parent=rid, ablation={"nodes": nodes, "adapt": adapt},
        )


def merge_adapters(run_dir: Path, graph: ir.Graph) -> tuple[str, dict]:
    """Fold LoRA into the base weights: plain model.py (no LoRA) + matching state dict."""
    import torch

    from weightsmith_codegen.project import _load_module

    mod = _load_module(run_dir / "model.py", f"ws_merge_{run_dir.name}")
    model = mod.Model()
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    for name, m in list(model.named_modules()):
        for child_name, child in list(m.named_children()):
            if type(child).__name__ == "LoRALinear":
                setattr(m, child_name, child.merged())
    plain = copy.deepcopy(graph)
    bodies = [plain["nodes"]] + [b["nodes"] for b in (plain.get("subgraphs") or {}).values()]
    for nodes in bodies:
        for n in nodes:
            n["adapt"] = [a for a in n.get("adapt", []) if a["kind"] != "lora" or a.get("disabled")]
    return compile_graph(plain).model_py, model.state_dict()


def export_zip(store: Store, rid: str, dest: Path, merge: bool = False) -> Path:
    """model.py + weights + config.yaml (+ train.py, data.py) as one folder, zipped.

    merge=True folds LoRA into the base weights; otherwise adapters.pt ships separately.
    """
    import torch

    src = store.run_dir(rid)
    stage = dest.parent / f".export-{rid}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for name in ("model.py", "train.py", "data.py", "config.yaml", "graph.json", "README.md", "model.map.json", "model.pt", "adapters.pt"):
        if (src / name).exists():
            shutil.copy2(src / name, stage / name)
    if merge and (src / "model.pt").exists() and (src / "adapters.pt").exists():
        graph = json.loads((src / "graph.json").read_text())
        model_py, state = merge_adapters(src, graph)
        (stage / "model.py").write_text(model_py)
        (stage / "model.map.json").unlink(missing_ok=True)
        (stage / "adapters.pt").unlink(missing_ok=True)
        torch.save(state, stage / "model.pt")
    base = shutil.make_archive(str(dest.with_suffix("")), "zip", stage)
    shutil.rmtree(stage, ignore_errors=True)
    return Path(base)
