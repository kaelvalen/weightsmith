"""Training loop for the model in model.py, configured by config.yaml.

    python train.py                      # train with config.yaml next to this file
    python train.py --resume checkpoints/step_500.pt

Copied verbatim into every run folder by Weightsmith. With --events it also
speaks the Weightsmith job protocol: JSON events on stdout (lines starting with
"@@ "), control commands (pause/resume/stop/checkpoint) as JSON lines on stdin.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import os
import platform
import queue
import random
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import data  # noqa: E402
from model import Model  # noqa: E402

try:  # optional: records per-layer diagnostics under scope/ (pip install trainscope)
    from trainscope import TrainScope
    from trainscope.core.config import load_config
except ImportError:
    TrainScope = None

EVENTS = False


def emit(kind: str, **payload) -> None:
    if EVENTS:
        print("@@ " + json.dumps({"type": kind, **payload}, default=float), flush=True)
    elif kind == "metric":
        parts = [f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in payload.items() if v is not None]
        print("  ".join(parts), flush=True)
    elif kind in ("status", "error", "checkpoint", "eval"):
        print(f"[{kind}] " + json.dumps(payload, default=float), flush=True)


def log(msg: str) -> None:
    if EVENTS:
        emit("log", message=msg)
    else:
        print(msg, flush=True)


# ------------------------------------------------------------------ control


class Control:
    """Reads JSON commands from stdin in a background thread."""

    def __init__(self, enabled: bool):
        self.q: queue.Queue[str] = queue.Queue()
        if enabled:
            threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in sys.stdin:
            line = line.strip()
            if line:
                try:
                    self.q.put(json.loads(line).get("cmd", ""))
                except json.JSONDecodeError:
                    self.q.put(line)

    def poll(self) -> list[str]:
        out = []
        while not self.q.empty():
            out.append(self.q.get_nowait())
        return out

    def wait(self) -> str:
        return self.q.get()


# -------------------------------------------------------------------- setup


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters: every node is frozen")
    name = cfg.get("name", "adamw").lower()
    lr = cfg.get("lr", 1e-3)
    wd = cfg.get("weight_decay", 0.0)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=tuple(cfg.get("betas", (0.9, 0.999))))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=cfg.get("momentum", 0.9))
    raise ValueError(f"unknown optimizer {name!r}")


def make_scheduler(opt: torch.optim.Optimizer, cfg: dict, total_steps: int):
    name = cfg.get("name", "constant")
    warmup = cfg.get("warmup_steps", 0)

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if name == "constant":
            return 1.0
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        if name == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        if name == "linear":
            return 1.0 - progress
        raise ValueError(f"unknown scheduler {name!r}")

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def enable_grad_checkpointing(model: torch.nn.Module) -> None:
    from torch.utils.checkpoint import checkpoint

    for m in model.modules():
        if m is model or not any(p.requires_grad for p in m.parameters(recurse=True)):
            continue
        if type(m).__module__ == "model" and not getattr(m, "_ws_ckpt", False):
            m.forward = functools.partial(checkpoint, m.forward, use_reentrant=False)
            m._ws_ckpt = True


def system_info(device: torch.device) -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "device": str(device)}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info |= {"gpu": p.name, "vram_mb": p.total_memory // 2**20, "cuda": torch.version.cuda}
    return info


def gpu_stats(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    out = {"gpu_mem_mb": torch.cuda.memory_allocated(device) / 2**20, "gpu_mem_peak_mb": torch.cuda.max_memory_allocated(device) / 2**20}
    try:
        out["gpu_util"] = float(torch.cuda.utilization(device))
    except Exception:
        pass
    return out


# -------------------------------------------------------------- checkpoints


def adapter_state(model: torch.nn.Module) -> dict:
    """Only the adaptation weights (LoRA A/B, bottleneck adapters)."""
    keys = ("lora_A", "lora_B", ".down.", ".up.")
    return {k: v for k, v in model.state_dict().items() if any(s in k for s in keys)}


def load_init(model: torch.nn.Module, path: Path) -> None:
    """Start from pretrained weights (e.g. imported from Hugging Face).

    Weights saved for the plain model also load into one with LoRA or adapters: a key
    that is missing as is is looked up without the wrapper's ".base." segment. Only
    adaptation parameters may stay at their initial values.
    """
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        sd = load_file(str(path))
        with safe_open(str(path), "pt") as f:
            for k, v in json.loads((f.metadata() or {}).get("weightsmith_aliases", "{}")).items():
                sd[k] = sd[v]
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    own = model.state_dict()
    mapped, used = {}, set()
    for k in own:
        src = k if k in sd else k.replace(".base.", ".")
        if src in sd:
            mapped[k] = sd[src]
            used.add(src)
    adapter = ("lora_A", "lora_B", ".down.", ".up.")
    missing = [k for k in own if k not in mapped and not any(a in k for a in adapter)]
    if missing:
        raise ValueError(f"init weights lack {len(missing)} tensors, e.g. {missing[:3]}")
    model.load_state_dict(mapped, strict=False)
    log(f"init: {len(mapped):,} tensors from {path.name}" + (f", {len(set(sd) - used)} unused" if set(sd) - used else ""))


@contextlib.contextmanager
def skip_init():
    """Build a model without filling its weights (they are about to be loaded)."""
    names = ["kaiming_uniform_", "kaiming_normal_", "uniform_", "normal_", "trunc_normal_", "xavier_uniform_", "xavier_normal_", "zeros_", "ones_"]
    saved = {n: getattr(torch.nn.init, n) for n in names}
    for n in names:
        setattr(torch.nn.init, n, lambda t, *a, **k: t)
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(torch.nn.init, n, f)


def cast_frozen(model: torch.nn.Module, dtype) -> None:
    """Store frozen matrices in a smaller dtype; trainable parameters and norms stay fp32."""
    for p in model.parameters():
        if not p.requires_grad and p.ndim >= 2:
            p.data = p.data.to(dtype)


@contextlib.contextmanager
def adapters_off(model: torch.nn.Module):
    """Within the block, every LoRA / adapter wrapper behaves like the layer it wraps."""
    mods = [m for m in model.modules() if hasattr(m, "enabled")]
    for m in mods:
        m.enabled = False
    try:
        yield
    finally:
        for m in mods:
            m.enabled = True


def last_logits(model, ids_list, cfg: dict) -> list[torch.Tensor]:
    """Next-token logits at the end of each prompt (fp32)."""
    lm = cfg["lm"]
    out = []
    for ids in ids_list:
        o = model(**{lm["input"]: ids[None].to(next(model.parameters()).device)})
        out.append(o[lm["logits"]][0, -1].float())
    return out


def jeffreys(pa: torch.Tensor, pb: torch.Tensor) -> float:
    a, b = torch.softmax(pa, -1), torch.softmax(pb, -1)
    return float((b * (b / a).log()).sum() + (a * (a / b).log()).sum())


def save_checkpoint(path: Path, model, opt, sched, scaler, step: int, epoch: int, batch_in_epoch: int, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "job_hash": cfg.get("job_hash"),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    adapters = adapter_state(model)
    if adapters:
        torch.save(adapters, path.with_name(path.stem + ".adapters.pt"))
    keep = cfg["train"].get("checkpoint", {}).get("keep", 2)
    ckpts = sorted(path.parent.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1].split(".")[0]))
    ckpts = [p for p in ckpts if not p.stem.endswith(".adapters")]
    for old in ckpts[:-keep] if keep else []:
        old.unlink(missing_ok=True)
        old.with_name(old.stem + ".adapters.pt").unlink(missing_ok=True)
    emit("checkpoint", step=step, path=str(path.relative_to(HERE)) if path.is_relative_to(HERE) else str(path))


def load_checkpoint(path: Path, model, opt, sched, scaler) -> tuple[int, int, int]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return state["step"], state["epoch"], state["batch_in_epoch"]


# ------------------------------------------------------------------- errors


def node_for_traceback(tb) -> str | None:
    """Map the innermost model.py frame back to a graph node id."""
    try:
        line_map = json.loads((HERE / "model.map.json").read_text())
    except OSError:
        return None
    node = None
    for frame in traceback.extract_tb(tb):
        if Path(frame.filename).name == "model.py":
            node = line_map.get(str(frame.lineno), node)
    return node


def classify(exc: BaseException) -> dict:
    msg = str(exc)
    if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in msg.lower():
        return {
            "class": "oom",
            "message": msg.splitlines()[0],
            "suggestions": ["halve batch_size", "enable grad_checkpointing", "use precision bf16", "freeze more nodes or use LoRA"],
        }
    shape_words = ("shape", "size mismatch", "mat1 and mat2", "dimension", "expected input")
    if isinstance(exc, (RuntimeError, ValueError, IndexError)) and any(w in msg for w in shape_words):
        return {"class": "shape", "message": msg.splitlines()[0]}
    return {"class": "runtime", "message": msg.splitlines()[0] if msg else type(exc).__name__}


# --------------------------------------------------------------------- loop


@torch.no_grad()
def evaluate(model, loader, device, amp, metrics: dict) -> dict:
    model.eval()
    total, n, correct, count = 0.0, 0, 0, 0
    acc = metrics.get("accuracy")
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with amp():
            out = model(**batch)
        bs = next(iter(batch.values())).shape[0]
        total += out["loss"].float().item() * bs
        n += bs
        if acc:
            pred = out[acc["logits"]].argmax(-1)
            target = batch[acc["target"]]
            keep = target != -100  # prompt and padding positions are not scored
            correct += ((pred == target) & keep).sum().item()
            count += keep.sum().item()
    model.train()
    res = {"val_loss": total / max(n, 1)}
    if acc and count:
        res["val_accuracy"] = correct / count
    return res


def main(argv: list[str] | None = None) -> int:
    global EVENTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    ap.add_argument("--events", action="store_true", help="speak the Weightsmith job protocol")
    ap.add_argument("--out", default=str(HERE), help="where checkpoints and weights go")
    args = ap.parse_args(argv)
    EVENTS = args.events
    cfg = yaml.safe_load(Path(args.config).read_text())
    tc = cfg["train"]
    out_dir = Path(args.out)
    ckpt_dir = out_dir / "checkpoints"
    control = Control(args.events)

    seed_everything(tc.get("seed", 0))
    device = pick_device(tc.get("device", "auto"))
    precision = tc.get("precision", "f32")
    if device.type != "cuda" and precision == "f16":
        precision = "f32"
    amp_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}.get(precision)
    amp = (lambda: torch.autocast(device.type, dtype=amp_dtype)) if amp_dtype else (lambda: torch.autocast(device.type, enabled=False))
    scaler = torch.amp.GradScaler(device.type) if precision == "f16" else None

    emit("status", state="preparing", system=system_info(device))
    train_ds, val_ds = data.build(cfg)
    bs = tc.get("batch_size", 64)
    workers = tc.get("num_workers", 0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=workers) if len(val_ds) else None

    init = cfg.get("init") or {}
    if init.get("weights"):
        t_init = time.perf_counter()
        with skip_init():  # every base tensor is overwritten by the checkpoint
            model = Model()
        load_init(model, Path(init["weights"]).expanduser())
        for m in model.modules():  # adapters are not in the checkpoint: initialise them properly
            if hasattr(m, "reset_adapter"):
                m.reset_adapter()
        log(f"init: model built and loaded in {time.perf_counter() - t_init:.1f}s")
    else:
        model = Model()
    small = {"bf16": torch.bfloat16, "f16": torch.float16}.get(tc.get("weights_dtype") or "")
    if small is not None:
        cast_frozen(model, small)
    model = model.to(device)
    if tc.get("grad_checkpointing"):
        enable_grad_checkpointing(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model: {n_params:,} parameters, {n_train:,} trainable · {len(train_ds):,} train / {len(val_ds):,} val samples · {device}")
    if tc.get("compile"):
        model = torch.compile(model)

    steps_per_epoch = math.ceil(len(train_ds) / bs) if not tc.get("drop_last") else len(train_ds) // bs
    epochs = tc.get("epochs", 1)
    total_steps = tc.get("max_steps") or steps_per_epoch * epochs
    opt = make_optimizer(model, tc.get("optimizer", {}))
    sched = make_scheduler(opt, tc.get("scheduler", {}), total_steps)

    step, start_epoch, skip = 0, 0, 0
    if args.resume:
        step, start_epoch, skip = load_checkpoint(Path(args.resume), model, opt, sched, scaler)
        log(f"resumed from {args.resume} at step {step}")

    dc = tc.get("diagnostics") or {}
    scope = None
    if TrainScope is not None and dc.get("trainscope"):
        try:  # diagnostics never stop training
            scope_cfg = load_config({"profile": dc.get("profile"), "run_dir": str(out_dir), "run_name": "scope",
                                     "resume": bool(args.resume)})
            # what run comparison diffs: this job's training settings and graph, not the scope's own config
            job = {"train": {k: v for k, v in tc.items() if k != "diagnostics"}, "graph": cfg.get("graph_hash"), "data": cfg["data"].get("source")}
            # the uncompiled module, so layer names stay the graph's attribute names
            scope = TrainScope(getattr(model, "_orig_mod", model), opt, scope_cfg, scaler=scaler, model_config=job).attach()
        except Exception as exc:
            log(f"trainscope: recording off ({type(exc).__name__}: {exc})")

    log_every = tc.get("log_every", 10)
    eval_every = tc.get("eval_every")
    ckpt_every = tc.get("checkpoint", {}).get("every_steps")
    clip = tc.get("grad_clip")
    metrics_cfg = cfg.get("metrics", {})

    # KL to the base model on fixed anchor prompts (Cerata): the base is this model with adapters off
    klc = tc.get("kl_to_base") or {}
    kl_ids, kl_base = [], []
    if klc.get("lambda") and klc.get("prompts"):
        tok = data.tokenizer(cfg)
        kl_ids = [data.chat_prompt_ids(tok, q, klc.get("system")) for q in klc["prompts"]]
        with torch.no_grad(), adapters_off(model), amp():
            model.eval()
            kl_base = last_logits(model, kl_ids, cfg)
            model.train()
        log(f"kl_to_base: λ={klc['lambda']} on {len(kl_ids)} anchor prompts")

    emit("status", state="running", step=step, total_steps=total_steps, params=n_params, trainable=n_train)
    model.train()
    t_last, samples = time.perf_counter(), 0
    state, epoch, i, last_eval = "completed", start_epoch, 0, None
    try:
        for epoch in range(start_epoch, epochs if not tc.get("max_steps") else 10**9):
            g = torch.Generator().manual_seed(tc.get("seed", 0) + epoch)  # same order on resume
            loader = DataLoader(train_ds, batch_size=bs, shuffle=tc.get("shuffle", True), generator=g, num_workers=workers,
                                drop_last=bool(tc.get("drop_last")))
            for i, batch in enumerate(loader):
                if epoch == start_epoch and i < skip:
                    continue
                for cmd in control.poll():
                    if cmd == "checkpoint":
                        save_checkpoint(ckpt_dir / f"step_{step}.pt", model, opt, sched, scaler, step, epoch, i, cfg)
                    elif cmd == "pause":
                        emit("status", state="paused", step=step)
                        while (c := control.wait()) not in ("resume", "stop"):
                            if c == "checkpoint":
                                save_checkpoint(ckpt_dir / f"step_{step}.pt", model, opt, sched, scaler, step, epoch, i, cfg)
                        if c == "stop":
                            raise KeyboardInterrupt
                        emit("status", state="running", step=step, total_steps=total_steps)
                        t_last, samples = time.perf_counter(), 0
                    elif cmd == "stop":
                        raise KeyboardInterrupt

                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with amp():
                    out = model(**batch)
                loss = out["loss"]
                if kl_base:
                    kl = 0.0
                    with amp():
                        anchored = last_logits(model, kl_ids, cfg)
                    for lg, base in zip(anchored, kl_base):
                        pb = torch.softmax(base, -1)
                        kl = kl + (pb * (pb.log() - torch.log_softmax(lg, -1))).sum()
                    kl = kl / len(kl_base)
                    out["kl_to_base"] = kl.detach()
                    loss = loss + klc["lambda"] * kl
                if not torch.isfinite(loss):
                    emit("error", **{"class": "nan", "message": f"loss is {loss.item()} at step {step}", "step": step,
                                     "suggestions": ["lower lr", "enable grad_clip", "use precision f32 or bf16"]})
                    state = "failed"
                    return 2
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                else:
                    loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip or float("inf"))
                spike = None
                if scope:
                    try:
                        spike = scope.step(loss.item(), batch_index=i, grad_norm_after_clip=float(grad_norm))
                    except Exception as exc:
                        log(f"trainscope: recording off at step {step} ({type(exc).__name__}: {exc})")
                        scope = None
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sched.step()
                step += 1
                if spike:
                    emit("spike", step=step, score=float(spike["spike_score"]))
                samples += next(iter(batch.values())).shape[0]

                if step % log_every == 0 or step == 1 or step >= total_steps:
                    now = time.perf_counter()
                    extra = {k: v.item() for k, v in out.items() if k != "loss" and torch.is_tensor(v) and v.ndim == 0}
                    emit(
                        "metric",
                        step=step,
                        epoch=epoch,
                        loss=loss.item(),
                        lr=sched.get_last_lr()[0],
                        grad_norm=float(grad_norm),
                        throughput=samples / max(now - t_last, 1e-9),
                        **{f"loss/{k}": v for k, v in extra.items()},
                        **gpu_stats(device),
                    )
                    t_last, samples = now, 0
                if eval_every and val_loader and step % eval_every == 0:
                    emit("eval", step=step, **evaluate(model, val_loader, device, amp, metrics_cfg))
                    last_eval = step
                if ckpt_every and step % ckpt_every == 0:
                    save_checkpoint(ckpt_dir / f"step_{step}.pt", model, opt, sched, scaler, step, epoch, i + 1, cfg)
                if step >= total_steps:
                    break
            skip = 0
            if step >= total_steps:
                break
    except KeyboardInterrupt:
        state = "stopped"
        save_checkpoint(ckpt_dir / f"step_{step}.pt", model, opt, sched, scaler, step, epoch, i, cfg)
    except Exception as exc:  # classified for the canvas
        info = classify(exc)
        info["node"] = node_for_traceback(exc.__traceback__)
        info["traceback"] = traceback.format_exc()
        emit("error", step=step, **info)
        if not EVENTS:
            raise
        return 1
    finally:
        if scope:
            try:
                scope.detach()
            except Exception as exc:
                log(f"trainscope: could not close the recording ({type(exc).__name__}: {exc})")

    if val_loader and state == "completed" and last_eval != step:
        emit("eval", step=step, **evaluate(model, val_loader, device, amp, metrics_cfg))
    raw = getattr(model, "_orig_mod", model)
    gate = tc.get("gate") or {}
    if state == "completed" and gate.get("prompts"):
        # divergence gate: held-out prompts, adapted vs base, Jeffreys divergence of the next-token distribution
        tok = data.tokenizer(cfg)
        ids = [data.chat_prompt_ids(tok, q, gate.get("system")) for q in gate["prompts"]]
        raw.eval()
        with torch.no_grad(), amp():
            adapted = last_logits(raw, ids, cfg)
            with adapters_off(raw):
                base = last_logits(raw, ids, cfg)
        ds = [jeffreys(a, b) for a, b in zip(adapted, base)]
        mean = sum(ds) / len(ds)
        passed = mean <= gate.get("kappa", float("inf"))
        emit("gate", mean=mean, max=max(ds), kappa=gate.get("kappa"), passed=passed, prompts=len(ds))
        if not passed:
            state = "rejected"
    torch.save(raw.state_dict(), out_dir / "model.pt")
    if adapter_state(raw):
        torch.save(adapter_state(raw), out_dir / "adapters.pt")
    emit("status", state=state, step=step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
