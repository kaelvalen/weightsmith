"""Weightsmith daemon: HTTP + WebSocket API over the core, codegen and job manager.

The canvas only talks to this API. A remote GPU box runs the same daemon behind
an SSH tunnel; nothing in the protocol assumes a local path.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask

from weightsmith_codegen import project as codegen
from weightsmith_codegen.emit import CodegenError, compile_graph
from weightsmith_core import diff as ir_diff
from weightsmith_core import ir, pynode, registry, validate

from .jobs import JobManager, export_zip
from .store import Store

PROTOCOL = 1


class GraphBody(BaseModel):
    graph: dict[str, Any]


class DiffBody(BaseModel):
    a: dict[str, Any]
    b: dict[str, Any]


class RunBody(BaseModel):
    graph: dict[str, Any]
    data: dict[str, Any] = {}
    train: dict[str, Any] = {}
    init: dict[str, Any] = {}  # {"weights": path} to start from pretrained weights
    project: str | None = None
    variant: str | None = None
    name: str | None = None
    rerun: bool = False


class ControlBody(BaseModel):
    cmd: str


class AblateBody(BaseModel):
    nodes: list[str] = []
    adapt: list[dict[str, Any]] = []


class ImportBody(BaseModel):
    source: str  # Hub repo id or local directory
    dtype: str | None = "bf16"
    name: str | None = None


class GenerateBody(BaseModel):
    prompt: str
    system: str | None = None
    adapters: bool = True
    max_new_tokens: int = 32


class RunPatch(BaseModel):
    note: str | None = None
    name: str | None = None


class PythonBody(BaseModel):
    params: dict[str, Any]
    inputs: dict[str, Any] | None = None


def _thumb(g: dict[str, Any]) -> list[list[float]]:
    """Node positions for a project card: the saved layout, else a simple layered one."""
    lay = g.get("layout") or {}
    nodes = g.get("nodes", [])
    if nodes and all(n["id"] in lay for n in nodes):
        return [[lay[n["id"]]["x"], lay[n["id"]]["y"]] for n in nodes][:400]
    preds: dict[str, list[str]] = {n["id"]: [] for n in nodes}
    for src, dst in g.get("edges", []):
        s, d = src.rsplit(".", 1)[0], dst.rsplit(".", 1)[0]
        if s in preds and d in preds:
            preds[d].append(s)
    depth: dict[str, int] = {}

    def level(nid: str, seen: frozenset = frozenset()) -> int:
        if nid not in depth:
            depth[nid] = 1 + max((level(p, seen | {nid}) for p in preds[nid] if p not in seen), default=-1)
        return depth[nid]

    rows: dict[int, int] = {}
    out = []
    for n in nodes:
        lv = level(n["id"])
        out.append([rows.get(lv, 0) * 216, lv * 104])
        rows[lv] = rows.get(lv, 0) + 1
    return out[:400]


def create_app(root: Path | None = None, web_dist: Path | None = None) -> FastAPI:
    store = Store(root)
    bridge = pynode.SandboxBridge()
    jobs: JobManager | None = None

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal jobs
        jobs = JobManager(store, bridge)
        jobs.start()
        app.state.jobs = jobs
        yield
        await jobs.shutdown()
        bridge.close()

    app = FastAPI(title="Weightsmith daemon", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def jm() -> JobManager:
        assert jobs is not None
        return jobs

    def analyze(graph: dict[str, Any]) -> validate.Analysis:
        try:
            g = ir.migrate(graph)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return validate.analyze(g, bridge)

    # ------------------------------------------------------------- graph

    @app.get("/api/health")
    def health():
        return {"ok": True, "protocol": PROTOCOL, "schema": ir.SCHEMA_VERSION, "workspace": str(store.root)}

    @app.get("/api/catalog")
    def catalog():
        return {"schema": ir.SCHEMA_VERSION, "types": registry.catalog()}

    @app.post("/api/validate")
    def validate_graph(body: GraphBody):
        a = analyze(body.graph)
        out = a.to_json()
        out["hash"] = ir.content_hash(body.graph) if not any(i.code == "schema" for i in a.issues) else None
        return out

    @app.post("/api/compile")
    def compile_endpoint(body: GraphBody):
        a = analyze(body.graph)
        if not a.ok:
            return {"ok": False, "analysis": a.to_json()}
        try:
            c = compile_graph(body.graph, a, bridge)
        except CodegenError as e:
            return {"ok": False, "analysis": e.analysis.to_json()}
        with tempfile.TemporaryDirectory() as tmp:
            try:
                smoke = codegen.smoke_test(c.model_py, c, ir.migrate(body.graph), Path(tmp) / "smoke")
            except Exception as e:  # generated code failed on the meta device
                smoke = {"ok": False, "issues": [{"node": None, "message": f"{type(e).__name__}: {e}"}]}
        return {"ok": True, "model_py": c.model_py, "line_map": c.line_map, "hash": c.graph_hash, "smoke": smoke, "analysis": a.to_json()}

    @app.post("/api/diff")
    def diff_endpoint(body: DiffBody):
        return ir_diff.diff(body.a, body.b)

    @app.post("/api/python/analyze")
    def python_analyze(body: PythonBody):
        return bridge.analyze(body.params, body.inputs)

    # ---------------------------------------------------------- projects

    @app.get("/api/projects")
    def list_projects():
        """Project cards: description, the active variant's size and a thumbnail of its layout."""
        out = []
        runs_by_project: dict[str, int] = {}
        for r in store.list_runs():
            runs_by_project[r.get("project") or ""] = runs_by_project.get(r.get("project") or "", 0) + 1
        for meta in store.list_projects():
            p = store.load_project(meta["slug"]) or {}
            card = {**meta, "description": p.get("description"), "active": p.get("active"), "runs": runs_by_project.get(meta["slug"], 0)}
            v = (p.get("variants") or {}).get(p.get("active")) or next(iter((p.get("variants") or {}).values()), None)
            if v:
                g = v["graph"]
                try:
                    t = validate.analyze(ir.migrate(g)).totals
                    card |= {"params": t.total, "trainable": t.trainable}
                except Exception:
                    pass
                card["nodes"] = len(g.get("nodes", []))
                card["thumb"] = _thumb(g)
            out.append(card)
        return out

    @app.get("/api/projects/{slug}")
    def get_project(slug: str):
        p = store.load_project(slug)
        if p is None:
            raise HTTPException(404, "no such project")
        return p

    @app.put("/api/projects/{slug}")
    def put_project(slug: str, project: dict[str, Any]):
        project.setdefault("name", slug)
        project["slug"] = slug
        for v in project.get("variants", {}).values():
            v["graph"] = ir.migrate(v["graph"])
        return {"slug": store.save_project(project)}

    @app.post("/api/import/hf")
    async def import_hf(body: ImportBody):
        """Download (if needed) and convert a Hugging Face model into a new project."""
        from .importing import import_to_project
        from weightsmith_importers.hf import ImportError_

        try:
            return await asyncio.to_thread(import_to_project, store, body.source, body.dtype, body.name)
        except ImportError_ as e:
            raise HTTPException(422, str(e))
        except Exception as e:  # download or conversion failure, shown to the user as is
            raise HTTPException(502, f"{type(e).__name__}: {e}")

    @app.delete("/api/projects/{slug}")
    def delete_project(slug: str):
        store.delete_project(slug)
        return {"ok": True}

    # -------------------------------------------------------------- runs

    @app.get("/api/runs")
    def list_runs(project: str | None = None):
        return store.list_runs(project)

    @app.post("/api/runs")
    async def create_run(body: RunBody):
        a = analyze(body.graph)
        if not a.ok:
            raise HTTPException(422, {"message": "graph does not validate", "analysis": a.to_json()})
        if not a.losses:
            raise HTTPException(422, {"message": "graph has no active loss node", "analysis": a.to_json()})
        try:
            return await jm().submit(
                {"graph": body.graph, "data": body.data, "train": body.train, "init": body.init},
                body.project, body.variant, body.name, rerun=body.rerun,
            )
        except (RuntimeError, ValueError, KeyError, OSError) as e:
            raise HTTPException(409, str(e))

    def _run_or_404(rid: str) -> dict[str, Any]:
        rec = store.get_run(rid)
        if rec is None:
            raise HTTPException(404, "no such run")
        return rec

    @app.get("/api/runs/{rid}")
    def get_run(rid: str):
        rec = _run_or_404(rid)
        job_path = store.run_dir(rid) / "job.json"
        rec["job"] = json.loads(job_path.read_text()) if job_path.exists() else None
        rec["children"] = [r["id"] for r in store.list_runs(rec.get("project")) if r.get("parent") == rid]
        return rec

    @app.patch("/api/runs/{rid}")
    def patch_run(rid: str, body: RunPatch):
        """A run's note and name: what was tried, and why."""
        _run_or_404(rid)
        if body.note is not None:
            store.update_run(rid, summary={"note": body.note})
        if body.name is not None:
            store.rename_run(rid, body.name)
        return store.get_run(rid)

    @app.delete("/api/runs/{rid}")
    async def delete_run(rid: str):
        rec = _run_or_404(rid)
        if rec["status"] in ("running", "paused", "preparing"):
            raise HTTPException(409, "stop the run first")
        store.delete_run(rid)
        return {"ok": True}

    @app.get("/api/runs/{rid}/events")
    def run_events(rid: str, since: int = 0, types: str | None = None):
        _run_or_404(rid)
        return store.events(rid, since, set(types.split(",")) if types else None)

    @app.post("/api/runs/{rid}/control")
    async def control(rid: str, body: ControlBody):
        try:
            return await jm().control(rid, body.cmd)
        except KeyError:
            raise HTTPException(404, "no such run")
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/runs/{rid}/ablate")
    async def ablate(rid: str, body: AblateBody):
        _run_or_404(rid)
        if not body.nodes and not body.adapt:
            raise HTTPException(400, "nothing to disable")
        try:
            return await jm().ablate(rid, body.nodes, body.adapt)
        except KeyError as e:
            raise HTTPException(404, f"no such node {e}")
        except CodegenError as e:
            raise HTTPException(422, {"message": str(e), "analysis": e.analysis.to_json()})

    @app.post("/api/runs/{rid}/generate")
    async def generate(rid: str, body: GenerateBody):
        """Greedy answer from a trained language model, adapters on or off."""
        from .playground import generate as gen

        rec = _run_or_404(rid)
        if rec["status"] not in ("completed", "rejected", "stopped"):
            raise HTTPException(409, "the run has no final weights yet")
        if not (store.run_dir(rid) / "model.pt").exists():
            raise HTTPException(404, "this run saved no weights")
        try:
            return await asyncio.to_thread(gen, store.run_dir(rid), body.prompt, body.system, body.adapters, min(body.max_new_tokens, 256))
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/api/runs/{rid}/files/{name}")
    def run_file(rid: str, name: str):
        _run_or_404(rid)
        allowed = {"model.py", "train.py", "data.py", "config.yaml", "graph.json", "job.json", "README.md"}
        if name not in allowed:
            raise HTTPException(404)
        return PlainTextResponse((store.run_dir(rid) / name).read_text())

    @app.get("/api/runs/{rid}/export")
    async def export(rid: str, merge: bool = False):
        _run_or_404(rid)
        tmp = Path(tempfile.mkdtemp())
        path = await asyncio.to_thread(export_zip, store, rid, tmp / f"weightsmith-{rid}.zip", merge)
        import shutil

        return FileResponse(path, filename=path.name, background=BackgroundTask(shutil.rmtree, tmp, True))

    # --------------------------------------------------------- websockets

    @app.websocket("/ws/runs/{rid}")
    async def ws_run(ws: WebSocket, rid: str, since: int = Query(0)):
        await ws.accept()
        if store.get_run(rid) is None:
            await ws.close(code=4404)
            return
        q = jm().subscribe(rid)
        try:
            last = since
            for ev in store.events(rid, since):  # replay what the client missed
                await ws.send_json(ev)
                last = ev["seq"]
            while True:
                ev = await q.get()
                if ev["seq"] > last:
                    await ws.send_json(ev)
                    last = ev["seq"]
        except WebSocketDisconnect:
            pass
        finally:
            jm().unsubscribe(rid, q)

    @app.websocket("/ws/status")
    async def ws_status(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        jm().status_subs.add(q)
        try:
            while True:
                await ws.send_json(await q.get())
        except WebSocketDisconnect:
            pass
        finally:
            jm().status_subs.discard(q)

    # ------------------------------------------------------------ the canvas

    dist = web_dist or Path(__file__).resolve().parents[2] / "web" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")

    return app


app = None


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app
