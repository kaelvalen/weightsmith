"""weightsmith CLI.

    weightsmith serve                         daemon + canvas on http://127.0.0.1:7860
    weightsmith validate graph.json
    weightsmith compile graph.json -o out/    model.py, train.py, data.py, config.yaml
    weightsmith train job.json                train through the experiment store, stream metrics
    weightsmith runs                          list runs
    weightsmith diff a.json b.json
    weightsmith export <run id> [--merge]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _load_job(path: Path) -> dict:
    """A job file, or a bare graph (then data/train take defaults). {"$ref": "x.json"} pulls a graph from a file."""
    from weightsmith_core import ir

    d = json.loads(path.read_text())
    if "nodes" in d:
        return {"graph": ir.migrate(d)}
    g = d.get("graph", {})
    if "$ref" in g:
        d["graph"] = ir.load(path.parent / g["$ref"])
    return d


def cmd_validate(args) -> int:
    from weightsmith_core import ir, pynode, validate

    a = validate.analyze(ir.load(args.graph), pynode.default_bridge())
    for i in a.issues:
        if i.severity == "peer":
            continue
        where = ".".join(x for x in (i.node, i.port) if x) or "graph"
        print(f"{i.severity:7} {where:24} {i.message}")
    t = a.totals
    print(f"{'ok' if a.ok else 'invalid'} · {t.total:,} parameters, {t.trainable:,} trainable · hash {ir.content_hash(ir.load(args.graph))[:12]}")
    return 0 if a.ok else 1


def cmd_compile(args) -> int:
    from weightsmith_codegen.emit import CodegenError
    from weightsmith_codegen.project import write_run
    from weightsmith_core import pynode

    job = _load_job(Path(args.graph))
    try:
        write_run(job, args.out, python=pynode.default_bridge())
    except CodegenError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"wrote {args.out}/model.py, train.py, data.py, config.yaml")
    return 0


def _fmt_metric(ev: dict) -> str:
    keys = ("step", "loss", "lr", "grad_norm", "throughput")
    return "  ".join(f"{k} {ev[k]:.4g}" if isinstance(ev[k], float) else f"{k} {ev[k]}" for k in keys if k in ev)


def cmd_train(args) -> int:
    from weightsmith_core import pynode

    from .jobs import JobManager
    from .store import Store

    job = _load_job(Path(args.job))
    for kv in args.set or []:
        k, _, v = kv.partition("=")
        node = job.setdefault("train", {})
        *parents, leaf = k.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        try:
            node[leaf] = json.loads(v)
        except json.JSONDecodeError:
            node[leaf] = v

    async def go() -> int:
        jm = JobManager(Store(Path(args.workspace) if args.workspace else None), pynode.default_bridge())
        jm.start()
        rec = await jm.submit(job, project=args.project, name=args.name, rerun=args.rerun)
        rid = rec["id"]
        if rec.get("existing") and rec["status"] == "completed":
            print(f"run {rid} already completed with this graph, config, data and seed (use --rerun)")
            return 0
        print(f"run {rid} · {jm.store.run_dir(rid)}")
        q = jm.subscribe(rid)
        last_state = None
        try:
            while True:
                ev = await q.get()
                t = ev["type"]
                if t == "metric":
                    print(_fmt_metric(ev))
                elif t == "eval":
                    print("eval   " + "  ".join(f"{k} {v:.4g}" for k, v in ev.items() if k.startswith("val_")))
                elif t == "log":
                    print(ev["message"])
                elif t == "error":
                    print(f"error  [{ev.get('class')}] {ev.get('message')}" + (f" (node {ev['node']})" if ev.get("node") else ""))
                    for s in ev.get("suggestions", []):
                        print(f"       → {s}")
                elif t == "status" and ev["state"] != last_state:
                    last_state = ev["state"]
                    print(f"status {ev['state']}")
                    if ev["state"] in ("completed", "stopped", "failed"):
                        return 0 if ev["state"] == "completed" else 1
        except (KeyboardInterrupt, asyncio.CancelledError):
            await jm.control(rid, "stop")
            return 130
        finally:
            await jm.shutdown()

    return asyncio.run(go())


def cmd_runs(args) -> int:
    from .store import Store

    for r in Store(Path(args.workspace) if args.workspace else None).list_runs(args.project):
        s = r["summary"]
        last = s.get("last", {})
        print(f"{r['id']}  {r['status']:11} {str(r.get('name') or '')[:32]:32} step {last.get('step', '-'):>6}  loss {last.get('loss', float('nan')):.4f}  val {s.get('best_val_loss', float('nan')):.4f}")
    return 0


def cmd_diff(args) -> int:
    from weightsmith_core import diff, ir

    d = diff.diff(ir.load(args.a), ir.load(args.b))
    if d["same"]:
        print("same graph")
    for c in d["changes"]:
        print(json.dumps(c))
    return 0


def cmd_export(args) -> int:
    from .jobs import export_zip
    from .store import Store

    store = Store(Path(args.workspace) if args.workspace else None)
    if store.get_run(args.run) is None:
        print(f"no run {args.run}", file=sys.stderr)
        return 1
    dest = Path(args.out or f"weightsmith-{args.run}.zip").resolve()
    print(export_zip(store, args.run, dest, merge=args.merge))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .server import create_app

    app = create_app(Path(args.workspace) if args.workspace else None)
    print(f"weightsmith daemon on http://{args.host}:{args.port}  (workspace {app.state.store.root})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="weightsmith", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", help="experiment store (default ~/.weightsmith or $WEIGHTSMITH_HOME)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("validate")
    p.add_argument("graph")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("compile")
    p.add_argument("graph", help="graph or job JSON")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(fn=cmd_compile)

    p = sub.add_parser("train")
    p.add_argument("job", help="job JSON (graph + data + train), or a bare graph")
    p.add_argument("--project")
    p.add_argument("--name")
    p.add_argument("--rerun", action="store_true", help="discard an existing run with the same id")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override train config, e.g. --set optimizer.lr=3e-4")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("runs")
    p.add_argument("--project")
    p.set_defaults(fn=cmd_runs)

    p = sub.add_parser("diff")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("export")
    p.add_argument("run")
    p.add_argument("-o", "--out")
    p.add_argument("--merge", action="store_true", help="fold LoRA into the base weights")
    p.set_defaults(fn=cmd_export)

    args = ap.parse_args(argv)
    # --workspace may be given before or after the subcommand
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
