"""Job spec → run folder (model.py, train.py, data.py, config.yaml) and the meta-device smoke test."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from weightsmith_core import ir
from weightsmith_core.registry import expand_defaults
from weightsmith_core.shapes import is_symbol

from .emit import Compiled, compile_graph, py_name

DEFAULT_TRAIN: dict[str, Any] = {
    "seed": 0,
    "epochs": 1,
    "max_steps": None,
    "batch_size": 64,
    "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.01},
    "scheduler": {"name": "cosine", "warmup_steps": 0},
    # bf16 autocast: the default for the primary test target (RTX 5060 Laptop, 8 GB). See docs/decisions.md.
    "precision": "bf16",
    "grad_clip": 1.0,
    "grad_checkpointing": False,
    "compile": False,
    "device": "auto",
    "log_every": 10,
    "eval_every": None,
    "checkpoint": {"every_steps": None, "keep": 2},
    # trainscope recording under <run>/scope/; not part of the run's identity
    "diagnostics": {"trainscope": True, "profile": "weightsmith"},
}

DEFAULT_DATA: dict[str, Any] = {"source": "synthetic", "split": {"val": 0.1, "seed": 0}, "preprocess": []}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def normalize_job(job: dict[str, Any]) -> dict[str, Any]:
    """Job spec = graph + data + train, with every default made explicit (so the hash is stable)."""
    return {
        "graph": ir.migrate(job["graph"]),
        "data": _merge(DEFAULT_DATA, job.get("data", {})),
        "train": _merge(DEFAULT_TRAIN, job.get("train", {})),
        "init": dict(job.get("init") or {}),
    }


def file_sha256(path: str | Path) -> str:
    """Content hash of a (possibly multi-GB) weights file, cached next to it by size and mtime."""
    import hashlib

    p = Path(path).expanduser()
    st = p.stat()
    side = p.with_name(f".{p.name}.sha256.json")
    try:
        c = json.loads(side.read_text())
        if c["size"] == st.st_size and c["mtime"] == st.st_mtime:
            return c["sha256"]
    except (OSError, ValueError, KeyError):
        pass
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    try:
        side.write_text(json.dumps({"size": st.st_size, "mtime": st.st_mtime, "sha256": h.hexdigest()}))
    except OSError:
        pass
    return h.hexdigest()


def job_hashes(job: dict[str, Any]) -> dict[str, str]:
    job = normalize_job(job)
    init = dict(job["init"])
    if init.get("weights"):
        init["sha256"] = file_sha256(init["weights"])
        init.pop("weights")  # the content identifies the run, not where the file lives
    return {
        "graph": ir.content_hash(job["graph"]),
        "config": ir.hash_obj({**{k: v for k, v in job["train"].items() if k not in ("seed", "diagnostics")}, "init": init}),
        "data_spec": ir.hash_obj(job["data"]),
        "seed": str(job["train"]["seed"]),
    }


def _template(name: str) -> str:
    return (resources.files("weightsmith_codegen") / "templates" / name).read_text()


def _config(job: dict[str, Any], compiled: Compiled) -> dict[str, Any]:
    graph = job["graph"]
    nodes = ir.node_index(graph)
    data = copy.deepcopy(job["data"])
    inputs = {}
    for field, nid in compiled.inputs.items():
        p = expand_defaults("Input", nodes[nid].get("params", {}))
        inputs[field] = {"shape": list(p["shape"][1:]), "dtype": p["dtype"]}
    data["inputs"] = inputs
    data["targets"] = [t for t in compiled.targets if t not in inputs]
    metrics: dict[str, Any] = {}
    for lid in compiled.analysis.losses:
        n = nodes[lid]
        if n["type"] == "CrossEntropy":
            src = next((e[0] for e in graph["edges"] if e[1] == f"{lid}.logits"), None)
            tgt = next((e[0] for e in graph["edges"] if e[1] == f"{lid}.target"), None)
            target_field = expand_defaults("CrossEntropy", n.get("params", {}))["target_field"]
            if tgt is not None:
                tn = nodes[ir.split_ref(tgt)[0]]
                target_field = expand_defaults("Input", tn.get("params", {}))["field"] if tn["type"] == "Input" else None
            if src and target_field:
                metrics["accuracy"] = {"logits": ir.split_ref(src)[0], "target": target_field}
                break
    lm = {}
    if metrics.get("accuracy"):
        lm = {"logits": metrics["accuracy"]["logits"], "input": next(iter(compiled.inputs), "x")}
    return {"graph_hash": compiled.graph_hash, "data": data, "train": job["train"], "metrics": metrics, "init": job.get("init") or {}, "lm": lm}


def write_run(job: dict[str, Any], out_dir: str | Path, job_hash: str | None = None, python=None) -> Compiled:
    """Emit a self-contained, re-runnable folder: python train.py works without Weightsmith."""
    job = normalize_job(job)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    compiled = compile_graph(job["graph"], python=python)
    cfg = _config(job, compiled)
    if job_hash:
        cfg["job_hash"] = job_hash
    (out / "model.py").write_text(compiled.model_py)
    (out / "model.map.json").write_text(json.dumps({str(k): v for k, v in compiled.line_map.items()}))
    # module attribute -> node id: trainscope layer names start with the attribute
    (out / "module_map.json").write_text(json.dumps({py_name(n["id"]): n["id"] for n in job["graph"]["nodes"]}))
    (out / "train.py").write_text(_template("train.py"))
    (out / "data.py").write_text(_template("data.py"))
    header = f"# Weightsmith job config. graph {compiled.graph_hash[:12]}" + (f" · job {job_hash[:12]}" if job_hash else "") + "\n"
    (out / "config.yaml").write_text(header + yaml.safe_dump(cfg, sort_keys=False))
    (out / "graph.json").write_text(json.dumps(job["graph"], indent=2))
    (out / "README.md").write_text(
        f"# {job['graph'].get('name') or 'graph'}\n\n"
        "Exported by Weightsmith. Re-run without the canvas:\n\n"
        "    pip install torch numpy pyyaml\n    python train.py\n\n"
        "- `model.py`: the model, one `# node:` comment per graph node\n"
        "- `scope/`: trainscope recording of the run (`pip install trainscope`, then `trainscope ui --run scope`)\n"
        "- `config.yaml`: data, optimiser, schedule, precision, seed\n"
        "- `graph.json`: the graph IR this code was generated from\n"
    )
    return compiled


# ------------------------------------------------------------- smoke test


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def concrete(shape: list, probe: dict[str, int]) -> list[int]:
    out = []
    for d in shape:
        if isinstance(d, int):
            out.append(d)
        elif is_symbol(d):
            out.append(probe.setdefault(d, 2 if d == "B" else 8))
        else:  # derived: "T+1", "2*B", "H/2"
            import re

            for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", d):
                probe.setdefault(name, 2 if name == "B" else 8)
            try:
                v = ir.eval_arg(d, probe)
            except ir.ArgError:
                raise ValueError(f"cannot probe derived dimension {d}") from None
            out.append(int(v))
    return out


def smoke_test(model_py: str, compiled: Compiled, graph: ir.Graph, workdir: Path) -> dict[str, Any]:
    """Instantiate on the meta device and run one forward pass with fake tensors.

    Checks that the generated code runs and that output shapes and parameter
    counts match what the analyzer predicted. Nothing is allocated.
    """
    import torch

    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "model.py"
    path.write_text(model_py)
    mod = _load_module(path, f"ws_smoke_{compiled.graph_hash[:8]}")
    nodes = ir.node_index(graph)
    probe: dict[str, int] = {}
    result: dict[str, Any] = {"ok": True, "issues": [], "params": {}}
    with torch.device("meta"):
        model = mod.Model()
        kwargs = {}
        for field, nid in compiled.inputs.items():
            p = expand_defaults("Input", nodes[nid].get("params", {}))
            dt = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16, "i64": torch.long, "i32": torch.int32, "bool": torch.bool}[p["dtype"]]
            kwargs[field] = torch.empty(concrete(compiled.analysis.ports[f"{nid}.out"].shape, probe), dtype=dt)
        outs = model(**kwargs)

    for name, ref in compiled.outputs.items():
        if name not in outs or ref not in compiled.analysis.ports:
            continue
        want = concrete(compiled.analysis.ports[ref].shape, dict(probe))
        got = list(outs[name].shape)
        if want != got:
            result["ok"] = False
            result["issues"].append({"node": ir.split_ref(ref)[0], "message": f"forward produced {got}, analyzer predicted {want}"})

    from .emit import py_name

    owned: set[int] = set()  # a tied parameter counts for the node that owns it (the first in order)
    for nid in compiled.analysis.order:
        if nid not in compiled.analysis.params:
            continue
        count = compiled.analysis.params[nid]
        mod_name = py_name(nid)
        sub = getattr(model, mod_name, None)
        if not isinstance(sub, torch.nn.Module):
            continue
        params = [p for p in sub.parameters() if id(p) not in owned]
        owned |= {id(p) for p in params}
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        result["params"][nid] = {"total": total, "trainable": trainable}
        if count.known and (total, trainable) != (count.total, count.trainable):
            result["issues"].append(
                {"node": nid, "message": f"parameter count {total}/{trainable} ≠ predicted {count.total}/{count.trainable}", "severity": "warning"}
            )
    result["total"] = sum(p.numel() for p in model.parameters())
    result["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    shutil.rmtree(workdir, ignore_errors=True)
    return result
