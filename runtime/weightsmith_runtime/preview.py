"""Data preview: build a few samples with the run's own data.py and check them against the graph.

Answers "will my data fit this model?" before a run is queued: which fields the dataset
has, their per-sample shapes and dtypes, a peek at the first sample, and what the graph's
Inputs and loss targets expect from it.
"""

from __future__ import annotations

import importlib.util
import math
from importlib import resources
from typing import Any

from weightsmith_codegen.emit import _target_fields
from weightsmith_core import ir, validate
from weightsmith_core.registry import expand_defaults

ROWS = 512  # enough to see the data, cheap enough to build on every request

_data_mod = None


def _data():
    global _data_mod
    if _data_mod is None:
        path = resources.files("weightsmith_codegen") / "templates" / "data.py"
        spec = importlib.util.spec_from_file_location("weightsmith_preview_data", str(path))
        _data_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_data_mod)
    return _data_mod


def _dtype(t) -> str:
    s = str(t.dtype).removeprefix("torch.")
    return {"float32": "f32", "float16": "f16", "bfloat16": "bf16", "int64": "i64", "int32": "i32", "bool": "bool", "uint8": "u8"}.get(s, s)


def _peek(x) -> str:
    flat = x.flatten()
    if x.dtype.is_floating_point:
        head = ", ".join(f"{v:.3g}" for v in flat[:6].tolist())
        return f"[{head}{', …' if len(flat) > 6 else ''}]  min {x.min().item():.3g} · max {x.max().item():.3g}"
    head = ", ".join(str(v) for v in flat[:8].tolist())
    return f"[{head}{', …' if len(flat) > 8 else ''}]"


def expects(graph: dict) -> tuple[dict[str, dict], list[str]]:
    """Per-sample shape/dtype of every Input, and the fields loss targets read when unconnected."""
    g = ir.migrate(graph)
    inputs = {}
    for n in g["nodes"]:
        if n["type"] == "Input":
            p = expand_defaults("Input", n.get("params", {}))
            inputs[p["field"]] = {"node": n["id"], "shape": list(p["shape"][1:]), "dtype": p["dtype"]}
    targets: list[str] = []
    try:
        a = validate.analyze(g)
        targets = [t for t in _target_fields(g, a) if t not in inputs]
    except Exception:  # noqa: BLE001, S110 - a broken graph still gets the field listing
        pass
    return inputs, targets


def preview(data: dict, graph: dict | None) -> dict[str, Any]:
    d = _data()
    cfg = dict(data)
    src = cfg.get("source", "synthetic")
    if src not in d.SOURCES:
        return {"ok": False, "error": f"unknown data source {src!r}"}
    if src == "synthetic":
        cfg["n"] = min(cfg.get("n", 4096), ROWS)
    elif src in ("mnist", "hf"):
        cfg["limit"] = min(cfg.get("limit") or ROWS, ROWS)
    try:
        fields = d.SOURCES[src](cfg)
        fields = {k: v[:ROWS] for k, v in fields.items()}
        fields = d._preprocess(fields, cfg.get("preprocess", []))
    except Exception as e:  # noqa: BLE001 - surface the data error, not a 500
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    out_fields = [
        {"name": k, "shape": list(v.shape[1:]), "dtype": _dtype(v), "peek": _peek(v[0]) if len(v) else ""}
        for k, v in fields.items()
    ]
    checks: list[dict] = []
    if graph is not None:
        inputs, targets = expects(graph)
        for f, spec in inputs.items():
            x = fields.get(f)
            if x is None:
                checks.append({"ok": False, "field": f, "message": f"Input {spec['node']} reads field {f!r}, the dataset has {', '.join(fields) or 'none'}"})
                continue
            want, have = spec["shape"], list(x.shape[1:])
            notes = []
            if all(isinstance(w, int) for w in want) and have != want:
                if math.prod(have) != math.prod(want):
                    checks.append({"ok": False, "field": f, "node": spec["node"], "have": have, "message": f"{f} is {have} per sample, {spec['node']} expects {want}"})
                    continue
                notes.append(f"reshaped {have} → {want}")
            elif not all(isinstance(w, int) for w in want) and len(have) != len(want):
                checks.append({"ok": False, "field": f, "node": spec["node"], "have": have, "message": f"{f} has {len(have)} dims per sample, {spec['node']} expects {len(want)} ({want})"})
                continue
            if (spec["dtype"] in ("f32", "f16", "bf16")) != x.dtype.is_floating_point:
                notes.append(f"cast {_dtype(x)} → {spec['dtype']}")
            checks.append({"ok": True, "field": f, "message": f"{f} → {spec['node']}" + (f" ({', '.join(notes)})" if notes else "")})
        for t in targets:
            if t in fields:
                checks.append({"ok": True, "field": t, "message": f"{t} → loss target"})
            else:
                checks.append({"ok": False, "field": t, "message": f"a loss reads target field {t!r}, the dataset has {', '.join(fields)}"})
    n = len(next(iter(fields.values()))) if fields else 0
    return {"ok": all(c["ok"] for c in checks), "rows": n, "capped": n >= ROWS, "fields": out_fields, "checks": checks}
