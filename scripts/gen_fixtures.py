"""Write the canvas's copies of Python truth:

- web/src/ir/catalog.json   node types, ports, params (from core/registry.py)
- web/src/ir/fixtures.json  graphs + the Python analyzer's results, for the TS mirror's cross-test

Run after changing the registry or validator:  uv run python scripts/gen_fixtures.py
tests/test_cross_ts.py fails when these files are stale.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from node_cases import cases as node_cases  # noqa: E402
from weightsmith_core import ir, registry, templates, validate  # noqa: E402

OUT = ROOT / "web" / "src" / "ir"


def ex(name: str) -> dict:
    return ir.load(ROOT / "examples" / f"{name}.json")


def cases() -> list[tuple[str, dict]]:
    out = [(n, ex(n)) for n in ("mlp_lora", "mnist_mlp", "transformer_reverse", "cerata_qwen2")]
    g = ex("cerata_qwen2")
    g["nodes"][4]["params"]["out"] = 1000
    out.append(("tie_shape_mismatch", g))
    g = ex("cerata_qwen2")
    next(n for n in g["subgraphs"]["qwen2_layer"]["nodes"] if n["id"] == "self_attn")["params"]["kv_heads"] = 5
    out.append(("kv_heads_not_divisor", g))
    g = ex("mlp_lora")
    g["nodes"][3]["params"]["in"] = 512
    out.append(("shape_mismatch", g))
    g = ex("mlp_lora")
    g["edges"] = [e for e in g["edges"] if e[1] != "fc2.in"]
    out.append(("unconnected", g))
    g = ex("mlp_lora")
    g["nodes"][1]["adapt"][0]["target"] = "q_proj"
    out.append(("lora_bad_target", g))
    g = ex("mlp_lora")
    g["nodes"][3]["frozen_params"] = ["bias"]
    g["nodes"][1]["trainable"] = False
    out.append(("freeze_masks", g))
    g = ex("transformer_reverse")
    g["nodes"][3]["params"]["tie_weights"] = True
    attn = next(n for n in g["subgraphs"]["block"]["nodes"] if n["id"] == "attn")
    attn["adapt"] = [{"kind": "lora", "target": "q_proj", "r": 4, "alpha": 8}, {"kind": "lora", "target": "v_proj", "r": 4, "alpha": 8}]
    mlp = next(n for n in g["subgraphs"]["block"]["nodes"] if n["id"] == "mlp")
    mlp["adapt"] = [{"kind": "adapter", "bottleneck": 8}]
    out.append(("tied_lora_adapter", g))
    g = ex("mnist_mlp")
    next(n for n in g["nodes"] if n["id"] == "fc1")["disabled"] = True
    out.append(("disable_changes_shape", g))
    g = ex("mnist_mlp")
    next(n for n in g["nodes"] if n["id"] == "drop")["disabled"] = True
    out.append(("disable_ok", g))
    out.append(("cycle", {
        "schema": 1,
        "nodes": [
            {"id": "x", "type": "Input", "params": {"shape": ["B", 4]}},
            {"id": "p", "type": "Add"},
            {"id": "q", "type": "Linear", "params": {"in": 4, "out": 4}},
        ],
        "edges": [["x.out", "p.a"], ["q.out", "p.b"], ["p.out", "q.in"]],
    }))
    out.append(("symbols_and_conv", {
        "schema": 1,
        "nodes": [
            {"id": "img", "type": "Input", "params": {"shape": ["B", 1, 28, 28]}},
            {"id": "c1", "type": "Conv2d", "params": {"in_channels": 1, "out_channels": 8}},
            {"id": "pool", "type": "MaxPool2d"},
            {"id": "flat", "type": "Flatten"},
            {"id": "fc", "type": "Linear", "params": {"in": 1568, "out": 10}},
            {"id": "a", "type": "Input", "params": {"shape": ["B", "D"], "field": "z"}},
            {"id": "cat", "type": "Concat"},
            {"id": "soft", "type": "Softmax"},
            {"id": "loss", "type": "CrossEntropy"},
        ],
        "edges": [["img.out", "c1.in"], ["c1.out", "pool.in"], ["pool.out", "flat.in"], ["flat.out", "fc.in"],
                  ["fc.out", "cat.a"], ["a.out", "cat.b"], ["cat.out", "soft.in"], ["soft.out", "loss.logits"]],
    }))
    out.append(("sequence_ops", {
        "schema": 1,
        "nodes": [
            {"id": "tok", "type": "Input", "params": {"shape": ["B", "T"], "dtype": "i64"}},
            {"id": "emb", "type": "Embedding", "params": {"num": 100, "dim": 32}},
            {"id": "norm", "type": "RMSNorm", "params": {"dim": 32}},
            {"id": "gate", "type": "Sigmoid"},
            {"id": "mul", "type": "Mul"},
            {"id": "pool", "type": "MeanPool"},
            {"id": "rs", "type": "Reshape", "params": {"shape": ["B", 4, -1]}},
            {"id": "flat", "type": "Flatten"},
            {"id": "head", "type": "Linear", "params": {"in": 32, "out": 2}},
            {"id": "loss", "type": "MSE"},
        ],
        "edges": [["tok.out", "emb.in"], ["emb.out", "norm.in"], ["norm.out", "gate.in"], ["norm.out", "mul.a"],
                  ["gate.out", "mul.b"], ["mul.out", "pool.in"], ["pool.out", "rs.in"], ["rs.out", "flat.in"],
                  ["flat.out", "head.in"], ["head.out", "loss.pred"]],
    }))
    out.append(("embedding_needs_ints", {
        "schema": 1,
        "nodes": [
            {"id": "x", "type": "Input", "params": {"shape": ["B", "T"]}},
            {"id": "emb", "type": "Embedding", "params": {"num": 10, "dim": 4}},
            {"id": "loss", "type": "MSE"},
        ],
        "edges": [["x.out", "emb.in"], ["emb.out", "loss.pred"]],
    }))
    out += [(f"node_{n}", g) for n, g in node_cases()]
    out += [(f"model_{t.id}", t.build()) for t in templates.MODELS]
    g = templates.resnet18()
    next(n for n in g["nodes"] if n["id"] == "layer1")["params"]["args"] = {"C": 64, "bogus": 1}
    out.append(("block_bad_arg", g))
    return out


def expected(graph: dict) -> dict:
    a = validate.analyze(graph)
    errs = sorted({f"{i.code}@{i.node or ''}.{i.port or ''}" for i in a.issues if i.severity == "error"})
    return {
        "ok": a.ok,
        "errors": errs,
        "ports": {k: v.to_json() for k, v in sorted(a.ports.items())},
        "params": {k: [v.total, v.trainable] for k, v in sorted(a.params.items())},
    }


def examples() -> str:
    """Bundled starter projects for the canvas: each example graph with its job config."""
    out = []
    for job_path in sorted((ROOT / "examples").glob("*.job.json")):
        job = json.loads(job_path.read_text())
        graph = ir.load(job_path.parent / job["graph"]["$ref"])
        out.append({"name": graph.get("name", job_path.stem), "graph": graph, "data": job.get("data", {}), "train": job.get("train", {})})
    return json.dumps(out, indent=1) + "\n"


def build() -> tuple[str, str]:
    catalog = json.dumps({"schema": ir.SCHEMA_VERSION, "types": registry.catalog()}, indent=1) + "\n"
    fixtures = json.dumps([{"name": n, "graph": g, "expected": expected(g)} for n, g in cases()], indent=1) + "\n"
    return catalog, fixtures


if __name__ == "__main__":
    catalog, fixtures = build()
    (OUT / "catalog.json").write_text(catalog)
    (OUT / "fixtures.json").write_text(fixtures)
    (OUT / "examples.json").write_text(examples())
    (OUT / "templates.json").write_text(json.dumps(templates.export(), indent=1) + "\n")
    print(f"wrote catalog.json, fixtures.json, examples.json, templates.json to {OUT}")
