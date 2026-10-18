"""Data preview: fields, shapes and the checks against the graph's Inputs and loss targets."""

import json
from pathlib import Path

from weightsmith_runtime.preview import preview

EX = Path(__file__).parent.parent / "examples"


def mlp(dim: int) -> dict:
    return {
        "schema": 1,
        "nodes": [
            {"id": "x", "type": "Input", "params": {"shape": ["B", dim], "dtype": "f32", "field": "x"}},
            {"id": "fc", "type": "Linear", "params": {"in": dim, "out": 4}},
            {"id": "loss", "type": "CrossEntropy"},
        ],
        "edges": [["x.out", "fc.in"], ["fc.out", "loss.logits"]],
    }


def test_fields_and_target():
    r = preview({"source": "synthetic", "dim": 16, "classes": 4}, mlp(16))
    assert r["ok"], r
    assert {f["name"]: (f["shape"], f["dtype"]) for f in r["fields"]} == {"x": ([16], "f32"), "y": ([], "i64")}
    assert r["rows"] == 512 and r["capped"]
    assert [c["message"] for c in r["checks"]] == ["x → x", "y → loss target"]


def test_mismatch_names_the_input():
    r = preview({"source": "synthetic", "dim": 16}, mlp(32))
    assert not r["ok"]
    bad = [c for c in r["checks"] if not c["ok"]]
    assert bad[0]["node"] == "x" and bad[0]["have"] == [16]


def test_reshape_is_fine():
    g = json.loads((EX / "mnist_mlp.json").read_text())
    r = preview({"source": "synthetic", "dim": 784}, g)
    assert r["ok"], r


def test_data_errors_are_reported():
    r = preview({"source": "text", "uri": "/nonexistent.txt"}, None)
    assert not r["ok"] and "nonexistent" in r["error"]
