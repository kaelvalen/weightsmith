import copy

import pytest

from weightsmith_core import diff, ir, select


def test_layout_and_labels_do_not_change_hash(mnist):
    moved = copy.deepcopy(mnist)
    moved["layout"] = {"fc1": {"x": 400, "y": 8}}
    moved["nodes"][1]["label"] = "renamed"
    moved["name"] = "other name"
    assert ir.content_hash(moved) == ir.content_hash(mnist)


def test_defaults_are_expanded_before_hashing(mnist):
    explicit = copy.deepcopy(mnist)
    explicit["nodes"][1]["params"]["bias"] = True
    explicit["nodes"][1]["trainable"] = True
    assert ir.content_hash(explicit) == ir.content_hash(mnist)


def test_order_does_not_change_hash(mnist):
    shuffled = copy.deepcopy(mnist)
    shuffled["nodes"].reverse()
    shuffled["edges"].reverse()
    assert ir.content_hash(shuffled) == ir.content_hash(mnist)


def test_param_change_changes_hash(mnist):
    other = copy.deepcopy(mnist)
    other["nodes"][1]["params"]["out"] = 128
    assert ir.content_hash(other) != ir.content_hash(mnist)


def test_schema_errors_are_reported():
    errs = ir.schema_errors({"schema": 1, "nodes": [{"id": "1bad", "type": "Linear"}], "edges": [["a", "b"]]})
    assert any("nodes/0/id" in e for e in errs)
    assert any("edges/0" in e for e in errs)


def test_migration_chain(monkeypatch):
    monkeypatch.setattr(ir, "SCHEMA_VERSION", 2)
    monkeypatch.setitem(ir.MIGRATIONS, 1, lambda g: {**g, "migrated": True})
    out = ir.migrate({"schema": 1, "nodes": [], "edges": []})
    assert out["schema"] == 2 and out["migrated"]
    with pytest.raises(ValueError):
        ir.migrate({"schema": 3, "nodes": [], "edges": []})


def test_diff_reports_param_adapt_and_edges(mlp_lora):
    b = copy.deepcopy(mlp_lora)
    b["nodes"][1]["adapt"][0]["r"] = 4
    b["nodes"][3]["params"]["out"] = 12
    b["nodes"].append({"id": "drop", "type": "Dropout"})
    b["layout"] = {"fc1": {"x": 1, "y": 2}}
    d = diff.diff(mlp_lora, b)
    kinds = {c["kind"] for c in d["changes"]}
    assert kinds == {"adapt_changed", "param_changed", "node_added"}
    assert diff.diff(mlp_lora, copy.deepcopy(mlp_lora))["same"]


def test_bulk_lora_on_attention_q_v(transformer):
    g, touched = select.apply_adapt(transformer, {"kind": "lora", "r": 8, "alpha": 16}, ["q_proj", "v_proj"], types=["Attention"])
    assert touched == ["block/attn:q_proj", "block/attn:v_proj"]
    attn = next(n for n in g["subgraphs"]["block"]["nodes"] if n["id"] == "attn")
    assert [a["target"] for a in attn["adapt"]] == ["q_proj", "v_proj"]
    again, touched2 = select.apply_adapt(g, {"kind": "lora", "r": 8, "alpha": 16}, ["q_proj", "v_proj"], types=["Attention"])
    assert touched2 == [] and again == g
