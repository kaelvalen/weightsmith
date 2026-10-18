import copy

from weightsmith_core import validate


def codes(a):
    return {(i.code, i.node, i.port) for i in a.issues if i.severity == "error"}


def test_plan_example_is_valid(mlp_lora):
    a = validate.analyze(mlp_lora)
    assert a.ok, a.issues
    assert a.ports["fc1.out"].shape == ["B", 1024]
    assert a.ports["loss.loss"].shape == []
    # LoRA r16 on 768→1024 adds 16·(768+1024); the base Linear is frozen
    assert a.params["fc1"].total == 768 * 1024 + 1024 + 16 * (768 + 1024)
    assert a.params["fc1"].trainable == 16 * (768 + 1024)
    assert a.totals.trainable == 16 * (768 + 1024) + 1024 * 10 + 10


def test_shape_mismatch_lands_on_port_and_edge(mlp_lora):
    g = copy.deepcopy(mlp_lora)
    g["nodes"][3]["params"]["in"] = 512
    a = validate.analyze(g)
    err = next(i for i in a.issues if i.code == "shape" and i.severity == "error")
    assert (err.node, err.port, err.edge) == ("fc2", "in", 2)
    assert "1024" in err.message and "512" in err.message
    peer = next(i for i in a.issues if i.severity == "peer")
    assert (peer.node, peer.port) == ("act", "out")


def test_symbols_unify_across_graph():
    g = {
        "schema": 1,
        "nodes": [
            {"id": "a", "type": "Input", "params": {"shape": ["B", "D"]}},
            {"id": "b", "type": "Input", "params": {"shape": ["B", 32], "field": "z"}},
            {"id": "sum", "type": "Add"},
            {"id": "fc", "type": "Linear", "params": {"in": 32, "out": 4}},
            {"id": "loss", "type": "MSE"},
        ],
        "edges": [["a.out", "sum.a"], ["b.out", "sum.b"], ["sum.out", "fc.in"], ["fc.out", "loss.pred"]],
    }
    a = validate.analyze(g)
    assert a.ok, a.issues
    assert a.ports["a.out"].shape == ["B", 32]  # D bound to 32 through the Add


def test_structural_errors():
    g = {
        "schema": 1,
        "nodes": [
            {"id": "x", "type": "Input", "params": {"shape": ["B", 4]}},
            {"id": "p", "type": "Add"},
            {"id": "q", "type": "Linear", "params": {"in": 4, "out": 4}},
            {"id": "r", "type": "ReLU"},
            {"id": "lone", "type": "Linear", "params": {"in": 4}},
            {"id": "what", "type": "Nope"},
        ],
        "edges": [["x.out", "p.a"], ["q.out", "p.b"], ["p.out", "q.in"], ["x.out", "r.in"], ["q.out", "r.in"], ["x.out", "ghost.in"]],
    }
    a = validate.analyze(g)
    c = {i.code for i in a.errors}
    assert {"multi_input", "cycle", "param_missing", "unknown_type", "edge_node"} <= c


def test_unconnected_required_port(mlp_lora):
    g = copy.deepcopy(mlp_lora)
    g["edges"] = [e for e in g["edges"] if e[1] != "fc2.in"]
    assert ("unconnected", "fc2", "in") in codes(validate.analyze(g))


def test_lora_target_must_exist(mlp_lora):
    g = copy.deepcopy(mlp_lora)
    g["nodes"][1]["adapt"][0]["target"] = "q_proj"
    assert any(i.code == "adapt_target" for i in validate.analyze(g).errors)


def test_freeze_mask_counts(mlp_lora):
    g = copy.deepcopy(mlp_lora)
    g["nodes"][3]["frozen_params"] = ["bias"]
    g["nodes"][1]["trainable"] = False
    a = validate.analyze(g)
    assert a.params["fc2"].trainable == 1024 * 10
    assert a.params["fc1"].trainable == 16 * (768 + 1024)  # LoRA stays trainable


def test_subgraph_repeat_and_tying(transformer):
    a = validate.analyze(transformer)
    assert a.ok, a.issues
    assert a.ports["blocks.out"].shape == ["B", 16, 64]
    per_block = a.subgraphs["block"].totals.total
    assert a.params["blocks"].total == 2 * per_block
    g = copy.deepcopy(transformer)
    g["nodes"][3]["params"]["tie_weights"] = True
    assert validate.analyze(g).params["blocks"].total == per_block


def test_subgraph_repeat_needs_stable_shape(transformer):
    g = copy.deepcopy(transformer)
    body = g["subgraphs"]["block"]
    body["nodes"].append({"id": "proj", "type": "Linear", "params": {"in": 64, "out": 32}})
    body["edges"] = [e for e in body["edges"] if e[1] != "out.in"] + [["res2.out", "proj.in"], ["proj.out", "out.in"]]
    errs = validate.analyze(g).errors
    assert any(i.node == "blocks" for i in errs)


def test_disable_requires_same_shape(mnist):
    g = copy.deepcopy(mnist)
    next(n for n in g["nodes"] if n["id"] == "drop")["disabled"] = True
    assert validate.analyze(g).ok
    next(n for n in g["nodes"] if n["id"] == "fc1")["disabled"] = True
    assert any(i.code == "disable" for i in validate.analyze(g).errors)


def test_duplicate_fields(mnist):
    g = copy.deepcopy(mnist)
    g["nodes"].append({"id": "img2", "type": "Input", "params": {"shape": ["B", 3]}})
    assert any(i.code == "field" for i in validate.analyze(g).errors)


def test_embedding_needs_integer_ids():
    g = {
        "schema": 1,
        "nodes": [
            {"id": "x", "type": "Input", "params": {"shape": ["B", "T"], "dtype": "f32"}},
            {"id": "emb", "type": "Embedding", "params": {"num": 10, "dim": 4}},
        ],
        "edges": [["x.out", "emb.in"]],
    }
    a = validate.analyze(g)
    assert any(i.code == "shape" and "integer" in i.message for i in a.errors)


def test_tie_to_must_match_embedding():
    from conftest import load_example

    g = load_example("cerata_qwen2")
    g["nodes"][4]["params"]["out"] = 1000
    assert any(i.code == "tie" for i in validate.analyze(g).errors)
    g = load_example("cerata_qwen2")
    g["nodes"][1]["trainable"] = True
    assert any(i.code == "tie" and i.severity == "warning" for i in validate.analyze(g).issues)
