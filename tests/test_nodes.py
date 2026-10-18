"""Every node type and every block template, run for real on CPU through the generated code.

The analyzer's predicted output shapes and parameter counts must match what PyTorch builds.
"""

import importlib.util

import pytest
import torch

from node_cases import cases
from weightsmith_codegen.emit import compile_graph
from weightsmith_codegen.project import concrete
from weightsmith_core import templates, validate
from weightsmith_core.registry import REGISTRY, expand_defaults

CASES = cases()


def test_every_node_type_has_a_case():
    covered = {n["type"] for _, g in CASES for n in g["nodes"]}
    covered |= {n["type"] for _, g in CASES for s in g.get("subgraphs", {}).values() for n in s["nodes"]}
    missing = set(REGISTRY) - covered - {"PythonModule"}  # covered by test_pynode.py
    assert not missing, missing


@pytest.mark.parametrize("name,graph", CASES, ids=[c[0] for c in CASES])
def test_node_runs_for_real(name, graph, tmp_path):
    a = validate.analyze(graph)
    assert a.ok, [f"{i.node}.{i.port}: {i.message}" for i in a.errors]
    c = compile_graph(graph, a)
    path = tmp_path / "model.py"
    path.write_text(c.model_py)
    spec = importlib.util.spec_from_file_location(f"m_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(0)
    model = mod.Model().eval()

    probe = {"B": 2, "T": 5, "S": 7}
    nodes = {n["id"]: n for n in graph["nodes"]}
    kwargs = {}
    for field, nid in c.inputs.items():
        p = expand_defaults("Input", nodes[nid]["params"])
        shape = concrete(a.ports[f"{nid}.out"].shape, probe)
        kwargs[field] = torch.randint(0, 5, shape) if p["dtype"] == "i64" else torch.rand(shape)
    with torch.no_grad():
        out = model(**kwargs)

    for oname, ref in c.outputs.items():
        want = concrete(a.ports[ref].shape, dict(probe))
        assert list(out[oname].shape) == want, f"{oname}: {list(out[oname].shape)} ≠ {want}"
    if a.losses:
        assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    assert sum(p.numel() for p in model.parameters()) == a.totals.total


@pytest.mark.parametrize("t", [t for t in templates.MODELS if t.reference], ids=lambda t: t.id)
def test_model_template_matches_published_count(t):
    a = validate.analyze(t.build())
    assert a.ok, [i.message for i in a.errors]
    assert a.totals.total == t.reference


def test_parametric_block_args_change_counts():
    g = templates.resnet18()
    a = validate.analyze(g)
    # layer2_1 (C=128) and layer1 (C=64, repeat 2): 2·(2·9·C² + 4C)
    assert a.params["layer2_1"].total == 2 * 9 * 128 * 128 + 4 * 128
    assert a.params["layer1"].total == 2 * (2 * 9 * 64 * 64 + 4 * 64)
    next(n for n in g["nodes"] if n["id"] == "layer1")["params"]["args"] = {"C": 64, "bogus": 1}
    assert any("bogus" in i.message for i in validate.analyze(g).errors)
