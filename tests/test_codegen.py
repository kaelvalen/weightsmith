import copy
import json
import os
from pathlib import Path

import pytest
import torch

from conftest import EXAMPLES, ROOT, load_example
from weightsmith_codegen.emit import CodegenError, compile_graph
from weightsmith_codegen.project import smoke_test, write_run
from weightsmith_core import select

SNAPSHOTS = ROOT / "tests" / "snapshots"
EXAMPLE_NAMES = ["mlp_lora", "mnist_mlp", "transformer_reverse"]


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_generated_code_matches_snapshot(name):
    """Generated code must stay readable; changes to it are reviewed as snapshot diffs.

    Regenerate with WS_UPDATE_SNAPSHOTS=1 pytest tests/test_codegen.py
    """
    src = compile_graph(load_example(name)).model_py
    path = SNAPSHOTS / f"{name}.py"
    if os.environ.get("WS_UPDATE_SNAPSHOTS") or not path.exists():
        path.write_text(src)
    assert src == path.read_text()


@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_meta_smoke_agrees_with_analyzer(name, tmp_path):
    g = load_example(name)
    c = compile_graph(g)
    res = smoke_test(c.model_py, c, g, tmp_path / "smoke")
    assert res["ok"], res["issues"]
    assert not res["issues"], res["issues"]
    assert res["total"] == c.analysis.totals.total
    assert res["trainable"] == c.analysis.totals.trainable


def test_bulk_lora_counts_match_meta_device(transformer, tmp_path):
    g, _ = select.apply_adapt(transformer, {"kind": "lora", "r": 4, "alpha": 8}, ["q_proj", "v_proj"], types=["Attention"])
    g["nodes"][1]["trainable"] = False  # freeze the embedding
    for n in g["subgraphs"]["block"]["nodes"]:
        if n["type"] == "MLP":
            n["adapt"] = [{"kind": "adapter", "bottleneck": 8}]
            n["frozen_params"] = ["fc1.bias"]
    c = compile_graph(g)
    res = smoke_test(c.model_py, c, g, tmp_path / "smoke")
    assert not res["issues"], res["issues"]
    assert (res["total"], res["trainable"]) == (c.analysis.totals.total, c.analysis.totals.trainable)
    assert "LoRALinear(self.attn.q_proj" in c.model_py


def test_lora_forward_starts_at_base(mlp_lora, tmp_path):
    out = tmp_path / "run"
    write_run({"graph": mlp_lora}, out)
    import importlib.util, sys

    spec = importlib.util.spec_from_file_location("m_lora", out / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.Model()
    x = torch.randn(4, 768)
    base = m.fc2(torch.nn.functional.gelu(m.fc1.base(x)))
    assert torch.allclose(m(x)["fc2"], base)  # B = 0
    assert not m.fc1.base.weight.requires_grad and m.fc1.lora_A.requires_grad
    merged = m.fc1.merged()
    torch.nn.init.normal_(m.fc1.lora_B)
    assert not torch.allclose(m.fc1.merged().weight, merged.weight)


def test_disabled_node_is_identity(mnist):
    g = copy.deepcopy(mnist)
    next(n for n in g["nodes"] if n["id"] == "drop")["disabled"] = True
    src = compile_graph(g).model_py
    assert "drop = act  # node: drop · disabled" in src
    assert "self.drop" not in src


def test_invalid_graph_refuses_to_compile(mlp_lora):
    g = copy.deepcopy(mlp_lora)
    g["nodes"][3]["params"]["in"] = 3
    with pytest.raises(CodegenError):
        compile_graph(g)


def test_run_folder_is_self_contained(mnist, tmp_path):
    out = tmp_path / "run"
    c = write_run({"graph": mnist, "data": {"source": "mnist"}}, out)
    assert {p.name for p in out.iterdir()} >= {"model.py", "train.py", "data.py", "config.yaml", "graph.json", "model.map.json"}
    line_map = json.loads((out / "model.map.json").read_text())
    fc1_line = next(i for i, l in enumerate(c.model_py.splitlines(), 1) if "self.fc1 = nn.Linear" in l)
    assert line_map[str(fc1_line)] == "fc1"
    assert "weightsmith" not in (out / "model.py").read_text().split('"""', 2)[2].lower()


def test_cerata_counts_match_qwen25_15b(tmp_path):
    """The Cerata example: frozen Qwen2.5-1.5B (GQA 12/2, RoPE, SwiGLU, tied head) + LoRA r16 on q/v."""
    g = load_example("cerata_qwen2")
    c = compile_graph(g)
    base = 1_543_714_304  # Qwen2.5-1.5B, tied embeddings counted once
    lora = 28 * 16 * ((1536 + 1536) + (1536 + 256))  # 2.18M, paper/tables/hparams.tex
    assert c.analysis.totals.total == base + lora
    assert c.analysis.totals.trainable == lora
    res = smoke_test(c.model_py, c, g, tmp_path / "smoke")
    assert not res["issues"], res["issues"]
    assert (res["total"], res["trainable"]) == (base + lora, lora)
    assert "self.lm_head.weight = self.embed.weight" in c.model_py


def test_gqa_rope_forward_is_causal(tmp_path):
    """A small GQA + RoPE + SwiGLU stack runs for real, and a causal model ignores the future."""
    g = load_example("cerata_qwen2")
    g["nodes"][1]["params"] = {"num": 50, "dim": 64}
    g["nodes"][4]["params"] |= {"in": 64, "out": 50}
    g["nodes"][3]["params"]["dim"] = 64
    g["nodes"][2]["params"]["repeat"] = 2
    for n in g["subgraphs"]["qwen2_layer"]["nodes"]:
        p = n.get("params", {})
        if "dim" in p:
            p["dim"] = 64
        if n["type"] == "Input":
            p["shape"] = ["B", "T", 64]
        if n["type"] == "Attention":
            p |= {"heads": 4, "kv_heads": 2}
        if n["type"] == "GatedMLP":
            p["hidden"] = 128
    out = tmp_path / "run"
    c = write_run({"graph": g}, out)
    import importlib.util

    spec = importlib.util.spec_from_file_location("m_gqa", out / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(0)
    m = mod.Model().eval()
    for blk in m.layers:
        torch.nn.init.normal_(blk.self_attn.q_proj.lora_B)  # make the adapters non-trivial
    x = torch.randint(0, 50, (1, 8))
    y = x.clone()
    y[0, 5:] = (y[0, 5:] + 1) % 50
    a, b = m(x)["lm_head"], m(y)["lm_head"]
    assert torch.allclose(a[0, :5], b[0, :5], atol=1e-5)  # positions before the change are unaffected
    assert not torch.allclose(a[0, 5:], b[0, 5:])
    assert m.lm_head.weight is m.embed.weight
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == c.analysis.totals.trainable


# config hashes before trainscope recording existed: diagnostics must not change run identity
TINY_CONFIG_HASH = "5bb762ae0017740cd54ad9e20d04091adf98337726480c0fa2690279c823c57e"
MNIST_CONFIG_HASH = "1c60f20bccaf93d9be499fd8cda88b12d651e546dca363d7f14cedf1450bb7e6"


def test_diagnostics_not_in_run_identity():
    from conftest import tiny_classifier
    from weightsmith_codegen.project import job_hashes

    job = tiny_classifier()
    off = tiny_classifier(diagnostics={"trainscope": False})
    assert job_hashes(job)["config"] == job_hashes(off)["config"] == TINY_CONFIG_HASH
    mnist_job = json.loads((EXAMPLES / "mnist_mlp.job.json").read_text())
    assert job_hashes(mnist_job)["config"] == MNIST_CONFIG_HASH


def test_module_map_covers_parameters(tmp_path):
    from weightsmith_codegen.project import _load_module

    job = json.loads((EXAMPLES / "transformer_reverse.job.json").read_text())
    job["graph"] = load_example("transformer_reverse")  # the job file references it via $ref
    out = tmp_path / "run"
    write_run(job, out)
    module_map = json.loads((out / "module_map.json").read_text())
    mod = _load_module(out / "model.py", "ws_module_map_probe")
    with torch.device("meta"):
        model = mod.Model()
    heads = {name.split(".")[0] for name, _ in model.named_parameters()}
    assert heads and heads <= set(module_map)
    assert set(module_map.values()) <= {n["id"] for n in job["graph"]["nodes"]}
