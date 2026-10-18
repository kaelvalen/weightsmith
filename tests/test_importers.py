"""Import tiny random Hugging Face models and check the generated code reproduces transformers.

Every family the importer supports: the graph comes from config.json, the weights from
safetensors, and the forward pass of Weightsmith's model.py must match transformers'.
"""

import importlib.util

import pytest
import torch

transformers = pytest.importorskip("transformers")

from weightsmith_codegen.emit import compile_graph  # noqa: E402
from weightsmith_core import validate  # noqa: E402
from weightsmith_importers import hf  # noqa: E402

LLAMA3 = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 16}


def _cfg(name, **kw):
    return getattr(transformers, name)(**kw)


CASES = {
    "llama": lambda: transformers.LlamaForCausalLM(_cfg("LlamaConfig", vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                                       rope_theta=10000.0, rope_scaling=LLAMA3, tie_word_embeddings=True)),
    "qwen2": lambda: transformers.Qwen2ForCausalLM(_cfg("Qwen2Config", vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                                       tie_word_embeddings=True)),
    "mistral": lambda: transformers.MistralForCausalLM(_cfg("MistralConfig", vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                                           num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                                           sliding_window=None, tie_word_embeddings=False)),
    "mixtral": lambda: transformers.MixtralForCausalLM(_cfg("MixtralConfig", vocab_size=97, hidden_size=32, intermediate_size=24, num_hidden_layers=2,
                                                           num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                                           num_local_experts=4, num_experts_per_tok=2, sliding_window=None)),
    "gpt2": lambda: transformers.GPT2LMHeadModel(_cfg("GPT2Config", vocab_size=97, n_embd=32, n_layer=2, n_head=4, n_positions=64)),
    "bert": lambda: transformers.BertModel(_cfg("BertConfig", vocab_size=97, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                               intermediate_size=48, max_position_embeddings=64)),
    "mpnet": lambda: transformers.MPNetModel(_cfg("MPNetConfig", vocab_size=97, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                                 intermediate_size=48, max_position_embeddings=64)),
    "mamba": lambda: transformers.MambaForCausalLM(_cfg("MambaConfig", vocab_size=97, hidden_size=32, state_size=4, num_hidden_layers=2,
                                                       expand=2, conv_kernel=4)),
}


def _load(model_py: str, tmp_path, name: str):
    path = tmp_path / f"{name}.py"
    path.write_text(model_py)
    spec = importlib.util.spec_from_file_location(f"imp_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("family", list(CASES))
def test_import_reproduces_transformers(family, tmp_path):
    torch.manual_seed(0)
    ref = CASES[family]().eval()
    src = tmp_path / "hf"
    ref.save_pretrained(src, safe_serialization=True)

    res = hf.import_model(src, tmp_path / "out")
    a = validate.analyze(res["graph"])
    assert a.ok, [f"{i.node}: {i.message}" for i in a.errors]
    c = compile_graph(res["graph"], a)
    model = _load(c.model_py, tmp_path, family).Model().eval()
    from safetensors.torch import load_file

    state = load_file(res["weights"])
    import json

    from safetensors import safe_open

    with safe_open(res["weights"], "pt") as f:
        aliases = json.loads(f.metadata().get("weightsmith_aliases", "{}"))
    for k, v in aliases.items():
        state[k] = state[v]
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in ref.parameters())

    ids = torch.randint(0, 97, (2, 9))
    with torch.no_grad():
        if family in ("bert", "mpnet"):
            mask = torch.ones_like(ids)
            mask[1, 6:] = 0  # padding on the second row
            kw = {"token_type": torch.zeros_like(ids)} if family == "bert" else {}
            got = model(x=ids, mask=mask, **kw)
            want = ref(input_ids=ids, attention_mask=mask)
            keep = mask.bool()
            torch.testing.assert_close(got["hidden"][keep], want.last_hidden_state[keep], atol=2e-5, rtol=1e-4)
            torch.testing.assert_close(got["pooled"], want.pooler_output, atol=2e-5, rtol=1e-4)
            # masked mean pooling ignores padding (sentence-transformers' pooling)
            m = mask.unsqueeze(-1).float()
            torch.testing.assert_close(got["embedding"] if "embedding" in got else got["mean_pool"],
                                       (want.last_hidden_state * m).sum(1) / m.sum(1)
                                       if family == "bert" else torch.nn.functional.normalize((want.last_hidden_state * m).sum(1) / m.sum(1), dim=-1),
                                       atol=2e-5, rtol=1e-4)
        else:
            got = model(x=ids)["lm_head"]
            want = ref(input_ids=ids).logits
            torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-4)


def test_unsupported_family_is_refused(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "t5"}')
    with pytest.raises(hf.ImportError_):
        hf.read_config(tmp_path)
