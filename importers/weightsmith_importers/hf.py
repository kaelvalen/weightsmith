"""Hugging Face → Weightsmith: graph from config.json, weights from safetensors.

Supported families (the scope is fixed on purpose; see importers/README.md):
llama, mistral, qwen2, mixtral, gpt2, bert, mpnet, mamba. Each family maps onto the
same templates the canvas offers, so an imported model is an ordinary editable graph
whose parameter names line up with the generated model.py.

    from weightsmith_importers import hf
    res = hf.import_model("Qwen/Qwen2.5-1.5B-Instruct", out_dir)   # → graph, weights file, tokenizer

tests/test_importers.py builds tiny random models with transformers, imports them and
checks the generated code reproduces transformers' outputs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from weightsmith_core import templates
from weightsmith_core.templates import Body

FAMILIES = ("llama", "mistral", "qwen2", "mixtral", "gpt2", "bert", "mpnet", "mamba")


class ImportError_(ValueError):
    pass


# ------------------------------------------------------------------ sources


def resolve(src: str | Path) -> Path:
    """A local model directory, or a Hub repo id downloaded into the HF cache."""
    p = Path(src).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(src), allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"]))


def read_config(path: Path) -> dict[str, Any]:
    cfg = json.loads((path / "config.json").read_text())
    if cfg.get("model_type") not in FAMILIES:
        raise ImportError_(f"model_type {cfg.get('model_type')!r} is not supported (supported: {', '.join(FAMILIES)})")
    return cfg


def _rope(cfg: dict[str, Any]) -> tuple[float, dict | None]:
    """rope_theta and scaling from either the v4 (rope_theta/rope_scaling) or v5 (rope_parameters) layout."""
    params = cfg.get("rope_parameters") or {}
    theta = float(cfg.get("rope_theta") or params.get("rope_theta") or 10000.0)
    scaling = cfg.get("rope_scaling") or {k: v for k, v in params.items() if k != "rope_theta"} or None
    if scaling:
        kind = scaling.get("rope_type") or scaling.get("type")
        if kind in (None, "default"):
            return theta, None
        if kind not in ("llama3", "linear"):
            raise ImportError_(f"RoPE scaling {kind!r} is not supported (llama3, linear)")
        scaling = {**scaling, "rope_type": kind}
        scaling.pop("type", None)
    return theta, scaling


def _head_dim_ok(cfg: dict[str, Any]) -> None:
    hd = cfg.get("head_dim")
    if hd and hd * cfg["num_attention_heads"] != cfg["hidden_size"]:
        raise ImportError_(f"head_dim {hd} ≠ hidden_size / heads; this attention layout is not supported")


# ------------------------------------------------------------------- graphs


def _decoder(cfg: dict[str, Any]) -> dict[str, Any]:
    _head_dim_ok(cfg)
    mt = cfg["model_type"]
    theta, scaling = _rope(cfg)
    args = {
        "D": cfg["hidden_size"], "H": cfg["num_attention_heads"], "KV": cfg.get("num_key_value_heads") or cfg["num_attention_heads"],
        "F": cfg["intermediate_size"], "theta": theta, "eps": cfg.get("rms_norm_eps", 1e-6),
    }
    if mt == "mixtral":
        block = templates.moe_block(**args, E=cfg["num_local_experts"], K=cfg["num_experts_per_tok"])
        name = "mixtral_layer"
        if scaling:
            raise ImportError_("RoPE scaling on mixtral is not supported")
    else:
        block = templates.llama_block(**args, qkv_bias=mt == "qwen2" or bool(cfg.get("attention_bias")), rope_scaling=scaling)
        name = f"{mt}_layer"
    tied = bool(cfg.get("tie_word_embeddings", mt == "qwen2"))
    return templates._decoder_lm(cfg.get("_name_or_path") or mt, cfg["vocab_size"], cfg["hidden_size"], cfg["num_hidden_layers"],
                                 name, block, tied, cfg.get("rms_norm_eps", 1e-6))


def _gpt2(cfg: dict[str, Any]) -> dict[str, Any]:
    d, n = cfg["n_embd"], cfg["n_layer"]
    b = Body()
    b.add("tokens", "Input", shape=["B", "T"], dtype="i64", field="x")
    b.add("wte", "Embedding", num=cfg["vocab_size"], dim=d)
    b.add("wpe", "PositionalEmbedding", max_len=cfg["n_positions"], dim=d)
    b.add("drop", "Dropout", p=cfg.get("embd_pdrop", 0.1))
    b.add("blocks", "Subgraph", label=f"{n}× block", graph="gpt2_block", repeat=n)
    b.add("ln_f", "LayerNorm", dim=d, eps=cfg.get("layer_norm_epsilon", 1e-5))
    b.chain("tokens", "wte", "wpe", "drop")
    b.wire("drop", "blocks.x")
    b.chain("blocks.out", "ln_f")
    templates._lm_head(b, "wte", d, cfg["vocab_size"], True, "ln_f")
    block = templates.pre_ln_block(D=d, H=cfg["n_head"], F=cfg.get("n_inner") or 4 * d, eps=cfg.get("layer_norm_epsilon", 1e-5),
                                   causal=True, act="gelu_tanh", dropout=cfg.get("resid_pdrop", 0.1))
    return b.graph(cfg.get("_name_or_path") or "gpt2", {"gpt2_block": block})


def _bert(cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg.get("hidden_act", "gelu") != "gelu":
        raise ImportError_(f"hidden_act {cfg.get('hidden_act')!r} is not supported")
    g = templates._bert_like(cfg.get("_name_or_path") or "bert", cfg["vocab_size"], cfg["hidden_size"], cfg["num_attention_heads"],
                             cfg["intermediate_size"], cfg["max_position_embeddings"], cfg.get("layer_norm_eps", 1e-12),
                             cfg.get("type_vocab_size", 2), sentence=True)
    _set_repeat(g, "encoder", cfg["num_hidden_layers"])
    return g


def _mpnet(cfg: dict[str, Any]) -> dict[str, Any]:
    g = templates.mpnet_base()
    g["name"] = cfg.get("_name_or_path") or "mpnet"
    d, h = cfg["hidden_size"], cfg["num_attention_heads"]
    for n in g["nodes"]:
        p = n.get("params", {})
        if n["id"] == "word":
            p |= {"num": cfg["vocab_size"], "dim": d}
        elif n["id"] == "pos":
            p |= {"max_len": cfg["max_position_embeddings"], "dim": d, "offset": cfg.get("pad_token_id", 1) + 1}
        elif n["id"] == "emb_norm":
            p |= {"dim": d, "eps": cfg.get("layer_norm_eps", 1e-5)}
        elif n["id"] == "rel_bias":
            p |= {"heads": h, "buckets": cfg.get("relative_attention_num_buckets", 32)}
        elif n["id"] == "pooler":
            p |= {"in": d, "out": d}
    body = g["subgraphs"]["mpnet_layer"]
    body["params"] |= {"D": d, "H": h, "F": cfg["intermediate_size"], "eps": cfg.get("layer_norm_eps", 1e-5)}
    _set_repeat(g, "encoder", cfg["num_hidden_layers"])
    return g


def _mamba(cfg: dict[str, Any]) -> dict[str, Any]:
    d = cfg["hidden_size"]
    block = templates.mamba_block(D=d, N=cfg.get("state_size", 16), eps=cfg.get("layer_norm_epsilon", 1e-5))
    for n in block["nodes"]:
        if n["type"] == "Mamba":
            n["params"] |= {"expand": cfg.get("expand", 2), "d_conv": cfg.get("conv_kernel", 4)}
            if cfg.get("time_step_rank") not in (None, "auto"):
                n["params"]["dt_rank"] = cfg["time_step_rank"]
    tied = bool(cfg.get("tie_word_embeddings", True))
    return templates._decoder_lm(cfg.get("_name_or_path") or "mamba", cfg["vocab_size"], d, cfg["num_hidden_layers"], "mamba_block",
                                 block, tied, cfg.get("layer_norm_epsilon", 1e-5), label=f"{cfg['num_hidden_layers']}× Mamba block")


def _set_repeat(g: dict[str, Any], nid: str, n: int) -> None:
    for node in g["nodes"]:
        if node["id"] == nid:
            node["params"]["repeat"] = n
            node["label"] = f"{n}× encoder layer"


def build_graph(cfg: dict[str, Any]) -> dict[str, Any]:
    mt = cfg["model_type"]
    if mt in ("llama", "mistral", "qwen2", "mixtral"):
        return _decoder(cfg)
    return {"gpt2": _gpt2, "bert": _bert, "mpnet": _mpnet, "mamba": _mamba}[mt](cfg)


# ------------------------------------------------------------------ weights

Get = Callable[[str], Any]  # hf key → tensor


def _layers(cfg: dict[str, Any]) -> int:
    return cfg.get("num_hidden_layers") or cfg.get("n_layer")


def _strip(keys: set[str], prefixes: tuple[str, ...]) -> str:
    """The prefix a checkpoint uses for the bare model ('model.', 'transformer.', 'bert.', ...)."""
    for pre in prefixes:
        if any(k.startswith(pre) for k in keys):
            return pre
    return ""


def _map_decoder(cfg, get: Get, keys: set[str]) -> Iterator[tuple[str, Any]]:
    import torch

    pre = _strip(keys, ("model.",))
    yield "embed.weight", get(f"{pre}embed_tokens.weight")
    for i in range(_layers(cfg)):
        src, dst = f"{pre}layers.{i}.", f"layers.{i}."
        yield dst + "input_norm.weight", get(src + "input_layernorm.weight")
        yield dst + "post_norm.weight", get(src + "post_attention_layernorm.weight")
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            yield f"{dst}self_attn.{proj}.weight", get(f"{src}self_attn.{proj}.weight")
            if f"{src}self_attn.{proj}.bias" in keys:
                yield f"{dst}self_attn.{proj}.bias", get(f"{src}self_attn.{proj}.bias")
        if cfg["model_type"] == "mixtral":
            moe = src + "block_sparse_moe."
            yield dst + "moe.router.weight", get(moe + "gate.weight")
            e = cfg["num_local_experts"]
            if f"{moe}experts.gate_up_proj" in keys:  # transformers v5 fused layout: E × 2H × D, E × D × H
                gu, dn = get(f"{moe}experts.gate_up_proj"), get(f"{moe}experts.down_proj")
                yield dst + "moe.experts.gate_up_proj", gu.transpose(1, 2).contiguous()
                yield dst + "moe.experts.down_proj", dn.transpose(1, 2).contiguous()
            else:  # per-expert w1 (gate), w3 (up), w2 (down)
                yield dst + "moe.experts.gate_up_proj", torch.stack(
                    [torch.cat([get(f"{moe}experts.{j}.w1.weight").T, get(f"{moe}experts.{j}.w3.weight").T], dim=1) for j in range(e)])
                yield dst + "moe.experts.down_proj", torch.stack([get(f"{moe}experts.{j}.w2.weight").T for j in range(e)])
        else:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield f"{dst}mlp.{proj}.weight", get(f"{src}mlp.{proj}.weight")
    yield "norm.weight", get(f"{pre}norm.weight")
    head = "lm_head.weight" if "lm_head.weight" in keys else f"{pre}embed_tokens.weight"
    yield "lm_head.weight", get(head)


def _map_gpt2(cfg, get: Get, keys: set[str]) -> Iterator[tuple[str, Any]]:
    pre = _strip(keys, ("transformer.",))
    d = cfg["n_embd"]
    yield "wte.weight", get(f"{pre}wte.weight")
    yield "wpe.weight", get(f"{pre}wpe.weight")
    for i in range(_layers(cfg)):
        src, dst = f"{pre}h.{i}.", f"blocks.{i}."
        for a, b in (("ln_1", "ln1"), ("ln_2", "ln2")):
            yield f"{dst}{b}.weight", get(f"{src}{a}.weight")
            yield f"{dst}{b}.bias", get(f"{src}{a}.bias")
        w, bias = get(src + "attn.c_attn.weight"), get(src + "attn.c_attn.bias")  # Conv1D: in × out, q|k|v
        for j, proj in enumerate(("q_proj", "k_proj", "v_proj")):
            yield f"{dst}attn.{proj}.weight", w[:, j * d : (j + 1) * d].T.contiguous()
            yield f"{dst}attn.{proj}.bias", bias[j * d : (j + 1) * d]
        yield f"{dst}attn.o_proj.weight", get(src + "attn.c_proj.weight").T.contiguous()
        yield f"{dst}attn.o_proj.bias", get(src + "attn.c_proj.bias")
        yield f"{dst}mlp.fc1.weight", get(src + "mlp.c_fc.weight").T.contiguous()
        yield f"{dst}mlp.fc1.bias", get(src + "mlp.c_fc.bias")
        yield f"{dst}mlp.fc2.weight", get(src + "mlp.c_proj.weight").T.contiguous()
        yield f"{dst}mlp.fc2.bias", get(src + "mlp.c_proj.bias")
    yield "ln_f.weight", get(f"{pre}ln_f.weight")
    yield "ln_f.bias", get(f"{pre}ln_f.bias")
    yield "lm_head.weight", get(f"{pre}wte.weight")


def _wb(dst: str, src: str, get: Get, keys: set[str]) -> Iterator[tuple[str, Any]]:
    yield f"{dst}.weight", get(f"{src}.weight")
    if f"{src}.bias" in keys:
        yield f"{dst}.bias", get(f"{src}.bias")


def _map_bert(cfg, get: Get, keys: set[str]) -> Iterator[tuple[str, Any]]:
    pre = _strip(keys, ("bert.",))
    e = f"{pre}embeddings."
    yield "word.weight", get(e + "word_embeddings.weight")
    yield "pos.weight", get(e + "position_embeddings.weight")
    yield "type_emb.weight", get(e + "token_type_embeddings.weight")
    yield from _wb("emb_norm", e + "LayerNorm", get, keys)
    for i in range(_layers(cfg)):
        src, dst = f"{pre}encoder.layer.{i}.", f"encoder.{i}."
        for a, b in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj")):
            yield from _wb(f"{dst}attn.{b}", f"{src}attention.self.{a}", get, keys)
        yield from _wb(f"{dst}attn.o_proj", f"{src}attention.output.dense", get, keys)
        yield from _wb(f"{dst}ln1", f"{src}attention.output.LayerNorm", get, keys)
        yield from _wb(f"{dst}mlp.fc1", f"{src}intermediate.dense", get, keys)
        yield from _wb(f"{dst}mlp.fc2", f"{src}output.dense", get, keys)
        yield from _wb(f"{dst}ln2", f"{src}output.LayerNorm", get, keys)
    yield from _wb("pooler", f"{pre}pooler.dense", get, keys)


def _map_mpnet(cfg, get: Get, keys: set[str]) -> Iterator[tuple[str, Any]]:
    pre = _strip(keys, ("mpnet.",))
    e = f"{pre}embeddings."
    yield "word.weight", get(e + "word_embeddings.weight")
    yield "pos.weight", get(e + "position_embeddings.weight")
    yield from _wb("emb_norm", e + "LayerNorm", get, keys)
    yield "rel_bias.table.weight", get(f"{pre}encoder.relative_attention_bias.weight")
    for i in range(_layers(cfg)):
        src, dst = f"{pre}encoder.layer.{i}.", f"encoder.{i}."
        for a, b in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj")):
            yield from _wb(f"{dst}attn.{b}", f"{src}attention.attn.{a}", get, keys)
        yield from _wb(f"{dst}ln1", f"{src}attention.LayerNorm", get, keys)
        yield from _wb(f"{dst}mlp.fc1", f"{src}intermediate.dense", get, keys)
        yield from _wb(f"{dst}mlp.fc2", f"{src}output.dense", get, keys)
        yield from _wb(f"{dst}ln2", f"{src}output.LayerNorm", get, keys)
    yield from _wb("pooler", f"{pre}pooler.dense", get, keys)


def _map_mamba(cfg, get: Get, keys: set[str]) -> Iterator[tuple[str, Any]]:
    pre = _strip(keys, ("backbone.",))
    yield "embed.weight", get(f"{pre}embeddings.weight")
    for i in range(_layers(cfg)):
        src, dst = f"{pre}layers.{i}.", f"layers.{i}."
        yield dst + "norm.weight", get(src + "norm.weight")
        for k in ("in_proj.weight", "conv1d.weight", "conv1d.bias", "x_proj.weight", "dt_proj.weight", "dt_proj.bias", "A_log", "D", "out_proj.weight"):
            if f"{src}mixer.{k}" in keys:
                yield f"{dst}mixer.{k}", get(f"{src}mixer.{k}")
    yield "norm.weight", get(f"{pre}norm_f.weight")
    yield "lm_head.weight", get("lm_head.weight") if "lm_head.weight" in keys else get(f"{pre}embeddings.weight")


MAPPERS = {"llama": _map_decoder, "mistral": _map_decoder, "qwen2": _map_decoder, "mixtral": _map_decoder,
           "gpt2": _map_gpt2, "bert": _map_bert, "mpnet": _map_mpnet, "mamba": _map_mamba}


def convert_weights(path: Path, cfg: dict[str, Any], dtype=None) -> dict[str, Any]:
    """Read every *.safetensors shard lazily and return Weightsmith-named tensors."""
    from safetensors import safe_open

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ImportError_(f"no *.safetensors in {path}")
    where: dict[str, Path] = {}
    for f in files:
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                where[k] = f
    handles: dict[Path, Any] = {}

    def get(key: str):
        if key not in where:
            raise ImportError_(f"checkpoint has no tensor {key!r}")
        f = where[key]
        if f not in handles:
            handles[f] = safe_open(str(f), "pt")
        t = handles[f].get_tensor(key)
        return t.to(dtype) if dtype is not None and t.is_floating_point() else t

    out = dict(MAPPERS[cfg["model_type"]](cfg, get, set(where)))
    # conv1d weights in Mamba are E × 1 × k in both; everything else is a direct copy
    return out


def save_weights(state: dict[str, Any], dest: Path) -> Path:
    from safetensors.torch import save_file

    dest.parent.mkdir(parents=True, exist_ok=True)
    # tied tensors share storage; safetensors wants each saved once
    seen: dict[int, str] = {}
    clean, aliases = {}, {}
    for k, v in state.items():
        ptr = v.data_ptr()
        if ptr in seen and v.shape == state[seen[ptr]].shape:
            aliases[k] = seen[ptr]
            continue
        seen[ptr] = k
        clean[k] = v.contiguous()
    save_file(clean, str(dest), metadata={"weightsmith_aliases": json.dumps(aliases)})
    return dest


def import_model(src: str | Path, out_dir: str | Path, dtype: str | None = None) -> dict[str, Any]:
    """Import a Hugging Face model: returns {graph, weights, tokenizer, config}."""
    import torch

    path = resolve(src)
    cfg = read_config(path)
    graph = build_graph(cfg)
    graph["name"] = str(src)
    td = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}.get(dtype or "") if dtype else None
    state = convert_weights(path, cfg, td)
    out = Path(out_dir)
    weights = save_weights(state, out / "weights.safetensors")
    (out / "graph.json").write_text(json.dumps(graph, indent=1))
    tokenizer = str(path) if (path / "tokenizer.json").exists() or (path / "tokenizer_config.json").exists() else None
    return {"graph": graph, "weights": str(weights), "tokenizer": tokenizer, "config": cfg, "source": str(src)}


def slug(src: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(src).lower()).strip("-")
