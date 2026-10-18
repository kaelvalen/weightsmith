"""The wider node catalogue: conv/pool/norm families, recurrent layers, MoE, Mamba,
cross-attention, positional schemes, tensor plumbing, generative pieces and losses.

Imported at the end of registry.py; nodes registered here with an existing name
(Conv2d, MaxPool2d, Flatten, GELU, Attention) replace the simpler originals.
Parameter names returned by `weights` match the generated nn.Module exactly, which
the meta-device smoke test checks.
"""

from __future__ import annotations

import math

from .registry import (
    IN,
    OUT,
    Ins,
    NodeType,
    Outs,
    Param,
    Params,
    Port,
    _float,
    _last_dim,
    _need,
    _rank_at_least,
    _same,
    _same_float,
    register,
)
from .shapes import (
    INT_DTYPES,
    ShapeError,
    SymbolEnv,
    TensorType,
    conv_out,
    conv_transpose_out,
    dim_add,
    dim_div,
    dim_mul,
    fmt_shape,
)


def _rank(x: TensorType, n: int, port: str, env: SymbolEnv, what: str) -> None:
    if len(x.shape) != n:
        raise ShapeError(f"expects {what}, got {fmt_shape(env.resolve_shape(x.shape))}", port)


def _channels(x: TensorType, c, env: SymbolEnv, pname: str = "in_channels") -> None:
    if not env.unify(x.shape[1], c):
        raise ShapeError(f"channels {env.resolve(x.shape[1])} ≠ {pname}={c}", "in")


def _norm_dim(d: int, rank: int) -> int:
    return d % rank if rank else 0


# ------------------------------------------------------------- convolutions

_CONV_PARAMS = {
    "in_channels": Param(None, "int", required=True),
    "out_channels": Param(None, "int", required=True),
    "kernel_size": Param(3, "int"),
    "stride": Param(1, "int"),
    "padding": Param(1, "int"),
    "dilation": Param(1, "int"),
    "groups": Param(1, "int", doc="groups = in_channels for a depthwise convolution"),
    "bias": Param(True, "bool"),
}


def _conv(nd: int):
    layout = {1: "B×C×L", 2: "B×C×H×W", 3: "B×C×D×H×W"}[nd]

    def infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
        x = _need(ins, "in")
        _float(x, "in")
        _rank(x, nd + 2, "in", env, layout)
        _channels(x, p["in_channels"], env)
        g = p["groups"]
        if p["in_channels"] % g or p["out_channels"] % g:
            raise ShapeError(f"groups={g} must divide in_channels and out_channels", "in")
        sp = [conv_out(env.resolve(d), p["kernel_size"], p["stride"], p["padding"], p["dilation"]) for d in x.shape[2:]]
        return {"out": TensorType([x.shape[0], p["out_channels"], *sp], x.dtype)}

    def weights(p: Params) -> dict[str, int]:
        w = {"weight": p["out_channels"] * (p["in_channels"] // p["groups"]) * p["kernel_size"] ** nd}
        if p.get("bias", True):
            w["bias"] = p["out_channels"]
        return w

    return infer, weights


for _nd in (1, 2, 3):
    _inf, _w = _conv(_nd)
    register(NodeType(f"Conv{_nd}d", "convolution", f"{_nd}D convolution over {['', 'B×C×L', 'B×C×H×W', 'B×C×D×H×W'][_nd]}.", IN, OUT, dict(_CONV_PARAMS), _inf, _w))


def _convT_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _rank(x, 4, "in", env, "B×C×H×W")
    _channels(x, p["in_channels"], env)
    sp = [conv_transpose_out(env.resolve(d), p["kernel_size"], p["stride"], p["padding"], p["output_padding"]) for d in x.shape[2:]]
    return {"out": TensorType([x.shape[0], p["out_channels"], *sp], x.dtype)}


register(
    NodeType(
        "ConvTranspose2d",
        "convolution",
        "Transposed 2D convolution (learned upsampling).",
        IN,
        OUT,
        {
            "in_channels": Param(None, "int", required=True),
            "out_channels": Param(None, "int", required=True),
            "kernel_size": Param(2, "int"),
            "stride": Param(2, "int"),
            "padding": Param(0, "int"),
            "output_padding": Param(0, "int"),
            "bias": Param(True, "bool"),
        },
        _convT_infer,
        lambda p: {"weight": p["in_channels"] * p["out_channels"] * p["kernel_size"] ** 2, **({"bias": p["out_channels"]} if p.get("bias", True) else {})},
    )
)


def _pool(nd: int):
    layout = {1: "B×C×L", 2: "B×C×H×W"}[nd]

    def infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
        x = _need(ins, "in")
        _rank(x, nd + 2, "in", env, layout)
        k, pad = p["kernel_size"], p["padding"]
        s = p.get("stride") or k
        sp = [conv_out(env.resolve(d), k, s, pad) for d in x.shape[2:]]
        return {"out": TensorType([x.shape[0], x.shape[1], *sp], x.dtype)}

    return infer


_POOL_PARAMS = {"kernel_size": Param(2, "int"), "stride": Param(None, "int", doc="Defaults to kernel_size"), "padding": Param(0, "int")}
for _nd in (1, 2):
    register(NodeType(f"MaxPool{_nd}d", "pooling", f"Max pooling over {'L' if _nd == 1 else 'H×W'}.", IN, OUT, dict(_POOL_PARAMS), _pool(_nd)))
    register(NodeType(f"AvgPool{_nd}d", "pooling", f"Average pooling over {'L' if _nd == 1 else 'H×W'}.", IN, OUT, dict(_POOL_PARAMS), _pool(_nd)))


def _adaptive(nd: int):
    def infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
        x = _need(ins, "in")
        _rank(x, nd + 2, "in", env, "B×C×L" if nd == 1 else "B×C×H×W")
        o = p["output_size"]
        return {"out": TensorType([x.shape[0], x.shape[1], *([o] * nd)], x.dtype)}

    return infer


register(NodeType("AdaptiveAvgPool1d", "pooling", "Average pooling to a fixed length.", IN, OUT, {"output_size": Param(1, "int")}, _adaptive(1)))
register(NodeType("AdaptiveAvgPool2d", "pooling", "Average pooling to a fixed H×W (1 = global average pooling).", IN, OUT, {"output_size": Param(1, "int")}, _adaptive(2)))


def _upsample_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _rank(x, 4, "in", env, "B×C×H×W")
    s = p["scale_factor"]
    return {"out": TensorType([x.shape[0], x.shape[1], dim_mul([s, env.resolve(x.shape[2])]), dim_mul([s, env.resolve(x.shape[3])])], x.dtype)}


register(
    NodeType("Upsample", "pooling", "Nearest or bilinear upsampling by an integer factor.", IN, OUT,
             {"scale_factor": Param(2, "int"), "mode": Param("nearest", "choice", choices=["nearest", "bilinear"])}, _upsample_infer)
)


def _pixelshuffle_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _rank(x, 4, "in", env, "B×C×H×W")
    r = p["upscale_factor"]
    try:
        c = dim_div(env.resolve(x.shape[1]), r * r)
    except ShapeError:
        raise ShapeError(f"channels {env.resolve(x.shape[1])} not divisible by upscale_factor²={r * r}", "in")
    return {"out": TensorType([x.shape[0], c, dim_mul([r, env.resolve(x.shape[2])]), dim_mul([r, env.resolve(x.shape[3])])], x.dtype)}


register(NodeType("PixelShuffle", "pooling", "Rearranges C·r² channels into an r× larger image.", IN, OUT, {"upscale_factor": Param(2, "int")}, _pixelshuffle_infer))


# ------------------------------------------------------------ normalisation


def _bn(nd: int):
    def infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
        x = _need(ins, "in")
        _float(x, "in")
        ok = len(x.shape) in ((2, 3) if nd == 1 else (4,))
        if not ok:
            raise ShapeError(f"expects {'B×C or B×C×L' if nd == 1 else 'B×C×H×W'}, got {fmt_shape(env.resolve_shape(x.shape))}", "in")
        _channels(x, p["features"], env, "features")
        return {"out": TensorType(list(x.shape), x.dtype)}

    return infer


_BN_PARAMS = {"features": Param(None, "int", required=True), "eps": Param(1e-5, "float"), "momentum": Param(0.1, "float")}
register(NodeType("BatchNorm1d", "norm", "Batch normalisation over B×C or B×C×L.", IN, OUT, dict(_BN_PARAMS), _bn(1), lambda p: {"weight": p["features"], "bias": p["features"]}))
register(NodeType("BatchNorm2d", "norm", "Batch normalisation over B×C×H×W.", IN, OUT, dict(_BN_PARAMS), _bn(2), lambda p: {"weight": p["features"], "bias": p["features"]}))


def _groupnorm_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _rank_at_least(x, 2, "in", env)
    _channels(x, p["channels"], env, "channels")
    if p["channels"] % p["groups"]:
        raise ShapeError(f"groups={p['groups']} must divide channels={p['channels']}", "in")
    return {"out": TensorType(list(x.shape), x.dtype)}


register(
    NodeType("GroupNorm", "norm", "Group normalisation over channel groups.", IN, OUT,
             {"groups": Param(32, "int"), "channels": Param(None, "int", required=True), "eps": Param(1e-5, "float")},
             _groupnorm_infer, lambda p: {"weight": p["channels"], "bias": p["channels"]})
)


def _instnorm_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _rank(x, 4, "in", env, "B×C×H×W")
    _channels(x, p["features"], env, "features")
    return {"out": TensorType(list(x.shape), x.dtype)}


register(
    NodeType("InstanceNorm2d", "norm", "Per-sample, per-channel normalisation (style transfer, GANs).", IN, OUT,
             {"features": Param(None, "int", required=True), "affine": Param(False, "bool"), "eps": Param(1e-5, "float")},
             _instnorm_infer, lambda p: {"weight": p["features"], "bias": p["features"]} if p.get("affine") else {})
)

register(NodeType("L2Normalize", "norm", "Scales vectors to unit length along a dimension (embeddings).", IN, OUT, {"dim": Param(-1, "int")}, _same_float))


# --------------------------------------------------------------- activation

register(NodeType("GELU", "activation", "Gaussian error linear unit.", IN, OUT, {"approximate": Param("none", "choice", choices=["none", "tanh"])}, _same_float))
register(NodeType("LeakyReLU", "activation", "max(x, slope·x).", IN, OUT, {"negative_slope": Param(0.01, "float")}, _same_float))
register(NodeType("ELU", "activation", "Exponential linear unit.", IN, OUT, {"alpha": Param(1.0, "float")}, _same_float))
register(NodeType("SELU", "activation", "Self-normalising ELU.", IN, OUT, {}, _same_float))
register(NodeType("Mish", "activation", "x · tanh(softplus(x)).", IN, OUT, {}, _same_float))
register(NodeType("Softplus", "activation", "log(1 + exp(β·x)) / β.", IN, OUT, {"beta": Param(1.0, "float")}, _same_float))
register(NodeType("Hardswish", "activation", "Piecewise-linear swish (MobileNetV3).", IN, OUT, {}, _same_float))
register(NodeType("LogSoftmax", "activation", "log(softmax(x)) along a dimension.", IN, OUT, {"dim": Param(-1, "int")}, _same_float))


def _glu_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    d = _norm_dim(p["dim"], len(x.shape))
    shape = env.resolve_shape(x.shape)
    try:
        shape[d] = dim_div(shape[d], 2)
    except ShapeError:
        raise ShapeError(f"dim {d} has odd size {shape[d]}", "in")
    return {"out": TensorType(shape, x.dtype)}


register(NodeType("GLU", "activation", "Gated linear unit: a · sigmoid(b) from the two halves of a dimension.", IN, OUT, {"dim": Param(-1, "int")}, _glu_infer))
register(NodeType("DropPath", "activation", "Stochastic depth: drops the whole residual branch per sample.", IN, OUT, {"p": Param(0.1, "float")}, _same))


# ---------------------------------------------------------------- embedding


def _sinus_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _rank(x, 3, "in", env, "B×T×D")
    _last_dim(x, p["dim"], "in", env, "dim")
    if p["dim"] % 2:
        raise ShapeError("dim must be even", "in")
    return {"out": TensorType(list(x.shape), x.dtype)}


register(NodeType("SinusoidalPositional", "embedding", "Adds fixed sine/cosine position encodings (no parameters).", IN, OUT, {"dim": Param(None, "int", required=True)}, _sinus_infer))


def _cls_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _rank(x, 3, "in", env, "B×T×D")
    _last_dim(x, p["dim"], "in", env, "dim")
    return {"out": TensorType([x.shape[0], dim_add(env.resolve(x.shape[1]), 1), x.shape[2]], x.dtype)}


register(NodeType("ClassToken", "embedding", "Prepends a learned [CLS] token: B×T×D → B×(T+1)×D.", IN, OUT, {"dim": Param(None, "int", required=True)}, _cls_infer, lambda p: {"token": p["dim"]}))


def _patch_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _rank(x, 4, "in", env, "B×C×H×W")
    _channels(x, p["in_channels"], env)
    ps = p["patch"]
    h, w = conv_out(env.resolve(x.shape[2]), ps, ps, 0), conv_out(env.resolve(x.shape[3]), ps, ps, 0)
    return {"out": TensorType([x.shape[0], dim_mul([h, w]), p["dim"]], x.dtype)}


register(
    NodeType("PatchEmbed", "embedding", "Image → sequence of patch embeddings (ViT): B×C×H×W → B×(H/p·W/p)×D.", IN, OUT,
             {"in_channels": Param(3, "int"), "dim": Param(None, "int", required=True), "patch": Param(16, "int")},
             _patch_infer, lambda p: {"proj.weight": p["dim"] * p["in_channels"] * p["patch"] ** 2, "proj.bias": p["dim"]},
             lambda p: {"proj": (p["in_channels"] * p["patch"] ** 2, p["dim"])}, lambda p: p["dim"])
)


def _relbias_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _rank_at_least(x, 2, "in", env)
    t = env.resolve(x.shape[1])
    return {"out": TensorType([p["heads"], t, t], "f32")}


register(
    NodeType("RelativePositionBias", "embedding", "Bucketed relative position bias (T5/MPNet), one table shared by all layers: → heads×T×T.", IN, OUT,
             {"heads": Param(None, "int", required=True), "buckets": Param(32, "int"), "max_distance": Param(128, "int"), "bidirectional": Param(True, "bool")},
             _relbias_infer, lambda p: {"table.weight": p["buckets"] * p["heads"]})
)


# ---------------------------------------------------------------- attention


def _kv_dim(p: Params) -> int:
    return (p.get("kv_heads") or p["heads"]) * (p["dim"] // p["heads"])


def _attn_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _rank(x, 3, "in", env, "B×T×D")
    _last_dim(x, p["dim"], "in", env, "dim")
    if p["dim"] % p["heads"]:
        raise ShapeError(f"dim={p['dim']} is not divisible by heads={p['heads']}", "in")
    kv = p.get("kv_heads") or p["heads"]
    if p["heads"] % kv:
        raise ShapeError(f"heads={p['heads']} is not divisible by kv_heads={kv}", "in")
    if p.get("rope") and (p["dim"] // p["heads"]) % 2:
        raise ShapeError("RoPE needs an even head dimension", "in")
    m = ins.get("mask")
    if m is not None:
        if len(m.shape) != 2 or not env.unify(m.shape[0], x.shape[0]) or not env.unify(m.shape[1], x.shape[1]):
            raise ShapeError(f"mask must be B×T, got {fmt_shape(env.resolve_shape(m.shape))}", "mask")
    b = ins.get("bias")
    if b is not None:
        t = env.resolve(x.shape[1])
        bs = env.resolve_shape(b.shape)
        if len(bs) not in (3, 4) or not env.unify(bs[-1], t) or not env.unify(bs[-2], t) or not env.unify(bs[-3], p["heads"]):
            raise ShapeError(f"bias must be heads×T×T ({p['heads']}×{t}×{t}), got {fmt_shape(bs)}", "bias")
    return {"out": TensorType(list(x.shape), x.dtype)}


def _attn_weights(p: Params) -> dict[str, int]:
    d, kvd = p["dim"], _kv_dim(p)
    w: dict[str, int] = {}
    for proj, out in (("q_proj", d), ("k_proj", kvd), ("v_proj", kvd), ("o_proj", d)):
        w[f"{proj}.weight"] = d * out
        if (p.get("o_bias", True) if proj == "o_proj" else p.get("bias", True)):
            w[f"{proj}.bias"] = out
    if p.get("qk_norm") == "rms":
        hd = d // p["heads"]
        w["q_norm.weight"] = hd
        w["k_norm.weight"] = hd
    return w


register(
    NodeType(
        "Attention",
        "attention",
        "Multi-head self-attention: grouped-query (kv_heads), RoPE or NoPE, QK-norm, chunked local attention, optional additive bias.",
        [Port("in"), Port("bias", optional=True, doc="Additive attention bias, heads×T×T (e.g. RelativePositionBias)"),
         Port("mask", optional=True, doc="Padding mask B×T: 1 = token, 0 = padding (keys)")],
        OUT,
        {
            "dim": Param(None, "int", required=True),
            "heads": Param(8, "int"),
            "kv_heads": Param(None, "int", doc="Grouped-query attention: key/value heads (defaults to heads)"),
            "causal": Param(False, "bool"),
            "dropout": Param(0.0, "float"),
            "bias": Param(True, "bool", doc="Bias on q/k/v"),
            "o_bias": Param(True, "bool", doc="Bias on the output projection"),
            "rope": Param(False, "bool", doc="Rotary position embedding on q and k"),
            "rope_theta": Param(10000.0, "float"),
            "rope_scaling": Param(None, "json", doc='e.g. {"type": "llama3", "factor": 32, "low_freq_factor": 1, "high_freq_factor": 4, "original_max_position_embeddings": 8192}'),
            "qk_norm": Param("none", "choice", choices=["none", "rms", "l2"], doc="Normalise q and k per head (rms: learned scale, l2: none)"),
            "chunk": Param(0, "int", doc="Local attention inside chunks of this many tokens (0 = full)"),
            "nope_temp": Param(False, "bool", doc="Llama 4 NoPE layers: scale q by log(position) for long context"),
        },
        _attn_infer,
        _attn_weights,
        lambda p: {"q_proj": (p["dim"], p["dim"]), "k_proj": (p["dim"], _kv_dim(p)), "v_proj": (p["dim"], _kv_dim(p)), "o_proj": (p["dim"], p["dim"])},
        lambda p: p["dim"],
    )
)


def _xattn_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x, c = _need(ins, "in"), _need(ins, "context")
    _float(x, "in")
    _rank(x, 3, "in", env, "B×T×D")
    _rank(c, 3, "context", env, "B×S×Dc")
    _last_dim(x, p["dim"], "in", env, "dim")
    _last_dim(c, p.get("context_dim") or p["dim"], "context", env, "context_dim")
    if not env.unify(x.shape[0], c.shape[0]):
        raise ShapeError("batch of context differs from input", "context")
    if p["dim"] % p["heads"]:
        raise ShapeError(f"dim={p['dim']} is not divisible by heads={p['heads']}", "in")
    return {"out": TensorType(list(x.shape), x.dtype)}


def _xattn_weights(p: Params) -> dict[str, int]:
    d, c = p["dim"], p.get("context_dim") or p["dim"]
    w = {"q_proj.weight": d * d, "k_proj.weight": c * d, "v_proj.weight": c * d, "o_proj.weight": d * d}
    if p.get("bias", True):
        w |= {"q_proj.bias": d, "k_proj.bias": d, "v_proj.bias": d, "o_proj.bias": d}
    return w


register(
    NodeType(
        "CrossAttention",
        "attention",
        "Queries from the input, keys/values from a context sequence (decoders, conditioning).",
        [Port("in"), Port("context")],
        OUT,
        {"dim": Param(None, "int", required=True), "heads": Param(8, "int"), "context_dim": Param(None, "int", doc="Defaults to dim"),
         "dropout": Param(0.0, "float"), "bias": Param(True, "bool")},
        _xattn_infer,
        _xattn_weights,
        lambda p: {"q_proj": (p["dim"], p["dim"]), "k_proj": (p.get("context_dim") or p["dim"], p["dim"]),
                   "v_proj": (p.get("context_dim") or p["dim"], p["dim"]), "o_proj": (p["dim"], p["dim"])},
        lambda p: p["dim"],
    )
)


# ------------------------------------------------------------ sequence mixers


def _rnn(gates: int, kind: str):
    def infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
        x = _need(ins, "in")
        _float(x, "in")
        _rank(x, 3, "in", env, "B×T×F")
        _last_dim(x, p["input"], "in", env, "input")
        h = p["hidden"] * (2 if p["bidirectional"] else 1)
        return {"out": TensorType([x.shape[0], x.shape[1], h], x.dtype), "last": TensorType([x.shape[0], h], x.dtype)}

    def weights(p: Params) -> dict[str, int]:
        h, dirs = p["hidden"], 2 if p["bidirectional"] else 1
        w: dict[str, int] = {}
        for layer in range(p["layers"]):
            fin = p["input"] if layer == 0 else h * dirs
            for d in range(dirs):
                sfx = f"l{layer}" + ("_reverse" if d else "")
                w[f"rnn.weight_ih_{sfx}"] = gates * h * fin
                w[f"rnn.weight_hh_{sfx}"] = gates * h * h
                if p.get("bias", True):
                    w[f"rnn.bias_ih_{sfx}"] = gates * h
                    w[f"rnn.bias_hh_{sfx}"] = gates * h
        return w

    register(
        NodeType(
            kind,
            "sequence",
            {"LSTM": "Long short-term memory.", "GRU": "Gated recurrent unit.", "RNN": "Elman RNN (tanh)."}[kind]
            + " out: every step; last: the final step (both directions concatenated).",
            IN,
            [Port("out"), Port("last")],
            {"input": Param(None, "int", required=True), "hidden": Param(None, "int", required=True), "layers": Param(1, "int"),
             "bidirectional": Param(False, "bool"), "dropout": Param(0.0, "float"), "bias": Param(True, "bool")},
            infer,
            weights,
            out_features=lambda p: p["hidden"] * (2 if p["bidirectional"] else 1),
        )
    )


_rnn(4, "LSTM")
_rnn(3, "GRU")
_rnn(1, "RNN")


def _mamba_dims(p: Params) -> tuple[int, int, int]:
    e = p["expand"] * p["dim"]
    r = p.get("dt_rank") or math.ceil(p["dim"] / 16)
    return e, r, p["d_state"]


def _mamba_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _rank(x, 3, "in", env, "B×T×D")
    _last_dim(x, p["dim"], "in", env, "dim")
    return {"out": TensorType(list(x.shape), x.dtype)}


def _mamba_weights(p: Params) -> dict[str, int]:
    e, r, n = _mamba_dims(p)
    d = p["dim"]
    return {
        "in_proj.weight": d * 2 * e,
        "conv1d.weight": e * p["d_conv"],
        "conv1d.bias": e,
        "x_proj.weight": e * (r + 2 * n),
        "dt_proj.weight": r * e,
        "dt_proj.bias": e,
        "A_log": e * n,
        "D": e,
        "out_proj.weight": e * d,
    }


register(
    NodeType(
        "Mamba",
        "sequence",
        "Selective state-space mixer (Mamba): causal, linear in sequence length.",
        IN,
        OUT,
        {"dim": Param(None, "int", required=True), "d_state": Param(16, "int"), "d_conv": Param(4, "int"), "expand": Param(2, "int"),
         "dt_rank": Param(None, "int", doc="Defaults to ceil(dim / 16)")},
        _mamba_infer,
        _mamba_weights,
        lambda p: {"in_proj": (p["dim"], 2 * _mamba_dims(p)[0]), "out_proj": (_mamba_dims(p)[0], p["dim"])},
        lambda p: p["dim"],
    )
)


def _moe_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _last_dim(x, p["dim"], "in", env, "dim")
    if not 1 <= p["top_k"] <= p["num_experts"]:
        raise ShapeError(f"top_k={p['top_k']} must be in 1..num_experts={p['num_experts']}", "in")
    return {"out": TensorType(list(x.shape), x.dtype)}


def _moe_weights(p: Params) -> dict[str, int]:
    d, h, e = p["dim"], p["hidden"], p["num_experts"]
    w = {"router.weight": d * e, "experts.gate_up_proj": e * d * 2 * h, "experts.down_proj": e * h * d}
    if p.get("shared_hidden"):
        s = p["shared_hidden"]
        w |= {"shared_expert.gate_proj.weight": d * s, "shared_expert.up_proj.weight": d * s, "shared_expert.down_proj.weight": s * d}
    return w


register(
    NodeType(
        "MoE",
        "layers",
        "Mixture of SwiGLU experts. softmax: top-k weights renormalised over the output (Mixtral); "
        "sigmoid: the score scales the expert's input (Llama 4). Optional always-on shared expert.",
        IN,
        OUT,
        {"dim": Param(None, "int", required=True), "hidden": Param(None, "int", required=True), "num_experts": Param(8, "int"),
         "top_k": Param(2, "int"), "router": Param("softmax", "choice", choices=["softmax", "sigmoid"]),
         "shared_hidden": Param(0, "int", doc="Shared expert width (0 = none)")},
        _moe_infer,
        _moe_weights,
        lambda p: {"router": (p["dim"], p["num_experts"])},
        lambda p: p["dim"],
    )
)


# ---------------------------------------------------------------- structure


def _binary_same(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    from .registry import _broadcast

    return {"out": _broadcast(_need(ins, "a"), _need(ins, "b"), env)}


register(NodeType("Sub", "structure", "Element-wise a − b (broadcasting).", [Port("a"), Port("b")], OUT, {}, _binary_same))


def _flatten_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    shape = env.resolve_shape(x.shape)
    n = len(shape)
    s, e = p["start_dim"] % n if n else 0, p["end_dim"] % n if n else 0
    if s > e:
        raise ShapeError(f"start_dim={p['start_dim']} is after end_dim={p['end_dim']} for {fmt_shape(shape)}", "in")
    return {"out": TensorType(shape[:s] + [dim_mul(shape[s : e + 1])] + shape[e + 1 :], x.dtype)}


register(NodeType("Flatten", "structure", "Merges dims start_dim..end_dim into one.", IN, OUT, {"start_dim": Param(1, "int"), "end_dim": Param(-1, "int")}, _flatten_infer))


def _unflatten_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    shape = env.resolve_shape(x.shape)
    d = p["dim"] % len(shape)
    sizes = list(p["sizes"])
    if sizes.count(-1) > 1:
        raise ShapeError("at most one -1 in sizes", "in")
    known = math.prod(s for s in sizes if s != -1)
    if -1 in sizes:
        sizes[sizes.index(-1)] = dim_div(shape[d], known)
    elif isinstance(shape[d], int) and shape[d] != known:
        raise ShapeError(f"cannot split {shape[d]} into {sizes}", "in")
    return {"out": TensorType(shape[:d] + sizes + shape[d + 1 :], x.dtype)}


register(NodeType("Unflatten", "structure", "Splits one dim into several (one size may be -1).", IN, OUT, {"dim": Param(1, "int"), "sizes": Param([-1, 2], "shape", required=True)}, _unflatten_infer))


def _transpose_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    s = env.resolve_shape(x.shape)
    a, b = p["dim0"] % len(s), p["dim1"] % len(s)
    s[a], s[b] = s[b], s[a]
    return {"out": TensorType(s, x.dtype)}


register(NodeType("Transpose", "structure", "Swaps two dimensions.", IN, OUT, {"dim0": Param(1, "int"), "dim1": Param(2, "int")}, _transpose_infer))


def _permute_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    s = env.resolve_shape(x.shape)
    dims = list(p["dims"])
    if sorted(d % len(s) for d in dims) != list(range(len(s))):
        raise ShapeError(f"dims {dims} is not a permutation of {len(s)} dimensions", "in")
    return {"out": TensorType([s[d] for d in dims], x.dtype)}


register(NodeType("Permute", "structure", "Reorders all dimensions.", IN, OUT, {"dims": Param([0, 2, 1], "shape", required=True)}, _permute_infer))


def _select_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    s = env.resolve_shape(x.shape)
    d = p["dim"] % len(s)
    if isinstance(s[d], int) and not -s[d] <= p["index"] < s[d]:
        raise ShapeError(f"index {p['index']} out of range for size {s[d]}", "in")
    return {"out": TensorType(s[:d] + s[d + 1 :], x.dtype)}


register(NodeType("Select", "structure", "Picks one index along a dimension, removing it ([CLS] token, last step).", IN, OUT, {"dim": Param(1, "int"), "index": Param(0, "int")}, _select_infer))


def _slice_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    s = env.resolve_shape(x.shape)
    d = p["dim"] % len(s)
    start, end = p["start"], p.get("end")
    if isinstance(s[d], int):
        stop = s[d] if end is None else min(end if end >= 0 else s[d] + end, s[d])
        s[d] = max(0, stop - start)
    elif end is not None and end >= 0:
        s[d] = end - start
    else:
        s[d] = f"{s[d]}-{start}" if start else s[d]
    return {"out": TensorType(s, x.dtype)}


register(NodeType("Slice", "structure", "x[start:end] along a dimension.", IN, OUT, {"dim": Param(1, "int"), "start": Param(0, "int"), "end": Param(None, "int")}, _slice_infer))


def _unsqueeze_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    s = env.resolve_shape(x.shape)
    d = p["dim"] % (len(s) + 1)
    return {"out": TensorType(s[:d] + [1] + s[d:], x.dtype)}


register(NodeType("Unsqueeze", "structure", "Inserts a size-1 dimension.", IN, OUT, {"dim": Param(1, "int")}, _unsqueeze_infer))


def _chunk_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    s = env.resolve_shape(x.shape)
    d = p["dim"] % len(s)
    try:
        s[d] = dim_div(s[d], 2)
    except ShapeError:
        raise ShapeError(f"dim {d} has odd size {s[d]}", "in")
    return {"a": TensorType(list(s), x.dtype), "b": TensorType(list(s), x.dtype)}


register(NodeType("Chunk", "structure", "Splits a dimension into two equal halves a and b.", IN, [Port("a"), Port("b")], {"dim": Param(-1, "int")}, _chunk_infer))


def _reduce_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    s = env.resolve_shape(x.shape)
    if len(s) < 1:
        raise ShapeError("cannot reduce a scalar", "in")
    d = p["dim"] % len(s)
    m = ins.get("mask")
    if m is not None:
        if len(m.shape) > len(s) or len(m.shape) <= d:
            raise ShapeError(f"mask {fmt_shape(env.resolve_shape(m.shape))} must cover the leading dims up to dim {d}", "mask")
        env.unify_shapes(m.shape, x.shape[: len(m.shape)], "mask", "mask vs input ")
    out = s[:d] + ([1] if p["keepdim"] else []) + s[d + 1 :]
    return {"out": TensorType(out, x.dtype)}


register(NodeType("Reduce", "structure", "mean, sum or max over one dimension (pooling over tokens); padding positions excluded when a mask is connected.",
                  [Port("in"), Port("mask", optional=True, doc="1 = keep, 0 = padding; covers the leading dims (e.g. B×T)")], OUT,
                  {"op": Param("mean", "choice", choices=["mean", "sum", "max"]), "dim": Param(1, "int"), "keepdim": Param(False, "bool")}, _reduce_infer))


def _matmul_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    a, b = _need(ins, "a"), _need(ins, "b")
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ShapeError("both operands need rank ≥ 2", "b")
    if not env.unify(a.shape[-1], b.shape[-2]):
        raise ShapeError(f"inner dims differ: {env.resolve(a.shape[-1])} vs {env.resolve(b.shape[-2])}", "b")
    from .registry import _broadcast

    batch = _broadcast(TensorType(a.shape[:-2], a.dtype), TensorType(b.shape[:-2], a.dtype), env).shape
    return {"out": TensorType(batch + [a.shape[-2], b.shape[-1]], a.dtype)}


register(NodeType("MatMul", "structure", "Batched matrix product a @ b.", [Port("a"), Port("b")], OUT, {}, _matmul_infer))
register(NodeType("Scale", "structure", "Multiplies by a constant.", IN, OUT, {"factor": Param(1.0, "float")}, _same))
register(NodeType("Identity", "structure", "Passes the input through (placeholders, ablations).", IN, OUT, {}, _same))


# --------------------------------------------------------------- generative


def _reparam_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    mu, lv = _need(ins, "mu"), _need(ins, "logvar")
    env.unify_shapes(mu.shape, lv.shape, "logvar", "logvar vs mu ")
    return {"out": TensorType(list(mu.shape), mu.dtype)}


register(NodeType("Reparameterize", "generative", "VAE sampling z = μ + ε·exp(½·log σ²) (returns μ at eval).", [Port("mu"), Port("logvar")], OUT, {}, _reparam_infer))


# --------------------------------------------------------------------- loss

_LOSS = {"weight": Param(1.0, "float", doc="Weight in the summed total loss")}
_TARGET = {"target_field": Param("y", "str")}


def _pred_target(pred_port: str, int_target: bool = False):
    def infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
        pred = _need(ins, pred_port)
        _float(pred, pred_port)
        t = ins.get("target")
        if t is not None:
            if int_target:
                if t.dtype not in INT_DTYPES:
                    raise ShapeError(f"target must be class ids (i64), got {t.dtype}", "target")
                env.unify_shapes(t.shape, pred.shape[:-1], "target", f"target vs {pred_port}[:-1] ")
            else:
                env.unify_shapes(t.shape, pred.shape, "target", f"target vs {pred_port} ")
        return {"loss": TensorType([], "f32")}

    return infer


_OPT_T = Port("target", optional=True, doc="Unconnected: dataset field y")
register(NodeType("BCEWithLogits", "loss", "Binary / multi-label cross-entropy on logits.", [Port("pred"), _OPT_T], [Port("loss")], {**_LOSS, **_TARGET}, _pred_target("pred"), loss=True))
register(NodeType("L1", "loss", "Mean absolute error.", [Port("pred"), _OPT_T], [Port("loss")], {**_LOSS, **_TARGET}, _pred_target("pred"), loss=True))
register(NodeType("Huber", "loss", "Smooth L1: quadratic below delta, linear above.", [Port("pred"), _OPT_T], [Port("loss")], {**_LOSS, **_TARGET, "delta": Param(1.0, "float")}, _pred_target("pred"), loss=True))
register(NodeType("NLL", "loss", "Negative log-likelihood on log-probabilities (after LogSoftmax).", [Port("log_probs"), _OPT_T], [Port("loss")], {**_LOSS, **_TARGET}, _pred_target("log_probs", True), loss=True))


def _kldiv_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    a, b = _need(ins, "log_probs"), _need(ins, "target_probs")
    env.unify_shapes(b.shape, a.shape, "target_probs", "target vs input ")
    return {"loss": TensorType([], "f32")}


register(NodeType("KLDiv", "loss", "KL(target ‖ input) for distillation; input is log-probabilities.", [Port("log_probs"), Port("target_probs")], [Port("loss")], {**_LOSS}, _kldiv_infer, loss=True))


def _contrastive_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    a, b = _need(ins, "a"), _need(ins, "b")
    _rank(a, 2, "a", env, "B×D")
    env.unify_shapes(b.shape, a.shape, "b", "b vs a ")
    n = ins.get("negatives")
    if n is not None:
        _rank(n, 2, "negatives", env, "N×D")
        if not env.unify(n.shape[1], a.shape[1]):
            raise ShapeError(f"negatives have dim {env.resolve(n.shape[1])}, a has {env.resolve(a.shape[1])}", "negatives")
    return {"loss": TensorType([], "f32")}


register(
    NodeType("Contrastive", "loss", "In-batch contrastive loss (InfoNCE / MNRL / CLIP): row i of a matches row i of b; "
             "every other row of b, and any hard negatives, are wrong answers.",
             [Port("a"), Port("b"), Port("negatives", optional=True, doc="Hard negatives, N×D, appended as extra wrong answers")],
             [Port("loss")], {**_LOSS, "scale": Param(20.0, "float", doc="1 / temperature"), "symmetric": Param(False, "bool")},
             _contrastive_infer, loss=True)
)


def _triplet_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    a = _need(ins, "anchor")
    for port in ("positive", "negative"):
        env.unify_shapes(_need(ins, port).shape, a.shape, port, f"{port} vs anchor ")
    return {"loss": TensorType([], "f32")}


register(NodeType("Triplet", "loss", "max(d(a,p) − d(a,n) + margin, 0).", [Port("anchor"), Port("positive"), Port("negative")], [Port("loss")], {**_LOSS, "margin": Param(1.0, "float")}, _triplet_infer, loss=True))


def _klg_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    mu, lv = _need(ins, "mu"), _need(ins, "logvar")
    env.unify_shapes(lv.shape, mu.shape, "logvar", "logvar vs mu ")
    return {"loss": TensorType([], "f32")}


register(NodeType("KLGaussian", "loss", "VAE prior term: KL(N(μ, σ²) ‖ N(0, 1)), summed over latent dims.", [Port("mu"), Port("logvar")], [Port("loss")], {**_LOSS}, _klg_infer, loss=True))

