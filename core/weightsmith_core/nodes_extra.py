"""More of the everyday catalogue: activations, element-wise math, tensor plumbing,
3-D pooling and norms, channel attention, attention biases, linear attention,
diffusion timesteps, vector quantisation, graph convolution and extra losses.

Imported at the end of nodes_more.py. Same contract: shape rules mirror
web/src/ir/rules_extra.ts, and `weights` names match the generated nn.Module.
"""

from __future__ import annotations

import math
import re

from .nodes_more import (
    _BN_PARAMS,
    _LOSS,
    _OPT_T,
    _POOL_PARAMS,
    _TARGET,
    _channels,
    _pred_target,
    _rank,
)
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
    dim_add,
    dim_mul,
    fmt_shape,
)


def _last(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _last_dim(x, p["dim"], "in", env, "dim")
    return {"out": TensorType(list(x.shape), x.dtype)}


# --------------------------------------------------------------- activation

register(NodeType("ReLU6", "activation", "min(max(x, 0), 6) (MobileNet).", IN, OUT, {}, _same_float))
register(NodeType("HardTanh", "activation", "Clips to [min, max] with a straight-through gradient inside.", IN, OUT, {"min": Param(-1.0, "float"), "max": Param(1.0, "float")}, _same_float))
register(NodeType("Hardsigmoid", "activation", "Piecewise-linear sigmoid (MobileNetV3 gates).", IN, OUT, {}, _same_float))
register(NodeType("Softsign", "activation", "x / (1 + |x|).", IN, OUT, {}, _same_float))
register(NodeType("Swish", "activation", "x · sigmoid(β·x) with a learned β per channel (β = 1 is SiLU).", IN, OUT, {"dim": Param(None, "int", required=True)}, _last, lambda p: {"beta": p["dim"]}))


def _prelu_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    if p["channels"] != 1:
        _rank_at_least(x, 2, "in", env)
        _channels(x, p["channels"], env, "channels")
    return {"out": TensorType(list(x.shape), x.dtype)}


register(NodeType("PReLU", "activation", "Leaky ReLU with a learned slope, one per channel (dim 1) or one shared (channels = 1).", IN, OUT,
                  {"channels": Param(1, "int"), "init": Param(0.25, "float")}, _prelu_infer, lambda p: {"weight": p["channels"]}))


def _gumbel_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    return {"out": TensorType(list(x.shape), x.dtype)}


register(NodeType("GumbelSoftmax", "activation", "Differentiable sample from a categorical over the last dim (hard: one-hot forward, soft gradient).", IN, OUT,
                  {"tau": Param(1.0, "float"), "hard": Param(False, "bool")}, _gumbel_infer))
register(NodeType("Dropout2d", "activation", "Drops whole channels of B×C×H×W.", IN, OUT, {"p": Param(0.1, "float")}, _same))


# --------------------------------------------------------------------- math

MATH_OPS = ["abs", "exp", "log", "log1p", "sqrt", "rsqrt", "square", "neg", "reciprocal", "sin", "cos", "sign", "erf"]
register(NodeType("Math", "math", "Element-wise function: abs, exp, log, sqrt, square, sin, …", IN, OUT, {"op": Param("exp", "choice", choices=MATH_OPS)}, _same_float))
register(NodeType("Clamp", "math", "Limits values to [min, max] (either may be empty).", IN, OUT, {"min": Param(None, "float"), "max": Param(None, "float")}, _same))
register(NodeType("Pow", "math", "x ** exponent.", IN, OUT, {"exponent": Param(2.0, "float")}, _same_float))
register(NodeType("AddConst", "math", "x + value (shifts, epsilons).", IN, OUT, {"value": Param(1.0, "float")}, _same))


def _binary(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    from .registry import _broadcast

    return {"out": _broadcast(_need(ins, "a"), _need(ins, "b"), env)}


AB = [Port("a"), Port("b")]
register(NodeType("Div", "math", "Element-wise a / b (broadcasting).", AB, OUT, {}, _binary))
register(NodeType("Maximum", "math", "Element-wise max(a, b).", AB, OUT, {}, _binary))
register(NodeType("Minimum", "math", "Element-wise min(a, b).", AB, OUT, {}, _binary))


def _einsum_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    eq = p["equation"].replace(" ", "")
    m = re.fullmatch(r"([a-zA-Z]+)(?:,([a-zA-Z]+))?->([a-zA-Z]*)", eq)
    if not m:
        raise ShapeError(f"equation {p['equation']!r} must look like 'bij,bjk->bik' (letters only)", "a")
    sizes: dict[str, object] = {}
    for port, letters in (("a", m.group(1)), ("b", m.group(2))):
        if letters is None:
            if ins.get("b") is not None:
                raise ShapeError("the equation names one operand, but b is connected", "b")
            continue
        x = _need(ins, port)
        if len(x.shape) != len(letters):
            raise ShapeError(f"{port} has rank {len(x.shape)}, the equation gives it {len(letters)} ({letters})", port)
        for ch, d in zip(letters, env.resolve_shape(x.shape)):
            if ch in sizes and not env.unify(sizes[ch], d):
                raise ShapeError(f"index {ch}: {env.resolve(sizes[ch])} ≠ {d}", port)
            sizes.setdefault(ch, d)
    out = m.group(3)
    missing = [c for c in out if c not in sizes]
    if missing:
        raise ShapeError(f"output index {''.join(missing)} appears in no input", "a")
    a = _need(ins, "a")
    return {"out": TensorType([env.resolve(sizes[c]) for c in out], a.dtype)}


register(NodeType("Einsum", "math", "torch.einsum over one or two operands, e.g. 'bhqd,bhkd->bhqk'.", [Port("a"), Port("b", optional=True)], OUT,
                  {"equation": Param("bij,bjk->bik", "str")}, _einsum_infer))


def _cumsum_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _rank_at_least(x, 1, "in", env)
    return {"out": TensorType(list(x.shape), x.dtype)}


register(NodeType("Cumsum", "math", "Running sum along a dimension.", IN, OUT, {"dim": Param(1, "int")}, _cumsum_infer))


# ---------------------------------------------------------------- structure


def _stack_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    a, b = _need(ins, "a"), _need(ins, "b")
    env.unify_shapes(b.shape, a.shape, "b", "b vs a ")
    s = env.resolve_shape(a.shape)
    d = p["dim"] % (len(s) + 1)
    return {"out": TensorType(s[:d] + [2] + s[d:], a.dtype)}


register(NodeType("Stack", "structure", "Stacks a and b along a new dimension.", AB, OUT, {"dim": Param(1, "int")}, _stack_infer))


def _squeeze_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    s = env.resolve_shape(x.shape)
    if not s:
        raise ShapeError("cannot squeeze a scalar", "in")
    d = p["dim"] % len(s)
    if s[d] != 1 and isinstance(s[d], int):
        raise ShapeError(f"dim {d} has size {s[d]}, not 1", "in")
    return {"out": TensorType(s[:d] + s[d + 1 :], x.dtype)}


register(NodeType("Squeeze", "structure", "Removes a size-1 dimension.", IN, OUT, {"dim": Param(1, "int")}, _squeeze_infer))


def _pad_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    s = env.resolve_shape(x.shape)
    pad = list(p["pad"])
    if len(pad) % 2 or len(pad) // 2 > len(s):
        raise ShapeError(f"pad {pad} needs (left, right) pairs for at most {len(s)} trailing dims", "in")
    for i in range(len(pad) // 2):
        d = len(s) - 1 - i
        if pad[2 * i] + pad[2 * i + 1]:
            s[d] = dim_add(s[d], pad[2 * i] + pad[2 * i + 1])
    return {"out": TensorType(s, x.dtype)}


register(NodeType("Pad", "structure", "F.pad: (left, right) pairs from the last dim backwards, e.g. [1, 1, 2, 2] pads W by 1 and H by 2.", IN, OUT,
                  {"pad": Param([1, 1], "shape", required=True), "mode": Param("constant", "choice", choices=["constant", "reflect", "replicate", "circular"]),
                   "value": Param(0.0, "float")}, _pad_infer))


def _repeat_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    s = env.resolve_shape(x.shape)
    reps = list(p["repeats"])
    if len(reps) != len(s):
        raise ShapeError(f"repeats {reps} needs one entry per dim of {fmt_shape(s)}", "in")
    return {"out": TensorType([d if r == 1 else dim_mul([r, d]) for d, r in zip(s, reps)], x.dtype)}


register(NodeType("Repeat", "structure", "Tiles the tensor: repeats[i] copies along dim i.", IN, OUT, {"repeats": Param([1, 2], "shape", required=True)}, _repeat_infer))


def _topk_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    s = env.resolve_shape(x.shape)
    if not s:
        raise ShapeError("cannot take top-k of a scalar", "in")
    d = p["dim"] % len(s)
    if isinstance(s[d], int) and p["k"] > s[d]:
        raise ShapeError(f"k={p['k']} > size {s[d]}", "in")
    s[d] = p["k"]
    return {"values": TensorType(list(s), x.dtype), "indices": TensorType(list(s), "i64")}


register(NodeType("TopK", "structure", "The k largest values along a dimension and their indices.", IN, [Port("values"), Port("indices")],
                  {"k": Param(1, "int"), "dim": Param(-1, "int")}, _topk_infer))


def _argmax_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    s = env.resolve_shape(x.shape)
    if not s:
        raise ShapeError("cannot argmax a scalar", "in")
    d = p["dim"] % len(s)
    return {"out": TensorType(s[:d] + ([1] if p["keepdim"] else []) + s[d + 1 :], "i64")}


register(NodeType("ArgMax", "structure", "Index of the largest value along a dimension (i64; not differentiable).", IN, OUT,
                  {"dim": Param(-1, "int"), "keepdim": Param(False, "bool")}, _argmax_infer))


def _onehot_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    if x.dtype not in INT_DTYPES:
        raise ShapeError(f"expects class ids (i64), got {x.dtype}", "in")
    return {"out": TensorType(list(x.shape) + [p["classes"]], "f32")}


register(NodeType("OneHot", "structure", "Class ids → one-hot float vectors (adds a last dim of size classes).", IN, OUT, {"classes": Param(None, "int", required=True)}, _onehot_infer))


def _cast_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    return {"out": TensorType(list(x.shape), p["dtype"])}


register(NodeType("Cast", "structure", "Changes the dtype (f32 for math on ids, i64 for indices).", IN, OUT,
                  {"dtype": Param("f32", "choice", choices=["f32", "f16", "bf16", "i64", "i32"])}, _cast_infer))
register(NodeType("StopGradient", "structure", "Passes values through, blocks gradients (x.detach(): targets, EMA teachers).", IN, OUT, {}, _same))


def _param_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    shape = list(p["shape"])
    if not shape or not all(isinstance(d, int) and d > 0 for d in shape):
        raise ShapeError(f"shape {shape} must be positive integers", None)
    return {"out": TensorType(shape, "f32")}


register(NodeType("Parameter", "structure", "A learned tensor with no input (queries, prompts, latents); broadcasts against the batch.", [], OUT,
                  {"shape": Param([1, 16], "shape", required=True), "init": Param("normal", "choice", choices=["normal", "zeros", "ones"]),
                   "std": Param(0.02, "float")}, _param_infer, lambda p: {"weight": math.prod(p["shape"])}))


# ------------------------------------------------------------------ pooling


def _pool3_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _rank(x, 5, "in", env, "B×C×D×H×W")
    k, pad = p["kernel_size"], p["padding"]
    st = p.get("stride") or k
    return {"out": TensorType([x.shape[0], x.shape[1], *[conv_out(env.resolve(d), k, st, pad) for d in x.shape[2:]]], x.dtype)}


def _bn3_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _rank(x, 5, "in", env, "B×C×D×H×W")
    _channels(x, p["features"], env, "features")
    return {"out": TensorType(list(x.shape), x.dtype)}


register(NodeType("MaxPool3d", "pooling", "3-D max pooling over B×C×D×H×W (video, volumes).", IN, OUT, dict(_POOL_PARAMS), _pool3_infer))
register(NodeType("AvgPool3d", "pooling", "3-D average pooling over B×C×D×H×W.", IN, OUT, dict(_POOL_PARAMS), _pool3_infer))


def _amax_infer(nd: int):
    def infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
        x = _need(ins, "in")
        _rank(x, nd + 2, "in", env, "B×C×L" if nd == 1 else "B×C×H×W")
        return {"out": TensorType([x.shape[0], x.shape[1]] + [p["output_size"]] * nd, x.dtype)}

    return infer


register(NodeType("AdaptiveMaxPool1d", "pooling", "Max pooling to a fixed length.", IN, OUT, {"output_size": Param(1, "int")}, _amax_infer(1)))
register(NodeType("AdaptiveMaxPool2d", "pooling", "Max pooling to a fixed H×W.", IN, OUT, {"output_size": Param(1, "int")}, _amax_infer(2)))


def _interp_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    s = env.resolve_shape(x.shape)
    if len(s) < 3:
        raise ShapeError(f"expects B×C×spatial…, got {fmt_shape(s)}", "in")
    size = list(p["size"])
    if len(size) != len(s) - 2:
        raise ShapeError(f"size {size} needs one entry per spatial dim ({len(s) - 2})", "in")
    return {"out": TensorType(s[:2] + size, x.dtype)}


register(NodeType("Interpolate", "pooling", "Resizes the spatial dims to an exact size (nearest, bilinear, …).", IN, OUT,
                  {"size": Param([16, 16], "shape", required=True), "mode": Param("nearest", "choice", choices=["nearest", "linear", "bilinear", "bicubic", "trilinear", "area"])},
                  _interp_infer))


# --------------------------------------------------------------------- norm

register(NodeType("BatchNorm3d", "norm", "Batch normalisation over B×C×D×H×W.", IN, OUT, dict(_BN_PARAMS), _bn3_infer,
                  lambda p: {"weight": p["features"], "bias": p["features"]}))
register(NodeType("LayerScale", "norm", "Learned per-channel scale on a residual branch (CaiT, ConvNeXt), starts small.", IN, OUT,
                  {"dim": Param(None, "int", required=True), "init": Param(1e-5, "float")}, _last, lambda p: {"gamma": p["dim"]}))


def _se_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _rank_at_least(x, 3, "in", env)
    _channels(x, p["channels"], env, "channels")
    return {"out": TensorType(list(x.shape), x.dtype)}


def _se_hidden(p: Params) -> int:
    return max(1, p["channels"] // p["reduction"])


register(NodeType("SqueezeExcite", "layers", "Channel attention: global pool → bottleneck MLP → sigmoid gate per channel (SENet, EfficientNet).", IN, OUT,
                  {"channels": Param(None, "int", required=True), "reduction": Param(4, "int")}, _se_infer,
                  lambda p: {"fc1.weight": p["channels"] * _se_hidden(p), "fc1.bias": _se_hidden(p), "fc2.weight": _se_hidden(p) * p["channels"], "fc2.bias": p["channels"]}))


# ---------------------------------------------------------------- attention


def _seq_bias_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _rank_at_least(x, 2, "in", env)
    t = env.resolve(x.shape[1])
    return {"out": TensorType([p["heads"], t, t], "f32")}


register(NodeType("ALiBi", "attention", "Attention with linear biases: a fixed per-head distance penalty → heads×T×T (into Attention.bias).", IN, OUT,
                  {"heads": Param(None, "int", required=True), "causal": Param(True, "bool")}, _seq_bias_infer))
register(NodeType("SlidingWindowBias", "attention", "Masks keys farther than window tokens away → heads×T×T (Mistral-style local attention; add to other biases).", IN, OUT,
                  {"heads": Param(None, "int", required=True), "window": Param(256, "int"), "causal": Param(True, "bool")}, _seq_bias_infer))


def _linattn_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _rank(x, 3, "in", env, "B×T×D")
    _last_dim(x, p["dim"], "in", env, "dim")
    if p["dim"] % p["heads"]:
        raise ShapeError(f"dim={p['dim']} is not divisible by heads={p['heads']}", "in")
    return {"out": TensorType(list(x.shape), x.dtype)}


register(NodeType("LinearAttention", "attention", "Kernelised attention, O(T) in sequence length (elu + 1 feature map; causal = running sums).", IN, OUT,
                  {"dim": Param(None, "int", required=True), "heads": Param(8, "int"), "causal": Param(False, "bool")}, _linattn_infer,
                  lambda p: {f"{k}.{w}": p["dim"] * p["dim"] if w == "weight" else p["dim"] for k in ("q_proj", "k_proj", "v_proj", "o_proj") for w in ("weight", "bias")},
                  lambda p: {k: (p["dim"], p["dim"]) for k in ("q_proj", "k_proj", "v_proj", "o_proj")}, lambda p: p["dim"]))


# ---------------------------------------------------------------- embedding


def _timestep_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    t = _need(ins, "in")
    if len(t.shape) != 1:
        raise ShapeError(f"expects one timestep per sample (B), got {fmt_shape(env.resolve_shape(t.shape))}", "in")
    if p["dim"] % 2:
        raise ShapeError("dim must be even", "in")
    return {"out": TensorType([t.shape[0], p["dim"]], "f32")}


def _ts_hidden(p: Params) -> int:
    return p.get("hidden") or 4 * p["dim"]


register(NodeType("TimestepEmbedding", "embedding", "Diffusion timestep t (B) → sinusoidal features → MLP → B×dim.", IN, OUT,
                  {"dim": Param(None, "int", required=True), "hidden": Param(None, "int", doc="Defaults to 4·dim"), "max_period": Param(10000.0, "float")},
                  _timestep_infer,
                  lambda p: {"fc1.weight": p["dim"] * _ts_hidden(p), "fc1.bias": _ts_hidden(p), "fc2.weight": _ts_hidden(p) * p["dim"], "fc2.bias": p["dim"]}))


# --------------------------------------------------------------- generative


def _vq_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _last_dim(x, p["dim"], "in", env, "dim")
    return {"out": TensorType(list(x.shape), x.dtype), "indices": TensorType(list(x.shape[:-1]), "i64"), "commit": TensorType([], "f32")}


register(NodeType("VectorQuantizer", "generative", "VQ-VAE codebook: snaps each vector to its nearest code (straight-through); "
                  "commit is the codebook + β·commitment loss, wire it into an AuxLoss.", IN,
                  [Port("out"), Port("indices"), Port("commit")],
                  {"codes": Param(512, "int"), "dim": Param(None, "int", required=True), "beta": Param(0.25, "float")}, _vq_infer,
                  lambda p: {"codebook.weight": p["codes"] * p["dim"]}))


def _diffuse_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _rank_at_least(x, 2, "in", env)
    return {"noisy": TensorType(list(x.shape), x.dtype), "t": TensorType([x.shape[0]], "i64"), "noise": TensorType(list(x.shape), x.dtype)}


register(NodeType("DiffusionForward", "generative", "DDPM forward process: random step t per sample, x_t = √ᾱ_t·x + √(1−ᾱ_t)·ε. "
                  "Train a denoiser on (noisy, t) to predict noise.", IN, [Port("noisy"), Port("t"), Port("noise")],
                  {"steps": Param(1000, "int"), "schedule": Param("linear", "choice", choices=["linear", "cosine"]),
                   "beta_start": Param(1e-4, "float"), "beta_end": Param(0.02, "float")}, _diffuse_infer))


# -------------------------------------------------------------------- graph


def _gconv_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x, a = _need(ins, "x"), _need(ins, "adj")
    _float(x, "x")
    _rank(x, 3, "x", env, "B×N×F")
    _last_dim(x, p["in"], "x", env, "in")
    _rank(a, 3, "adj", env, "B×N×N")
    n = x.shape[1]
    if not env.unify(a.shape[0], x.shape[0]) or not env.unify(a.shape[1], n) or not env.unify(a.shape[2], n):
        raise ShapeError(f"adj must be B×N×N for x {fmt_shape(env.resolve_shape(x.shape))}, got {fmt_shape(env.resolve_shape(a.shape))}", "adj")
    return {"out": TensorType([x.shape[0], n, p["out"]], x.dtype)}


register(NodeType("GraphConv", "graph", "Graph convolution (Kipf & Welling): Â·X·W with a symmetric-normalised adjacency plus self loops.",
                  [Port("x", doc="Node features B×N×F"), Port("adj", doc="Adjacency B×N×N (0/1 or weights)")], OUT,
                  {"in": Param(None, "int", required=True), "out": Param(None, "int", required=True), "bias": Param(True, "bool")}, _gconv_infer,
                  lambda p: {"lin.weight": p["in"] * p["out"], **({"lin.bias": p["out"]} if p["bias"] else {})},
                  lambda p: {"lin": (p["in"], p["out"])}, lambda p: p["out"]))


def _gpool_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _rank(x, 3, "in", env, "B×N×F")
    return {"out": TensorType([x.shape[0], x.shape[2]], x.dtype)}


register(NodeType("GraphPool", "graph", "Readout: mean, sum or max over the nodes of each graph, B×N×F → B×F.", IN, OUT,
                  {"op": Param("mean", "choice", choices=["mean", "sum", "max"])}, _gpool_infer))


# --------------------------------------------------------------------- loss


def _aux_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    return {"loss": TensorType([], "f32")}


register(NodeType("AuxLoss", "loss", "Adds any tensor (its mean) to the total loss: VQ commitment, MoE balance, regularisers.", IN, [Port("loss")], {**_LOSS}, _aux_infer, loss=True))
register(NodeType("Focal", "loss", "Cross-entropy that down-weights easy examples: (1 − p)^γ · CE (detection, imbalance).", [Port("logits"), _OPT_T], [Port("loss")],
                  {**_LOSS, **_TARGET, "gamma": Param(2.0, "float")}, _pred_target("logits", True), loss=True))
register(NodeType("Dice", "loss", "1 − Dice overlap of sigmoid(pred) and a 0/1 target (segmentation).", [Port("pred"), _OPT_T], [Port("loss")],
                  {**_LOSS, **_TARGET, "smooth": Param(1.0, "float")}, _pred_target("pred"), loss=True))


def _gnll_infer(p: Params, ins: Ins, env: SymbolEnv) -> Outs:
    out = _pred_target("mean")(p, ins, env)
    env.unify_shapes(_need(ins, "var").shape, _need(ins, "mean").shape, "var", "var vs mean ")
    return out


register(NodeType("GaussianNLL", "loss", "Negative log-likelihood of the target under N(mean, var): regression with learned uncertainty.",
                  [Port("mean"), Port("var", doc="Positive variance (e.g. after Softplus)"), _OPT_T], [Port("loss")], {**_LOSS, **_TARGET}, _gnll_infer, loss=True))
register(NodeType("CosineEmbedding", "loss", "1 − cos(a, b): pulls paired embeddings together (target-free).", AB, [Port("loss")], {**_LOSS},
                  lambda p, ins, env: (env.unify_shapes(_need(ins, "b").shape, _need(ins, "a").shape, "b", "b vs a "), {"loss": TensorType([], "f32")})[1], loss=True))
