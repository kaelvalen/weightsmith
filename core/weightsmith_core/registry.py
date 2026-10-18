"""Node type registry: ports, parameters, shape rules and parameter counts.

This module is the single source of truth for node semantics. The canvas keeps
a TypeScript mirror (web/src/ir/registry.ts) for instant feedback; both are
checked against the same fixtures (tests/test_cross_ts.py + web/src/ir/cross.test.ts).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .shapes import (
    FLOAT_DTYPES,
    INT_DTYPES,
    ShapeError,
    SymbolEnv,
    TensorType,
    conv_out,
    dim_add,
    dim_mul,
    fmt_shape,
    numel,
)

Params = dict[str, Any]
Ins = dict[str, TensorType | None]
Outs = dict[str, TensorType]


@dataclass
class Port:
    name: str
    optional: bool = False
    doc: str = ""


@dataclass
class Param:
    default: Any = None
    kind: str = "int"  # int | float | bool | str | shape | choice | code | json
    required: bool = False
    choices: list[Any] | None = None
    doc: str = ""


@dataclass
class NodeType:
    name: str
    category: str
    doc: str
    inputs: list[Port]
    outputs: list[Port]
    params: dict[str, Param] = field(default_factory=dict)
    infer: Callable[[Params, Ins, SymbolEnv], Outs] | None = None
    # parameter name -> element count, named exactly as the generated nn.Module names them
    weights: Callable[[Params], dict[str, int]] = lambda p: {}
    # LoRA-able linear sub-modules: target -> (in_features, out_features)
    linear_targets: Callable[[Params], dict[str, tuple[int, int]]] = lambda p: {}
    # output feature size, for bottleneck adapters
    out_features: Callable[[Params], int | None] = lambda p: None
    loss: bool = False


REGISTRY: dict[str, NodeType] = {}


def register(t: NodeType) -> NodeType:
    REGISTRY[t.name] = t
    return t


def get(type_name: str) -> NodeType | None:
    return REGISTRY.get(type_name)


def expand_defaults(type_name: str, params: Params) -> Params:
    t = REGISTRY.get(type_name)
    out = dict(params)
    if t is None:
        return out
    for k, p in t.params.items():
        if k not in out and not p.required:
            out[k] = p.default
    return out


# ------------------------------------------------------------------ helpers


def _need(ins: Ins, port: str) -> TensorType:
    t = ins.get(port)
    if t is None:
        raise ShapeError("input not connected", port)
    return t


def _float(t: TensorType, port: str) -> None:
    if t.dtype not in FLOAT_DTYPES:
        raise ShapeError(f"expects a float tensor, got {t.dtype}", port)


def _rank_at_least(t: TensorType, n: int, port: str, env: SymbolEnv) -> None:
    if len(t.shape) < n:
        raise ShapeError(f"expects rank ≥ {n}, got {fmt_shape(env.resolve_shape(t.shape))}", port)


def _last_dim(t: TensorType, size: Any, port: str, env: SymbolEnv, pname: str) -> None:
    _rank_at_least(t, 1, port, env)
    if not env.unify(t.shape[-1], size):
        raise ShapeError(
            f"last dim {env.resolve(t.shape[-1])} ≠ {pname}={size} (input {fmt_shape(env.resolve_shape(t.shape))})",
            port,
        )


def _same(params: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    return {"out": TensorType(list(x.shape), x.dtype)}


def _same_float(params: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    return {"out": TensorType(list(x.shape), x.dtype)}


IN = [Port("in")]
OUT = [Port("out")]


# ----------------------------------------------------------------------- io


def _input_infer(params: Params, ins: Ins, env: SymbolEnv) -> Outs:
    return {"out": TensorType(list(params["shape"]), params.get("dtype", "f32"))}


register(
    NodeType(
        "Input",
        "io",
        "Graph input. Bound to a dataset field at training time.",
        [],
        OUT,
        {
            "shape": Param(["B", 784], "shape", required=True, doc="Leading B is the batch dimension"),
            "dtype": Param("f32", "choice", choices=["f32", "f16", "bf16", "i64", "i32", "bool"]),
            "field": Param("x", "str", doc="Dataset field fed into this input"),
        },
        _input_infer,
    )
)

register(
    NodeType(
        "Output",
        "io",
        "Named graph output. Marks a tensor the model returns.",
        [Port("in")],
        [],
        {},
        lambda p, ins, env: {},
    )
)


# ------------------------------------------------------------------- layers


def _linear_infer(params: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _last_dim(x, params["in"], "in", env, "in")
    return {"out": TensorType(list(x.shape[:-1]) + [params["out"]], x.dtype)}


def _linear_weights(p: Params) -> dict[str, int]:
    w = {"weight": 0 if p.get("tie_to") else p["in"] * p["out"]}  # a tied weight is counted on its owner
    if p.get("bias", True):
        w["bias"] = p["out"]
    return w


register(
    NodeType(
        "Linear",
        "layers",
        "y = x Wᵀ + b over the last dimension.",
        IN,
        OUT,
        {
            "in": Param(None, "int", required=True),
            "out": Param(None, "int", required=True),
            "bias": Param(True, "bool"),
            "tie_to": Param(None, "str", doc="Share the weight of an Embedding node (e.g. an LM head tied to the input embedding)"),
        },
        _linear_infer,
        _linear_weights,
        lambda p: {"weight": (p["in"], p["out"])},
        lambda p: p["out"],
    )
)


def _mlp_infer(params: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _last_dim(x, params["dim"], "in", env, "dim")
    out = params.get("out") or params["dim"]
    return {"out": TensorType(list(x.shape[:-1]) + [out], x.dtype)}


def _mlp_weights(p: Params) -> dict[str, int]:
    d, h, o = p["dim"], p["hidden"], p.get("out") or p["dim"]
    return {"fc1.weight": d * h, "fc1.bias": h, "fc2.weight": h * o, "fc2.bias": o}


register(
    NodeType(
        "MLP",
        "layers",
        "Two-layer feed-forward: fc2(act(fc1(x))).",
        IN,
        OUT,
        {
            "dim": Param(None, "int", required=True),
            "hidden": Param(None, "int", required=True),
            "out": Param(None, "int", doc="Defaults to dim"),
            "act": Param("gelu", "choice", choices=["gelu", "gelu_tanh", "relu", "silu", "tanh"]),
            "dropout": Param(0.0, "float"),
        },
        _mlp_infer,
        _mlp_weights,
        lambda p: {"fc1": (p["dim"], p["hidden"]), "fc2": (p["hidden"], p.get("out") or p["dim"])},
        lambda p: p.get("out") or p["dim"],
    )
)


def _gated_weights(p: Params) -> dict[str, int]:
    d, h, o = p["dim"], p["hidden"], p.get("out") or p["dim"]
    w = {"gate_proj.weight": d * h, "up_proj.weight": d * h, "down_proj.weight": h * o}
    if p.get("bias"):
        w |= {"gate_proj.bias": h, "up_proj.bias": h, "down_proj.bias": o}
    return w


register(
    NodeType(
        "GatedMLP",
        "layers",
        "Gated feed-forward (SwiGLU): down(act(gate(x)) · up(x)).",
        IN,
        OUT,
        {
            "dim": Param(None, "int", required=True),
            "hidden": Param(None, "int", required=True),
            "out": Param(None, "int", doc="Defaults to dim"),
            "act": Param("silu", "choice", choices=["silu", "gelu"]),
            "bias": Param(False, "bool"),
        },
        _mlp_infer,
        _gated_weights,
        lambda p: {"gate_proj": (p["dim"], p["hidden"]), "up_proj": (p["dim"], p["hidden"]), "down_proj": (p["hidden"], p.get("out") or p["dim"])},
        lambda p: p.get("out") or p["dim"],
    )
)


def _embedding_infer(params: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    if x.dtype not in INT_DTYPES:
        raise ShapeError(f"expects integer token ids, got {x.dtype}", "in")
    return {"out": TensorType(list(x.shape) + [params["dim"]], "f32")}


register(
    NodeType(
        "Embedding",
        "embedding",
        "Lookup table: integer ids → vectors.",
        IN,
        OUT,
        {"num": Param(None, "int", required=True, doc="Vocabulary size"), "dim": Param(None, "int", required=True)},
        _embedding_infer,
        lambda p: {"weight": p["num"] * p["dim"]},
        out_features=lambda p: p["dim"],
    )
)


def _posemb_infer(params: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    if len(x.shape) != 3:
        raise ShapeError(f"expects B×T×D, got {fmt_shape(env.resolve_shape(x.shape))}", "in")
    _last_dim(x, params["dim"], "in", env, "dim")
    t = env.resolve(x.shape[1])
    if isinstance(t, int) and t + params.get("offset", 0) > params["max_len"]:
        raise ShapeError(f"sequence length {t} + offset {params.get('offset', 0)} > max_len={params['max_len']}", "in")
    return {"out": TensorType(list(x.shape), x.dtype)}


register(
    NodeType(
        "PositionalEmbedding",
        "embedding",
        "Adds a learned position vector to each step of B×T×D.",
        IN,
        OUT,
        {"max_len": Param(512, "int"), "dim": Param(None, "int", required=True),
         "offset": Param(0, "int", doc="First position id (RoBERTa/MPNet start at 2)")},
        _posemb_infer,
        lambda p: {"weight": p["max_len"] * p["dim"]},
        out_features=lambda p: p["dim"],
    )
)


# --------------------------------------------------------------------- norm


def _norm_infer(params: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    _last_dim(x, params["dim"], "in", env, "dim")
    return {"out": TensorType(list(x.shape), x.dtype)}


register(
    NodeType(
        "LayerNorm",
        "norm",
        "Layer normalisation over the last dimension.",
        IN,
        OUT,
        {"dim": Param(None, "int", required=True), "eps": Param(1e-5, "float")},
        _norm_infer,
        lambda p: {"weight": p["dim"], "bias": p["dim"]},
        out_features=lambda p: p["dim"],
    )
)

register(
    NodeType(
        "RMSNorm",
        "norm",
        "Root-mean-square normalisation over the last dimension.",
        IN,
        OUT,
        {"dim": Param(None, "int", required=True), "eps": Param(1e-6, "float")},
        _norm_infer,
        lambda p: {"weight": p["dim"]},
        out_features=lambda p: p["dim"],
    )
)


# --------------------------------------------------------------- activation

for _act, _doc in [
    ("ReLU", "max(0, x)."),
    ("SiLU", "x · sigmoid(x)."),
    ("Tanh", "Hyperbolic tangent."),
    ("Sigmoid", "Logistic sigmoid."),
]:
    register(NodeType(_act, "activation", _doc, IN, OUT, {}, _same_float))

register(
    NodeType("Softmax", "activation", "Softmax along a dimension.", IN, OUT, {"dim": Param(-1, "int")}, _same_float)
)

register(NodeType("Dropout", "activation", "Zeroes elements with probability p during training.", IN, OUT, {"p": Param(0.1, "float")}, _same))


# ---------------------------------------------------------------- attention


# ---------------------------------------------------------------- structure


def _broadcast(a: TensorType, b: TensorType, env: SymbolEnv) -> TensorType:
    sa, sb = env.resolve_shape(a.shape), env.resolve_shape(b.shape)
    n = max(len(sa), len(sb))
    sa = [1] * (n - len(sa)) + sa
    sb = [1] * (n - len(sb)) + sb
    out = []
    for x, y in zip(sa, sb):
        if x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        elif env.unify(x, y):
            out.append(env.resolve(x))
        else:
            raise ShapeError(f"cannot broadcast {fmt_shape(sa)} with {fmt_shape(sb)}", "b")
    if a.dtype != b.dtype:
        raise ShapeError(f"dtype mismatch {a.dtype} vs {b.dtype}", "b")
    return TensorType(out, a.dtype)


def _binary_infer(params: Params, ins: Ins, env: SymbolEnv) -> Outs:
    return {"out": _broadcast(_need(ins, "a"), _need(ins, "b"), env)}


register(NodeType("Add", "structure", "Element-wise a + b (broadcasting). Residual connections.", [Port("a"), Port("b")], OUT, {}, _binary_infer))
register(NodeType("Mul", "structure", "Element-wise a · b (broadcasting). Gating.", [Port("a"), Port("b")], OUT, {}, _binary_infer))


def _concat_infer(params: Params, ins: Ins, env: SymbolEnv) -> Outs:
    a, b = _need(ins, "a"), _need(ins, "b")
    if len(a.shape) != len(b.shape):
        raise ShapeError(f"rank mismatch {fmt_shape(a.shape)} vs {fmt_shape(b.shape)}", "b")
    d = params["dim"] % len(a.shape)
    out = []
    for i, (x, y) in enumerate(zip(a.shape, b.shape)):
        if i == d:
            out.append(dim_add(env.resolve(x), env.resolve(y)))
        elif env.unify(x, y):
            out.append(env.resolve(x))
        else:
            raise ShapeError(f"dim {i} mismatch {env.resolve(x)} vs {env.resolve(y)}", "b")
    return {"out": TensorType(out, a.dtype)}


register(NodeType("Concat", "structure", "Concatenate a and b along a dimension.", [Port("a"), Port("b")], OUT, {"dim": Param(-1, "int")}, _concat_infer))


def _reshape_infer(params: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    target = list(params["shape"])
    if target.count(-1) > 1:
        raise ShapeError("at most one -1 in reshape", "in")
    a = numel(env.resolve_shape(x.shape[1:]))
    rest = [d for d in target[1:] if d != -1]
    b = numel(rest)
    if -1 in target:
        if a is not None and b:
            if a % b:
                raise ShapeError(f"cannot reshape {fmt_shape(env.resolve_shape(x.shape))} into {fmt_shape(target)}", "in")
            target[target.index(-1)] = a // b
        else:
            target[target.index(-1)] = dim_mul(env.resolve_shape(x.shape[1:]))
    elif a is not None and b is not None and a != b:
        raise ShapeError(f"cannot reshape {fmt_shape(env.resolve_shape(x.shape))} into {fmt_shape(target)}", "in")
    if target and not env.unify(x.shape[0], target[0]):
        raise ShapeError("reshape must keep the batch dimension", "in")
    return {"out": TensorType(target, x.dtype)}


register(NodeType("Reshape", "structure", "Reshape, keeping the leading batch dimension.", IN, OUT, {"shape": Param(["B", -1], "shape", required=True)}, _reshape_infer))


def _meanpool_infer(params: Params, ins: Ins, env: SymbolEnv) -> Outs:
    x = _need(ins, "in")
    _float(x, "in")
    d = params["dim"] % max(len(x.shape), 1)
    if len(x.shape) < 2:
        raise ShapeError("expects rank ≥ 2", "in")
    return {"out": TensorType([s for i, s in enumerate(x.shape) if i != d], x.dtype)}


register(NodeType("MeanPool", "structure", "Mean over one dimension (e.g. the sequence).", IN, OUT, {"dim": Param(1, "int")}, _meanpool_infer))


# --------------------------------------------------------------------- loss


def _ce_infer(params: Params, ins: Ins, env: SymbolEnv) -> Outs:
    logits = _need(ins, "logits")
    _float(logits, "logits")
    _rank_at_least(logits, 2, "logits", env)
    target = ins.get("target")
    if target is not None:
        if target.dtype not in INT_DTYPES:
            raise ShapeError(f"target must be class ids (i64), got {target.dtype}", "target")
        env.unify_shapes(target.shape, logits.shape[:-1], "target", "target vs logits[:-1] ")
    return {"loss": TensorType([], "f32")}


def _mse_infer(params: Params, ins: Ins, env: SymbolEnv) -> Outs:
    pred = _need(ins, "pred")
    _float(pred, "pred")
    target = ins.get("target")
    if target is not None:
        env.unify_shapes(target.shape, pred.shape, "target", "target vs pred ")
    return {"loss": TensorType([], "f32")}


_LOSS_PARAMS = {"weight": Param(1.0, "float", doc="Weight in the summed total loss")}

register(
    NodeType(
        "CrossEntropy",
        "loss",
        "Classification loss. target defaults to the dataset label field.",
        [Port("logits"), Port("target", optional=True, doc="Unconnected: dataset field y")],
        [Port("loss")],
        {**_LOSS_PARAMS, "label_smoothing": Param(0.0, "float"), "ignore_index": Param(-100, "int", doc="Target value that is skipped (padding, prompt tokens)"),
         "target_field": Param("y", "str")},
        _ce_infer,
        loss=True,
    )
)

register(
    NodeType(
        "MSE",
        "loss",
        "Mean squared error. target defaults to the dataset label field.",
        [Port("pred"), Port("target", optional=True, doc="Unconnected: dataset field y")],
        [Port("loss")],
        {**_LOSS_PARAMS, "target_field": Param("y", "str")},
        _mse_infer,
        loss=True,
    )
)


# --------------------------------------------------- graph-level node types
# Ports and shapes of these depend on graph context; validate.py resolves them.

register(
    NodeType(
        "Subgraph",
        "structure",
        "Instance of a reusable block, optionally repeated N× with weight tying.",
        [],
        [],
        {
            "graph": Param(None, "str", required=True),
            "repeat": Param(1, "int"),
            "tie_weights": Param(False, "bool", doc="All repeats share one set of weights"),
            "args": Param({}, "json", doc="Values for the block's $parameters, e.g. {\"C\": 128}"),
        },
    )
)

register(
    NodeType(
        "PythonModule",
        "custom",
        "Escape hatch: your own nn.Module plus an infer_shape contract.",
        [],
        [],
        {
            "code": Param("", "code", required=True),
            "class_name": Param("Block", "str"),
            "init": Param({}, "json", doc="Keyword arguments for the constructor"),
            "signature": Param(None, "json", doc="Cached port signature from the daemon"),
        },
    )
)


def catalog() -> list[dict[str, Any]]:
    """Registry as JSON, served to the canvas."""
    out = []
    for t in REGISTRY.values():
        out.append(
            {
                "type": t.name,
                "category": t.category,
                "doc": t.doc,
                "inputs": [{"name": p.name, "optional": p.optional, "doc": p.doc} for p in t.inputs],
                "outputs": [{"name": p.name} for p in t.outputs],
                "params": {
                    k: {"default": p.default, "kind": p.kind, "required": p.required, "choices": p.choices, "doc": p.doc}
                    for k, p in t.params.items()
                },
                "loss": t.loss,
            }
        )
    return out


# the wider catalogue registers itself (and replaces a few simpler originals)
from . import nodes_more  # noqa: E402,F401
