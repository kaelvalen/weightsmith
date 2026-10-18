"""IR → readable PyTorch.

Each node type has a template that produces constructor lines and a forward
expression. Nodes are emitted in topological order and every emitted line
carries a ``# node: <id>`` comment; ``line_map`` maps model.py line numbers back
to node ids so runtime errors land on the canvas.
"""

from __future__ import annotations

import keyword
import textwrap
from dataclasses import dataclass, field
from typing import Any

from weightsmith_core import ir
from weightsmith_core.registry import REGISTRY, expand_defaults
from weightsmith_core.shapes import fmt_shape
from weightsmith_core.validate import Analysis, analyze

from .blocks import BLOCKS

_RESERVED = {
    "self", "training", "forward", "parameters", "modules", "children", "state_dict", "train", "eval", "to", "cuda",
    "float", "half", "type", "apply", "zero_grad", "buffers", "named_parameters", "out", "loss", "F", "nn", "torch",
    "math", "input", "sum", "min", "max", "len", "range", "list", "dict", "id",
}


class CodegenError(Exception):
    def __init__(self, analysis: Analysis):
        msgs = "; ".join(f"{i.node or ''}: {i.message}" for i in analysis.errors[:5])
        super().__init__(f"graph does not validate: {msgs}")
        self.analysis = analysis


def py_name(node_id: str) -> str:
    name = node_id.replace("-", "_")
    if keyword.iskeyword(name) or name in _RESERVED:
        name += "_"
    return name


def class_name(graph_name: str) -> str:
    parts = [p for p in "".join(c if c.isalnum() else " " for c in graph_name).split() if p]
    name = "".join(p[:1].upper() + p[1:] for p in parts) or "Block"
    return name if name[0].isalpha() else "B" + name


@dataclass
class Emitter:
    lines: list[str] = field(default_factory=list)
    line_nodes: dict[int, str] = field(default_factory=dict)

    def emit(self, text: str = "", node: str | None = None) -> None:
        for ln in text.split("\n"):
            self.lines.append(ln)
            if node is not None:
                self.line_nodes[len(self.lines)] = node

    def source(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


def _r(v: Any) -> str:
    return repr(v)


class Sym:
    """A block parameter inside a parametric subgraph class: renders as its Python expression."""

    def __init__(self, expr: str):
        self.expr = expr

    def __repr__(self) -> str:
        return self.expr

    __str__ = __repr__


def _kw(p: dict[str, Any], key: str, default: Any, name: str | None = None) -> str:
    """", key=value" only when it differs from the constructor default (keeps the code short)."""
    v = p.get(key)
    return "" if v == default and not isinstance(v, Sym) else f", {name or key}={_r(v)}"


# block dependencies: a block's source uses these other blocks
_NEEDS = {"MoE": {"MoE", "GatedMLP"}}


def _module_ctor(t: str, p: dict[str, Any]) -> tuple[str | None, set[str]]:
    """(constructor expression, building blocks needed). None: stateless node."""
    match t:
        case "Linear":
            return f"nn.Linear({p['in']}, {p['out']}{'' if p['bias'] else ', bias=False'})", set()
        case "Conv1d" | "Conv2d" | "Conv3d":
            extra = _kw(p, "stride", 1) + _kw(p, "padding", 0) + _kw(p, "dilation", 1) + _kw(p, "groups", 1) + _kw(p, "bias", True)
            return f"nn.{t}({p['in_channels']}, {p['out_channels']}, kernel_size={p['kernel_size']}{extra})", set()
        case "ConvTranspose2d":
            extra = _kw(p, "padding", 0) + _kw(p, "output_padding", 0) + _kw(p, "bias", True)
            return f"nn.ConvTranspose2d({p['in_channels']}, {p['out_channels']}, kernel_size={p['kernel_size']}, stride={p['stride']}{extra})", set()
        case "MaxPool1d" | "MaxPool2d" | "AvgPool1d" | "AvgPool2d":
            return f"nn.{t}({p['kernel_size']}, stride={p.get('stride') or p['kernel_size']}{_kw(p, 'padding', 0)})", set()
        case "AdaptiveAvgPool1d" | "AdaptiveAvgPool2d":
            return f"nn.{t}({p['output_size']})", set()
        case "Upsample":
            return f"nn.Upsample(scale_factor={p['scale_factor']}, mode={_r(p['mode'])})", set()
        case "PixelShuffle":
            return f"nn.PixelShuffle({p['upscale_factor']})", set()
        case "Embedding":
            return f"nn.Embedding({p['num']}, {p['dim']})", set()
        case "PositionalEmbedding":
            return f"PositionalEmbedding({p['max_len']}, {p['dim']})", {"PositionalEmbedding"}
        case "SinusoidalPositional":
            return f"SinusoidalPositional({p['dim']})", {"SinusoidalPositional"}
        case "ClassToken":
            return f"ClassToken({p['dim']})", {"ClassToken"}
        case "PatchEmbed":
            return f"PatchEmbed({p['in_channels']}, {p['dim']}, patch={p['patch']})", {"PatchEmbed"}
        case "RelativePositionBias":
            extra = _kw(p, "buckets", 32) + _kw(p, "max_distance", 128) + _kw(p, "bidirectional", True)
            return f"RelativePositionBias({p['heads']}{extra})", {"RelativePositionBias"}
        case "LayerNorm":
            return f"nn.LayerNorm({p['dim']}, eps={_r(p['eps'])})", set()
        case "RMSNorm":
            return f"nn.RMSNorm({p['dim']}, eps={_r(p['eps'])})", set()
        case "BatchNorm1d" | "BatchNorm2d":
            return f"nn.{t}({p['features']}{_kw(p, 'eps', 1e-5)}{_kw(p, 'momentum', 0.1)})", set()
        case "GroupNorm":
            return f"nn.GroupNorm({p['groups']}, {p['channels']}{_kw(p, 'eps', 1e-5)})", set()
        case "InstanceNorm2d":
            return f"nn.InstanceNorm2d({p['features']}{_kw(p, 'affine', False)}{_kw(p, 'eps', 1e-5)})", set()
        case "MLP":
            out = p.get("out") or p["dim"]
            return f"MLP({p['dim']}, {p['hidden']}, {out}, act={_r(p['act'])}, dropout={_r(p['dropout'])})", {"MLP"}
        case "GatedMLP":
            out = p.get("out") or p["dim"]
            b = ", bias=True" if p.get("bias") else ""
            return f"GatedMLP({p['dim']}, {p['hidden']}, {out}, act={_r(p['act'])}{b})", {"GatedMLP"}
        case "MoE":
            extra = _kw(p, "router", "softmax") + _kw(p, "shared_hidden", 0)
            return f"MoE({p['dim']}, {p['hidden']}, num_experts={p['num_experts']}, top_k={p['top_k']}{extra})", _NEEDS["MoE"]
        case "Attention":
            extra = ""
            if p.get("kv_heads") and p["kv_heads"] != p["heads"]:
                extra += f", kv_heads={p['kv_heads']}"
            if not p.get("o_bias", True):
                extra += ", o_bias=False"
            if p.get("rope"):
                extra += f", rope=True, rope_theta={_r(p['rope_theta'])}"
            extra += _kw(p, "qk_norm", "none") + _kw(p, "chunk", 0) + _kw(p, "nope_temp", False)
            return (
                f"Attention({p['dim']}, heads={p['heads']}{extra}, causal={p['causal']}, dropout={_r(p['dropout'])}, bias={p['bias']})",
                {"Attention"},
            )
        case "CrossAttention":
            extra = _kw(p, "context_dim", None) + _kw(p, "dropout", 0.0) + _kw(p, "bias", True)
            return f"CrossAttention({p['dim']}, heads={p['heads']}{extra})", {"CrossAttention"}
        case "LSTM" | "GRU" | "RNN":
            extra = _kw(p, "layers", 1) + _kw(p, "bidirectional", False) + _kw(p, "dropout", 0.0) + _kw(p, "bias", True)
            return f"Recurrent({t!r}, {p['input']}, {p['hidden']}{extra})", {"Recurrent"}
        case "Mamba":
            extra = _kw(p, "d_state", 16) + _kw(p, "d_conv", 4) + _kw(p, "expand", 2) + _kw(p, "dt_rank", None)
            return f"Mamba({p['dim']}{extra})", {"Mamba"}
        case "Dropout":
            return f"nn.Dropout({_r(p['p'])})", set()
        case "DropPath":
            return f"DropPath({_r(p['p'])})", {"DropPath"}
    return None, set()


_MODULE_CALL = {
    "Linear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose2d", "MaxPool1d", "MaxPool2d", "AvgPool1d", "AvgPool2d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "Upsample", "PixelShuffle", "Embedding", "PositionalEmbedding",
    "SinusoidalPositional", "ClassToken", "PatchEmbed", "RelativePositionBias", "LayerNorm", "RMSNorm", "BatchNorm1d",
    "BatchNorm2d", "GroupNorm", "InstanceNorm2d", "MLP", "GatedMLP", "MoE", "Mamba", "Dropout", "DropPath",
    "LSTM", "GRU", "RNN",
}


def _expr(t: str, p: dict[str, Any], mod: str, a: dict[str, str], rank: int | None = None) -> str:
    """Forward expression. ``a`` maps input port name → variable."""
    x = a.get("in")
    if t in _MODULE_CALL:
        return f"{mod}({x})"
    match t:
        case "Attention":
            return f"{mod}({x}, {a['bias']})" if a.get("bias") else f"{mod}({x})"
        case "CrossAttention":
            return f"{mod}({x}, {a['context']})"
        case "GELU":
            return f"F.gelu({x}, approximate='tanh')" if p.get("approximate") == "tanh" else f"F.gelu({x})"
        case "ReLU":
            return f"F.relu({x})"
        case "LeakyReLU":
            return f"F.leaky_relu({x}, {_r(p['negative_slope'])})"
        case "ELU":
            return f"F.elu({x}, alpha={_r(p['alpha'])})"
        case "SELU":
            return f"F.selu({x})"
        case "SiLU":
            return f"F.silu({x})"
        case "Mish":
            return f"F.mish({x})"
        case "Softplus":
            return f"F.softplus({x}, beta={_r(p['beta'])})"
        case "Hardswish":
            return f"F.hardswish({x})"
        case "Tanh":
            return f"torch.tanh({x})"
        case "Sigmoid":
            return f"torch.sigmoid({x})"
        case "Softmax":
            return f"F.softmax({x}, dim={p['dim']})"
        case "LogSoftmax":
            return f"F.log_softmax({x}, dim={p['dim']})"
        case "GLU":
            return f"F.glu({x}, dim={p['dim']})"
        case "L2Normalize":
            return f"F.normalize({x}, dim={p['dim']})"
        case "Add":
            return f"{a['a']} + {a['b']}"
        case "Sub":
            return f"{a['a']} - {a['b']}"
        case "Mul":
            return f"{a['a']} * {a['b']}"
        case "MatMul":
            return f"{a['a']} @ {a['b']}"
        case "Scale":
            return f"{x} * {_r(p['factor'])}"
        case "Identity":
            return f"{x}"
        case "Concat":
            return f"torch.cat([{a['a']}, {a['b']}], dim={p['dim']})"
        case "Flatten":
            return f"{x}.flatten({p['start_dim']}{'' if p['end_dim'] == -1 else ', ' + str(p['end_dim'])})"
        case "Unflatten":
            return f"{x}.unflatten({p['dim']}, {tuple(p['sizes'])!r})"
        case "Reshape":
            dims = ", ".join(f"{x}.shape[0]" if i == 0 else str(d) for i, d in enumerate(p["shape"]))
            return f"{x}.reshape({dims})"
        case "Transpose":
            return f"{x}.transpose({p['dim0']}, {p['dim1']})"
        case "Permute":
            return f"{x}.permute({', '.join(str(d) for d in p['dims'])})"
        case "Select":
            return f"{x}.select({p['dim']}, {p['index']})"
        case "Slice":
            d = p["dim"] % rank if rank and isinstance(p["dim"], int) else p["dim"]
            end = "" if p.get("end") is None else p["end"]
            if isinstance(d, int):
                start = p["start"] or ""
                index = ", ".join([":"] * d + [f"{start}:{end}"])
                return f"{x}[{index}]"
            return f"{x}.narrow({d}, {p['start']}, {x}.shape[{d}] - {p['start']})"
        case "Unsqueeze":
            return f"{x}.unsqueeze({p['dim']})"
        case "Chunk":
            return f"{x}.chunk(2, dim={p['dim']})"
        case "Reduce":
            op = {"mean": "mean", "sum": "sum", "max": "amax"}[p["op"]]
            return f"{x}.{op}(dim={p['dim']}{', keepdim=True' if p['keepdim'] else ''})"
        case "MeanPool":
            return f"{x}.mean(dim={p['dim']})"
        case "Reparameterize":
            return f"({a['mu']} + torch.randn_like({a['mu']}) * torch.exp(0.5 * {a['logvar']})) if self.training else {a['mu']}"
    raise NotImplementedError(t)


def _loss_expr(t: str, p: dict[str, Any], a: dict[str, str]) -> str:
    match t:
        case "CrossEntropy":
            ls = f", label_smoothing={_r(p['label_smoothing'])}" if p["label_smoothing"] else ""
            return f"F.cross_entropy({a['logits']}.flatten(0, -2).float(), {a['target']}.flatten(){ls})"
        case "MSE":
            return f"F.mse_loss({a['pred']}.float(), {a['target']}.float())"
        case "BCEWithLogits":
            return f"F.binary_cross_entropy_with_logits({a['pred']}.float(), {a['target']}.float())"
        case "L1":
            return f"F.l1_loss({a['pred']}.float(), {a['target']}.float())"
        case "Huber":
            return f"F.huber_loss({a['pred']}.float(), {a['target']}.float(), delta={_r(p['delta'])})"
        case "NLL":
            return f"F.nll_loss({a['log_probs']}.flatten(0, -2).float(), {a['target']}.flatten())"
        case "KLDiv":
            return f"F.kl_div({a['log_probs']}.float(), {a['target_probs']}.float(), reduction='batchmean')"
        case "Contrastive":
            cands = f"torch.cat([{a['b']}, {a['negatives']}]).float()" if a.get("negatives") else f"{a['b']}.float()"
            logits = f"{_r(p['scale'])} * {a['a']}.float() @ {cands}.T"
            labels = f"torch.arange({a['a']}.shape[0], device={a['a']}.device)"
            if p["symmetric"]:
                return f"(F.cross_entropy({logits}, {labels}) + F.cross_entropy(({logits}).T, {labels})) / 2"
            return f"F.cross_entropy({logits}, {labels})"
        case "Triplet":
            return f"F.triplet_margin_loss({a['anchor']}.float(), {a['positive']}.float(), {a['negative']}.float(), margin={_r(p['margin'])})"
        case "KLGaussian":
            return f"(-0.5 * (1 + {a['logvar']} - {a['mu']}.pow(2) - {a['logvar']}.exp()).sum(-1)).mean()"
    raise NotImplementedError(t)


@dataclass
class Body:
    """Shared emission state for the top-level model and for subgraph classes."""

    nodes: dict[str, ir.Node]
    edges: list[ir.Edge]
    analysis: Analysis
    scope: str  # "" for top level, "<subgraph>/" inside a subgraph class
    blocks: set[str] = field(default_factory=set)
    user_code: dict[str, str] = field(default_factory=dict)  # class name → source
    symbolic: bool = False  # parametric subgraph: $params render as Python names

    def params(self, n: ir.Node) -> dict[str, Any]:
        p = expand_defaults(n["type"], n.get("params", {}))
        return ir.substitute(p, {}, render=Sym) if self.symbolic else p

    def incoming(self) -> dict[str, str]:
        return {dst: src for src, dst in self.edges}


def _shape_note(analysis: Analysis, nid: str, port: str) -> str:
    t = analysis.ports.get(f"{nid}.{port}")
    return f"  {fmt_shape(t.shape)}" if t else ""


def _emit_init(e: Emitter, body: Body, subclasses: dict[str, str], ind: str) -> None:
    for nid in body.analysis.order:
        n = body.nodes[nid]
        t = n["type"]
        p = body.params(n)
        tag = body.scope + nid
        name = py_name(nid)
        if n.get("disabled") or REGISTRY[t].loss or t in ("Input", "Output"):
            continue
        if t == "Subgraph":
            cls = subclasses[p["graph"]]
            kw = ", ".join(f"{k}={_r(v)}" for k, v in (p.get("args") or {}).items())
            if p["repeat"] == 1:
                e.emit(f"{ind}self.{name} = {cls}({kw})  # node: {nid}", tag)
            elif p["tie_weights"]:
                e.emit(f"{ind}self.{name} = {cls}({kw})  # node: {nid} · applied {p['repeat']}× with tied weights", tag)
            else:
                e.emit(f"{ind}self.{name} = nn.ModuleList([{cls}({kw}) for _ in range({p['repeat']})])  # node: {nid}", tag)
        elif t == "PythonModule":
            kwargs = ", ".join(f"{k}={_r(v)}" for k, v in (p.get("init") or {}).items())
            body.user_code[p["class_name"]] = p["code"]
            e.emit(f"{ind}self.{name} = {p['class_name']}({kwargs})  # node: {nid} · python", tag)
        else:
            ctor, needs = _module_ctor(t, p)
            body.blocks |= needs
            if ctor is None:
                continue
            e.emit(f"{ind}self.{name} = {ctor}  # node: {nid}", tag)
            if t == "Linear" and p.get("tie_to"):
                e.emit(f"{ind}self.{name}.weight = self.{py_name(p['tie_to'])}.weight  # node: {nid} · tied to {p['tie_to']}", tag)
        _emit_trainability(e, n, name, nid, tag, ind)
        _emit_adapters(e, n, p, name, nid, tag, body, ind)


def _emit_trainability(e: Emitter, n: ir.Node, name: str, nid: str, tag: str, ind: str) -> None:
    if n.get("trainable") is False:
        e.emit(f"{ind}self.{name}.requires_grad_(False)  # node: {nid} · frozen", tag)
    for mask in n.get("frozen_params", []):
        e.emit(f"{ind}self.{name}.{mask}.requires_grad_(False)  # node: {nid} · frozen_params", tag)


def _emit_adapters(e: Emitter, n: ir.Node, p: dict[str, Any], name: str, nid: str, tag: str, body: Body, ind: str) -> None:
    t = REGISTRY[n["type"]]
    for i, ad in enumerate(n.get("adapt", [])):
        ad = {**ir.adapt_defaults(ad["kind"]), **ad}
        if body.symbolic:
            ad = ir.substitute(ad, {}, render=Sym)
        if ad.get("disabled"):
            e.emit(f"{ind}# adapt[{i}] {ad['kind']} disabled (ablation)  # node: {nid}", tag)
            continue
        tgt = ad.get("target", "weight")
        path = f"self.{name}" if tgt in ("weight", "out") else f"self.{name}.{tgt}"
        if ad["kind"] == "lora":
            body.blocks.add("LoRALinear")
            args = f"r={ad['r']}, alpha={_r(ad['alpha'])}"
            if ad["dropout"]:
                args += f", dropout={_r(ad['dropout'])}"
            if not ad["freeze"]:
                args += ", freeze=False"
            if ad["init"] != "zeros_b":
                args += f", init={_r(ad['init'])}"
            e.emit(f"{ind}{path} = LoRALinear({path}, {args})  # node: {nid} · adapt[{i}]", tag)
        elif ad["kind"] == "adapter":
            body.blocks.add("BottleneckAdapter")
            feats = t.out_features(p)
            frz = "" if ad["freeze"] else ", freeze=False"
            e.emit(f"{ind}{path} = BottleneckAdapter({path}, {feats}, {ad['bottleneck']}{frz})  # node: {nid} · adapt[{i}]", tag)
        elif ad["kind"] == "custom":
            cls = ad.get("class_name") or "CustomAdapter"
            body.user_code[cls] = ad["code"]
            e.emit(f"{ind}{path} = {cls}().wrap({path})  # node: {nid} · adapt[{i}] custom", tag)


def _emit_forward(e: Emitter, body: Body, ind: str, var_of_input: dict[str, str]) -> dict[str, str]:
    """Emit forward statements. Returns output-port ref → variable name."""
    inc = body.incoming()
    var: dict[str, str] = {}
    for nid in body.analysis.order:
        n = body.nodes[nid]
        t = n["type"]
        nt = REGISTRY[t]
        p = body.params(n)
        tag = body.scope + nid
        name = py_name(nid)
        sig = body.analysis.signatures[nid]
        args = {port: var[inc[f"{nid}.{port}"]] for port in sig["inputs"] if f"{nid}.{port}" in inc}
        if t == "Input":
            var[f"{nid}.out"] = var_of_input[nid]
            continue
        if t == "Output" or nt.loss:
            continue
        outs = sig["outputs"]
        target = name if len(outs) == 1 else ", ".join(f"{name}_{o}" for o in outs)
        note = _shape_note(body.analysis, nid, outs[0]) if len(outs) == 1 and not body.symbolic else ""
        if n.get("disabled"):
            first = args.get(sig["inputs"][0]) if sig["inputs"] else None
            e.emit(f"{ind}{name} = {first}  # node: {nid} · disabled (ablation){note}", tag)
            var[f"{nid}.{outs[0]}"] = name
            continue
        if t == "Subgraph":
            call_args = ", ".join(args[i] for i in sig["inputs"])
            if p["repeat"] == 1:
                e.emit(f"{ind}{target} = self.{name}({call_args})  # node: {nid}{note}", tag)
            else:
                carried = [args[i] for i in sig["inputs"][: len(outs)]]
                side = [args[i] for i in sig["inputs"][len(outs) :]]  # same value for every repeat
                e.emit(f"{ind}{target} = {', '.join(carried)}  # node: {nid}", tag)
                it = f"self.{name}" if not p["tie_weights"] else f"[self.{name}] * {p['repeat']}"
                e.emit(f"{ind}for block in {it}:  # node: {nid} · {p['repeat']}×", tag)
                e.emit(f"{ind}    {target} = block({', '.join([target, *side])})  # node: {nid}{note}", tag)
        elif t == "PythonModule":
            call_args = ", ".join(f"{k}={v}" for k, v in args.items())
            if len(outs) == 1:
                e.emit(f"{ind}{name} = self.{name}({call_args})  # node: {nid} · python{note}", tag)
            else:
                e.emit(f"{ind}{name}_out = self.{name}({call_args})  # node: {nid} · python", tag)
                for o in outs:
                    e.emit(f"{ind}{name}_{o} = {name}_out[{o!r}]  # node: {nid}", tag)
        else:
            src = body.analysis.ports.get(f"{nid}.in")
            rank = len(src.shape) if src is not None else None
            e.emit(f"{ind}{target} = {_expr(t, p, 'self.' + name, args, rank)}  # node: {nid}{note}", tag)
        for o in outs:
            var[f"{nid}.{o}"] = name if len(outs) == 1 else f"{name}_{o}"
    return var


@dataclass
class Compiled:
    model_py: str
    line_map: dict[int, str]
    inputs: dict[str, str]  # forward arg → dataset field (same name)
    targets: list[str]  # dataset fields used as loss targets when not connected
    outputs: dict[str, str]  # output name → port ref
    analysis: Analysis
    graph_hash: str


def compile_graph(graph: ir.Graph, analysis: Analysis | None = None, python=None) -> Compiled:
    graph = ir.migrate(graph)
    analysis = analysis or analyze(graph, python)
    if not analysis.ok:
        raise CodegenError(analysis)
    ghash = ir.content_hash(graph)
    e = Emitter()
    nodes = ir.node_index(graph)
    subs = graph.get("subgraphs") or {}
    used_subs = sorted({n["params"]["graph"] for n in graph["nodes"] if n["type"] == "Subgraph"})
    subclasses = {s: class_name(s) for s in used_subs}

    # Render bodies first (into separate emitters) to learn which blocks are needed.
    sub_emitters: list[tuple[Emitter, Body]] = []
    all_blocks: set[str] = set()
    user_code: dict[str, str] = {}
    for s in used_subs:
        se, sb = _render_subgraph(s, subs[s], analysis.subgraphs[s], subclasses[s])
        sub_emitters.append((se, sb))
        all_blocks |= sb.blocks
        user_code |= sb.user_code

    body = Body(nodes, graph["edges"], analysis, "")
    me = Emitter()
    _render_model(me, body, graph, subclasses)
    all_blocks |= body.blocks
    user_code |= body.user_code

    name = graph.get("name") or "graph"
    e.emit(f'"""{name} — generated by Weightsmith from graph {ghash[:12]}.')
    e.emit("")
    e.emit("Readable on purpose: edit it, or keep working on the canvas. Each line notes the")
    e.emit('node it came from (`# node: <id>`) and the inferred output shape."""')
    e.emit("")
    e.emit("import math")
    e.emit("")
    e.emit("import torch")
    e.emit("import torch.nn.functional as F")
    e.emit("from torch import nn")
    e.emit("")
    order = ["LoRALinear", "BottleneckAdapter", "Attention", "CrossAttention", "MLP", "GatedMLP", "MoE", "Mamba", "Recurrent",
             "PositionalEmbedding", "SinusoidalPositional", "ClassToken", "PatchEmbed", "RelativePositionBias", "DropPath"]
    for b in order:
        if b in all_blocks:
            _gap(e)
            e.emit(BLOCKS[b].strip("\n"))
    for cls, code in user_code.items():
        _gap(e)
        e.emit(f"# ---- user code: {cls}")
        e.emit(textwrap.dedent(code).strip("\n"))
    for se, _ in sub_emitters:
        _append(e, se)
    _append(e, me)

    inputs = _input_args(graph, analysis)
    targets = _target_fields(graph, analysis)
    outputs = _output_names(graph, analysis)
    return Compiled(e.source(), e.line_nodes, inputs, targets, outputs, analysis, ghash)


def _gap(e: Emitter) -> None:
    """Exactly two blank lines before the next top-level definition."""
    while e.lines and e.lines[-1] == "":
        e.lines.pop()
    e.emit("")
    e.emit("")


def _append(dst: Emitter, src: Emitter) -> None:
    _gap(dst)
    offset = len(dst.lines)
    dst.lines.extend(src.lines)
    for ln, node in src.line_nodes.items():
        dst.line_nodes[ln + offset] = node


def _render_subgraph(name: str, body_def: dict[str, Any], analysis: Analysis, cls: str) -> tuple[Emitter, Body]:
    e = Emitter()
    nodes = {n["id"]: n for n in body_def["nodes"]}
    bparams = body_def.get("params") or {}
    body = Body(nodes, body_def["edges"], analysis, f"{name}/", symbolic=bool(bparams))
    in_ids = [n["id"] for n in body_def["nodes"] if n["type"] == "Input"]
    out_ids = [n["id"] for n in body_def["nodes"] if n["type"] == "Output"]
    e.emit(f"class {cls}(nn.Module):")
    doc = body_def.get("doc") or f"Subgraph {name!r}."
    e.emit(f'    """{doc}"""')
    e.emit("")
    sig = "".join(f", {k}={_r(v)}" for k, v in bparams.items())
    e.emit(f"    def __init__(self{sig}):")
    e.emit("        super().__init__()")
    n0 = len(e.lines)
    _emit_init(e, body, {}, "        ")
    if len(e.lines) == n0:
        e.emit("        pass")
    e.emit("")
    args = ", ".join(py_name(i) for i in in_ids)
    e.emit(f"    def forward(self, {args}):")
    var = _emit_forward(e, body, "        ", {i: py_name(i) for i in in_ids})
    inc = body.incoming()
    rets = [var[inc[f"{o}.in"]] for o in out_ids]
    e.emit(f"        return {', '.join(rets)}")
    return e, body


def _input_args(graph: ir.Graph, analysis: Analysis) -> dict[str, str]:
    out = {}
    for nid in analysis.order:
        n = ir.node_index(graph)[nid]
        if n["type"] == "Input":
            f = expand_defaults("Input", n.get("params", {}))["field"]
            out[f] = nid
    return out


def _target_fields(graph: ir.Graph, analysis: Analysis) -> list[str]:
    nodes = ir.node_index(graph)
    connected = {dst for _, dst in graph["edges"]}
    fields = []
    for lid in analysis.losses:
        n = nodes[lid]
        if "target" in analysis.signatures[lid]["optional"] and f"{lid}.target" not in connected:
            f = expand_defaults(n["type"], n.get("params", {}))["target_field"]
            if f not in fields:
                fields.append(f)
    return fields


def _output_names(graph: ir.Graph, analysis: Analysis) -> dict[str, str]:
    """Model output name → the port ref whose tensor it is."""
    nodes = ir.node_index(graph)
    names: dict[str, str] = {}
    for e in graph["edges"]:
        dn, dp = ir.split_ref(e[1])
        t = REGISTRY.get(nodes[dn]["type"]) if dn in nodes else None
        if t is None:
            continue
        if t.name == "Output":
            names.setdefault(dn, e[0])
        elif t.loss and dp != "target":
            sn, sp = ir.split_ref(e[0])
            if nodes[sn]["type"] == "Input":
                continue
            names.setdefault(sn if len(analysis.signatures[sn]["outputs"]) == 1 else f"{sn}_{sp}", e[0])
    return names


def _render_model(e: Emitter, body: Body, graph: ir.Graph, subclasses: dict[str, str]) -> None:
    analysis = body.analysis
    inputs = _input_args(graph, analysis)
    targets = [f for f in _target_fields(graph, analysis) if f not in inputs]
    e.emit("class Model(nn.Module):")
    in_doc = ", ".join(f"{f} ({fmt_shape(analysis.ports[f'{nid}.out'].shape)})" for f, nid in inputs.items())
    e.emit(f'    """Inputs: {in_doc}. Returns a dict of outputs; includes "loss" when targets are given."""')
    e.emit("")
    e.emit("    def __init__(self):")
    e.emit("        super().__init__()")
    n0 = len(e.lines)
    _emit_init(e, body, subclasses, "        ")
    if len(e.lines) == n0:
        e.emit("        pass")
    e.emit("")
    sig = ", ".join([*inputs] + [f"{t}=None" for t in targets])
    e.emit(f"    def forward(self, {sig}, **unused):")
    var_of_input = {}
    for f, nid in inputs.items():
        var_of_input[nid] = py_name(f)
    var = _emit_forward(e, body, "        ", var_of_input)

    inc = body.incoming()
    nodes = body.nodes
    outs = []
    for oname, ref in _output_names(graph, analysis).items():
        outs.append((oname, var[ref]))
    items = ", ".join(f"{k!r}: {v}" for k, v in outs)
    e.emit(f"        out = {{{items}}}")

    losses = analysis.losses
    if losses:
        conds = []
        for lid in losses:
            if "target" in analysis.signatures[lid]["optional"] and f"{lid}.target" not in inc:
                f = expand_defaults(nodes[lid]["type"], nodes[lid].get("params", {}))["target_field"]
                if f not in inputs and f not in conds:
                    conds.append(f)
        guard = " and ".join(f"{c} is not None" for c in conds)
        ind = "        "
        if guard:
            e.emit(f"        if {guard}:")
            ind = "            "
        terms = []
        for lid in losses:
            n = nodes[lid]
            p = expand_defaults(n["type"], n.get("params", {}))
            sig_in = analysis.signatures[lid]["inputs"]
            args = {}
            for port in sig_in:
                ref = f"{lid}.{port}"
                if ref in inc:
                    args[port] = var[inc[ref]]
                elif port == "target":
                    args[port] = py_name(p["target_field"])
            lname = py_name(lid)
            e.emit(f"{ind}{lname} = {_loss_expr(n['type'], p, args)}  # node: {lid}", lid)
            if len(losses) > 1:
                e.emit(f"{ind}out[{lid!r}] = {lname}", lid)
            terms.append(lname if p["weight"] == 1 else f"{_r(p['weight'])} * {lname}")
        e.emit(f"{ind}out['loss'] = {' + '.join(terms)}")
    e.emit("        return out")
