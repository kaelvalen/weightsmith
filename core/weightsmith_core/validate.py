"""IR validator and shape inference.

``analyze(graph)`` runs every check the canvas shows and the compiler relies on:
schema, ids, types, params, edges, cycles, unconnected ports, shape/dtype
inference and parameter counts. Issues carry the node/port/edge they belong to
so the canvas can paint them in place.
"""

from __future__ import annotations

import keyword
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from . import ir
from .registry import REGISTRY, NodeType, expand_defaults
from .shapes import ShapeError, SymbolEnv, TensorType, fmt_shape


@dataclass
class Issue:
    code: str
    message: str
    node: str | None = None
    port: str | None = None
    edge: int | None = None
    severity: str = "error"


@dataclass
class ParamCount:
    total: int = 0
    trainable: int = 0
    known: bool = True


@dataclass
class Analysis:
    issues: list[Issue] = field(default_factory=list)
    ports: dict[str, TensorType] = field(default_factory=dict)
    params: dict[str, ParamCount] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    signatures: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    losses: list[str] = field(default_factory=list)
    subgraphs: dict[str, "Analysis"] = field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def totals(self) -> ParamCount:
        t = ParamCount()
        for c in self.params.values():
            t.total += c.total
            t.trainable += c.trainable
            t.known = t.known and c.known
        return t

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": [asdict(i) for i in self.issues],
            "ports": {k: v.to_json() for k, v in sorted(self.ports.items())},
            "params": {k: asdict(v) for k, v in sorted(self.params.items())},
            "totals": asdict(self.totals),
            "order": self.order,
            "signatures": self.signatures,
            "losses": self.losses,
            "subgraphs": {k: v.to_json() for k, v in self.subgraphs.items()},
        }


# integer parameters that may be negative (Python-style indexing from the end)
SIGNED_PARAMS = frozenset({"dim", "start_dim", "end_dim", "dim0", "dim1", "index", "start", "end"})


class PythonBridge(Protocol):
    """Evaluates PythonModule nodes. Implemented by pynode.SandboxBridge."""

    def signature(self, params: dict[str, Any]) -> dict[str, Any]: ...

    def infer(self, params: dict[str, Any], ins: dict[str, TensorType]) -> dict[str, TensorType]: ...


def analyze(graph: ir.Graph, python: PythonBridge | None = None) -> Analysis:
    schema_issues = ir.schema_errors(graph)
    if schema_issues:
        a = Analysis()
        a.issues = [Issue("schema", m) for m in schema_issues]
        return a
    return _analyze_body(graph["nodes"], graph["edges"], graph.get("subgraphs") or {}, python, top=True)


# --------------------------------------------------------------- internals


def _signature(node: ir.Node, subgraphs: dict[str, Any], python: PythonBridge | None) -> tuple[list[str], list[str], list[str]]:
    """(inputs, optional inputs, outputs) for a node."""
    t = REGISTRY[node["type"]]
    params = node.get("params", {})
    if t.name == "Subgraph":
        body = subgraphs.get(params.get("graph"), {"nodes": []})
        ins = [n["id"] for n in body["nodes"] if n["type"] == "Input"]
        outs = [n["id"] for n in body["nodes"] if n["type"] == "Output"]
        return ins, [], outs
    if t.name == "PythonModule":
        sig = params.get("signature")
        if python is not None and params.get("code"):
            sig = python.signature(params)
        sig = sig or {"inputs": ["in"], "outputs": ["out"]}
        return list(sig["inputs"]), list(sig.get("optional", [])), list(sig["outputs"])
    return [p.name for p in t.inputs], [p.name for p in t.inputs if p.optional], [p.name for p in t.outputs]


def _check_params(node: ir.Node, t: NodeType, a: Analysis) -> bool:
    ok = True
    params = node.get("params", {})
    for k, p in t.params.items():
        if p.required and params.get(k) in (None, ""):
            a.issues.append(Issue("param_missing", f"parameter '{k}' is required", node["id"]))
            ok = False
            continue
        v = params.get(k, p.default)
        if v is None:
            continue
        if p.kind == "int" and (not isinstance(v, int) or isinstance(v, bool)):
            a.issues.append(Issue("param_type", f"'{k}' must be an integer", node["id"]))
            ok = False
        elif p.kind == "int" and k not in SIGNED_PARAMS and v < 0:
            a.issues.append(Issue("param_value", f"'{k}' must be ≥ 0", node["id"]))
            ok = False
        elif p.kind == "float" and not isinstance(v, (int, float)):
            a.issues.append(Issue("param_type", f"'{k}' must be a number", node["id"]))
            ok = False
        elif p.kind == "bool" and not isinstance(v, bool):
            a.issues.append(Issue("param_type", f"'{k}' must be true/false", node["id"]))
            ok = False
        elif p.kind == "choice" and p.choices and v not in p.choices:
            a.issues.append(Issue("param_value", f"'{k}' must be one of {', '.join(map(str, p.choices))}", node["id"]))
            ok = False
        elif p.kind == "shape" and not isinstance(v, list):
            a.issues.append(Issue("param_type", f"'{k}' must be a shape list", node["id"]))
            ok = False
    for k in params:
        if k not in t.params:
            a.issues.append(Issue("param_unknown", f"unknown parameter '{k}'", node["id"], severity="warning"))
    return ok


def _count_params(
    node: ir.Node, params: dict[str, Any], t: NodeType, sub_counts: ParamCount | None, py_params: int | None = None
) -> ParamCount:
    if t.name == "Subgraph":
        assert sub_counts is not None
        mult = 1 if params.get("tie_weights") else params.get("repeat", 1)
        c = ParamCount(sub_counts.total * mult, sub_counts.trainable * mult, sub_counts.known)
        if node.get("trainable") is False:
            c.trainable = 0
        return c
    if t.name == "PythonModule":
        n = py_params
        if n is None:
            return ParamCount(0, 0, known=False)
        return ParamCount(n, 0 if node.get("trainable") is False else n)

    weights = t.weights(params)
    frozen: set[str] = set()
    if node.get("trainable") is False:
        frozen = set(weights)
    for mask in node.get("frozen_params", []):
        frozen |= {w for w in weights if w == mask or w.startswith(mask + ".")}

    extra_total = extra_trainable = 0
    known = True
    targets = t.linear_targets(params)
    for ad in node.get("adapt", []):
        ad = {**ir.adapt_defaults(ad["kind"]), **ad}
        if ad.get("disabled"):
            continue
        tgt = ad.get("target", "weight")
        if ad["kind"] == "lora":
            if tgt not in targets:
                continue  # reported by _check_adapt
            fin, fout = targets[tgt]
            n = ad["r"] * (fin + fout)
            prefix = "" if tgt == "weight" else tgt + "."
        elif ad["kind"] == "adapter":
            o = t.out_features(params)
            if o is None:
                continue
            b = ad["bottleneck"]
            n = o * b + b + b * o + o
            prefix = ""
        else:
            known = False
            continue
        extra_total += n
        extra_trainable += n
        if ad.get("freeze", True):
            frozen |= {w for w in weights if w.startswith(prefix)} if prefix else set(weights)

    total = sum(weights.values()) + extra_total
    trainable = sum(v for k, v in weights.items() if k not in frozen) + extra_trainable
    return ParamCount(total, trainable, known)


def _check_adapt(node: ir.Node, params: dict[str, Any], t: NodeType, a: Analysis) -> None:
    targets = t.linear_targets(params)
    for i, ad in enumerate(node.get("adapt", [])):
        kind = ad["kind"]
        tgt = ad.get("target", ir.adapt_defaults(kind).get("target"))
        if kind == "lora" and tgt not in targets:
            have = ", ".join(targets) or "none"
            a.issues.append(Issue("adapt_target", f"adapt[{i}]: LoRA target '{tgt}' not found (targets: {have})", node["id"]))
        if kind == "adapter" and t.out_features(params) is None:
            a.issues.append(Issue("adapt_target", f"adapt[{i}]: {t.name} has no feature dimension for an adapter", node["id"]))
        if kind == "custom" and not ad.get("code"):
            a.issues.append(Issue("adapt_code", f"adapt[{i}]: custom adapter needs code with a wrap(module) method", node["id"]))


def _analyze_body(
    nodes: list[ir.Node],
    edges: list[ir.Edge],
    subgraphs: dict[str, Any],
    python: PythonBridge | None,
    top: bool,
    bound_inputs: dict[str, TensorType] | None = None,
    env: SymbolEnv | None = None,
) -> Analysis:
    a = Analysis()
    env = env or SymbolEnv()

    # ---- nodes: ids, types, params
    by_id: dict[str, ir.Node] = {}
    for n in nodes:
        if n["id"] in by_id:
            a.issues.append(Issue("duplicate_id", f"duplicate node id '{n['id']}'", n["id"]))
            continue
        by_id[n["id"]] = n
    sigs: dict[str, tuple[list[str], list[str], list[str]]] = {}
    py_counts: dict[str, int] = {}
    valid_params: dict[str, bool] = {}
    for nid, n in by_id.items():
        t = REGISTRY.get(n["type"])
        if t is None:
            a.issues.append(Issue("unknown_type", f"unknown node type '{n['type']}'", nid))
            continue
        valid_params[nid] = _check_params(n, t, a)
        if t.name == "Subgraph" and n.get("params", {}).get("graph") not in subgraphs:
            a.issues.append(Issue("subgraph_missing", f"subgraph '{n.get('params', {}).get('graph')}' is not defined", nid))
            valid_params[nid] = False
        if t.name == "Subgraph" and not top:
            a.issues.append(Issue("subgraph_nested", "nested subgraphs are not supported in schema 1", nid))
            valid_params[nid] = False
        try:
            sigs[nid] = _signature(n, subgraphs, python)
        except ShapeError as e:
            a.issues.append(Issue("python_node", str(e), nid))
            sigs[nid] = ([], [], [])
            valid_params[nid] = False
        a.signatures[nid] = {"inputs": sigs[nid][0], "optional": sigs[nid][1], "outputs": sigs[nid][2]}
        if t.name == "PythonModule" and python is not None and valid_params[nid]:
            try:
                sig = python.signature(n.get("params", {}))
                py_counts[nid] = sig.get("params")
                declared = (n.get("params") or {}).get("class_name") or "Block"
                if sig.get("class_name") and sig["class_name"] != declared:
                    a.issues.append(Issue("python_node", f"class_name is '{declared}' but the code defines '{sig['class_name']}'; Check & apply sets it", nid))
                    valid_params[nid] = False
            except ShapeError:
                pass
        if valid_params[nid]:
            _check_adapt(n, expand_defaults(t.name, n.get("params", {})), t, a)

    # ---- dataset field bindings (model.forward takes fields as keyword arguments)
    if top:
        fields: dict[str, str] = {}
        for nid, n in by_id.items():
            if n["type"] != "Input":
                continue
            f = expand_defaults("Input", n.get("params", {})).get("field", "x")
            if not isinstance(f, str) or not f.isidentifier() or keyword.iskeyword(f):
                a.issues.append(Issue("field", f"field '{f}' must be a Python identifier", nid))
            elif f in fields:
                a.issues.append(Issue("field", f"field '{f}' is already bound by input '{fields[f]}'", nid))
            elif f in by_id and f != nid:
                a.issues.append(Issue("field", f"field '{f}' collides with node id '{f}'", nid))
            fields[f] = nid

    # ---- weight tying: a Linear may share an Embedding's weight
    for nid, n in by_id.items():
        tie = (n.get("params") or {}).get("tie_to") if n["type"] == "Linear" else None
        if not tie:
            continue
        owner = by_id.get(tie)
        if owner is None or owner["type"] != "Embedding":
            a.issues.append(Issue("tie", f"tie_to '{tie}' must name an Embedding node in the same graph", nid))
            valid_params[nid] = False
            continue
        lp, ep = n.get("params", {}), owner.get("params", {})
        if (lp.get("in"), lp.get("out")) != (ep.get("dim"), ep.get("num")):
            a.issues.append(Issue("tie", f"tied weight shape: Linear {lp.get('in')}→{lp.get('out')} needs Embedding dim={lp.get('in')}, num={lp.get('out')}", nid))
            valid_params[nid] = False
        elif (n.get("trainable") is False) != (owner.get("trainable") is False):
            a.issues.append(Issue("tie", f"'{nid}' and '{tie}' share a weight but differ in trainability; freezing one freezes both", nid, severity="warning"))

    # ---- edges
    incoming: dict[str, tuple[str, int]] = {}  # "dst.port" -> ("src.port", edge index)
    succ: dict[str, set[str]] = defaultdict(set)
    for i, e in enumerate(edges):
        src, dst = e
        sn, sp = ir.split_ref(src)
        dn, dp = ir.split_ref(dst)
        if sn not in sigs or dn not in sigs:
            missing = sn if sn not in by_id else dn
            a.issues.append(Issue("edge_node", f"edge {src} → {dst}: node '{missing}' does not exist", edge=i))
            continue
        if sp not in sigs[sn][2]:
            a.issues.append(Issue("edge_port", f"'{sn}' has no output port '{sp}'", sn, sp, i))
            continue
        if dp not in sigs[dn][0]:
            a.issues.append(Issue("edge_port", f"'{dn}' has no input port '{dp}'", dn, dp, i))
            continue
        if sn == dn:
            a.issues.append(Issue("cycle", "node is connected to itself", sn, edge=i))
            continue
        if dst in incoming:
            a.issues.append(Issue("multi_input", f"input '{dst}' already has a connection", dn, dp, i))
            continue
        incoming[dst] = (src, i)
        succ[sn].add(dn)

    # ---- topological order (Kahn), cycle detection
    indeg = {nid: 0 for nid in sigs}
    for s, ds in succ.items():
        for d in ds:
            indeg[d] += 1
    queue = deque(sorted(nid for nid, d in indeg.items() if d == 0))
    while queue:
        nid = queue.popleft()
        a.order.append(nid)
        for d in sorted(succ[nid]):
            indeg[d] -= 1
            if indeg[d] == 0:
                queue.append(d)
    for nid in sorted(set(sigs) - set(a.order)):
        a.issues.append(Issue("cycle", "node is part of a cycle", nid))

    # ---- subgraph bodies, analysed once standalone for inspection
    if top:
        for name, body in subgraphs.items():
            try:
                inst = ir.instantiate(body)
            except ir.ArgError as e:
                sub = Analysis()
                sub.issues.append(Issue("block_param", str(e)))
                a.subgraphs[name] = sub
                continue
            a.subgraphs[name] = _analyze_body(inst["nodes"], inst["edges"], {}, python, top=False)

    # ---- shape inference
    for nid in a.order:
        n = by_id[nid]
        t = REGISTRY[n["type"]]
        if not valid_params.get(nid):
            continue
        params = expand_defaults(t.name, n.get("params", {}))
        inputs, optional, outputs = sigs[nid]
        ins: dict[str, TensorType | None] = {}
        blocked = False
        for p in inputs:
            ref = f"{nid}.{p}"
            if ref in incoming:
                src_ref, _ = incoming[ref]
                if src_ref in a.ports:
                    ins[p] = a.ports[src_ref]
                    a.ports[ref] = a.ports[src_ref]
                else:
                    blocked = True  # upstream failed; don't cascade errors
            elif p in optional:
                ins[p] = None
            else:
                a.issues.append(Issue("unconnected", f"input '{p}' is not connected", nid, p))
                blocked = True
        if blocked:
            continue

        sub_counts: ParamCount | None = None
        try:
            if t.name == "Input" and bound_inputs is not None and nid in bound_inputs:
                declared = _input_type(params)
                env.unify_shapes(declared.shape, bound_inputs[nid].shape, "out", "subgraph input ")
                outs = {"out": TensorType(env.resolve_shape(bound_inputs[nid].shape), bound_inputs[nid].dtype)}
            elif t.name == "Subgraph":
                outs, sub_counts = _infer_subgraph(n, params, ins, subgraphs, python, a)  # type: ignore[arg-type]
            elif t.name == "PythonModule":
                if python is None:
                    outs = {}
                else:
                    outs = python.infer(params, {k: v for k, v in ins.items() if v is not None})
            else:
                assert t.infer is not None
                outs = t.infer(params, ins, env)
        except ShapeError as e:
            port = e.port
            edge = incoming.get(f"{nid}.{port}", (None, None))[1] if port else None
            a.issues.append(Issue("shape", str(e), nid, port, edge))
            if port and edge is not None:
                src_node, src_port = ir.split_ref(edges[edge][0])
                a.issues.append(Issue("shape", str(e), src_node, src_port, edge, severity="peer"))
            continue

        if n.get("disabled"):
            outs = _disabled_outs(n, t, ins, outs, inputs, outputs, env, a)
            if outs is None:
                continue
        for p, tt in outs.items():
            a.ports[f"{nid}.{p}"] = TensorType(env.resolve_shape(tt.shape), tt.dtype)

        if t.loss and not n.get("disabled"):
            a.losses.append(nid)
        a.params[nid] = _count_params(n, params, t, sub_counts, py_counts.get(nid))

    # resolve every port once more: later unifications may have bound symbols
    for k, v in a.ports.items():
        a.ports[k] = TensorType(env.resolve_shape(v.shape), v.dtype)

    # ---- graph-level checks
    if top:
        if not a.losses and not any(i.code == "cycle" for i in a.issues):
            a.issues.append(Issue("no_loss", "graph has no active loss node; it can be compiled but not trained", severity="warning"))
        used = {src for src, _ in incoming.values()}
        for nid in a.order:
            t = REGISTRY.get(by_id[nid]["type"])
            if t is None or t.loss or t.name == "Output":
                continue
            for p in sigs[nid][2]:
                if f"{nid}.{p}" not in used:
                    a.issues.append(Issue("unused_output", f"output '{p}' is not used", nid, p, severity="warning"))
    return a


def _input_type(params: dict[str, Any]) -> TensorType:
    return TensorType(list(params["shape"]), params.get("dtype", "f32"))


def _disabled_outs(n, t, ins, outs, inputs, outputs, env, a) -> dict[str, TensorType] | None:
    """Ablation: a disabled node passes its first input through unchanged."""
    if t.loss:
        return outs
    if t.name == "Input":
        a.issues.append(Issue("disable", "inputs cannot be disabled", n["id"]))
        return None
    if not inputs or not outputs:
        return {}
    x = ins.get(inputs[0])
    if x is None:
        return None
    o = outs.get(outputs[0])
    if o is not None and (env.resolve_shape(o.shape) != env.resolve_shape(x.shape) or o.dtype != x.dtype):
        a.issues.append(
            Issue(
                "disable",
                f"cannot disable: output {fmt_shape(env.resolve_shape(o.shape))} differs from input {fmt_shape(env.resolve_shape(x.shape))}",
                n["id"],
            )
        )
        return None
    return {outputs[0]: x}


def _infer_subgraph(n, params, ins, subgraphs, python, parent: Analysis):
    try:
        body = ir.instantiate(subgraphs[params["graph"]], params.get("args"))
    except ir.ArgError as e:
        raise ShapeError(f"{params['graph']}: {e}")
    in_ids = [m["id"] for m in body["nodes"] if m["type"] == "Input"]
    out_ids = [m["id"] for m in body["nodes"] if m["type"] == "Output"]
    repeat = params.get("repeat", 1)
    if repeat > 1 and len(out_ids) > len(in_ids):
        raise ShapeError(f"repeat={repeat} needs at least as many inputs as outputs ({len(in_ids)} vs {len(out_ids)})")
    # with repeat, the first len(outputs) inputs are carried from block to block; the rest are
    # side inputs every repeat receives unchanged (an encoder memory, a shared position bias)
    carried = in_ids[: len(out_ids)]
    current = {k: v for k, v in ins.items() if v is not None}
    counts = ParamCount()
    for r in range(min(repeat, 2)):  # a second pass proves the block is shape-stable
        sub = _analyze_body(body["nodes"], body["edges"], {}, python, top=False, bound_inputs=current)
        errs = sub.errors
        if errs:
            first = errs[0]
            raise ShapeError(f"in {params['graph']}/{first.node}: {first.message}")
        outs: dict[str, TensorType] = {}
        for oid in out_ids:
            src = next((e[0] for e in body["edges"] if e[1] == f"{oid}.in"), None)
            if src is None or src not in sub.ports:
                raise ShapeError(f"{params['graph']}: output '{oid}' is not connected")
            outs[oid] = sub.ports[src]
        counts = sub.totals
        if repeat > 1:
            nxt = dict(current)
            for iid, oid in zip(carried, out_ids):
                if outs[oid].shape != current[iid].shape:
                    raise ShapeError(
                        f"repeat={repeat}: block output {fmt_shape(outs[oid].shape)} must equal its input {fmt_shape(current[iid].shape)}"
                    )
                nxt[iid] = outs[oid]
            current = nxt
    return outs, counts
