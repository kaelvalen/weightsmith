"""IR diff between two graphs (variants, or a run and its ablation).

Works on canonical forms, so layout and labels never show up as changes.
"""

from __future__ import annotations

from typing import Any

from . import ir


def _body_diff(a: dict[str, Any], b: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    na = {n["id"]: n for n in a.get("nodes", [])}
    nb = {n["id"]: n for n in b.get("nodes", [])}
    for nid in sorted(nb.keys() - na.keys()):
        changes.append({"kind": "node_added", "node": scope + nid, "type": nb[nid]["type"]})
    for nid in sorted(na.keys() - nb.keys()):
        changes.append({"kind": "node_removed", "node": scope + nid, "type": na[nid]["type"]})
    for nid in sorted(na.keys() & nb.keys()):
        x, y = na[nid], nb[nid]
        if x["type"] != y["type"]:
            changes.append({"kind": "type_changed", "node": scope + nid, "from": x["type"], "to": y["type"]})
            continue
        for k in sorted(set(x["params"]) | set(y["params"])):
            if x["params"].get(k) != y["params"].get(k):
                changes.append({"kind": "param_changed", "node": scope + nid, "key": k, "from": x["params"].get(k), "to": y["params"].get(k)})
        for flag in ("trainable", "disabled", "frozen_params"):
            if x.get(flag) != y.get(flag):
                changes.append({"kind": f"{flag}_changed", "node": scope + nid, "from": x.get(flag), "to": y.get(flag)})
        if x["adapt"] != y["adapt"]:
            changes.append({"kind": "adapt_changed", "node": scope + nid, "from": x["adapt"], "to": y["adapt"]})
    ea = {tuple(e) for e in a.get("edges", [])}
    eb = {tuple(e) for e in b.get("edges", [])}
    for e in sorted(eb - ea):
        changes.append({"kind": "edge_added", "edge": [scope + e[0], scope + e[1]]})
    for e in sorted(ea - eb):
        changes.append({"kind": "edge_removed", "edge": [scope + e[0], scope + e[1]]})
    return changes


def diff(a: ir.Graph, b: ir.Graph) -> dict[str, Any]:
    ca, cb = ir.canonical(ir.migrate(a)), ir.canonical(ir.migrate(b))
    changes = _body_diff(ca, cb, "")
    sa, sb = ca.get("subgraphs", {}), cb.get("subgraphs", {})
    for name in sorted(sb.keys() - sa.keys()):
        changes.append({"kind": "subgraph_added", "subgraph": name})
    for name in sorted(sa.keys() - sb.keys()):
        changes.append({"kind": "subgraph_removed", "subgraph": name})
    for name in sorted(sa.keys() & sb.keys()):
        changes += _body_diff(sa[name], sb[name], f"{name}/")
    return {"same": not changes, "a": ir.hash_obj(ca), "b": ir.hash_obj(cb), "changes": changes}
