"""Bulk selection: "LoRA on every attention q/v" in one move.

A query (node type + id/label glob) runs over the graph and its subgraphs; the
result is written back as one annotation per matched node, so the IR stays
explicit and diffable.
"""

from __future__ import annotations

import copy
import fnmatch
from typing import Any

from . import ir
from .registry import REGISTRY, expand_defaults


def match(graph: ir.Graph, types: list[str] | None = None, pattern: str = "*") -> list[tuple[str, str]]:
    """(scope, node id) pairs; scope is "" for the top level or the subgraph name."""
    out = []
    bodies = [("", graph.get("nodes", []))] + [(k, v["nodes"]) for k, v in (graph.get("subgraphs") or {}).items()]
    for scope, nodes in bodies:
        for n in nodes:
            if types and n["type"] not in types:
                continue
            names = [n["id"], n.get("label") or n["id"]]
            if any(fnmatch.fnmatchcase(x, pattern) for x in names):
                out.append((scope, n["id"]))
    return out


def apply_adapt(
    graph: ir.Graph, adapt: dict[str, Any], targets: list[str], types: list[str] | None = None, pattern: str = "*"
) -> tuple[ir.Graph, list[str]]:
    """Add ``adapt`` (once per target) to each matched node that has the target. Idempotent."""
    g = copy.deepcopy(graph)
    hits = set(match(g, types, pattern))
    touched = []
    subs = g.get("subgraphs") or {}
    bodies = [("", g["nodes"])] + [(k, v["nodes"]) for k, v in subs.items()]
    for scope, nodes in bodies:
        env = ir.block_args(subs[scope], None) if scope else {}  # $params resolved with the block's defaults
        for n in nodes:
            if (scope, n["id"]) not in hits:
                continue
            t = REGISTRY.get(n["type"])
            if t is None:
                continue
            params = ir.substitute(expand_defaults(t.name, n.get("params", {})), env) if scope else expand_defaults(t.name, n.get("params", {}))
            available = t.linear_targets(params) if adapt["kind"] == "lora" else {"out": None}
            for tgt in targets:
                if tgt not in available:
                    continue
                entry = {**adapt, "target": tgt}
                existing = n.setdefault("adapt", [])
                if any(e.get("kind") == entry["kind"] and e.get("target") == tgt for e in existing):
                    continue
                existing.append(entry)
                touched.append(f"{scope}/{n['id']}:{tgt}" if scope else f"{n['id']}:{tgt}")
    return g, touched
