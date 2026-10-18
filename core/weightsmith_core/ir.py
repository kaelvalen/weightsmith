"""Graph IR: loading, schema validation, migrations, canonical form and content hash.

The IR is plain JSON (dicts and lists). Nothing here depends on PyTorch.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

Graph = dict[str, Any]
Node = dict[str, Any]
Edge = list[str]

# Fields that never affect computation and therefore never enter the content hash.
_COSMETIC_GRAPH_FIELDS = ("layout", "name")
_COSMETIC_NODE_FIELDS = ("label",)


def schema_path() -> Path:
    """Locate ir.schema.json both in a source checkout and in an installed wheel."""
    repo = Path(__file__).resolve().parents[2] / "schema" / "ir.schema.json"
    if repo.exists():
        return repo
    return Path(str(resources.files("weightsmith_core") / "_schema" / "ir.schema.json"))


_validator = None


def schema_errors(graph: Graph) -> list[str]:
    """Structural (JSON Schema) errors. Semantic checks live in validate.py."""
    global _validator
    if _validator is None:
        import jsonschema

        schema = json.loads(schema_path().read_text())
        _validator = jsonschema.Draft202012Validator(schema)
    out = []
    for err in sorted(_validator.iter_errors(graph), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in err.absolute_path) or "(root)"
        out.append(f"{where}: {err.message}")
    return out


# ---------------------------------------------------------------- migrations

# MIGRATIONS[n] upgrades a graph from schema n to n+1. Every schema bump adds an
# entry here so old project files keep opening.
MIGRATIONS: dict[int, Callable[[Graph], Graph]] = {}


def migrate(graph: Graph) -> Graph:
    g = copy.deepcopy(graph)
    version = g.get("schema", SCHEMA_VERSION)
    if version > SCHEMA_VERSION:
        raise ValueError(f"graph schema {version} is newer than this build ({SCHEMA_VERSION})")
    while version < SCHEMA_VERSION:
        if version not in MIGRATIONS:
            raise ValueError(f"no migration from schema {version}")
        g = MIGRATIONS[version](g)
        version += 1
        g["schema"] = version
    return g


def load(path: str | Path) -> Graph:
    return migrate(json.loads(Path(path).read_text()))


# ------------------------------------------------------------ edges & ports


def split_ref(ref: str) -> tuple[str, str]:
    node, _, port = ref.rpartition(".")
    return node, port


def node_index(graph: Graph) -> dict[str, Node]:
    return {n["id"]: n for n in graph.get("nodes", [])}


# ------------------------------------------------------------ canonical form


def canonical(graph: Graph) -> Graph:
    """Sorted, defaults expanded, cosmetic fields removed.

    Two graphs that compute the same thing produce the same canonical form, so
    dragging a node or renaming its label never creates a new run.
    """
    from .registry import expand_defaults  # local import: registry imports ir

    def canon_body(nodes: list[Node], edges: list[Edge]) -> dict[str, Any]:
        cn = []
        for node in nodes:
            n = {k: v for k, v in node.items() if k not in _COSMETIC_NODE_FIELDS}
            n["params"] = expand_defaults(n["type"], n.get("params", {}))
            n.setdefault("adapt", [])
            n["adapt"] = [_canon_adapt(a) for a in n["adapt"]]
            n.setdefault("trainable", True)
            n["frozen_params"] = sorted(n.get("frozen_params", []))
            n.setdefault("disabled", False)
            cn.append(n)
        cn.sort(key=lambda n: n["id"])
        ce = sorted([list(e) for e in edges])
        return {"nodes": cn, "edges": ce}

    out: Graph = {"schema": graph.get("schema", SCHEMA_VERSION)}
    out.update(canon_body(graph.get("nodes", []), graph.get("edges", [])))
    subs = graph.get("subgraphs") or {}
    out["subgraphs"] = {k: {**canon_body(v["nodes"], v["edges"]), "params": dict(sorted((v.get("params") or {}).items()))} for k, v in sorted(subs.items())}
    return out


_ADAPT_DEFAULTS = {
    "lora": {"target": "weight", "r": 8, "alpha": 16, "dropout": 0.0, "freeze": True, "init": "zeros_b"},
    "adapter": {"target": "out", "bottleneck": 64, "freeze": True},
    "custom": {"target": "weight", "freeze": True},
}


def adapt_defaults(kind: str) -> dict[str, Any]:
    return dict(_ADAPT_DEFAULTS.get(kind, {}))


def _canon_adapt(a: dict[str, Any]) -> dict[str, Any]:
    full = adapt_defaults(a["kind"])
    full.update(a)
    full.setdefault("disabled", False)
    return full


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(graph: Graph) -> str:
    return hashlib.sha256(canonical_json(canonical(graph)).encode()).hexdigest()


def hash_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


# ------------------------------------------------------- parametric subgraphs

import ast as _ast
import operator as _op

_OPS = {_ast.Add: _op.add, _ast.Sub: _op.sub, _ast.Mult: _op.mul, _ast.FloorDiv: _op.floordiv, _ast.Div: _op.truediv,
        _ast.Mod: _op.mod, _ast.Pow: _op.pow, _ast.USub: _op.neg}


class ArgError(ValueError):
    pass


def eval_arg(expr: str, env: dict[str, Any]) -> Any:
    """Evaluate a block parameter expression such as "$C*2" with plain arithmetic only."""

    def ev(n):
        if isinstance(n, _ast.Expression):
            return ev(n.body)
        if isinstance(n, _ast.Constant) and isinstance(n.value, (int, float, bool, str)):
            return n.value
        if isinstance(n, _ast.Name):
            if n.id not in env:
                raise ArgError(f"unknown block parameter ${n.id}")
            return env[n.id]
        if isinstance(n, _ast.BinOp) and type(n.op) in _OPS:
            v = _OPS[type(n.op)](ev(n.left), ev(n.right))
            return int(v) if isinstance(v, float) and v.is_integer() and isinstance(n.op, _ast.Div) else v
        if isinstance(n, _ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ArgError(f"unsupported expression in ${expr}")

    try:
        return ev(_ast.parse(expr, mode="eval"))
    except SyntaxError as e:
        raise ArgError(f"cannot parse ${expr}: {e.msg}") from None


def substitute(value: Any, env: dict[str, Any], render=None) -> Any:
    """Replace "$expr" strings (also inside lists and dicts). `render` keeps them symbolic (codegen)."""
    if isinstance(value, str) and value.startswith("$"):
        return render(value[1:]) if render else eval_arg(value[1:], env)
    if isinstance(value, list):
        return [substitute(v, env, render) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, env, render) for k, v in value.items()}
    return value


def block_args(body: dict[str, Any], args: dict[str, Any] | None) -> dict[str, Any]:
    defaults = dict(body.get("params") or {})
    for k in args or {}:
        if k not in defaults:
            raise ArgError(f"block has no parameter '{k}' (has: {', '.join(defaults) or 'none'})")
    return {**defaults, **(args or {})}


def instantiate(body: dict[str, Any], args: dict[str, Any] | None = None) -> dict[str, Any]:
    """A subgraph body with its $parameters replaced by values (defaults overridden by args)."""
    env = block_args(body, args)
    out = copy.deepcopy(body)
    for n in out["nodes"]:
        if "params" in n:
            n["params"] = substitute(n["params"], env)
        if "adapt" in n:
            n["adapt"] = substitute(n["adapt"], env)
    return out
