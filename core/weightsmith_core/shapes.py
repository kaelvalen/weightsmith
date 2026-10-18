"""Symbolic shapes.

A dimension is an int or a symbol name (``"B"``, ``"T"``). Symbols are shared
across a graph: once ``T`` is bound to 128 anywhere, every ``T`` is 128.
Derived dimensions that cannot be simplified (``"C*H*W"``) are opaque strings
compared by equality.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

Dim = int | str
Shape = list[Dim]

_SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

FLOAT_DTYPES = ("f32", "f16", "bf16")
INT_DTYPES = ("i64", "i32")


class ShapeError(Exception):
    def __init__(self, message: str, port: str | None = None):
        super().__init__(message)
        self.port = port


@dataclass
class TensorType:
    shape: Shape
    dtype: str = "f32"

    def to_json(self) -> dict:
        return {"shape": list(self.shape), "dtype": self.dtype}


def is_symbol(d: Dim) -> bool:
    return isinstance(d, str) and bool(_SYMBOL.match(d))


def fmt_shape(shape: Shape) -> str:
    return "×".join(str(d) for d in shape) if shape else "scalar"


@dataclass
class SymbolEnv:
    """Union-find over symbols, with optional concrete values."""

    parent: dict[str, Dim] = field(default_factory=dict)

    def resolve(self, d: Dim) -> Dim:
        seen = []
        while is_symbol(d) and d in self.parent:
            seen.append(d)
            d = self.parent[d]
        for s in seen:  # path compression
            self.parent[s] = d
        return d

    def resolve_shape(self, shape: Shape) -> Shape:
        return [self.resolve(d) for d in shape]

    def unify(self, a: Dim, b: Dim) -> bool:
        a, b = self.resolve(a), self.resolve(b)
        if a == b:
            return True
        if is_symbol(a):
            self.parent[a] = b
            return True
        if is_symbol(b):
            self.parent[b] = a
            return True
        return False  # two different ints, or opaque expressions that differ

    def unify_shapes(self, a: Shape, b: Shape, port: str | None = None, what: str = "") -> None:
        if len(a) != len(b):
            raise ShapeError(
                f"{what}rank mismatch: {fmt_shape(self.resolve_shape(a))} vs {fmt_shape(self.resolve_shape(b))}",
                port,
            )
        for x, y in zip(a, b):
            if not self.unify(x, y):
                raise ShapeError(
                    f"{what}shape mismatch: {fmt_shape(self.resolve_shape(a))} vs {fmt_shape(self.resolve_shape(b))}",
                    port,
                )


def dim_mul(dims: list[Dim]) -> Dim:
    ints = [d for d in dims if isinstance(d, int)]
    syms = [str(d) for d in dims if not isinstance(d, int)]
    prod = math.prod(ints) if ints else 1
    if not syms:
        return prod
    parts = ([str(prod)] if prod != 1 else []) + syms
    return "*".join(parts)


def dim_add(a: Dim, b: Dim) -> Dim:
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    return f"{a}+{b}"


def conv_out(size: Dim, kernel: int, stride: int, padding: int, dilation: int = 1) -> Dim:
    if isinstance(size, int):
        return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    if stride == 1 and 2 * padding == dilation * (kernel - 1):
        return size
    return f"conv_{size}_k{kernel}s{stride}p{padding}"


def numel(shape: Shape) -> int | None:
    if all(isinstance(d, int) for d in shape):
        return math.prod(shape)  # type: ignore[arg-type]
    return None


def dim_div(d: Dim, k: int) -> Dim:
    """d / k for an exact division; symbolic 'k*X' becomes X."""
    if isinstance(d, int):
        if d % k:
            raise ShapeError(f"{d} is not divisible by {k}")
        return d // k
    prefix = f"{k}*"
    if d.startswith(prefix) and is_symbol(d[len(prefix):]):
        return d[len(prefix):]
    return f"{d}/{k}"


def conv_transpose_out(size: Dim, kernel: int, stride: int, padding: int, output_padding: int = 0, dilation: int = 1) -> Dim:
    if isinstance(size, int):
        return (size - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    if stride == 1 and 2 * padding == dilation * (kernel - 1) and not output_padding:
        return size
    if kernel == stride and not padding and not output_padding:
        return dim_mul([stride, size])
    return f"convT_{size}_k{kernel}s{stride}"
