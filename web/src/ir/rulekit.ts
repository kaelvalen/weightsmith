// Helpers shared by the shape rules (registry.ts, rules_more.ts). Mirror of the
// helpers at the top of core/weightsmith_core/registry.py.

import { FLOAT_DTYPES, ShapeError, SymbolEnv, TensorType, fmtShape } from "./shapes";

export type P = Record<string, any>;
export type Ins = Record<string, TensorType | null>;
export type Outs = Record<string, TensorType>;

export const mod = (a: number, n: number) => ((a % n) + n) % n;

export function need(ins: Ins, port: string): TensorType {
  const t = ins[port];
  if (!t) throw new ShapeError("input not connected", port);
  return t;
}
export function float(t: TensorType, port: string) {
  if (!FLOAT_DTYPES.includes(t.dtype)) throw new ShapeError(`expects a float tensor, got ${t.dtype}`, port);
}
export function rankAtLeast(t: TensorType, n: number, port: string, env: SymbolEnv) {
  if (t.shape.length < n) throw new ShapeError(`expects rank ≥ ${n}, got ${fmtShape(env.resolveShape(t.shape))}`, port);
}
export function lastDim(t: TensorType, size: any, port: string, env: SymbolEnv, pname: string) {
  rankAtLeast(t, 1, port, env);
  if (!env.unify(t.shape[t.shape.length - 1], size))
    throw new ShapeError(
      `last dim ${env.resolve(t.shape[t.shape.length - 1])} ≠ ${pname}=${size} (input ${fmtShape(env.resolveShape(t.shape))})`,
      port,
    );
}
export const same = (_p: P, ins: Ins): Outs => {
  const x = need(ins, "in");
  return { out: { shape: [...x.shape], dtype: x.dtype } };
};
export const sameFloat = (_p: P, ins: Ins): Outs => {
  const x = need(ins, "in");
  float(x, "in");
  return { out: { shape: [...x.shape], dtype: x.dtype } };
};
export const norm = (p: P, ins: Ins, env: SymbolEnv): Outs => {
  const x = need(ins, "in");
  float(x, "in");
  lastDim(x, p.dim, "in", env, "dim");
  return { out: { shape: [...x.shape], dtype: x.dtype } };
};
export function broadcast(a: TensorType, b: TensorType, env: SymbolEnv): TensorType {
  let sa = env.resolveShape(a.shape);
  let sb = env.resolveShape(b.shape);
  const n = Math.max(sa.length, sb.length);
  sa = [...Array(n - sa.length).fill(1), ...sa];
  sb = [...Array(n - sb.length).fill(1), ...sb];
  const out = sa.map((x, i) => {
    const y = sb[i];
    if (x === 1) return y;
    if (y === 1) return x;
    if (env.unify(x, y)) return env.resolve(x);
    throw new ShapeError(`cannot broadcast ${fmtShape(sa)} with ${fmtShape(sb)}`, "b");
  });
  if (a.dtype !== b.dtype) throw new ShapeError(`dtype mismatch ${a.dtype} vs ${b.dtype}`, "b");
  return { shape: out, dtype: a.dtype };
}
export const binary = (_p: P, ins: Ins, env: SymbolEnv): Outs => ({ out: broadcast(need(ins, "a"), need(ins, "b"), env) });

