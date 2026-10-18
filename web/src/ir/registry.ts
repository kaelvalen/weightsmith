// Shape rules and parameter counts, mirrored from core/weightsmith_core/registry.py.
// Ports, params and docs are not duplicated: they come from catalog.json, which
// scripts/gen_fixtures.py writes from the Python registry.

import catalogJson from "./catalog.json";
import { INT_DTYPES, ShapeError, SymbolEnv, dimAdd, dimMul, fmtShape, numel } from "./shapes";
import { P, binary, float, lastDim, mod, need, norm, rankAtLeast, same, sameFloat, type Ins, type Outs } from "./rulekit";

import { MORE_RULES } from "./rules_more";

export type { P };

export interface ParamSpec {
  default: unknown;
  kind: string;
  required: boolean;
  choices: unknown[] | null;
  doc: string;
}

export interface TypeSpec {
  type: string;
  category: string;
  doc: string;
  inputs: { name: string; optional: boolean; doc: string }[];
  outputs: { name: string }[];
  params: Record<string, ParamSpec>;
  loss: boolean;
}

export const CATALOG: TypeSpec[] = (catalogJson as unknown as { types: TypeSpec[] }).types;
export const TYPES: Record<string, TypeSpec> = Object.fromEntries(CATALOG.map((t) => [t.type, t]));

export function expandDefaults(type: string, params: P = {}): P {
  const t = TYPES[type];
  const out: P = { ...params };
  if (!t) return out;
  for (const [k, p] of Object.entries(t.params)) if (!(k in out) && !p.required) out[k] = p.default;
  return out;
}

export interface Rules {
  infer?: (p: P, ins: Ins, env: SymbolEnv) => Outs;
  weights?: (p: P) => Record<string, number>;
  linearTargets?: (p: P) => Record<string, [number, number]>;
  outFeatures?: (p: P) => number | null;
}


export const RULES: Record<string, Rules> = {
  Input: { infer: (p) => ({ out: { shape: [...p.shape], dtype: p.dtype ?? "f32" } }) },
  Output: { infer: () => ({}) },
  Linear: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      lastDim(x, p.in, "in", env, "in");
      return { out: { shape: [...x.shape.slice(0, -1), p.out], dtype: x.dtype } };
    },
    weights: (p): Record<string, number> => {
      const weight = p.tie_to ? 0 : p.in * p.out; // a tied weight is counted on its owner
      return p.bias ?? true ? { weight, bias: p.out } : { weight };
    },
    linearTargets: (p) => ({ weight: [p.in, p.out] }),
    outFeatures: (p) => p.out,
  },
  MLP: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      lastDim(x, p.dim, "in", env, "dim");
      return { out: { shape: [...x.shape.slice(0, -1), p.out || p.dim], dtype: x.dtype } };
    },
    weights: (p) => {
      const o = p.out || p.dim;
      return { "fc1.weight": p.dim * p.hidden, "fc1.bias": p.hidden, "fc2.weight": p.hidden * o, "fc2.bias": o };
    },
    linearTargets: (p) => ({ fc1: [p.dim, p.hidden], fc2: [p.hidden, p.out || p.dim] }),
    outFeatures: (p) => p.out || p.dim,
  },
  GatedMLP: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      lastDim(x, p.dim, "in", env, "dim");
      return { out: { shape: [...x.shape.slice(0, -1), p.out || p.dim], dtype: x.dtype } };
    },
    weights: (p) => {
      const o = p.out || p.dim;
      const w: Record<string, number> = { "gate_proj.weight": p.dim * p.hidden, "up_proj.weight": p.dim * p.hidden, "down_proj.weight": p.hidden * o };
      if (p.bias) Object.assign(w, { "gate_proj.bias": p.hidden, "up_proj.bias": p.hidden, "down_proj.bias": o });
      return w;
    },
    linearTargets: (p) => ({ gate_proj: [p.dim, p.hidden], up_proj: [p.dim, p.hidden], down_proj: [p.hidden, p.out || p.dim] }),
    outFeatures: (p) => p.out || p.dim,
  },
  Embedding: {
    infer: (p, ins) => {
      const x = need(ins, "in");
      if (!INT_DTYPES.includes(x.dtype)) throw new ShapeError(`expects integer token ids, got ${x.dtype}`, "in");
      return { out: { shape: [...x.shape, p.dim], dtype: "f32" } };
    },
    weights: (p) => ({ weight: p.num * p.dim }),
    outFeatures: (p) => p.dim,
  },
  PositionalEmbedding: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      if (x.shape.length !== 3) throw new ShapeError(`expects B×T×D, got ${fmtShape(env.resolveShape(x.shape))}`, "in");
      lastDim(x, p.dim, "in", env, "dim");
      const t = env.resolve(x.shape[1]);
      if (typeof t === "number" && t > p.max_len) throw new ShapeError(`sequence length ${t} > max_len=${p.max_len}`, "in");
      return { out: { shape: [...x.shape], dtype: x.dtype } };
    },
    weights: (p) => ({ weight: p.max_len * p.dim }),
    outFeatures: (p) => p.dim,
  },
  LayerNorm: { infer: norm, weights: (p) => ({ weight: p.dim, bias: p.dim }), outFeatures: (p) => p.dim },
  RMSNorm: { infer: norm, weights: (p) => ({ weight: p.dim }), outFeatures: (p) => p.dim },
  ReLU: { infer: sameFloat },
  SiLU: { infer: sameFloat },
  Tanh: { infer: sameFloat },
  Sigmoid: { infer: sameFloat },
  Softmax: { infer: sameFloat },
  Dropout: { infer: same },
  Add: { infer: binary },
  Mul: { infer: binary },
  Concat: {
    infer: (p, ins, env) => {
      const a = need(ins, "a");
      const b = need(ins, "b");
      if (a.shape.length !== b.shape.length) throw new ShapeError(`rank mismatch ${fmtShape(a.shape)} vs ${fmtShape(b.shape)}`, "b");
      const d = mod(p.dim, a.shape.length);
      const out = a.shape.map((x, i) => {
        const y = b.shape[i];
        if (i === d) return dimAdd(env.resolve(x), env.resolve(y));
        if (env.unify(x, y)) return env.resolve(x);
        throw new ShapeError(`dim ${i} mismatch ${env.resolve(x)} vs ${env.resolve(y)}`, "b");
      });
      return { out: { shape: out, dtype: a.dtype } };
    },
  },
  Reshape: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const target = [...p.shape];
      if (target.filter((d) => d === -1).length > 1) throw new ShapeError("at most one -1 in reshape", "in");
      const a = numel(env.resolveShape(x.shape.slice(1)));
      const rest = target.slice(1).filter((d) => d !== -1);
      const b = numel(rest);
      const bad = () => new ShapeError(`cannot reshape ${fmtShape(env.resolveShape(x.shape))} into ${fmtShape(target)}`, "in");
      if (target.includes(-1)) {
        if (a !== null && b) {
          if (a % b) throw bad();
          target[target.indexOf(-1)] = a / b;
        } else target[target.indexOf(-1)] = dimMul(env.resolveShape(x.shape.slice(1)));
      } else if (a !== null && b !== null && a !== b) throw bad();
      if (target.length && !env.unify(x.shape[0], target[0])) throw new ShapeError("reshape must keep the batch dimension", "in");
      return { out: { shape: target, dtype: x.dtype } };
    },
  },
  MeanPool: {
    infer: (p, ins) => {
      const x = need(ins, "in");
      float(x, "in");
      const d = mod(p.dim, Math.max(x.shape.length, 1));
      if (x.shape.length < 2) throw new ShapeError("expects rank ≥ 2", "in");
      return { out: { shape: x.shape.filter((_, i) => i !== d), dtype: x.dtype } };
    },
  },
  CrossEntropy: {
    infer: (_p, ins, env) => {
      const logits = need(ins, "logits");
      float(logits, "logits");
      rankAtLeast(logits, 2, "logits", env);
      const target = ins.target;
      if (target) {
        if (!INT_DTYPES.includes(target.dtype)) throw new ShapeError(`target must be class ids (i64), got ${target.dtype}`, "target");
        env.unifyShapes(target.shape, logits.shape.slice(0, -1), "target", "target vs logits[:-1] ");
      }
      return { loss: { shape: [], dtype: "f32" } };
    },
  },
  MSE: {
    infer: (_p, ins, env) => {
      const pred = need(ins, "pred");
      float(pred, "pred");
      if (ins.target) env.unifyShapes(ins.target.shape, pred.shape, "target", "target vs pred ");
      return { loss: { shape: [], dtype: "f32" } };
    },
  },
  Subgraph: {},
  PythonModule: {},
};

Object.assign(RULES, MORE_RULES);

export const weightsOf = (type: string, p: P) => RULES[type]?.weights?.(p) ?? {};
export const linearTargetsOf = (type: string, p: P) => RULES[type]?.linearTargets?.(p) ?? {};
export const outFeaturesOf = (type: string, p: P) => RULES[type]?.outFeatures?.(p) ?? null;
