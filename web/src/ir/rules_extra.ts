// Mirror of core/weightsmith_core/nodes_extra.py: activations, element-wise math, tensor
// plumbing, 3-D pooling and norms, attention biases, linear attention, timesteps, VQ,
// graph convolution and more losses.

import type { Rules } from "./registry";
import { channels, predTarget, rank, scalarLoss } from "./rules_more";
import { P, binary, float, lastDim, mod, need, rankAtLeast, same, sameFloat, type Ins, type Outs } from "./rulekit";
import { INT_DTYPES, ShapeError, SymbolEnv, convOut, dimAdd, dimMul, fmtShape } from "./shapes";

const last = (p: P, ins: Ins, env: SymbolEnv): Outs => {
  const x = need(ins, "in");
  float(x, "in");
  lastDim(x, p.dim, "in", env, "dim");
  return { out: { shape: [...x.shape], dtype: x.dtype } };
};

const seqBias = (p: P, ins: Ins, env: SymbolEnv): Outs => {
  const x = need(ins, "in");
  rankAtLeast(x, 2, "in", env);
  const t = env.resolve(x.shape[1]);
  return { out: { shape: [p.heads, t, t], dtype: "f32" } };
};

const amax = (nd: number): Rules => ({
  infer: (p, ins, env) => {
    const x = need(ins, "in");
    rank(x, nd + 2, "in", env, nd === 1 ? "B×C×L" : "B×C×H×W");
    return { out: { shape: [x.shape[0], x.shape[1], ...Array(nd).fill(p.output_size)], dtype: x.dtype } };
  },
});

const pool3: Rules = {
  infer: (p, ins, env) => {
    const x = need(ins, "in");
    rank(x, 5, "in", env, "B×C×D×H×W");
    const k = p.kernel_size;
    const s = p.stride || k;
    return { out: { shape: [x.shape[0], x.shape[1], ...x.shape.slice(2).map((d) => convOut(env.resolve(d), k, s, p.padding))], dtype: x.dtype } };
  },
};

const seHidden = (p: P) => Math.max(1, Math.floor(p.channels / p.reduction));
const tsHidden = (p: P) => p.hidden || 4 * p.dim;
const prod = (xs: number[]) => xs.reduce((a, b) => a * b, 1);

export const EXTRA_RULES: Record<string, Rules> = {
  ReLU6: { infer: sameFloat },
  HardTanh: { infer: sameFloat },
  Hardsigmoid: { infer: sameFloat },
  Softsign: { infer: sameFloat },
  Swish: { infer: last, weights: (p) => ({ beta: p.dim }) },
  PReLU: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      if (p.channels !== 1) {
        rankAtLeast(x, 2, "in", env);
        channels(x, p.channels, env, "channels");
      }
      return { out: { shape: [...x.shape], dtype: x.dtype } };
    },
    weights: (p) => ({ weight: p.channels }),
  },
  GumbelSoftmax: { infer: sameFloat },
  Dropout2d: { infer: same },

  Math: { infer: sameFloat },
  Clamp: { infer: same },
  Pow: { infer: sameFloat },
  AddConst: { infer: same },
  Div: { infer: binary },
  Maximum: { infer: binary },
  Minimum: { infer: binary },
  Einsum: {
    infer: (p, ins, env) => {
      const eq = String(p.equation).replace(/ /g, "");
      const m = /^([a-zA-Z]+)(?:,([a-zA-Z]+))?->([a-zA-Z]*)$/.exec(eq);
      if (!m) throw new ShapeError(`equation '${p.equation}' must look like 'bij,bjk->bik' (letters only)`, "a");
      const sizes = new Map<string, any>();
      for (const [port, letters] of [["a", m[1]], ["b", m[2]]] as [string, string | undefined][]) {
        if (letters === undefined) {
          if (ins.b) throw new ShapeError("the equation names one operand, but b is connected", "b");
          continue;
        }
        const x = need(ins, port);
        if (x.shape.length !== letters.length) throw new ShapeError(`${port} has rank ${x.shape.length}, the equation gives it ${letters.length} (${letters})`, port);
        const s = env.resolveShape(x.shape);
        [...letters].forEach((ch, i) => {
          if (sizes.has(ch) && !env.unify(sizes.get(ch), s[i])) throw new ShapeError(`index ${ch}: ${env.resolve(sizes.get(ch))} ≠ ${s[i]}`, port);
          if (!sizes.has(ch)) sizes.set(ch, s[i]);
        });
      }
      const missing = [...m[3]].filter((c) => !sizes.has(c));
      if (missing.length) throw new ShapeError(`output index ${missing.join("")} appears in no input`, "a");
      return { out: { shape: [...m[3]].map((c) => env.resolve(sizes.get(c))), dtype: need(ins, "a").dtype } };
    },
  },
  Cumsum: {
    infer: (_p, ins, env) => {
      const x = need(ins, "in");
      rankAtLeast(x, 1, "in", env);
      return { out: { shape: [...x.shape], dtype: x.dtype } };
    },
  },

  Stack: {
    infer: (p, ins, env) => {
      const a = need(ins, "a");
      const b = need(ins, "b");
      env.unifyShapes(b.shape, a.shape, "b", "b vs a ");
      const s = env.resolveShape(a.shape);
      const d = mod(p.dim, s.length + 1);
      return { out: { shape: [...s.slice(0, d), 2, ...s.slice(d)], dtype: a.dtype } };
    },
  },
  Squeeze: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const s = env.resolveShape(x.shape);
      if (!s.length) throw new ShapeError("cannot squeeze a scalar", "in");
      const d = mod(p.dim, s.length);
      if (s[d] !== 1 && typeof s[d] === "number") throw new ShapeError(`dim ${d} has size ${s[d]}, not 1`, "in");
      return { out: { shape: [...s.slice(0, d), ...s.slice(d + 1)], dtype: x.dtype } };
    },
  },
  Pad: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const s = env.resolveShape(x.shape);
      const pad: number[] = [...p.pad];
      if (pad.length % 2 || pad.length / 2 > s.length) throw new ShapeError(`pad ${pad} needs (left, right) pairs for at most ${s.length} trailing dims`, "in");
      for (let i = 0; i < pad.length / 2; i++) {
        const d = s.length - 1 - i;
        if (pad[2 * i] + pad[2 * i + 1]) s[d] = dimAdd(s[d], pad[2 * i] + pad[2 * i + 1]);
      }
      return { out: { shape: s, dtype: x.dtype } };
    },
  },
  Repeat: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const s = env.resolveShape(x.shape);
      const reps: number[] = [...p.repeats];
      if (reps.length !== s.length) throw new ShapeError(`repeats ${reps} needs one entry per dim of ${fmtShape(s)}`, "in");
      return { out: { shape: s.map((d, i) => (reps[i] === 1 ? d : dimMul([reps[i], d]))), dtype: x.dtype } };
    },
  },
  TopK: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const s = env.resolveShape(x.shape);
      if (!s.length) throw new ShapeError("cannot take top-k of a scalar", "in");
      const d = mod(p.dim, s.length);
      if (typeof s[d] === "number" && p.k > (s[d] as number)) throw new ShapeError(`k=${p.k} > size ${s[d]}`, "in");
      s[d] = p.k;
      return { values: { shape: [...s], dtype: x.dtype }, indices: { shape: [...s], dtype: "i64" } };
    },
  },
  ArgMax: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const s = env.resolveShape(x.shape);
      if (!s.length) throw new ShapeError("cannot argmax a scalar", "in");
      const d = mod(p.dim, s.length);
      return { out: { shape: [...s.slice(0, d), ...(p.keepdim ? [1] : []), ...s.slice(d + 1)], dtype: "i64" } };
    },
  },
  OneHot: {
    infer: (p, ins) => {
      const x = need(ins, "in");
      if (!INT_DTYPES.includes(x.dtype)) throw new ShapeError(`expects class ids (i64), got ${x.dtype}`, "in");
      return { out: { shape: [...x.shape, p.classes], dtype: "f32" } };
    },
  },
  Cast: {
    infer: (p, ins) => {
      const x = need(ins, "in");
      return { out: { shape: [...x.shape], dtype: p.dtype } };
    },
  },
  StopGradient: { infer: same },
  Parameter: {
    infer: (p) => {
      const shape: any[] = [...(p.shape ?? [])];
      if (!shape.length || !shape.every((d) => typeof d === "number" && Number.isInteger(d) && d > 0)) throw new ShapeError(`shape ${shape} must be positive integers`, null);
      return { out: { shape, dtype: "f32" } };
    },
    weights: (p) => ({ weight: prod(p.shape) }),
  },

  MaxPool3d: pool3,
  AvgPool3d: pool3,
  AdaptiveMaxPool1d: amax(1),
  AdaptiveMaxPool2d: amax(2),
  Interpolate: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      const s = env.resolveShape(x.shape);
      if (s.length < 3) throw new ShapeError(`expects B×C×spatial…, got ${fmtShape(s)}`, "in");
      const size: number[] = [...p.size];
      if (size.length !== s.length - 2) throw new ShapeError(`size ${size} needs one entry per spatial dim (${s.length - 2})`, "in");
      return { out: { shape: [...s.slice(0, 2), ...size], dtype: x.dtype } };
    },
  },

  BatchNorm3d: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      rank(x, 5, "in", env, "B×C×D×H×W");
      channels(x, p.features, env, "features");
      return { out: { shape: [...x.shape], dtype: x.dtype } };
    },
    weights: (p) => ({ weight: p.features, bias: p.features }),
  },
  LayerScale: { infer: last, weights: (p) => ({ gamma: p.dim }) },
  SqueezeExcite: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      rankAtLeast(x, 3, "in", env);
      channels(x, p.channels, env, "channels");
      return { out: { shape: [...x.shape], dtype: x.dtype } };
    },
    weights: (p) => ({ "fc1.weight": p.channels * seHidden(p), "fc1.bias": seHidden(p), "fc2.weight": seHidden(p) * p.channels, "fc2.bias": p.channels }),
  },

  ALiBi: { infer: seqBias },
  SlidingWindowBias: { infer: seqBias },
  LinearAttention: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      rank(x, 3, "in", env, "B×T×D");
      lastDim(x, p.dim, "in", env, "dim");
      if (p.dim % p.heads) throw new ShapeError(`dim=${p.dim} is not divisible by heads=${p.heads}`, "in");
      return { out: { shape: [...x.shape], dtype: x.dtype } };
    },
    weights: (p) => {
      const w: Record<string, number> = {};
      for (const k of ["q_proj", "k_proj", "v_proj", "o_proj"]) {
        w[`${k}.weight`] = p.dim * p.dim;
        w[`${k}.bias`] = p.dim;
      }
      return w;
    },
    linearTargets: (p) => Object.fromEntries(["q_proj", "k_proj", "v_proj", "o_proj"].map((k) => [k, [p.dim, p.dim] as [number, number]])),
    outFeatures: (p) => p.dim,
  },

  TimestepEmbedding: {
    infer: (p, ins, env) => {
      const t = need(ins, "in");
      if (t.shape.length !== 1) throw new ShapeError(`expects one timestep per sample (B), got ${fmtShape(env.resolveShape(t.shape))}`, "in");
      if (p.dim % 2) throw new ShapeError("dim must be even", "in");
      return { out: { shape: [t.shape[0], p.dim], dtype: "f32" } };
    },
    weights: (p) => ({ "fc1.weight": p.dim * tsHidden(p), "fc1.bias": tsHidden(p), "fc2.weight": tsHidden(p) * p.dim, "fc2.bias": p.dim }),
  },

  VectorQuantizer: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      lastDim(x, p.dim, "in", env, "dim");
      return { out: { shape: [...x.shape], dtype: x.dtype }, indices: { shape: x.shape.slice(0, -1), dtype: "i64" }, commit: { shape: [], dtype: "f32" } };
    },
    weights: (p) => ({ "codebook.weight": p.codes * p.dim }),
  },

  DiffusionForward: {
    infer: (_p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      rankAtLeast(x, 2, "in", env);
      return { noisy: { shape: [...x.shape], dtype: x.dtype }, t: { shape: [x.shape[0]], dtype: "i64" }, noise: { shape: [...x.shape], dtype: x.dtype } };
    },
  },

  GraphConv: {
    infer: (p, ins, env) => {
      const x = need(ins, "x");
      const a = need(ins, "adj");
      float(x, "x");
      rank(x, 3, "x", env, "B×N×F");
      lastDim(x, p.in, "x", env, "in");
      rank(a, 3, "adj", env, "B×N×N");
      const n = x.shape[1];
      if (!env.unify(a.shape[0], x.shape[0]) || !env.unify(a.shape[1], n) || !env.unify(a.shape[2], n))
        throw new ShapeError(`adj must be B×N×N for x ${fmtShape(env.resolveShape(x.shape))}, got ${fmtShape(env.resolveShape(a.shape))}`, "adj");
      return { out: { shape: [x.shape[0], n, p.out], dtype: x.dtype } };
    },
    weights: (p) => ({ "lin.weight": p.in * p.out, ...(p.bias ? { "lin.bias": p.out } : {}) }),
    linearTargets: (p) => ({ lin: [p.in, p.out] }),
    outFeatures: (p) => p.out,
  },
  GraphPool: {
    infer: (_p, ins, env) => {
      const x = need(ins, "in");
      rank(x, 3, "in", env, "B×N×F");
      return { out: { shape: [x.shape[0], x.shape[2]], dtype: x.dtype } };
    },
  },

  AuxLoss: {
    infer: (_p, ins) => {
      float(need(ins, "in"), "in");
      return scalarLoss();
    },
  },
  Focal: { infer: predTarget("logits", true) },
  Dice: { infer: predTarget("pred") },
  GaussianNLL: {
    infer: (p, ins, env) => {
      const out = predTarget("mean")!(p, ins, env);
      env.unifyShapes(need(ins, "var").shape, need(ins, "mean").shape, "var", "var vs mean ");
      return out;
    },
  },
  CosineEmbedding: {
    infer: (_p, ins, env) => {
      env.unifyShapes(need(ins, "b").shape, need(ins, "a").shape, "b", "b vs a ");
      return scalarLoss();
    },
  },
};
