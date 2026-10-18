// Mirror of core/weightsmith_core/nodes_more.py: the wider node catalogue.
// Parameter names in `weights` match the generated nn.Module (frozen_params masks use them).

import type { Rules } from "./registry";
import { P, binary, broadcast, float, lastDim, mod, need, rankAtLeast, same, sameFloat, type Ins, type Outs } from "./rulekit";
import { INT_DTYPES, ShapeError, SymbolEnv, TensorType, convOut, convTransposeOut, dimAdd, dimDiv, dimMul, fmtShape } from "./shapes";

function rank(x: TensorType, n: number, port: string, env: SymbolEnv, what: string) {
  if (x.shape.length !== n) throw new ShapeError(`expects ${what}, got ${fmtShape(env.resolveShape(x.shape))}`, port);
}
function channels(x: TensorType, c: any, env: SymbolEnv, pname = "in_channels") {
  if (!env.unify(x.shape[1], c)) throw new ShapeError(`channels ${env.resolve(x.shape[1])} ≠ ${pname}=${c}`, "in");
}
const bias = (p: P, key = "bias") => p[key] ?? true;

function conv(nd: number): Rules {
  const layout = ["", "B×C×L", "B×C×H×W", "B×C×D×H×W"][nd];
  return {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      rank(x, nd + 2, "in", env, layout);
      channels(x, p.in_channels, env);
      const g = p.groups;
      if (p.in_channels % g || p.out_channels % g) throw new ShapeError(`groups=${g} must divide in_channels and out_channels`, "in");
      const sp = x.shape.slice(2).map((d) => convOut(env.resolve(d), p.kernel_size, p.stride, p.padding, p.dilation));
      return { out: { shape: [x.shape[0], p.out_channels, ...sp], dtype: x.dtype } };
    },
    weights: (p) => {
      const w: Record<string, number> = { weight: p.out_channels * (p.in_channels / p.groups) * p.kernel_size ** nd };
      if (bias(p)) w.bias = p.out_channels;
      return w;
    },
  };
}

function pool(nd: number): Rules {
  return {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      rank(x, nd + 2, "in", env, nd === 1 ? "B×C×L" : "B×C×H×W");
      const k = p.kernel_size;
      const s = p.stride || k;
      const sp = x.shape.slice(2).map((d) => convOut(env.resolve(d), k, s, p.padding));
      return { out: { shape: [x.shape[0], x.shape[1], ...sp], dtype: x.dtype } };
    },
  };
}

function adaptive(nd: number): Rules {
  return {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      rank(x, nd + 2, "in", env, nd === 1 ? "B×C×L" : "B×C×H×W");
      return { out: { shape: [x.shape[0], x.shape[1], ...Array(nd).fill(p.output_size)], dtype: x.dtype } };
    },
  };
}

function bn(nd: number): Rules {
  return {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      const ok = nd === 1 ? [2, 3].includes(x.shape.length) : x.shape.length === 4;
      if (!ok) throw new ShapeError(`expects ${nd === 1 ? "B×C or B×C×L" : "B×C×H×W"}, got ${fmtShape(env.resolveShape(x.shape))}`, "in");
      channels(x, p.features, env, "features");
      return { out: { shape: [...x.shape], dtype: x.dtype } };
    },
    weights: (p) => ({ weight: p.features, bias: p.features }),
  };
}

const kvDim = (p: P) => (p.kv_heads || p.heads) * (p.dim / p.heads);
const mambaDims = (p: P): [number, number, number] => [p.expand * p.dim, p.dt_rank || Math.ceil(p.dim / 16), p.d_state];

function rnn(gates: number): Rules {
  return {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      rank(x, 3, "in", env, "B×T×F");
      lastDim(x, p.input, "in", env, "input");
      const h = p.hidden * (p.bidirectional ? 2 : 1);
      return { out: { shape: [x.shape[0], x.shape[1], h], dtype: x.dtype }, last: { shape: [x.shape[0], h], dtype: x.dtype } };
    },
    weights: (p) => {
      const h = p.hidden;
      const dirs = p.bidirectional ? 2 : 1;
      const w: Record<string, number> = {};
      for (let layer = 0; layer < p.layers; layer++) {
        const fin = layer === 0 ? p.input : h * dirs;
        for (let d = 0; d < dirs; d++) {
          const sfx = `l${layer}${d ? "_reverse" : ""}`;
          w[`rnn.weight_ih_${sfx}`] = gates * h * fin;
          w[`rnn.weight_hh_${sfx}`] = gates * h * h;
          if (bias(p)) {
            w[`rnn.bias_ih_${sfx}`] = gates * h;
            w[`rnn.bias_hh_${sfx}`] = gates * h;
          }
        }
      }
      return w;
    },
    outFeatures: (p) => p.hidden * (p.bidirectional ? 2 : 1),
  };
}

function predTarget(predPort: string, intTarget = false): Rules["infer"] {
  return (_p, ins, env) => {
    const pred = need(ins, predPort);
    float(pred, predPort);
    const t = ins.target;
    if (t) {
      if (intTarget) {
        if (!INT_DTYPES.includes(t.dtype)) throw new ShapeError(`target must be class ids (i64), got ${t.dtype}`, "target");
        env.unifyShapes(t.shape, pred.shape.slice(0, -1), "target", `target vs ${predPort}[:-1] `);
      } else env.unifyShapes(t.shape, pred.shape, "target", `target vs ${predPort} `);
    }
    return { loss: { shape: [], dtype: "f32" } };
  };
}
const scalarLoss = (): Outs => ({ loss: { shape: [], dtype: "f32" } });

function halve(x: TensorType, dim: number, env: SymbolEnv): TensorType["shape"] {
  const s = env.resolveShape(x.shape);
  const d = mod(dim, s.length);
  try {
    s[d] = dimDiv(s[d], 2);
  } catch {
    throw new ShapeError(`dim ${d} has odd size ${s[d]}`, "in");
  }
  return s;
}

export const MORE_RULES: Record<string, Rules> = {
  Conv1d: conv(1),
  Conv2d: conv(2),
  Conv3d: conv(3),
  ConvTranspose2d: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      rank(x, 4, "in", env, "B×C×H×W");
      channels(x, p.in_channels, env);
      const sp = x.shape.slice(2).map((d) => convTransposeOut(env.resolve(d), p.kernel_size, p.stride, p.padding, p.output_padding));
      return { out: { shape: [x.shape[0], p.out_channels, ...sp], dtype: x.dtype } };
    },
    weights: (p): Record<string, number> => ({ weight: p.in_channels * p.out_channels * p.kernel_size ** 2, ...(bias(p) ? { bias: p.out_channels } : {}) }),
  },
  MaxPool1d: pool(1),
  MaxPool2d: pool(2),
  AvgPool1d: pool(1),
  AvgPool2d: pool(2),
  AdaptiveAvgPool1d: adaptive(1),
  AdaptiveAvgPool2d: adaptive(2),
  Upsample: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      rank(x, 4, "in", env, "B×C×H×W");
      const s = p.scale_factor;
      return { out: { shape: [x.shape[0], x.shape[1], dimMul([s, env.resolve(x.shape[2])]), dimMul([s, env.resolve(x.shape[3])])], dtype: x.dtype } };
    },
  },
  PixelShuffle: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      rank(x, 4, "in", env, "B×C×H×W");
      const r = p.upscale_factor;
      let c;
      try {
        c = dimDiv(env.resolve(x.shape[1]), r * r);
      } catch {
        throw new ShapeError(`channels ${env.resolve(x.shape[1])} not divisible by upscale_factor²=${r * r}`, "in");
      }
      return { out: { shape: [x.shape[0], c, dimMul([r, env.resolve(x.shape[2])]), dimMul([r, env.resolve(x.shape[3])])], dtype: x.dtype } };
    },
  },
  BatchNorm1d: bn(1),
  BatchNorm2d: bn(2),
  GroupNorm: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      rankAtLeast(x, 2, "in", env);
      channels(x, p.channels, env, "channels");
      if (p.channels % p.groups) throw new ShapeError(`groups=${p.groups} must divide channels=${p.channels}`, "in");
      return { out: { shape: [...x.shape], dtype: x.dtype } };
    },
    weights: (p) => ({ weight: p.channels, bias: p.channels }),
  },
  InstanceNorm2d: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      rank(x, 4, "in", env, "B×C×H×W");
      channels(x, p.features, env, "features");
      return { out: { shape: [...x.shape], dtype: x.dtype } };
    },
    weights: (p): Record<string, number> => (p.affine ? { weight: p.features, bias: p.features } : {}),
  },
  L2Normalize: { infer: sameFloat },
  GELU: { infer: sameFloat },
  LeakyReLU: { infer: sameFloat },
  ELU: { infer: sameFloat },
  SELU: { infer: sameFloat },
  Mish: { infer: sameFloat },
  Softplus: { infer: sameFloat },
  Hardswish: { infer: sameFloat },
  LogSoftmax: { infer: sameFloat },
  GLU: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      return { out: { shape: halve(x, p.dim, env), dtype: x.dtype } };
    },
  },
  DropPath: { infer: same },
  SinusoidalPositional: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      rank(x, 3, "in", env, "B×T×D");
      lastDim(x, p.dim, "in", env, "dim");
      if (p.dim % 2) throw new ShapeError("dim must be even", "in");
      return { out: { shape: [...x.shape], dtype: x.dtype } };
    },
  },
  ClassToken: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      rank(x, 3, "in", env, "B×T×D");
      lastDim(x, p.dim, "in", env, "dim");
      return { out: { shape: [x.shape[0], dimAdd(env.resolve(x.shape[1]), 1), x.shape[2]], dtype: x.dtype } };
    },
    weights: (p) => ({ token: p.dim }),
  },
  PatchEmbed: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      rank(x, 4, "in", env, "B×C×H×W");
      channels(x, p.in_channels, env);
      const h = convOut(env.resolve(x.shape[2]), p.patch, p.patch, 0);
      const w = convOut(env.resolve(x.shape[3]), p.patch, p.patch, 0);
      return { out: { shape: [x.shape[0], dimMul([h, w]), p.dim], dtype: x.dtype } };
    },
    weights: (p) => ({ "proj.weight": p.dim * p.in_channels * p.patch ** 2, "proj.bias": p.dim }),
    linearTargets: (p) => ({ proj: [p.in_channels * p.patch ** 2, p.dim] }),
    outFeatures: (p) => p.dim,
  },
  RelativePositionBias: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      rankAtLeast(x, 2, "in", env);
      const t = env.resolve(x.shape[1]);
      return { out: { shape: [p.heads, t, t], dtype: "f32" } };
    },
    weights: (p) => ({ "table.weight": p.buckets * p.heads }),
  },
  Attention: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      rank(x, 3, "in", env, "B×T×D");
      lastDim(x, p.dim, "in", env, "dim");
      if (p.dim % p.heads) throw new ShapeError(`dim=${p.dim} is not divisible by heads=${p.heads}`, "in");
      const kv = p.kv_heads || p.heads;
      if (p.heads % kv) throw new ShapeError(`heads=${p.heads} is not divisible by kv_heads=${kv}`, "in");
      if (p.rope && (p.dim / p.heads) % 2) throw new ShapeError("RoPE needs an even head dimension", "in");
      const b = ins.bias;
      if (b) {
        const t = env.resolve(x.shape[1]);
        const bs = env.resolveShape(b.shape);
        const n = bs.length;
        if (![3, 4].includes(n) || !env.unify(bs[n - 1], t) || !env.unify(bs[n - 2], t) || !env.unify(bs[n - 3], p.heads))
          throw new ShapeError(`bias must be heads×T×T (${p.heads}×${t}×${t}), got ${fmtShape(bs)}`, "bias");
      }
      return { out: { shape: [...x.shape], dtype: x.dtype } };
    },
    weights: (p) => {
      const kvd = kvDim(p);
      const w: Record<string, number> = {};
      for (const [k, out] of [["q_proj", p.dim], ["k_proj", kvd], ["v_proj", kvd], ["o_proj", p.dim]] as [string, number][]) {
        w[`${k}.weight`] = p.dim * out;
        if (k === "o_proj" ? (p.o_bias ?? true) : bias(p)) w[`${k}.bias`] = out;
      }
      if (p.qk_norm === "rms") {
        w["q_norm.weight"] = p.dim / p.heads;
        w["k_norm.weight"] = p.dim / p.heads;
      }
      return w;
    },
    linearTargets: (p) => ({ q_proj: [p.dim, p.dim], k_proj: [p.dim, kvDim(p)], v_proj: [p.dim, kvDim(p)], o_proj: [p.dim, p.dim] }),
    outFeatures: (p) => p.dim,
  },
  CrossAttention: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const c = need(ins, "context");
      float(x, "in");
      rank(x, 3, "in", env, "B×T×D");
      rank(c, 3, "context", env, "B×S×Dc");
      lastDim(x, p.dim, "in", env, "dim");
      lastDim(c, p.context_dim || p.dim, "context", env, "context_dim");
      if (!env.unify(x.shape[0], c.shape[0])) throw new ShapeError("batch of context differs from input", "context");
      if (p.dim % p.heads) throw new ShapeError(`dim=${p.dim} is not divisible by heads=${p.heads}`, "in");
      return { out: { shape: [...x.shape], dtype: x.dtype } };
    },
    weights: (p) => {
      const d = p.dim;
      const c = p.context_dim || p.dim;
      const w: Record<string, number> = { "q_proj.weight": d * d, "k_proj.weight": c * d, "v_proj.weight": c * d, "o_proj.weight": d * d };
      if (bias(p)) Object.assign(w, { "q_proj.bias": d, "k_proj.bias": d, "v_proj.bias": d, "o_proj.bias": d });
      return w;
    },
    linearTargets: (p) => ({ q_proj: [p.dim, p.dim], k_proj: [p.context_dim || p.dim, p.dim], v_proj: [p.context_dim || p.dim, p.dim], o_proj: [p.dim, p.dim] }),
    outFeatures: (p) => p.dim,
  },
  LSTM: rnn(4),
  GRU: rnn(3),
  RNN: rnn(1),
  Mamba: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      rank(x, 3, "in", env, "B×T×D");
      lastDim(x, p.dim, "in", env, "dim");
      return { out: { shape: [...x.shape], dtype: x.dtype } };
    },
    weights: (p) => {
      const [e, r, n] = mambaDims(p);
      const d = p.dim;
      return {
        "in_proj.weight": d * 2 * e,
        "conv1d.weight": e * p.d_conv,
        "conv1d.bias": e,
        "x_proj.weight": e * (r + 2 * n),
        "dt_proj.weight": r * e,
        "dt_proj.bias": e,
        A_log: e * n,
        D: e,
        "out_proj.weight": e * d,
      };
    },
    linearTargets: (p) => ({ in_proj: [p.dim, 2 * mambaDims(p)[0]], out_proj: [mambaDims(p)[0], p.dim] }),
    outFeatures: (p) => p.dim,
  },
  MoE: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      float(x, "in");
      lastDim(x, p.dim, "in", env, "dim");
      if (!(p.top_k >= 1 && p.top_k <= p.num_experts)) throw new ShapeError(`top_k=${p.top_k} must be in 1..num_experts=${p.num_experts}`, "in");
      return { out: { shape: [...x.shape], dtype: x.dtype } };
    },
    weights: (p) => {
      const [d, h, e] = [p.dim, p.hidden, p.num_experts];
      const w: Record<string, number> = { "router.weight": d * e, "experts.gate_up_proj": e * d * 2 * h, "experts.down_proj": e * h * d };
      if (p.shared_hidden) {
        const s = p.shared_hidden;
        Object.assign(w, { "shared_expert.gate_proj.weight": d * s, "shared_expert.up_proj.weight": d * s, "shared_expert.down_proj.weight": s * d });
      }
      return w;
    },
    linearTargets: (p) => ({ router: [p.dim, p.num_experts] }),
    outFeatures: (p) => p.dim,
  },
  Sub: { infer: binary },
  Flatten: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const shape = env.resolveShape(x.shape);
      const n = shape.length;
      const s = n ? mod(p.start_dim, n) : 0;
      const e = n ? mod(p.end_dim, n) : 0;
      if (s > e) throw new ShapeError(`start_dim=${p.start_dim} is after end_dim=${p.end_dim} for ${fmtShape(shape)}`, "in");
      return { out: { shape: [...shape.slice(0, s), dimMul(shape.slice(s, e + 1)), ...shape.slice(e + 1)], dtype: x.dtype } };
    },
  },
  Unflatten: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const shape = env.resolveShape(x.shape);
      const d = mod(p.dim, shape.length);
      const sizes = [...p.sizes];
      if (sizes.filter((v: number) => v === -1).length > 1) throw new ShapeError("at most one -1 in sizes", "in");
      const known = sizes.filter((v: number) => v !== -1).reduce((a: number, b: number) => a * b, 1);
      if (sizes.includes(-1)) sizes[sizes.indexOf(-1)] = dimDiv(shape[d], known);
      else if (typeof shape[d] === "number" && shape[d] !== known) throw new ShapeError(`cannot split ${shape[d]} into ${sizes}`, "in");
      return { out: { shape: [...shape.slice(0, d), ...sizes, ...shape.slice(d + 1)], dtype: x.dtype } };
    },
  },
  Transpose: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const s = env.resolveShape(x.shape);
      const [a, b] = [mod(p.dim0, s.length), mod(p.dim1, s.length)];
      [s[a], s[b]] = [s[b], s[a]];
      return { out: { shape: s, dtype: x.dtype } };
    },
  },
  Permute: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const s = env.resolveShape(x.shape);
      const dims: number[] = [...p.dims];
      const sorted = dims.map((d) => mod(d, s.length)).sort((a, b) => a - b);
      if (sorted.length !== s.length || sorted.some((d, i) => d !== i)) throw new ShapeError(`dims ${dims} is not a permutation of ${s.length} dimensions`, "in");
      return { out: { shape: dims.map((d) => s[d < 0 ? d + s.length : d]), dtype: x.dtype } };
    },
  },
  Select: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const s = env.resolveShape(x.shape);
      const d = mod(p.dim, s.length);
      if (typeof s[d] === "number" && !(-s[d] <= p.index && p.index < s[d])) throw new ShapeError(`index ${p.index} out of range for size ${s[d]}`, "in");
      return { out: { shape: [...s.slice(0, d), ...s.slice(d + 1)], dtype: x.dtype } };
    },
  },
  Slice: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const s = env.resolveShape(x.shape);
      const d = mod(p.dim, s.length);
      const start = p.start;
      const end = p.end ?? null;
      const size = s[d];
      if (typeof size === "number") {
        const stop = end === null ? size : Math.min(end >= 0 ? end : size + end, size);
        s[d] = Math.max(0, stop - start);
      } else if (end !== null && end >= 0) s[d] = end - start;
      else s[d] = start ? `${size}-${start}` : size;
      return { out: { shape: s, dtype: x.dtype } };
    },
  },
  Unsqueeze: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const s = env.resolveShape(x.shape);
      const d = mod(p.dim, s.length + 1);
      return { out: { shape: [...s.slice(0, d), 1, ...s.slice(d)], dtype: x.dtype } };
    },
  },
  Chunk: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const s = halve(x, p.dim, env);
      return { a: { shape: [...s], dtype: x.dtype }, b: { shape: [...s], dtype: x.dtype } };
    },
  },
  Reduce: {
    infer: (p, ins, env) => {
      const x = need(ins, "in");
      const s = env.resolveShape(x.shape);
      if (s.length < 1) throw new ShapeError("cannot reduce a scalar", "in");
      const d = mod(p.dim, s.length);
      return { out: { shape: [...s.slice(0, d), ...(p.keepdim ? [1] : []), ...s.slice(d + 1)], dtype: x.dtype } };
    },
  },
  MatMul: {
    infer: (_p, ins, env) => {
      const a = need(ins, "a");
      const b = need(ins, "b");
      if (a.shape.length < 2 || b.shape.length < 2) throw new ShapeError("both operands need rank ≥ 2", "b");
      if (!env.unify(a.shape[a.shape.length - 1], b.shape[b.shape.length - 2]))
        throw new ShapeError(`inner dims differ: ${env.resolve(a.shape[a.shape.length - 1])} vs ${env.resolve(b.shape[b.shape.length - 2])}`, "b");
      const batch = broadcast({ shape: a.shape.slice(0, -2), dtype: a.dtype }, { shape: b.shape.slice(0, -2), dtype: a.dtype }, env).shape;
      return { out: { shape: [...batch, a.shape[a.shape.length - 2], b.shape[b.shape.length - 1]], dtype: a.dtype } };
    },
  },
  Scale: { infer: same },
  Identity: { infer: same },
  Reparameterize: {
    infer: (_p, ins, env) => {
      const mu = need(ins, "mu");
      const lv = need(ins, "logvar");
      env.unifyShapes(mu.shape, lv.shape, "logvar", "logvar vs mu ");
      return { out: { shape: [...mu.shape], dtype: mu.dtype } };
    },
  },
  BCEWithLogits: { infer: predTarget("pred") },
  L1: { infer: predTarget("pred") },
  Huber: { infer: predTarget("pred") },
  NLL: { infer: predTarget("log_probs", true) },
  KLDiv: {
    infer: (_p: P, ins: Ins, env: SymbolEnv) => {
      env.unifyShapes(need(ins, "target_probs").shape, need(ins, "log_probs").shape, "target_probs", "target vs input ");
      return scalarLoss();
    },
  },
  Contrastive: {
    infer: (_p, ins, env) => {
      const a = need(ins, "a");
      const b = need(ins, "b");
      rank(a, 2, "a", env, "B×D");
      env.unifyShapes(b.shape, a.shape, "b", "b vs a ");
      const n = ins.negatives;
      if (n) {
        rank(n, 2, "negatives", env, "N×D");
        if (!env.unify(n.shape[1], a.shape[1]))
          throw new ShapeError(`negatives have dim ${env.resolve(n.shape[1])}, a has ${env.resolve(a.shape[1])}`, "negatives");
      }
      return scalarLoss();
    },
  },
  Triplet: {
    infer: (_p, ins, env) => {
      const a = need(ins, "anchor");
      for (const port of ["positive", "negative"]) env.unifyShapes(need(ins, port).shape, a.shape, port, `${port} vs anchor `);
      return scalarLoss();
    },
  },
  KLGaussian: {
    infer: (_p, ins, env) => {
      env.unifyShapes(need(ins, "logvar").shape, need(ins, "mu").shape, "logvar", "logvar vs mu ");
      return scalarLoss();
    },
  },
};
