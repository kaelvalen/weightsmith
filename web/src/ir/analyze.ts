// Mirror of core/weightsmith_core/validate.py for instant feedback while editing.
// The daemon's analysis is authoritative (and the only one that can run Python
// nodes); on disagreement Python wins. cross.test.ts keeps the two in step.

import type { Adapt, Graph, Node } from "./types";
import { RULES, TYPES, expandDefaults, linearTargetsOf, outFeaturesOf, weightsOf } from "./registry";
import { ShapeError, SymbolEnv, TensorType, fmtShape } from "./shapes";
import { ArgError, instantiate } from "./args";

export interface Issue {
  code: string;
  message: string;
  node: string | null;
  port: string | null;
  edge: number | null;
  severity: "error" | "warning" | "peer";
}

export interface ParamCount {
  total: number;
  trainable: number;
  known: boolean;
}

export interface Signature {
  inputs: string[];
  optional: string[];
  outputs: string[];
}

export interface Analysis {
  ok: boolean;
  issues: Issue[];
  ports: Record<string, TensorType>;
  params: Record<string, ParamCount>;
  totals: ParamCount;
  order: string[];
  signatures: Record<string, Signature>;
  losses: string[];
  subgraphs: Record<string, Analysis>;
}

const ADAPT_DEFAULTS: Record<string, Partial<Adapt>> = {
  lora: { target: "weight", r: 8, alpha: 16, dropout: 0, freeze: true, init: "zeros_b" },
  adapter: { target: "out", bottleneck: 64, freeze: true },
  custom: { target: "weight", freeze: true },
};

export const adaptDefaults = (kind: string): Partial<Adapt> => ({ ...(ADAPT_DEFAULTS[kind] ?? {}) });

// integer parameters that may be negative (Python-style indexing from the end)
const SIGNED_PARAMS = new Set(["dim", "start_dim", "end_dim", "dim0", "dim1", "index", "start", "end", "ignore_index"]);

const KEYWORDS = new Set(
  "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(
    " ",
  ),
);

export const splitRef = (ref: string): [string, string] => {
  const i = ref.lastIndexOf(".");
  return [ref.slice(0, i), ref.slice(i + 1)];
};

/** Raised when a result depends on a Python node, which only the daemon can evaluate. */
class Deferred extends Error {}

const issue = (
  code: string,
  message: string,
  node: string | null = null,
  port: string | null = null,
  edge: number | null = null,
  severity: Issue["severity"] = "error",
): Issue => ({ code, message, node, port, edge, severity });

function totals(params: Record<string, ParamCount>): ParamCount {
  const t = { total: 0, trainable: 0, known: true };
  for (const c of Object.values(params)) {
    t.total += c.total;
    t.trainable += c.trainable;
    t.known = t.known && c.known;
  }
  return t;
}

function finish(a: Omit<Analysis, "ok" | "totals">): Analysis {
  return { ...a, ok: !a.issues.some((i) => i.severity === "error"), totals: totals(a.params) };
}

export function analyze(graph: Graph): Analysis {
  return body(graph.nodes, graph.edges as [string, string][], graph.subgraphs ?? {}, true);
}

function signature(n: Node, subgraphs: Graph["subgraphs"] & object): Signature {
  const p = (n.params ?? {}) as Record<string, any>;
  if (n.type === "Subgraph") {
    const b = subgraphs[p.graph] ?? { nodes: [] };
    return {
      inputs: b.nodes.filter((m) => m.type === "Input").map((m) => m.id),
      optional: [],
      outputs: b.nodes.filter((m) => m.type === "Output").map((m) => m.id),
    };
  }
  if (n.type === "PythonModule") {
    const s = p.signature ?? { inputs: ["in"], outputs: ["out"] };
    return { inputs: [...s.inputs], optional: [...(s.optional ?? [])], outputs: [...s.outputs] };
  }
  const t = TYPES[n.type];
  return { inputs: t.inputs.map((x) => x.name), optional: t.inputs.filter((x) => x.optional).map((x) => x.name), outputs: t.outputs.map((x) => x.name) };
}

function checkParams(n: Node, issues: Issue[]): boolean {
  const t = TYPES[n.type];
  const params = (n.params ?? {}) as Record<string, any>;
  let ok = true;
  for (const [k, p] of Object.entries(t.params)) {
    if (p.required && (params[k] === undefined || params[k] === null || params[k] === "")) {
      issues.push(issue("param_missing", `parameter '${k}' is required`, n.id));
      ok = false;
      continue;
    }
    const v = k in params ? params[k] : p.default;
    if (v === null || v === undefined) continue;
    const bad = (code: string, msg: string) => {
      issues.push(issue(code, msg, n.id));
      ok = false;
    };
    if (p.kind === "int" && !Number.isInteger(v)) bad("param_type", `'${k}' must be an integer`);
    else if (p.kind === "int" && !SIGNED_PARAMS.has(k) && v < 0) bad("param_value", `'${k}' must be ≥ 0`);
    else if (p.kind === "float" && typeof v !== "number") bad("param_type", `'${k}' must be a number`);
    else if (p.kind === "bool" && typeof v !== "boolean") bad("param_type", `'${k}' must be true/false`);
    else if (p.kind === "choice" && p.choices && !p.choices.includes(v)) bad("param_value", `'${k}' must be one of ${p.choices.join(", ")}`);
    else if (p.kind === "shape" && !Array.isArray(v)) bad("param_type", `'${k}' must be a shape list`);
  }
  for (const k of Object.keys(params))
    if (!(k in t.params)) issues.push(issue("param_unknown", `unknown parameter '${k}'`, n.id, null, null, "warning"));
  return ok;
}

function checkAdapt(n: Node, p: Record<string, any>, issues: Issue[]) {
  const targets = linearTargetsOf(n.type, p);
  (n.adapt ?? []).forEach((ad, i) => {
    const tgt = ad.target ?? adaptDefaults(ad.kind).target!;
    if (ad.kind === "lora" && !(tgt in targets)) {
      const have = Object.keys(targets).join(", ") || "none";
      issues.push(issue("adapt_target", `adapt[${i}]: LoRA target '${tgt}' not found (targets: ${have})`, n.id));
    }
    if (ad.kind === "adapter" && outFeaturesOf(n.type, p) === null)
      issues.push(issue("adapt_target", `adapt[${i}]: ${n.type} has no feature dimension for an adapter`, n.id));
    if (ad.kind === "custom" && !ad.code)
      issues.push(issue("adapt_code", `adapt[${i}]: custom adapter needs code with a wrap(module) method`, n.id));
  });
}

export function countParams(n: Node, p: Record<string, any>, sub: ParamCount | null): ParamCount {
  if (n.type === "Subgraph") {
    const mult = p.tie_weights ? 1 : p.repeat ?? 1;
    const c = { total: sub!.total * mult, trainable: sub!.trainable * mult, known: sub!.known };
    if (n.trainable === false) c.trainable = 0;
    return c;
  }
  if (n.type === "PythonModule") {
    const count = p.signature?.params;
    if (count === undefined || count === null) return { total: 0, trainable: 0, known: false };
    return { total: count, trainable: n.trainable === false ? 0 : count, known: true };
  }
  const weights = weightsOf(n.type, p);
  const names = Object.keys(weights);
  const frozen = new Set<string>();
  if (n.trainable === false) names.forEach((w) => frozen.add(w));
  for (const mask of n.frozen_params ?? []) names.filter((w) => w === mask || w.startsWith(mask + ".")).forEach((w) => frozen.add(w));
  let extra = 0;
  let known = true;
  const targets = linearTargetsOf(n.type, p);
  for (const raw of n.adapt ?? []) {
    const ad = { ...adaptDefaults(raw.kind), ...raw };
    if (ad.disabled) continue;
    const tgt = ad.target ?? "weight";
    let count: number;
    let prefix = "";
    if (ad.kind === "lora") {
      if (!(tgt in targets)) continue;
      const [fin, fout] = targets[tgt];
      count = ad.r! * (fin + fout);
      prefix = tgt === "weight" ? "" : tgt + ".";
    } else if (ad.kind === "adapter") {
      const o = outFeaturesOf(n.type, p);
      if (o === null) continue;
      const b = ad.bottleneck!;
      count = o * b + b + b * o + o;
    } else {
      known = false;
      continue;
    }
    extra += count;
    if (ad.freeze ?? true) names.filter((w) => !prefix || w.startsWith(prefix)).forEach((w) => frozen.add(w));
  }
  const total = names.reduce((s, k) => s + weights[k], 0) + extra;
  const trainable = names.filter((k) => !frozen.has(k)).reduce((s, k) => s + weights[k], 0) + extra;
  return { total, trainable, known };
}

function body(
  nodes: Node[],
  edges: [string, string][],
  subgraphs: NonNullable<Graph["subgraphs"]>,
  top: boolean,
  bound?: Record<string, TensorType>,
): Analysis {
  const issues: Issue[] = [];
  const ports: Record<string, TensorType> = {};
  const params: Record<string, ParamCount> = {};
  const order: string[] = [];
  const signatures: Record<string, Signature> = {};
  const losses: string[] = [];
  const subs: Record<string, Analysis> = {};
  const env = new SymbolEnv();

  const byId = new Map<string, Node>();
  for (const n of nodes) {
    if (byId.has(n.id)) {
      issues.push(issue("duplicate_id", `duplicate node id '${n.id}'`, n.id));
      continue;
    }
    byId.set(n.id, n);
  }
  const valid = new Map<string, boolean>();
  for (const [id, n] of byId) {
    if (!TYPES[n.type]) {
      issues.push(issue("unknown_type", `unknown node type '${n.type}'`, id));
      continue;
    }
    let ok = checkParams(n, issues);
    const p = (n.params ?? {}) as Record<string, any>;
    if (n.type === "Subgraph" && !(p.graph in subgraphs)) {
      issues.push(issue("subgraph_missing", `subgraph '${p.graph}' is not defined`, id));
      ok = false;
    }
    if (n.type === "Subgraph" && !top) {
      issues.push(issue("subgraph_nested", "nested subgraphs are not supported in schema 1", id));
      ok = false;
    }
    valid.set(id, ok);
    signatures[id] = signature(n, subgraphs);
    if (ok) checkAdapt(n, expandDefaults(n.type, p), issues);
  }

  if (top) {
    const fields = new Map<string, string>();
    for (const [id, n] of byId) {
      if (n.type !== "Input") continue;
      const f = expandDefaults("Input", n.params as any).field ?? "x";
      if (typeof f !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(f) || KEYWORDS.has(f))
        issues.push(issue("field", `field '${f}' must be a Python identifier`, id));
      else if (fields.has(f)) issues.push(issue("field", `field '${f}' is already bound by input '${fields.get(f)}'`, id));
      else if (byId.has(f) && f !== id) issues.push(issue("field", `field '${f}' collides with node id '${f}'`, id));
      fields.set(f, id);
    }
  }

  for (const [id, n] of byId) {
    const tie = n.type === "Linear" ? (n.params as any)?.tie_to : null;
    if (!tie) continue;
    const owner = byId.get(tie);
    if (!owner || owner.type !== "Embedding") {
      issues.push(issue("tie", `tie_to '${tie}' must name an Embedding node in the same graph`, id));
      valid.set(id, false);
      continue;
    }
    const lp = (n.params ?? {}) as any;
    const ep = (owner.params ?? {}) as any;
    if (lp.in !== ep.dim || lp.out !== ep.num) {
      issues.push(issue("tie", `tied weight shape: Linear ${lp.in}→${lp.out} needs Embedding dim=${lp.in}, num=${lp.out}`, id));
      valid.set(id, false);
    } else if ((n.trainable === false) !== (owner.trainable === false))
      issues.push(issue("tie", `'${id}' and '${tie}' share a weight but differ in trainability; freezing one freezes both`, id, null, null, "warning"));
  }

  const incoming = new Map<string, [string, number]>();
  const succ = new Map<string, Set<string>>();
  edges.forEach(([src, dst], i) => {
    const [sn, sp] = splitRef(src);
    const [dn, dp] = splitRef(dst);
    if (!signatures[sn] || !signatures[dn]) {
      const missing = !byId.has(sn) ? sn : dn;
      issues.push(issue("edge_node", `edge ${src} → ${dst}: node '${missing}' does not exist`, null, null, i));
      return;
    }
    if (!signatures[sn].outputs.includes(sp)) return void issues.push(issue("edge_port", `'${sn}' has no output port '${sp}'`, sn, sp, i));
    if (!signatures[dn].inputs.includes(dp)) return void issues.push(issue("edge_port", `'${dn}' has no input port '${dp}'`, dn, dp, i));
    if (sn === dn) return void issues.push(issue("cycle", "node is connected to itself", sn, null, i));
    if (incoming.has(dst)) return void issues.push(issue("multi_input", `input '${dst}' already has a connection`, dn, dp, i));
    incoming.set(dst, [src, i]);
    if (!succ.has(sn)) succ.set(sn, new Set());
    succ.get(sn)!.add(dn);
  });

  const indeg = new Map<string, number>(Object.keys(signatures).map((id) => [id, 0]));
  for (const ds of succ.values()) for (const d of ds) indeg.set(d, indeg.get(d)! + 1);
  const queue = [...indeg].filter(([, d]) => d === 0).map(([id]) => id).sort();
  while (queue.length) {
    const id = queue.shift()!;
    order.push(id);
    for (const d of [...(succ.get(id) ?? [])].sort()) {
      indeg.set(d, indeg.get(d)! - 1);
      if (indeg.get(d) === 0) queue.push(d);
    }
  }
  for (const id of Object.keys(signatures).filter((id) => !order.includes(id)).sort())
    issues.push(issue("cycle", "node is part of a cycle", id));

  if (top)
    for (const [name, b] of Object.entries(subgraphs)) {
      try {
        const inst = instantiate(b as any);
        subs[name] = body(inst.nodes, inst.edges as [string, string][], {}, false);
      } catch (e) {
        if (!(e instanceof ArgError)) throw e;
        subs[name] = finish({ issues: [issue("block_param", e.message)], ports: {}, params: {}, order: [], signatures: {}, losses: [], subgraphs: {} });
      }
    }

  for (const id of order) {
    const n = byId.get(id)!;
    if (!valid.get(id)) continue;
    const p = expandDefaults(n.type, (n.params ?? {}) as any);
    const sig = signatures[id];
    const ins: Record<string, TensorType | null> = {};
    let blocked = false;
    for (const port of sig.inputs) {
      const ref = `${id}.${port}`;
      const inc = incoming.get(ref);
      if (inc) {
        if (ports[inc[0]]) {
          ins[port] = ports[inc[0]];
          ports[ref] = ports[inc[0]];
        } else blocked = true;
      } else if (sig.optional.includes(port)) ins[port] = null;
      else {
        issues.push(issue("unconnected", `input '${port}' is not connected`, id, port));
        blocked = true;
      }
    }
    if (blocked) continue;

    let sub: ParamCount | null = null;
    let outs: Record<string, TensorType>;
    try {
      if (n.type === "Input" && bound && bound[id]) {
        env.unifyShapes([...p.shape], bound[id].shape, "out", "subgraph input ");
        outs = { out: { shape: env.resolveShape(bound[id].shape), dtype: bound[id].dtype } };
      } else if (n.type === "Subgraph") {
        [outs, sub] = inferSubgraph(p, ins, subgraphs);
      } else if (n.type === "PythonModule") {
        outs = {}; // shapes come from the daemon, which runs the user's infer_shape
      } else outs = RULES[n.type].infer!(p, ins, env);
    } catch (e) {
      if (e instanceof Deferred) continue;
      if (!(e instanceof ShapeError)) throw e;
      const edge = e.port ? (incoming.get(`${id}.${e.port}`)?.[1] ?? null) : null;
      issues.push(issue("shape", e.message, id, e.port, edge));
      if (e.port && edge !== null) {
        const [sn, sp] = splitRef(edges[edge][0]);
        issues.push(issue("shape", e.message, sn, sp, edge, "peer"));
      }
      continue;
    }
    if (n.disabled) {
      const d = disabledOuts(n, ins, outs, sig, env, issues);
      if (d === null) continue;
      outs = d;
    }
    for (const [port, t] of Object.entries(outs)) ports[`${id}.${port}`] = { shape: env.resolveShape(t.shape), dtype: t.dtype };
    if (TYPES[n.type].loss && !n.disabled) losses.push(id);
    params[id] = countParams(n, p, sub);
  }
  for (const k of Object.keys(ports)) ports[k] = { shape: env.resolveShape(ports[k].shape), dtype: ports[k].dtype };

  if (top) {
    if (!losses.length && !issues.some((i) => i.code === "cycle"))
      issues.push(issue("no_loss", "graph has no active loss node; it can be compiled but not trained", null, null, null, "warning"));
    const used = new Set([...incoming.values()].map(([s]) => s));
    for (const id of order) {
      const t = TYPES[byId.get(id)!.type];
      if (!t || t.loss || t.type === "Output") continue;
      for (const port of signatures[id].outputs)
        if (!used.has(`${id}.${port}`)) issues.push(issue("unused_output", `output '${port}' is not used`, id, port, null, "warning"));
    }
  }
  return finish({ issues, ports, params, order, signatures, losses, subgraphs: subs });
}

function disabledOuts(
  n: Node,
  ins: Record<string, TensorType | null>,
  outs: Record<string, TensorType>,
  sig: Signature,
  env: SymbolEnv,
  issues: Issue[],
): Record<string, TensorType> | null {
  if (TYPES[n.type].loss) return outs;
  if (n.type === "Input") {
    issues.push(issue("disable", "inputs cannot be disabled", n.id));
    return null;
  }
  if (!sig.inputs.length || !sig.outputs.length) return {};
  const x = ins[sig.inputs[0]];
  if (!x) return null;
  const o = outs[sig.outputs[0]];
  if (o) {
    const a = env.resolveShape(o.shape);
    const b = env.resolveShape(x.shape);
    if (a.length !== b.length || a.some((d, i) => d !== b[i]) || o.dtype !== x.dtype) {
      issues.push(issue("disable", `cannot disable: output ${fmtShape(a)} differs from input ${fmtShape(b)}`, n.id));
      return null;
    }
  }
  return { [sig.outputs[0]]: x };
}

function inferSubgraph(
  p: Record<string, any>,
  ins: Record<string, TensorType | null>,
  subgraphs: NonNullable<Graph["subgraphs"]>,
): [Record<string, TensorType>, ParamCount] {
  let b: any;
  try {
    b = instantiate(subgraphs[p.graph] as any, p.args);
  } catch (e) {
    if (e instanceof ArgError) throw new ShapeError(`${p.graph}: ${e.message}`);
    throw e;
  }
  const inIds: string[] = b.nodes.filter((m: Node) => m.type === "Input").map((m: Node) => m.id);
  const outIds: string[] = b.nodes.filter((m: Node) => m.type === "Output").map((m: Node) => m.id);
  const repeat = p.repeat ?? 1;
  if (repeat > 1 && outIds.length > inIds.length)
    throw new ShapeError(`repeat=${repeat} needs at least as many inputs as outputs (${inIds.length} vs ${outIds.length})`);
  // with repeat, the first outputs.length inputs are carried; the rest are side inputs fed to every repeat
  const carried = inIds.slice(0, outIds.length);
  let current: Record<string, TensorType> = Object.fromEntries(Object.entries(ins).filter(([, v]) => v !== null)) as Record<string, TensorType>;
  let outs: Record<string, TensorType> = {};
  let counts: ParamCount = { total: 0, trainable: 0, known: true };
  for (let r = 0; r < Math.min(repeat, 2); r++) {
    const sub = body(b.nodes, b.edges as [string, string][], {}, false, current);
    const errs = sub.issues.filter((i) => i.severity === "error");
    if (errs.length) throw new ShapeError(`in ${p.graph}/${errs[0].node}: ${errs[0].message}`);
    outs = {};
    for (const oid of outIds) {
      const src = (b.edges as [string, string][]).find((e) => e[1] === `${oid}.in`)?.[0];
      if (src && !sub.ports[src] && b.nodes.some((m: Node) => m.type === "PythonModule")) throw new Deferred();
      if (!src || !sub.ports[src]) throw new ShapeError(`${p.graph}: output '${oid}' is not connected`);
      outs[oid] = sub.ports[src];
    }
    counts = sub.totals;
    if (repeat > 1) {
      const next: Record<string, TensorType> = { ...current };
      carried.forEach((iid, k) => {
        const o = outs[outIds[k]];
        if (JSON.stringify(o.shape) !== JSON.stringify(current[iid].shape))
          throw new ShapeError(`repeat=${repeat}: block output ${fmtShape(o.shape)} must equal its input ${fmtShape(current[iid].shape)}`);
        next[iid] = o;
      });
      current = next;
    }
  }
  return [outs, counts];
}
