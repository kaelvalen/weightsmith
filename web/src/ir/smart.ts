// Editing that understands the graph: chain a new node after the selection, size it from
// the tensor that flows into it, splice into an edge, keep chains intact on delete, and
// turn common issues into one-click fixes. Everything is a pure Graph → Graph function,
// so it goes through the same operation log (and undo) as any other edit.

import type { Analysis, Issue } from "./analyze";
import { analyze } from "./analyze";
import { addNode, getBody, withBody } from "./ops";
import { TYPES, expandDefaults } from "./registry";
import type { Dim, TensorType } from "./shapes";
import type { Graph, Node, Position } from "./types";

type Where = "last" | "channel";
/** Which parameter of a node is "the size of what comes in", per input port. */
const SIZE_PARAMS: Record<string, Record<string, [string, Where][]>> = {
  Linear: { in: [["in", "last"]] },
  MLP: { in: [["dim", "last"]] },
  GatedMLP: { in: [["dim", "last"]] },
  MoE: { in: [["dim", "last"]] },
  LayerNorm: { in: [["dim", "last"]] },
  RMSNorm: { in: [["dim", "last"]] },
  Attention: { in: [["dim", "last"]] },
  CrossAttention: { in: [["dim", "last"]], context: [["context_dim", "last"]] },
  LinearAttention: { in: [["dim", "last"]] },
  PositionalEmbedding: { in: [["dim", "last"]] },
  SinusoidalPositional: { in: [["dim", "last"]] },
  ClassToken: { in: [["dim", "last"]] },
  LayerScale: { in: [["dim", "last"]] },
  Swish: { in: [["dim", "last"]] },
  VectorQuantizer: { in: [["dim", "last"]] },
  Mamba: { in: [["dim", "last"]] },
  LSTM: { in: [["input", "last"]] },
  GRU: { in: [["input", "last"]] },
  RNN: { in: [["input", "last"]] },
  GraphConv: { x: [["in", "last"]] },
  Conv1d: { in: [["in_channels", "channel"]] },
  Conv2d: { in: [["in_channels", "channel"]] },
  Conv3d: { in: [["in_channels", "channel"]] },
  ConvTranspose2d: { in: [["in_channels", "channel"]] },
  PatchEmbed: { in: [["in_channels", "channel"]] },
  BatchNorm1d: { in: [["features", "channel"]] },
  BatchNorm2d: { in: [["features", "channel"]] },
  BatchNorm3d: { in: [["features", "channel"]] },
  GroupNorm: { in: [["channels", "channel"]] },
  InstanceNorm2d: { in: [["features", "channel"]] },
  SqueezeExcite: { in: [["channels", "channel"]] },
  PReLU: { in: [["channels", "channel"]] },
};

/** Parameters that should follow the input size when a node is first created (keeps a chain consistent). */
const FOLLOW_ON_CREATE: Record<string, string[]> = { Linear: ["out"], Conv1d: ["out_channels"], Conv2d: ["out_channels"], Conv3d: ["out_channels"] };

function sizeOf(t: TensorType | undefined, where: Where): number | null {
  if (!t) return null;
  const d: Dim | undefined = where === "last" ? t.shape[t.shape.length - 1] : t.shape[1];
  return typeof d === "number" ? d : null;
}

function scopeAnalysis(g: Graph, scope: string): Analysis {
  const a = analyze(g);
  return scope ? (a.subgraphs[scope] ?? a) : a;
}

function largestDivisor(n: number, atMost: number): number {
  for (let k = Math.min(atMost, n); k >= 1; k--) if (n % k === 0) return k;
  return 1;
}

/**
 * Set a node's input-size parameters from the tensors connected to it.
 * Returns the graph and a human description of what changed (empty if nothing).
 */
export function autoSize(g: Graph, scope: string, id: string, fresh = false): [Graph, string[]] {
  const n = getBody(g, scope).nodes.find((x) => x.id === id);
  const spec = n && SIZE_PARAMS[n.type];
  if (!n || !spec) return [g, []];
  const a = scopeAnalysis(g, scope);
  const p = expandDefaults(n.type, (n.params ?? {}) as Record<string, unknown>) as Record<string, any>;
  const changes: [string, unknown][] = [];
  for (const [port, params] of Object.entries(spec)) {
    const t = a.ports[`${id}.${port}`];
    for (const [param, where] of params) {
      const size = sizeOf(t, where);
      if (size !== null && p[param] !== size) changes.push([param, size]);
    }
  }
  if (fresh) {
    const base = changes.find(([k]) => k === spec.in?.[0]?.[0] || k === spec.x?.[0]?.[0])?.[1];
    for (const k of FOLLOW_ON_CREATE[n.type] ?? []) if (typeof base === "number" && p[k] !== base) changes.push([k, base]);
  }
  if (!changes.length) return [g, []];
  const notes = changes.map(([k, v]) => `${k} = ${v}`);
  const next = withBody(g, scope, (b) => {
    const m = b.nodes.find((x) => x.id === id)!;
    const params = { ...(m.params ?? {}) } as Record<string, any>;
    for (const [k, v] of changes) params[k] = v;
    // keep dependent parameters valid
    if (m.type === "Attention" || m.type === "CrossAttention" || m.type === "LinearAttention") {
      const heads = params.heads ?? p.heads ?? 8;
      if (params.dim % heads) params.heads = largestDivisor(params.dim, heads);
      const kv = params.kv_heads;
      if (kv && params.heads % kv) params.kv_heads = largestDivisor(params.heads, kv);
    }
    if (m.type === "GroupNorm" && params.channels % (params.groups ?? 32)) params.groups = largestDivisor(params.channels, params.groups ?? 32);
    m.params = params;
  });
  return [next, notes];
}

function firstOutput(g: Graph, scope: string, id: string): string | null {
  const a = scopeAnalysis(g, scope);
  return a.signatures[id]?.outputs[0] ?? null;
}

function firstInput(type: string): string | null {
  const t = TYPES[type];
  const p = t?.inputs.find((x) => !x.optional) ?? t?.inputs[0];
  return p?.name ?? null;
}

/** Add `type` below `after` (its first output → the new node's first input), sized from it. */
export function appendAfter(g: Graph, scope: string, type: string, after: string, at: Position): [Graph, string, string[]] {
  const out = firstOutput(g, scope, after);
  const inp = firstInput(type);
  let id = "";
  let next = g;
  [next, id] = addNode(next, scope, type, at);
  if (!out || !inp) return [next, id, []];
  next = withBody(next, scope, (b) => {
    b.edges.push([`${after}.${out}`, `${id}.${inp}`]);
  });
  const [sized, notes] = autoSize(next, scope, id, true);
  return [sized, id, notes];
}

/** Put `type` into an edge: src → new → dst. */
export function spliceInto(g: Graph, scope: string, type: string, edge: [string, string], at: Position): [Graph, string, string[]] {
  const inp = firstInput(type);
  let id = "";
  let next = g;
  [next, id] = addNode(next, scope, type, at);
  if (!inp) return [next, id, []];
  next = withBody(next, scope, (b) => {
    b.edges = b.edges.filter((e) => !(e[0] === edge[0] && e[1] === edge[1]));
    b.edges.push([edge[0], `${id}.${inp}`]);
  });
  const out = firstOutput(next, scope, id);
  if (out) next = withBody(next, scope, (b) => void b.edges.push([`${id}.${out}`, edge[1]]));
  const [sized, notes] = autoSize(next, scope, id, true);
  return [sized, id, notes];
}

/**
 * Delete nodes but keep chains whole: a removed node with one incoming edge hands its
 * consumers to whatever fed it (resolving runs of removed nodes).
 */
export function dissolve(g: Graph, scope: string, ids: string[]): Graph {
  const gone = new Set(ids);
  const body = getBody(g, scope);
  const nodeOf = (ref: string) => ref.slice(0, ref.lastIndexOf("."));
  const feeder = (id: string, seen = new Set<string>()): string | null => {
    const inc = body.edges.filter(([, d]) => nodeOf(d) === id);
    if (inc.length !== 1 || seen.has(id)) return null;
    const src = inc[0][0];
    if (!gone.has(nodeOf(src))) return src;
    seen.add(id);
    return feeder(nodeOf(src), seen);
  };
  return withBody(g, scope, (b) => {
    const added: [string, string][] = [];
    for (const [s, d] of b.edges) {
      if (gone.has(nodeOf(s)) && !gone.has(nodeOf(d))) {
        const src = feeder(nodeOf(s));
        if (src) added.push([src, d]);
      }
    }
    b.nodes = b.nodes.filter((n) => !gone.has(n.id));
    b.edges = b.edges.filter(([s, d]) => !gone.has(nodeOf(s)) && !gone.has(nodeOf(d))).concat(added);
    if (b.layout) for (const id of ids) delete b.layout[id];
  });
}

/** Re-size every node fed by `id` after it changed (used after connecting). */
export function sizeConsumers(g: Graph, scope: string, src: string): [Graph, string[]] {
  const body = getBody(g, scope);
  let next = g;
  const notes: string[] = [];
  for (const [s, d] of body.edges) {
    if (s.slice(0, s.lastIndexOf(".")) !== src) continue;
    const dst = d.slice(0, d.lastIndexOf("."));
    const a = scopeAnalysis(next, scope);
    const shapeErr = a.issues.some((i) => i.node === dst && i.code === "shape");
    if (!shapeErr) continue;
    const [g2, ns] = autoSize(next, scope, dst);
    next = g2;
    notes.push(...ns.map((x) => `${dst}.${x}`));
  }
  return [next, notes];
}

/** Re-size nodes along the graph until shape errors that sizing can fix are gone. */
export function healShapes(g: Graph, scope: string): [Graph, string[]] {
  let next = g;
  const notes: string[] = [];
  const tried = new Set<string>();
  for (let i = 0; i < 64; i++) {
    const a = scopeAnalysis(next, scope);
    const body = getBody(next, scope);
    const issue = a.issues.find((x) => x.code === "shape" && x.node && !tried.has(x.node) && SIZE_PARAMS[body.nodes.find((n) => n.id === x.node)?.type ?? ""]);
    if (!issue) break;
    tried.add(issue.node!);
    const [g2, ns] = autoSize(next, scope, issue.node!);
    next = g2;
    notes.push(...ns.map((x) => `${issue.node}.${x}`));
  }
  return [next, notes];
}

/** Give an Input a new per-sample shape (batch dim kept) and re-size what follows it. */
export function reshapeInput(g: Graph, id: string, perSample: (number | string)[]): Graph {
  const set = withBody(g, "", (b) => {
    const n = b.nodes.find((x) => x.id === id);
    if (!n) return;
    const shape = ((n.params as any)?.shape ?? ["B"]) as (number | string)[];
    n.params = { ...(n.params ?? {}), shape: [shape[0] ?? "B", ...perSample] };
  });
  return healShapes(set, "")[0];
}

export interface Fix {
  label: string;
  apply: (g: Graph) => Graph;
}

/** One-click fixes for an issue, when there is an obvious one. */
export function fixesFor(issue: Issue, g: Graph, scope: string): Fix[] {
  const body = getBody(g, scope);
  const n = issue.node ? body.nodes.find((x) => x.id === issue.node) : undefined;
  const fixes: Fix[] = [];
  if ((issue.code === "shape" || issue.code === "param_missing") && n && SIZE_PARAMS[n.type]) {
    const [, notes] = autoSize(g, scope, n.id);
    if (notes.length) fixes.push({ label: `Set ${notes.join(", ")}`, apply: (gr) => autoSize(gr, scope, n.id)[0] });
  }
  if (issue.code === "unconnected" && n && issue.port) {
    const pos = body.layout?.[n.id];
    const a = scopeAnalysis(g, scope);
    const candidates = body.nodes
      .filter((m) => m.id !== n.id && (a.signatures[m.id]?.outputs.length ?? 0) > 0 && !TYPES[m.type]?.loss)
      .map((m) => ({ m, p: body.layout?.[m.id] }))
      .filter(({ p }) => p && pos && p.y < pos.y)
      .sort((x, y) => y.p!.y - x.p!.y || Math.abs(x.p!.x - pos!.x) - Math.abs(y.p!.x - pos!.x));
    const c = candidates[0]?.m;
    if (c) {
      const out = a.signatures[c.id].outputs[0];
      fixes.push({
        label: `Connect from ${c.label || c.id}`,
        apply: (gr) => {
          const wired = withBody(gr, scope, (b) => void b.edges.push([`${c.id}.${out}`, `${n.id}.${issue.port}`]));
          return autoSize(wired, scope, n.id)[0];
        },
      });
    }
  }
  if (issue.code === "no_loss") {
    const a = scopeAnalysis(g, scope);
    const consumed = new Set(body.edges.map(([s]) => s));
    const tail = [...a.order].reverse().find((id) => {
      const out = a.signatures[id]?.outputs[0];
      const t = out ? a.ports[`${id}.${out}`] : undefined;
      return t && !consumed.has(`${id}.${out}`) && t.dtype.startsWith("f") && t.shape.length >= 2;
    });
    if (tail) {
      const at = body.layout?.[tail] ?? { x: 0, y: 0 };
      fixes.push({ label: `Add CrossEntropy after ${tail}`, apply: (gr) => appendAfter(gr, scope, "CrossEntropy", tail, { x: at.x, y: at.y + 104 })[0] });
      fixes.push({ label: `Add MSE after ${tail}`, apply: (gr) => appendAfter(gr, scope, "MSE", tail, { x: at.x, y: at.y + 104 })[0] });
    }
  }
  return fixes;
}

export function below(g: Graph, scope: string, id: string): Position {
  const p = getBody(g, scope).layout?.[id] ?? { x: 0, y: 0 };
  return { x: p.x, y: p.y + 104 };
}

export type { Node };
