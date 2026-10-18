// Pure IR operations. The canvas never edits its own state: every gesture is
// one of these functions applied to the graph, recorded in the operation log.

import type { Adapt, Graph, Node, Position } from "./types";
import { TYPES, expandDefaults, linearTargetsOf, outFeaturesOf } from "./registry";

export interface Body {
  nodes: Node[];
  edges: [string, string][];
  layout?: Record<string, Position>;
}

const clone = <T>(x: T): T => structuredClone(x);

/** The node list a scope edits: "" is the top level, otherwise a subgraph name. */
export function getBody(g: Graph, scope: string): Body {
  if (!scope) return g as unknown as Body;
  return g.subgraphs![scope] as unknown as Body;
}

export function withBody(g: Graph, scope: string, fn: (b: Body) => void): Graph {
  const next = clone(g);
  fn(getBody(next, scope));
  return next;
}

const SNAP = 8;
export const snap = (v: number) => Math.round(v / SNAP) * SNAP;

const NODE_BOX = { w: 184, h: 88 };

/** The nearest grid position at or right of `at` where a node does not cover another. */
export function freeSpot(layout: Record<string, Position> | undefined, at: Position, ignore: Set<string> = new Set()): Position {
  const taken = Object.entries(layout ?? {})
    .filter(([id]) => !ignore.has(id))
    .map(([, p]) => p);
  const hits = (p: Position) => taken.some((q) => Math.abs(q.x - p.x) < NODE_BOX.w + 16 && Math.abs(q.y - p.y) < NODE_BOX.h);
  let p = { x: snap(at.x), y: snap(at.y) };
  for (let i = 0; i < 200 && hits(p); i++) p = { x: p.x + 216, y: p.y };
  return p;
}

export function newId(b: Body, type: string): string {
  const base =
    { Linear: "fc", Attention: "attn", LayerNorm: "ln", RMSNorm: "norm", Embedding: "emb", CrossEntropy: "loss", MSE: "loss", Input: "in", Output: "out", Subgraph: "block", PythonModule: "py", Conv2d: "conv", Dropout: "drop", PositionalEmbedding: "pos" }[type] ??
    type.toLowerCase();
  const ids = new Set(b.nodes.map((n) => n.id));
  for (let i = 1; ; i++) {
    const id = `${base}${i}`;
    if (!ids.has(id)) return id;
  }
}

function starterParams(type: string): Record<string, unknown> {
  const t = TYPES[type];
  const p: Record<string, unknown> = {};
  for (const [k, spec] of Object.entries(t.params)) if (spec.required) p[k] = spec.default ?? (spec.kind === "int" ? 64 : "");
  if (type === "PythonModule") {
    p.code = PYTHON_TEMPLATE;
    p.class_name = "Block";
    p.init = { dim: 64 };
  }
  if (type === "Input") p.shape = ["B", 64];
  return p;
}

export const PYTHON_TEMPLATE = `class Block(nn.Module):
    """Your module. forward() parameters become input ports."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.proj(x)

    @staticmethod
    def infer_shape(inputs):
        # symbolic shapes in, symbolic shapes out: {"x": ["B", "T", 64]}
        return {"out": inputs["x"]}
`;

export function addNode(g: Graph, scope: string, type: string, at: Position, params?: Record<string, unknown>): [Graph, string] {
  let id = "";
  const next = withBody(g, scope, (b) => {
    id = newId(b, type);
    b.nodes.push({ id, type, params: params ?? starterParams(type) });
    b.layout = { ...(b.layout ?? {}), [id]: freeSpot(b.layout, at) };
  });
  return [next, id];
}

export function removeNodes(g: Graph, scope: string, ids: string[]): Graph {
  const gone = new Set(ids);
  return withBody(g, scope, (b) => {
    b.nodes = b.nodes.filter((n) => !gone.has(n.id));
    b.edges = b.edges.filter(([s, d]) => !gone.has(s.split(".")[0]) && !gone.has(d.split(".")[0]));
    if (b.layout) for (const id of ids) delete b.layout[id];
  });
}

export function connect(g: Graph, scope: string, src: string, dst: string): Graph {
  return withBody(g, scope, (b) => {
    b.edges = b.edges.filter(([, d]) => d !== dst); // an input takes one edge; the new one replaces it
    b.edges.push([src, dst]);
  });
}

export function disconnect(g: Graph, scope: string, edges: [string, string][]): Graph {
  const key = (e: [string, string]) => e.join("→");
  const gone = new Set(edges.map(key));
  return withBody(g, scope, (b) => {
    b.edges = b.edges.filter((e) => !gone.has(key(e)));
  });
}

export function updateNode(g: Graph, scope: string, id: string, fn: (n: Node) => void): Graph {
  return withBody(g, scope, (b) => {
    const n = b.nodes.find((x) => x.id === id);
    if (n) fn(n);
  });
}

export function setParam(g: Graph, scope: string, id: string, key: string, value: unknown): Graph {
  return updateNode(g, scope, id, (n) => {
    n.params = { ...(n.params ?? {}) };
    if (value === undefined) delete n.params[key];
    else n.params[key] = value;
  });
}

export function renameNode(g: Graph, scope: string, id: string, next: string): Graph {
  if (!/^[A-Za-z_][A-Za-z0-9_-]*$/.test(next) || id === next) return g;
  return withBody(g, scope, (b) => {
    if (b.nodes.some((n) => n.id === next)) return;
    const n = b.nodes.find((x) => x.id === id);
    if (!n) return;
    n.id = next;
    const re = (r: string) => (r.split(".")[0] === id ? `${next}.${r.split(".")[1]}` : r);
    b.edges = b.edges.map(([s, d]) => [re(s), re(d)]);
    if (b.layout?.[id]) {
      b.layout[next] = b.layout[id];
      delete b.layout[id];
    }
  });
}

export function moveNodes(g: Graph, scope: string, pos: Record<string, Position>): Graph {
  return withBody(g, scope, (b) => {
    b.layout = { ...(b.layout ?? {}) };
    for (const [id, p] of Object.entries(pos)) b.layout[id] = { x: snap(p.x), y: snap(p.y) };
  });
}

/** Bulk selection, mirroring core/weightsmith_core/select.py. */
export function matchNodes(g: Graph, types: string[] | null, pattern: string): [string, string][] {
  const re = new RegExp("^" + pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".") + "$");
  const bodies: [string, Node[]][] = [["", g.nodes], ...Object.entries(g.subgraphs ?? {}).map(([k, v]) => [k, v.nodes] as [string, Node[]])];
  const out: [string, string][] = [];
  for (const [scope, nodes] of bodies)
    for (const n of nodes)
      if ((!types || types.includes(n.type)) && (re.test(n.id) || re.test(n.label ?? n.id))) out.push([scope, n.id]);
  return out;
}

export function bulkAdapt(g: Graph, adapt: Adapt, targets: string[], types: string[] | null, pattern = "*"): [Graph, string[]] {
  const next = clone(g);
  const hits = new Set(matchNodes(next, types, pattern).map(([s, id]) => `${s}/${id}`));
  const touched: string[] = [];
  const bodies: [string, Node[]][] = [["", next.nodes], ...Object.entries(next.subgraphs ?? {}).map(([k, v]) => [k, v.nodes] as [string, Node[]])];
  for (const [scope, nodes] of bodies)
    for (const n of nodes) {
      if (!hits.has(`${scope}/${n.id}`) || !TYPES[n.type]) continue;
      const p = expandDefaults(n.type, n.params as any);
      const available = adapt.kind === "lora" ? Object.keys(linearTargetsOf(n.type, p)) : outFeaturesOf(n.type, p) !== null ? ["out"] : [];
      for (const tgt of targets) {
        if (!available.includes(tgt)) continue;
        n.adapt = n.adapt ?? [];
        if (n.adapt.some((a) => a.kind === adapt.kind && a.target === tgt)) continue;
        n.adapt.push({ ...adapt, target: tgt });
        touched.push(scope ? `${scope}/${n.id}:${tgt}` : `${n.id}:${tgt}`);
      }
    }
  return [next, touched];
}

/**
 * Layered top-to-bottom layout for graphs without coordinates: longest-path layers, then
 * nodes ordered inside each layer by the mean column of their parents (fewer crossings),
 * so a residual stream stays in one column and side branches step to the right.
 */
export function autoLayout(b: Body, onlyMissing = true): Record<string, Position> {
  const ids = b.nodes.map((n) => n.id);
  const preds = new Map<string, string[]>(ids.map((id) => [id, []]));
  for (const [s, d] of b.edges) {
    const sn = s.split(".")[0];
    const dn = d.split(".")[0];
    if (preds.has(dn) && preds.has(sn)) preds.get(dn)!.push(sn);
  }
  const layer = new Map<string, number>();
  const visit = (id: string, stack: Set<string>): number => {
    if (layer.has(id)) return layer.get(id)!;
    if (stack.has(id)) return 0;
    stack.add(id);
    const l = Math.max(-1, ...preds.get(id)!.map((p) => visit(p, stack))) + 1;
    stack.delete(id);
    layer.set(id, l);
    return l;
  };
  ids.forEach((id) => visit(id, new Set()));
  const layers: string[][] = [];
  for (const id of ids) (layers[layer.get(id)!] ??= []).push(id);
  const typeOf = new Map(b.nodes.map((n) => [n.id, n.type]));
  const MERGE = new Set(["Add", "Sub", "Mul", "Concat"]);
  const children = new Map<string, string[]>(ids.map((id) => [id, []]));
  for (const [id, ps] of preds) for (const p of ps) children.get(p)!.push(id);
  // a branch start: its parent also feeds a merge node further down (a skip connection)
  const branchStart = (id: string, p: string) =>
    !MERGE.has(typeOf.get(id)!) && children.get(p)!.some((c) => c !== id && MERGE.has(typeOf.get(c)!) && layer.get(c)! > layer.get(id)!);
  const col = new Map<string, number>();
  for (const row of layers) {
    if (!row) continue;
    const key = (id: string) => {
      const ps = preds.get(id)!.filter((p) => col.has(p));
      if (!ps.length) return 0;
      if (MERGE.has(typeOf.get(id)!)) return Math.min(...ps.map((p) => col.get(p)!));
      if (ps.length === 1 && branchStart(id, ps[0])) return col.get(ps[0])! + 1;
      return ps.reduce((s, p) => s + col.get(p)!, 0) / ps.length;
    };
    const sorted = [...row].sort((a, c) => key(a) - key(c));
    // keep each node as close to its parents' column as the row allows
    let next = 0;
    for (const id of sorted) {
      const want = Math.max(next, Math.round(key(id)));
      col.set(id, want);
      next = want + 1;
    }
  }
  const out: Record<string, Position> = {};
  for (const id of ids) {
    if (onlyMissing && b.layout?.[id]) out[id] = b.layout[id];
    else out[id] = { x: col.get(id)! * 216, y: layer.get(id)! * 104 };
  }
  return out;
}

export function ensureLayout(g: Graph): Graph {
  const next = clone(g);
  const fill = (b: Body) => {
    const missing = b.nodes.some((n) => !b.layout?.[n.id]);
    if (missing) b.layout = autoLayout(b, Object.keys(b.layout ?? {}).length > 0);
  };
  fill(next as unknown as Body);
  for (const s of Object.values(next.subgraphs ?? {})) fill(s as unknown as Body);
  return next;
}

/** Wrap the selected nodes into a new subgraph definition and replace them by one instance. */
export function groupIntoSubgraph(
  g: Graph,
  ids: string[],
  name: string,
  typeOf: (ref: string) => { shape: (string | number)[]; dtype: string } | undefined = () => undefined,
): Graph {
  const next = clone(g);
  const sel = new Set(ids);
  const inner = next.nodes.filter((n) => sel.has(n.id));
  if (!inner.length || next.subgraphs?.[name]) return g;
  const edges = next.edges as [string, string][];
  const internal = edges.filter(([s, d]) => sel.has(s.split(".")[0]) && sel.has(d.split(".")[0]));
  const entering = edges.filter(([s, d]) => !sel.has(s.split(".")[0]) && sel.has(d.split(".")[0]));
  const leaving = edges.filter(([s, d]) => sel.has(s.split(".")[0]) && !sel.has(d.split(".")[0]));
  const bodyNodes: Node[] = clone(inner);
  const bodyEdges: [string, string][] = [...internal];
  const instEdges: [string, string][] = [];
  const srcToIn = new Map<string, string>();
  entering.forEach(([s, d]) => {
    let iid = srcToIn.get(s);
    if (!iid) {
      iid = `in_${s.replace(/\./g, "_")}`;
      srcToIn.set(s, iid);
      const t = typeOf(s);
      bodyNodes.push({ id: iid, type: "Input", params: { shape: t?.shape ?? ["B", "D"], dtype: t?.dtype ?? "f32" } });
      instEdges.push([s, `${name}.${iid}`]);
    }
    bodyEdges.push([`${iid}.out`, d]);
  });
  const srcToOut = new Map<string, string>();
  leaving.forEach(([s, d]) => {
    let oid = srcToOut.get(s);
    if (!oid) {
      oid = `out_${s.replace(/\./g, "_")}`;
      srcToOut.set(s, oid);
      bodyNodes.push({ id: oid, type: "Output" });
      bodyEdges.push([s, `${oid}.in`]);
    }
    instEdges.push([`${name}.${oid}`, d]);
  });
  const layout = (next.layout ?? {}) as Record<string, Position>;
  const pos = ids.map((id) => layout[id]).filter(Boolean);
  const at = pos.length ? { x: Math.min(...pos.map((p) => p.x)), y: Math.min(...pos.map((p) => p.y)) } : { x: 0, y: 0 };
  next.subgraphs = { ...(next.subgraphs ?? {}), [name]: { nodes: bodyNodes, edges: bodyEdges } };
  next.nodes = [...next.nodes.filter((n) => !sel.has(n.id)), { id: name, type: "Subgraph", params: { graph: name, repeat: 1 } }];
  next.edges = [...edges.filter(([s, d]) => !sel.has(s.split(".")[0]) && !sel.has(d.split(".")[0])), ...instEdges];
  next.layout = { ...Object.fromEntries(Object.entries(layout).filter(([k]) => !sel.has(k))), [name]: at };
  return next;
}

// ------------------------------------------------------------ templates

function freeName(taken: Set<string>, base: string): string {
  if (!taken.has(base)) return base;
  for (let i = 2; ; i++) if (!taken.has(`${base}_${i}`)) return `${base}_${i}`;
}

/** Add a block template as a subgraph definition plus one instance node in the current scope. */
export function insertBlock(g: Graph, scope: string, blockId: string, block: any, at: Position): [Graph, string] {
  if (scope) return [g, ""]; // nested subgraphs are not part of schema 1
  const next = clone(g);
  next.subgraphs = next.subgraphs ?? {};
  // reuse an identical definition if the graph already has it
  const same = Object.entries(next.subgraphs).find(([, d]) => JSON.stringify({ ...d, layout: undefined }) === JSON.stringify({ ...block, layout: undefined }));
  const name = same ? same[0] : freeName(new Set(Object.keys(next.subgraphs)), blockId);
  if (!same) {
    const laid = ensureLayout({ schema: 1, nodes: block.nodes, edges: block.edges } as Graph);
    next.subgraphs[name] = { nodes: laid.nodes, edges: laid.edges, layout: laid.layout, params: block.params, doc: block.doc } as any;
  }
  const id = freeName(new Set(next.nodes.map((n) => n.id)), name);
  next.nodes.push({ id, type: "Subgraph", params: { graph: name, repeat: 1 } });
  next.layout = { ...(next.layout ?? {}), [id]: freeSpot(next.layout, at) };
  return [next, id];
}

/** Merge a whole model into the current top-level graph, prefixing ids and block names. */
export function insertModelInline(g: Graph, model: Graph, prefix: string, at: Position): [Graph, string[]] {
  const next = clone(g);
  const taken = new Set(next.nodes.map((n) => n.id));
  const subTaken = new Set(Object.keys(next.subgraphs ?? {}));
  const idMap = new Map<string, string>();
  for (const n of model.nodes) idMap.set(n.id, freeName(taken, `${prefix}_${n.id}`)), taken.add(idMap.get(n.id)!);
  const subMap = new Map<string, string>();
  for (const s of Object.keys(model.subgraphs ?? {})) subMap.set(s, freeName(subTaken, s)), subTaken.add(subMap.get(s)!);
  const fieldsTaken = new Set(next.nodes.filter((n) => n.type === "Input").map((n) => ((n.params as any)?.field ?? "x") as string));
  const laid = ensureLayout(model);
  const ids: string[] = [];
  const existing = Object.values(next.layout ?? {});
  if (existing.length) at = { x: Math.max(at.x, Math.max(...existing.map((p) => p.x)) + 232), y: at.y };
  for (const n of model.nodes) {
    const m = clone(n);
    m.id = idMap.get(n.id)!;
    if (m.type === "Subgraph") m.params = { ...(m.params ?? {}), graph: subMap.get((m.params as any).graph) };
    if (m.type === "Linear" && (m.params as any)?.tie_to) m.params = { ...m.params, tie_to: idMap.get((m.params as any).tie_to) };
    if (m.type === "Input") {
      const f = ((m.params as any)?.field ?? "x") as string;
      const nf = fieldsTaken.has(f) ? freeName(fieldsTaken, `${prefix}_${f}`) : f;
      fieldsTaken.add(nf);
      m.params = { ...(m.params ?? {}), field: nf };
    }
    next.nodes.push(m);
    ids.push(m.id);
    const p = laid.layout?.[n.id] ?? { x: 0, y: 0 };
    next.layout = { ...(next.layout ?? {}), [m.id]: { x: snap(at.x + p.x), y: snap(at.y + p.y) } };
  }
  for (const [s, d] of model.edges as [string, string][]) {
    const [sn, sp] = [s.slice(0, s.lastIndexOf(".")), s.slice(s.lastIndexOf(".") + 1)];
    const [dn, dp] = [d.slice(0, d.lastIndexOf(".")), d.slice(d.lastIndexOf(".") + 1)];
    next.edges.push([`${idMap.get(sn)}.${sp}`, `${idMap.get(dn)}.${dp}`]);
  }
  for (const [s, def] of Object.entries(model.subgraphs ?? {})) {
    const laid = ensureLayout({ schema: 1, ...clone(def) } as Graph) as any;
    delete laid.schema;
    next.subgraphs = { ...(next.subgraphs ?? {}), [subMap.get(s)!]: laid };
  }
  return [next, ids];
}

// ------------------------------------------------------- clipboard

export interface Clip {
  nodes: Node[];
  edges: [string, string][];
  layout: Record<string, Position>;
}

export function copySelection(g: Graph, scope: string, ids: string[]): Clip {
  const b = getBody(g, scope);
  const sel = new Set(ids);
  return {
    nodes: clone(b.nodes.filter((n) => sel.has(n.id))),
    edges: b.edges.filter(([s, d]) => sel.has(s.split(".")[0]) && sel.has(d.split(".")[0])),
    layout: Object.fromEntries(ids.map((id) => [id, b.layout?.[id] ?? { x: 0, y: 0 }])),
  };
}

export function pasteClip(g: Graph, scope: string, clip: Clip, offset = 32): [Graph, string[]] {
  let fresh: string[] = [];
  const next = withBody(g, scope, (b) => {
    const taken = new Set(b.nodes.map((n) => n.id));
    const map = new Map<string, string>();
    for (const n of clip.nodes) {
      const base = n.id.replace(/_\d+$/, "");
      const id = freeName(taken, base);
      taken.add(id);
      map.set(n.id, id);
    }
    b.layout = { ...(b.layout ?? {}) };
    // shift the pasted group as a whole until none of it covers an existing node
    let dx = offset;
    const covers = (d: number) =>
      clip.nodes.some((n) => {
        const p = clip.layout[n.id] ?? { x: 0, y: 0 };
        const spot = freeSpot(b.layout, { x: p.x + d, y: p.y + offset });
        return spot.x !== snap(p.x + d) || spot.y !== snap(p.y + offset);
      });
    for (let i = 0; i < 50 && covers(dx); i++) dx += 216;
    for (const n of clip.nodes) {
      const m = clone(n);
      m.id = map.get(n.id)!;
      b.nodes.push(m);
      const p = clip.layout[n.id] ?? { x: 0, y: 0 };
      b.layout[m.id] = { x: snap(p.x + dx), y: snap(p.y + offset) };
    }
    for (const [s, d] of clip.edges) {
      const re = (r: string) => `${map.get(r.slice(0, r.lastIndexOf(".")))}.${r.slice(r.lastIndexOf(".") + 1)}`;
      b.edges.push([re(s), re(d)]);
    }
    fresh = [...map.values()];
  });
  return [next, fresh];
}
