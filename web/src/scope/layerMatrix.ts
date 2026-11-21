// Row shaping for the layer heatmap: grouping by block when there are too many tensors,
// first-crossing order, the colour ramp and an accessible summary.
import type { LayerMatrix } from "./types";

const Z_MAX = 6;

export interface HeatRow {
  key: string;
  label: string;
  isGroup: boolean;
  z: (number | null)[];
  values: (number | null)[];
  children?: string[];
}

/** `blocks.3.attn.qkv.weight` → `blocks.3`; names without a numbered segment stay ungrouped. */
function groupKey(name: string): string | null {
  const parts = name.split(".");
  const i = parts.findIndex((p) => /^\d+$/.test(p));
  return i < 0 || i === parts.length - 1 ? null : parts.slice(0, i + 1).join(".");
}

function maxRows(rows: HeatRow[], field: "z" | "values"): (number | null)[] {
  const width = rows[0]?.[field].length ?? 0;
  return Array.from({ length: width }, (_, b) => {
    let best: number | null = null;
    for (const row of rows) {
      const v = row[field][b];
      if (v != null && (best == null || v > best)) best = v;
    }
    return best;
  });
}

export function groupRows(matrix: LayerMatrix, { maxRows: limit = 24, expanded = new Set<string>() }: { maxRows?: number; expanded?: Set<string> } = {}): HeatRow[] {
  const leaves: HeatRow[] = matrix.layers.map((name, i) => ({ key: name, label: name, isGroup: false, z: matrix.z[i], values: matrix.values[i] }));
  if (leaves.length <= limit) return leaves;
  const out: (HeatRow & { members?: HeatRow[] })[] = [];
  const groups = new Map<string, HeatRow & { members: HeatRow[]; children: string[] }>();
  for (const leaf of leaves) {
    const key = groupKey(leaf.key);
    if (key == null) {
      out.push(leaf);
      continue;
    }
    if (!groups.has(key)) {
      const group = { key, label: key, isGroup: true, z: [], values: [], children: [] as string[], members: [] as HeatRow[] };
      groups.set(key, group);
      out.push(group);
    }
    const group = groups.get(key)!;
    group.children.push(leaf.key);
    group.members.push(leaf);
  }
  return out.flatMap((row) => {
    if (!row.isGroup) return [row];
    const { members = [], ...group } = row;
    const merged: HeatRow = { ...group, z: maxRows(members, "z"), values: maxRows(members, "values") };
    return expanded.has(row.key) ? [merged, ...members] : [merged];
  });
}

export function firstCrossings(rows: HeatRow[], bucketStarts: number[], threshold = 3): Map<string, number | null> {
  const out = new Map<string, number | null>();
  for (const row of rows) {
    const b = row.z.findIndex((z) => z != null && z >= threshold);
    out.set(row.key, b < 0 ? null : bucketStarts[b]);
  }
  return out;
}

export function sortRows(rows: HeatRow[], mode: string, crossings: Map<string, number | null>): HeatRow[] {
  if (mode !== "first") return rows;
  const at = (row: HeatRow) => crossings.get(row.key) ?? Infinity;
  return [...rows].sort((a, b) => at(a) - at(b));
}

export function zColor(z: number | null, surfaceRgb: number[], redRgb: number[]): string {
  const t = z == null ? 0 : Math.min(Math.max(z / Z_MAX, 0), 1);
  return `rgb(${surfaceRgb.map((c, i) => Math.round(c + (redRgb[i] - c) * t)).join(",")})`;
}

export function heatmapSummary(rows: HeatRow[], crossings: Map<string, number | null>): string {
  const first = rows
    .filter((r) => crossings.get(r.key) != null)
    .sort((a, b) => (crossings.get(a.key) as number) - (crossings.get(b.key) as number))
    .slice(0, 3);
  if (first.length === 0) return "No layer crossed 3σ";
  return `First above 3σ: ${first.map((r) => `${r.label} at step ${crossings.get(r.key)}`).join(", ")}`;
}
