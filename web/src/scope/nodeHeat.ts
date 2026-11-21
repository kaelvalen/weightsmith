// From layer metrics to graph nodes: how anomalous each node got, and which node's layer went
// first before the spike. Frozen layers (gradient always 0) are recorded but never blamed.
import { nodeFor } from "./nodeMap";
import type { LayerMatrix } from "./types";

const THRESHOLD = 3;

const trainable = (values: (number | null)[]) => values.some((v) => v != null && v > 0);

export function layersByNode(layers: string[], moduleMap: Record<string, string>): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const l of layers) {
    const node = nodeFor(l, moduleMap);
    if (node) (out[node] ??= []).push(l);
  }
  return out;
}

/** Largest z-score each node's trainable layers reached over the run. */
export function nodeHeat(matrix: LayerMatrix, moduleMap: Record<string, string>): Record<string, number> {
  const out: Record<string, number> = {};
  matrix.layers.forEach((layer, i) => {
    const node = nodeFor(layer, moduleMap);
    if (!node || !trainable(matrix.values[i])) return;
    const peak = Math.max(...matrix.z[i].map((z) => z ?? 0));
    out[node] = Math.max(out[node] ?? 0, peak);
  });
  return out;
}

/** The node whose trainable layer first crossed 3σ at or before the end of the spike window. */
export function spikeSourceNode(matrix: LayerMatrix, moduleMap: Record<string, string>, window: { from: number; to: number } | null): string | null {
  if (!window) return null;
  const starts = matrix.bucket_starts ?? [];
  let best: { step: number; node: string } | null = null;
  matrix.layers.forEach((layer, i) => {
    const node = nodeFor(layer, moduleMap);
    if (!node || !trainable(matrix.values[i])) return;
    const b = matrix.z[i].findIndex((z) => z != null && z >= THRESHOLD);
    if (b < 0 || starts[b] > window.to) return;
    if (!best || starts[b] < best.step) best = { step: starts[b], node };
  });
  return best ? (best as { node: string }).node : null;
}
