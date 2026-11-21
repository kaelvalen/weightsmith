import type { GlobalRow } from "./types";

/**
 * The detector produces no scores until it has seen `min_observations` losses, so spikes in that
 * window are invisible. The band spans the first recorded step to the min_observations-th one.
 */
export function computeWarmupBand(meta: { detector?: { min_observations?: number; [k: string]: unknown }; [k: string]: unknown } | null | undefined, globalData: GlobalRow[] = []) {
  const minObservations = meta?.detector?.min_observations;
  if (!Number.isInteger(minObservations) || (minObservations as number) <= 0) return null;
  if (!Array.isArray(globalData) || globalData.length === 0) return null;
  const startRow = globalData[0];
  const endRow = globalData[Math.min(minObservations as number, globalData.length) - 1];
  if (startRow?.step == null || endRow?.step == null) return null;
  return { startStep: startRow.step, endStep: endRow.step };
}
