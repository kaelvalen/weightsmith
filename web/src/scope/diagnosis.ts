// Spike cascade diagnosis: a pre/post-spike window as a chronological failure narrative.
import type { GlobalRow, SpikeEvent } from "./types";

export interface Diagnosis {
  primaryTrigger: string;
  cascadePath: string;
  badgeVariant: "accent" | "warning" | "danger";
  desc: string;
}

export function diagnoseSpike(globalWindow: GlobalRow[] | null | undefined, selectedSpike: number | null, currentEvent?: Pick<SpikeEvent, "earlyWarningWindow"> & Partial<SpikeEvent> | null): Diagnosis | null {
  if (!globalWindow || globalWindow.length === 0) return null;
  let firstDriftStep: number | null = null;
  let firstGradExplosionStep: number | null = null;
  let firstNanStep: number | null = null;
  const baselineLoss = globalWindow[0]?.loss || 1.0;
  const baselineGrad = globalWindow[0]?.grad_norm_before_clip || 1.0;

  for (const row of globalWindow) {
    const loss = row.loss as number;
    if (firstNanStep == null && ((row.grad_nan_inf_ratio ?? 0) > 0 || !Number.isFinite(loss))) firstNanStep = row.step;
    if (firstGradExplosionStep == null && (row.grad_norm_before_clip ?? 0) > Math.max(5.0, baselineGrad * 3.0)) firstGradExplosionStep = row.step;
    // spike_score is the detector's own anomaly score; runs from before it existed fall back to a heuristic
    const zScore = row.spike_score ?? (loss - baselineLoss) / Math.max(1e-6, baselineLoss * 0.1);
    if (firstDriftStep == null && (zScore >= 3.5 || loss > baselineLoss * 1.5)) firstDriftStep = row.step;
  }

  const events: { name: string; step: number }[] = [];
  if (firstDriftStep != null) events.push({ name: "Loss Shift", step: firstDriftStep });
  if (firstGradExplosionStep != null) events.push({ name: "Gradient Explosion", step: firstGradExplosionStep });
  if (firstNanStep != null) events.push({ name: "NaN / Inf Collapse", step: firstNanStep });
  events.sort((a, b) => a.step - b.step);

  if (events.length === 0) {
    return { primaryTrigger: "Distributional Shift", cascadePath: `Step ${selectedSpike}`, badgeVariant: "accent", desc: "Anomalous deviation detected in loss stream baseline." };
  }
  const primaryTrigger = events[0].name;
  const cascadePath = events.map((e) => `${e.name} (Step ${e.step})`).join(" → ");
  const badgeVariant = primaryTrigger === "NaN / Inf Collapse" ? "danger" : primaryTrigger === "Gradient Explosion" ? "warning" : "accent";
  let desc = `Chronological Failure Cascade: ${cascadePath}.`;
  if (currentEvent && currentEvent.earlyWarningWindow > 0) {
    desc += ` The detected anomaly window began ${currentEvent.earlyWarningWindow} steps before the highest observed loss at Step ${currentEvent.peakStep}.`;
  }
  return { primaryTrigger, cascadePath, badgeVariant, desc };
}
