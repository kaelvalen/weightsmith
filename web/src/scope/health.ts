import type { RunSummary } from "./types";

export type Health = "—" | "diverged" | "spiked" | "training" | "stable";

/** One word per run from its recording summary; "—" when the run has no recording. */
export function runHealth(summary: RunSummary | undefined, status: string): Health {
  if (!summary) return "—";
  if (summary.last_loss_finite === false) return "diverged";
  if (summary.spike_count > 0) return "spiked";
  return status === "running" || status === "preparing" ? "training" : "stable";
}
