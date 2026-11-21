// Raw spike detections grouped into spike events (anomaly cascades).
import type { Band } from "../ui/chartOptions";
import type { GlobalRow, SpikeEvent } from "./types";

interface Item {
  step: number;
  loss: number | null;
  zScore: number | null;
}

/** Detections closer than `maxGap` steps merge into one event; `spikeList` is the fallback before global rows load. */
export function groupSpikes(globalData: GlobalRow[] = [], spikeList: (number | { step: number })[] = [], maxGap = 5): SpikeEvent[] {
  let items: Item[] = [];
  if (Array.isArray(globalData) && globalData.some((r) => r.is_spike)) {
    items = globalData.filter((r) => r.is_spike).map((r) => ({ step: r.step, loss: r.loss ?? null, zScore: r.spike_score ?? null }));
  } else if (Array.isArray(spikeList) && spikeList.length > 0) {
    items = spikeList.map((s) => ({ step: typeof s === "number" ? s : s.step, loss: null, zScore: null }));
  }
  if (items.length === 0) return [];
  items.sort((a, b) => a.step - b.step);
  const events: SpikeEvent[] = [];
  let group = [items[0]];
  for (let i = 1; i < items.length; i++) {
    if (items[i].step - items[i - 1].step <= maxGap) group.push(items[i]);
    else {
      events.push(createSpikeEvent(group, events.length));
      group = [items[i]];
    }
  }
  events.push(createSpikeEvent(group, events.length));
  return events;
}

function createSpikeEvent(group: Item[], index: number): SpikeEvent {
  const steps = group.map((item) => item.step);
  const startStep = steps[0];
  const endStep = steps[steps.length - 1];
  // peak: highest loss, else highest score
  let peak = group[0];
  for (const item of group) {
    if (item.loss != null && (peak.loss == null || item.loss > peak.loss)) peak = item;
    else if (item.loss == null && item.zScore != null && (peak.zScore == null || item.zScore > peak.zScore)) peak = item;
  }
  const earlyWarningWindow = Math.max(0, peak.step - startStep);
  const label =
    startStep === endStep
      ? `Step ${startStep}`
      : earlyWarningWindow > 0
        ? `Steps ${startStep}–${endStep} (${earlyWarningWindow}-step detection lead)`
        : `Steps ${startStep}–${endStep}`;
  return { id: `event_${index}`, startStep, endStep, count: steps.length, steps, peakStep: peak.step, peakLoss: peak.loss, earlyWarningWindow, label };
}

/** Grouped spike events as chart bands. */
export function spikeBands(events: { startStep: number; endStep: number }[] = []): Band[] {
  return events.map((e) => ({ x0: e.startStep, x1: e.endStep, tone: "spike" }));
}
