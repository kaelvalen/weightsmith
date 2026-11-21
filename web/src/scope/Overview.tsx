// What happened in this run: status, loss and grad norm on one step axis, the layer heatmap
// below them, and the signals that crossed their threshold before the first spike.
import { useEffect, useMemo, useState } from "react";
import { fmt, lossYMax } from "../ui/chartOptions";
import { LineChart } from "../ui/LineChart";
import { Select } from "../ui/Select";
import { scopeApi } from "./api";
import { LayerHeatmap } from "./LayerHeatmap";
import { runStatus } from "./runStatus";
import { spikeBands } from "./spikeCluster";
import type { LayerMatrix } from "./types";
import type { ScopeRun } from "./useScopeRun";
import type { SpikeEvent } from "./types";
import { computeWarmupBand } from "./warmup";

const SIGNAL_LABELS: Record<string, string> = {
  kurtosis: "activation kurtosis",
  grad_norm: "grad norm",
  concentration: "routing concentration",
  loss: "loss",
};

/** The layer matrix, refetched at most every 5 s while steps arrive. */
function useMatrix(base: string, run: string, length: number, metric: string): LayerMatrix | null {
  const [matrix, setMatrix] = useState<LayerMatrix | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const t = window.setTimeout(() => setTick((k) => k + 1), tick === 0 ? 0 : 5000);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [length]);
  useEffect(() => {
    let alive = true;
    scopeApi(base, run)
      .layerMatrix({ metric })
      .then((m) => alive && setMatrix(m ?? { layers: [], z: [], values: [] }))
      .catch(() => alive && setMatrix((m) => m ?? { layers: [], z: [], values: [] }));
    return () => {
      alive = false;
    };
  }, [base, run, tick, metric]);
  return matrix;
}

const METRICS = [
  { value: "grad_l2_norm", label: "gradient norm" },
  { value: "weight_l2_norm", label: "weight norm" },
  { value: "act_kurtosis", label: "activation kurtosis" },
  { value: "grad_max_abs", label: "max |gradient|" },
];

export interface CanvasLink {
  paint: boolean;
  setPaint: (on: boolean) => void;
  /** the matrix behind the heatmap and the first spike window, to paint graph nodes */
  onMatrix: (m: LayerMatrix, spike: { from: number; to: number } | null) => void;
}

export function Overview({ base, run, data, live, onSelectLayer, highlight, link }: { base: string; run: string; data: ScopeRun & { events: SpikeEvent[] }; live: boolean; onSelectLayer?: (layer: string) => void; highlight?: string | null; link?: CanvasLink }) {
  const { global, events, meta } = data;
  const [signals, setSignals] = useState<Record<string, any> | null>(null);
  const [metric, setMetric] = useState("grad_l2_norm");
  const matrix = useMatrix(base, run, global.length, metric);
  const firstWindow = events[0] ? { from: events[0].startStep, to: events[0].endStep } : null;
  useEffect(() => {
    if (matrix && link) link.onMatrix(matrix, firstWindow);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [matrix, firstWindow?.from, firstWindow?.to]);
  useEffect(() => {
    let alive = true;
    scopeApi(base, run)
      .signals()
      .then((s) => alive && setSignals(s ?? { first_steps: {} }))
      .catch(() => alive && setSignals({ first_steps: {} }));
    return () => {
      alive = false;
    };
  }, [base, run, events.length]);

  const steps = useMemo(() => global.map((r) => r.step), [global]);
  const losses = useMemo(() => global.map((r) => r.loss ?? null), [global]);
  const first = events[0] ?? null;
  const status = runStatus({ globalData: global, spikeEvents: events, live });
  const signalRows = useMemo(
    () =>
      Object.entries((signals?.first_steps ?? {}) as Record<string, number>)
        .sort((a, b) => a[1] - b[1])
        .map(([name, step]) => ({ name, label: SIGNAL_LABELS[name] ?? name, step, lead: first && first.startStep - step >= 0 ? first.startStep - step : null })),
    [signals, first],
  );
  const firstWarning = signalRows.find((r) => r.name !== "loss")?.step ?? null;
  const warmup = useMemo(() => computeWarmupBand(meta, global), [meta, global]);
  const bands = useMemo(() => [...(warmup ? [{ x0: warmup.startStep, x1: warmup.endStep, tone: "warn" as const }] : []), ...spikeBands(events)], [warmup, events]);
  const markers = useMemo(() => (firstWarning == null ? [] : [{ x: firstWarning, kind: "note" as const }]), [firstWarning]);
  const yMax = useMemo(() => lossYMax(steps, losses, first?.startStep ?? null), [steps, losses, first]);
  const lossSeries = useMemo(() => [{ id: "loss", label: "loss", points: global.filter((r) => r.loss != null).map((r) => [r.step, r.loss as number] as [number, number]) }], [global]);
  const gradSeries = useMemo(
    () => [{ id: "grad", label: "grad norm", points: global.filter((r) => r.grad_norm_before_clip != null).map((r) => [r.step, r.grad_norm_before_clip as number] as [number, number]) }],
    [global],
  );

  if (!global.length) return <p className="m-0 p-4 text-small text-muted">Waiting for the first recorded step…</p>;
  const last = global[global.length - 1];
  const stat = (label: string, value: string) => (
    <div className="min-w-0">
      <div className="text-label text-muted">{label}</div>
      <div className="num truncate text-small">{value}</div>
    </div>
  );

  return (
    <div className="grid gap-x-6 gap-y-3 px-4 py-2 lg:grid-cols-[minmax(0,1fr)_260px]">
      <div className="min-w-0 space-y-2">
        <div className="text-small">
          <span style={{ color: status.kind === "diverged" || status.kind === "spiked" ? "var(--error)" : status.kind === "training" ? "var(--accent)" : undefined }}>{status.label}</span>
          <span className="ml-2 text-muted">{status.detail}</span>
        </div>
        <div className="grid grid-cols-4 gap-4">
          {stat("final loss", fmt(last.loss ?? NaN))}
          {stat("grad norm", fmt(last.grad_norm_before_clip ?? NaN))}
          {stat("first warning", firstWarning == null ? "none" : `step ${firstWarning}`)}
          {stat("spike window", first ? (events.length > 1 ? `${events.length} spikes, first ${first.startStep}` : first.label.toLowerCase()) : "no spike")}
        </div>
        <LineChart title="loss" series={lossSeries} height={150} yMax={yMax} bands={bands} markers={markers} syncKey={`diag-${run}`} />
        <LineChart title="grad norm" series={gradSeries} height={110} bands={bands} syncKey={`diag-${run}`} />
        <div>
          <div className="flex items-center gap-3 pb-1 text-label text-muted">
            <div className="w-48 shrink-0">
              <Select value={metric} options={METRICS} onChange={setMetric} ariaLabel="Heatmap metric" />
            </div>
            <span>by layer · z-score against the run's own normal</span>
            {link && (
              <button className="link ml-auto" aria-pressed={link.paint} style={{ color: link.paint ? "var(--ink)" : undefined }} onClick={() => link.setPaint(!link.paint)}>
                paint canvas
              </button>
            )}
          </div>
          <LayerHeatmap
            matrix={matrix}
            domain={[steps[0], steps[steps.length - 1]]}
            warningStep={firstWarning}
            spikeWindow={first ? { from: first.startStep, to: first.endStep } : null}
            highlight={highlight}
            onSelect={(layer) => onSelectLayer?.(layer)}
          />
        </div>
      </div>
      <aside className="min-w-0">
        <div className="pb-1 text-label text-muted">signals before the spike</div>
        {signals == null ? (
          <p className="m-0 text-small text-muted">Loading…</p>
        ) : signalRows.length === 0 ? (
          <p className="m-0 text-small text-muted">No signal crossed its threshold.</p>
        ) : (
          <table className="num w-full border-collapse text-small">
            <thead>
              <tr className="text-left text-label text-muted">
                <th className="py-1 font-normal">signal</th>
                <th className="py-1 text-right font-normal">crossed</th>
                <th className="py-1 text-right font-normal">lead</th>
              </tr>
            </thead>
            <tbody>
              {signalRows.map((r) => (
                <tr key={r.name} data-signal={r.name} className="border-t border-line">
                  <td className="py-1">{r.label}</td>
                  <td className="py-1 text-right">{r.step}</td>
                  <td className="py-1 text-right" style={{ color: (r.lead ?? 0) > 0 ? "var(--error)" : "var(--muted)" }}>
                    {r.lead ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="mt-3 text-label text-muted">
          Lead is how many steps before the first spike a signal crossed. Click a heatmap row to see that layer
          {onSelectLayer ? " and its node" : ""}.
        </p>
      </aside>
    </div>
  );
}
