// One tensor over the run: gradient, weight and activation statistics.
import { useEffect, useMemo, useState } from "react";
import { LineChart } from "../ui/LineChart";
import { Select } from "../ui/Select";
import { scopeApi } from "./api";
import { spikeBands } from "./spikeCluster";
import type { SpikeEvent } from "./types";

const METRICS: [string, string][] = [
  ["grad_l2_norm", "gradient L2 norm"],
  ["weight_l2_norm", "weight L2 norm"],
  ["act_kurtosis", "activation kurtosis"],
  ["grad_max_abs", "max |gradient|"],
];

export function Layers({ base, run, layers, events, layer, onLayer }: { base: string; run: string; layers: string[]; events: SpikeEvent[]; layer: string | null; onLayer: (l: string) => void }) {
  const [rows, setRows] = useState<Record<string, any>[] | null>(null);
  const current = layer && layers.includes(layer) ? layer : layers[0] ?? null;
  useEffect(() => {
    if (!current) return;
    let alive = true;
    setRows(null);
    scopeApi(base, run)
      .layer(current)
      .then((r) => alive && setRows(r ?? []));
    return () => {
      alive = false;
    };
  }, [base, run, current]);
  const bands = useMemo(() => spikeBands(events), [events]);
  if (!current) return <p className="m-0 p-4 text-small text-muted">No layer metrics recorded for this run.</p>;
  return (
    <div className="px-4 py-2">
      <div className="flex items-center gap-3 pb-2">
        <span className="text-label text-muted">layer</span>
        <Select value={current} options={layers} onChange={onLayer} ariaLabel="Layer" mono className="w-96" />
      </div>
      {rows == null ? (
        <p className="m-0 text-small text-muted">Loading…</p>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fit,minmax(260px,1fr))] gap-x-6 gap-y-2">
          {METRICS.map(([key, title]) => (
            <LineChart
              key={key}
              title={title}
              series={[{ id: key, label: title, points: rows.filter((r) => typeof r[key] === "number").map((r) => [r.step, r[key]] as [number, number]) }]}
              height={120}
              bands={bands}
              syncKey={`layer-${run}`}
            />
          ))}
        </div>
      )}
    </div>
  );
}
