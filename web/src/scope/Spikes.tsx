// Each spike event: the window around it, the failure cascade read from it, and the layers
// recorded at full resolution inside it.
import { useEffect, useState } from "react";
import { LineChart } from "../ui/LineChart";
import { scopeApi } from "./api";
import { diagnoseSpike } from "./diagnosis";
import type { GlobalRow, SpikeEvent } from "./types";

export function Spikes({ base, run, events, files, onSelectLayer }: { base: string; run: string; events: SpikeEvent[]; files: number[]; onSelectLayer?: (layer: string) => void }) {
  const [sel, setSel] = useState(0);
  const [win, setWin] = useState<GlobalRow[] | null>(null);
  const [layers, setLayers] = useState<string[]>([]);
  const ev = events[Math.min(sel, events.length - 1)];
  // a recorded window file for this event: the first detection inside it that has one
  const at = ev ? (files.find((f) => f >= ev.startStep && f <= ev.endStep) ?? ev.startStep) : 0;
  useEffect(() => {
    if (!ev) return;
    let alive = true;
    const api = scopeApi(base, run);
    setWin(null);
    Promise.all([api.spike(at), api.spikeLayerNames(at)]).then(([w, l]) => {
      if (!alive) return;
      setWin(w ?? []);
      setLayers(l ?? []);
    });
    return () => {
      alive = false;
    };
  }, [base, run, at, ev]);

  if (!events.length) return <p className="m-0 p-4 text-small text-muted">No spike was detected in this run.</p>;
  const d = win && win.length ? diagnoseSpike(win, at, ev) : null;
  return (
    <div className="grid gap-x-6 px-4 py-2 lg:grid-cols-[220px_minmax(0,1fr)]">
      <ul className="m-0 list-none p-0 text-small">
        {events.map((e, i) => (
          <li key={e.id} data-spike={e.startStep} className="border-t border-line first:border-t-0">
            <button className="link w-full py-1 text-left" style={{ color: i === sel ? "var(--ink)" : undefined, fontWeight: i === sel ? 500 : 400 }} onClick={() => setSel(i)}>
              {e.label}
            </button>
          </li>
        ))}
      </ul>
      <div className="min-w-0 space-y-2">
        {d && (
          <div className="text-small">
            <div style={{ color: d.badgeVariant === "accent" ? "var(--ink)" : "var(--error)" }}>{d.primaryTrigger}</div>
            <div className="text-muted">{d.desc}</div>
          </div>
        )}
        <LineChart
          title={`loss around step ${at}`}
          series={[{ id: "w", label: "loss", points: (win ?? []).filter((r) => r.loss != null).map((r) => [r.step, r.loss as number] as [number, number]) }]}
          height={140}
          markers={[{ x: at, kind: "spike" }]}
        />
        {layers.length > 0 && (
          <div className="text-small">
            <span className="text-label text-muted">layers recorded in the window </span>
            {layers.slice(0, 24).map((l) => (
              <button key={l} className="link num mr-3" onClick={() => onSelectLayer?.(l)}>
                {l}
              </button>
            ))}
            {layers.length > 24 && <span className="text-muted">+{layers.length - 24} more</span>}
          </div>
        )}
      </div>
    </div>
  );
}
