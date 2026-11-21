// How far each layer's weight distribution moved between two steps (KL of the recorded histograms).
import { useEffect, useState } from "react";
import { fmt } from "../ui/chartOptions";
import { scopeApi } from "./api";

export function WeightsDiff({ base, run, steps, onSelectLayer }: { base: string; run: string; steps: number[]; onSelectLayer?: (layer: string) => void }) {
  const [a, setA] = useState(steps[0] ?? 0);
  const [b, setB] = useState(steps[steps.length - 1] ?? 0);
  const [rows, setRows] = useState<Record<string, any>[] | null>(null);
  useEffect(() => {
    let alive = true;
    scopeApi(base, run)
      .diff(a, b)
      .then((r) => alive && setRows((r ?? []).slice().sort((x, y) => (y.kl_divergence ?? 0) - (x.kl_divergence ?? 0))))
      .catch(() => alive && setRows([]));
    return () => {
      alive = false;
    };
  }, [base, run, a, b]);
  const field = (label: string, v: number, set: (n: number) => void) => (
    <label className="flex items-center gap-2">
      <span className="text-label text-muted">{label}</span>
      <input className="num w-20 border border-line bg-bg px-1 text-small" type="number" value={v} onChange={(e) => set(Number(e.target.value))} />
    </label>
  );
  return (
    <div className="px-4 py-2">
      <div className="flex gap-4 pb-2">
        {field("from step", a, setA)}
        {field("to step", b, setB)}
        <span className="text-label text-muted">histograms are recorded every few steps; the nearest recorded ones are compared</span>
      </div>
      {rows == null ? (
        <p className="m-0 text-small text-muted">Loading…</p>
      ) : rows.length === 0 ? (
        <p className="m-0 text-small text-muted">No histograms at these steps.</p>
      ) : (
        <table className="num w-full border-collapse text-small">
          <thead>
            <tr className="text-left text-label text-muted">
              <th className="py-1 font-normal">layer</th>
              <th className="py-1 text-right font-normal">KL divergence</th>
              <th className="py-1 text-right font-normal">grad norm change</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 50).map((r) => (
              <tr key={r.layer} className="border-t border-line">
                <td className="py-1">
                  <button className="link" onClick={() => onSelectLayer?.(r.layer)}>
                    {r.layer}
                  </button>
                </td>
                <td className="py-1 text-right">{fmt(r.kl_divergence ?? NaN)}</td>
                <td className="py-1 text-right">{fmt(r.grad_norm_change ?? NaN)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
