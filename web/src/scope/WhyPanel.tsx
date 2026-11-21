// Compare's "why did it blow up": trainscope's multi-run analysis over the ticked runs. Config
// fields are this job's training settings (train.py records them), so the answer reads like
// "every run that spiked had lr 1.0; the nearest stable run had 0.05".
import { useEffect, useState } from "react";
import { scopeMulti } from "./api";
import type { RunSummary } from "./types";

const field = (f: string) => f.replace(/^model\.(train\.)?/, "").replace(/^config\./, "trainscope.");
const short = (id: string) => id.slice(0, 8);
const val = (v: unknown) => (typeof v === "number" ? String(+v.toPrecision(4)) : v == null ? "—" : String(v));

export function WhyPanel({ base, runs }: { base: string; runs: string[] }) {
  const [known, setKnown] = useState<RunSummary[] | null>(null);
  const [cmp, setCmp] = useState<Record<string, any> | null>(null);
  const [clusters, setClusters] = useState<Record<string, any> | null>(null);
  const [ce, setCe] = useState<Record<string, any> | null>(null);
  const key = runs.join(",");
  useEffect(() => {
    let alive = true;
    const m = scopeMulti(base);
    setCmp(null);
    setCe(null);
    m.runs().then(async (all) => {
      if (!alive) return;
      const list = (all ?? []).filter((r) => runs.includes(r.name));
      setKnown(list);
      const recorded = list.map((r) => r.name);
      if (recorded.length >= 2) m.compare(recorded).then((c) => alive && setCmp(c)).catch(() => {});
      if (recorded.length) m.clusters(recorded).then((c) => alive && setClusters(c)).catch(() => {});
      const blown = list.find((r) => r.spike_count > 0 || r.last_loss_finite === false);
      // the nearest stable run is searched over every recorded run, not only the ticked ones
      if (blown) m.counterexample(blown.name).then((c) => alive && setCe(c)).catch(() => alive && setCe(null));
    });
    return () => {
      alive = false;
    };
  }, [base, key]);

  if (!known) return null;
  const missing = runs.filter((id) => !known.some((r) => r.name === id));
  return (
    <section className="text-small">
      <div className="text-label text-muted">Why did it blow up · trainscope</div>
      {known.length === 0 && <p className="m-0 text-muted">None of these runs has a diagnostics recording.</p>}
      {cmp?.divergence && (
        <p className="m-0">
          Loss curves separate at step <span className="num">{cmp.divergence.step}</span>.
        </p>
      )}
      {cmp && cmp.common_cause?.length > 0 && (
        <div className="mt-1">
          Every run that spiked differs from the stable ones in:
          <ul className="m-0 list-none p-0">
            {cmp.common_cause.map((c: any) => (
              <li key={c.field} className="num">
                {field(c.field)}: {val(c.spiked_value)} spiked, {val(c.stable_value)} stable
              </li>
            ))}
          </ul>
        </div>
      )}
      {cmp && !cmp.common_cause?.length && cmp.config_diff?.length > 0 && (
        <div className="mt-1 text-muted">Settings that differ: {cmp.config_diff.map((d: any) => field(d.field)).join(", ")}</div>
      )}
      {ce && (
        <p className="m-0 mt-1" data-counterexample={ce.counterexample_run}>
          Nearest stable run to <span className="num">{short(ce.query_run)}</span> is <span className="num">{short(ce.counterexample_run)}</span>
          {ce.config_diff?.length ? (
            <>
              ; it differs only in{" "}
              {ce.config_diff.map((d: any, i: number) => (
                <span key={d.field} className="num">
                  {i ? ", " : ""}
                  {field(d.field)} {val(d.query_value)} → {val(d.stable_value)}
                </span>
              ))}
            </>
          ) : null}
          {ce.first_signal ? `. First signal: ${ce.first_signal} at step ${ce.crossing_step}.` : "."}
        </p>
      )}
      {clusters && clusters.clusters?.length > 0 && (
        <div className="mt-1 text-muted">
          {clusters.clusters.map((c: any) => (
            <div key={c.label}>
              {c.label}: {c.runs.map(short).join(", ")}
              {c.typical_lead_steps != null ? ` · first signal ~${c.typical_lead_steps} steps before the spike` : ""}
            </div>
          ))}
        </div>
      )}
      {missing.map((id) => (
        <div key={id} data-no-recording={id} className="text-muted">
          <span className="num">{short(id)}</span>: no recording
        </div>
      ))}
    </section>
  );
}
