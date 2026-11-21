// The Diagnose tab: trainscope's view of one run. Weightsmith passes onSelectLayer so a layer
// can be shown on the canvas; the standalone trainscope UI renders the same component without it.
import { useState } from "react";
import { Layers } from "./Layers";
import { Overview } from "./Overview";
import { Replay } from "./Replay";
import { Routing } from "./Routing";
import { Spikes } from "./Spikes";
import { useScopeRun } from "./useScopeRun";
import { WeightsDiff } from "./WeightsDiff";

type Sub = "overview" | "layers" | "spikes" | "routing" | "replay" | "weights";

export function Diagnose({ base, run, live = false, onSelectLayer }: { base: string; run: string | null; live?: boolean; onSelectLayer?: (layer: string) => void }) {
  const [sub, setSub] = useState<Sub>("overview");
  const [layer, setLayer] = useState<string | null>(null);
  const data = useScopeRun(base, run, live);
  if (!run) return <p className="m-0 p-4 text-small text-muted" data-testid="diagnose">Select a run in Runs, or press Run.</p>;
  if (data.loading) return <p className="m-0 p-4 text-small text-muted" data-testid="diagnose">Loading the recording…</p>;
  if (data.missing)
    return (
      <div className="p-4 text-small" data-testid="diagnose">
        <p className="m-0">No diagnostics were recorded for this run.</p>
        <p className="m-0 mt-1 text-muted">Recording is on by default for new runs; it is off when the job sets diagnostics.trainscope to false.</p>
      </div>
    );
  const pick = (l: string) => {
    setLayer(l);
    onSelectLayer?.(l);
  };
  const open = (l: string) => {
    pick(l);
    setSub("layers");
  };
  const subs: [Sub, string][] = [
    ["overview", "Overview"],
    ["layers", "Layers"],
    ["spikes", "Spikes"],
    ...(data.moe.length ? ([["routing", "Routing"]] as [Sub, string][]) : []),
    ["weights", "Weights"],
    ["replay", "Replay"],
  ];
  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="diagnose">
      <div className="flex shrink-0 gap-4 px-4 pt-1 text-label">
        {subs.map(([k, label]) => (
          <button key={k} className="link" style={{ color: sub === k ? "var(--ink)" : undefined, fontWeight: sub === k ? 500 : 400 }} onClick={() => setSub(k)}>
            {label}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {sub === "overview" && <Overview base={base} run={run} data={data} live={live} onSelectLayer={pick} highlight={layer} />}
        {sub === "layers" && <Layers base={base} run={run} layers={data.layers} events={data.events} layer={layer} onLayer={pick} />}
        {sub === "spikes" && <Spikes base={base} run={run} events={data.events} files={data.spikes} onSelectLayer={open} />}
        {sub === "routing" && <Routing moe={data.moe} />}
        {sub === "weights" && <WeightsDiff base={base} run={run} steps={data.global.map((r) => r.step)} onSelectLayer={open} />}
        {sub === "replay" && <Replay base={base} run={run} />}
      </div>
    </div>
  );
}
