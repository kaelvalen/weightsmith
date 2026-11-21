// The strip at the bottom of every stage: the run that matters now (a live one, else the
// selected one) with its progress, loss and spikes, and the controls to steer it.
import { api } from "../api/client";
import { useStore } from "../state/store";
import { Status } from "./RunViews";

const f4 = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toPrecision(4)) : "—");

export function RunStrip() {
  const runs = useStore((s) => s.runs);
  const activeRun = useStore((s) => s.activeRun);
  const setView = useStore((s) => s.setView);
  const refresh = useStore((s) => s.refreshRuns);
  const notify = useStore((s) => s.notify);
  const run = runs.find((r) => r.status === "running" || r.status === "preparing") ?? runs.find((r) => r.id === activeRun);
  const cmd = async (c: string) => {
    try {
      await api.control(run!.id, c);
      refresh();
    } catch (e: any) {
      notify(e.message, true);
    }
  };
  if (!run)
    return (
      <footer data-testid="run-strip" className="flex h-8 shrink-0 items-center gap-3 border-t border-line px-4 text-small text-muted">
        Nothing trained yet · <span className="num">⌘↵</span> runs the current variant
      </footer>
    );
  const s = run.status;
  const step = run.summary?.last?.step ?? 0;
  const total = run.summary?.total_steps;
  const spikes = run.summary?.spikes ?? 0;
  return (
    <footer data-testid="run-strip" className="flex h-8 shrink-0 items-center gap-4 border-t border-line px-4 text-small">
      <Status s={s} />
      <button className="link num" onClick={() => (useStore.setState({ activeRun: run.id }), setView("runs"))} title="Open in Runs">
        {run.id.slice(0, 8)}
        {run.variant ? ` · ${run.variant}` : ""}
      </button>
      <span className="num text-muted">
        step {step}
        {total ? `/${total}` : ""}
      </span>
      {total ? (
        <span className="h-px w-32 bg-line" aria-hidden>
          <span className="block h-px bg-accent" style={{ width: `${Math.min(100, (100 * step) / total)}%` }} />
        </span>
      ) : null}
      <span className="num">loss {f4(run.summary?.last?.loss)}</span>
      {spikes > 0 && (
        <button className="link num" style={{ color: "var(--error)" }} onClick={() => (useStore.setState({ activeRun: run.id }), setView("diagnose"))}>
          spiked at step {run.summary?.first_spike_step} · why?
        </button>
      )}
      {run.error?.message && <span className="truncate text-error">[{run.error.class}] {run.error.message}</span>}
      <span className="ml-auto flex items-center gap-2">
        {(s === "running" || s === "preparing") && <button onClick={() => cmd("pause")}>Pause</button>}
        {s === "paused" && <button onClick={() => cmd("resume")}>Resume</button>}
        {["running", "paused", "queued", "preparing"].includes(s) && <button onClick={() => cmd("stop")}>Stop</button>}
      </span>
    </footer>
  );
}
