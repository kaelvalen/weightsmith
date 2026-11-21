// The standalone trainscope UI (`trainscope ui`, shipped in the PyPI wheel): the same Diagnose
// view Weightsmith shows, over the runs a trainscope server lists, without the canvas.
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "../index.css";
import { Select } from "../ui/Select";
import { scopeMulti } from "./api";
import { Diagnose } from "./Diagnose";
import { runHealth } from "./health";
import type { RunSummary } from "./types";
import { WhyPanel } from "./WhyPanel";

const BASE = "/api";
const fromHash = () => new URLSearchParams(location.hash.slice(1)).get("run");

function App() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [run, setRun] = useState<string | null>(fromHash);
  useEffect(() => {
    let alive = true;
    const load = () =>
      scopeMulti(BASE)
        .runs()
        .then((r) => alive && setRuns(r ?? []))
        .catch(() => alive && setRuns([]));
    load();
    const t = window.setInterval(load, 5000);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, []);
  const current = run && runs?.some((r) => r.name === run) ? run : (runs?.[0]?.name ?? null);
  const pick = (name: string) => {
    location.hash = `run=${encodeURIComponent(name)}`;
    setRun(name);
  };
  return (
    <div className="flex h-screen flex-col bg-bg text-body text-ink">
      <header className="flex h-10 shrink-0 items-center gap-4 border-b border-line px-4">
        <span className="font-medium">trainscope</span>
        {runs && runs.length > 0 && current && (
          <div className="w-80">
            <Select
              value={current}
              options={runs.map((r) => ({ value: r.name, hint: runHealth(r, "completed") }))}
              onChange={pick}
              ariaLabel="Run"
              mono
            />
          </div>
        )}
        <span className="text-label text-muted">{runs ? `${runs.length} run${runs.length === 1 ? "" : "s"}` : "…"}</span>
      </header>
      {runs && runs.length === 0 ? (
        <p className="m-0 p-4 text-small text-muted">
          No runs here yet. Attach <code className="num">TrainScope</code> to a training loop, or run <code className="num">trainscope demo</code>.
        </p>
      ) : (
        <div className="grid min-h-0 flex-1 lg:grid-cols-[minmax(0,1fr)_320px]">
          <main className="min-h-0">
            <Diagnose key={current ?? "none"} base={BASE} run={current} live />
          </main>
          {runs && runs.length > 1 && (
            <aside className="min-h-0 overflow-y-auto border-l border-line p-4">
              <WhyPanel base={BASE} runs={runs.map((r) => r.name)} label={(n) => n} />
            </aside>
          )}
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
