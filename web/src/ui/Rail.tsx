// The left rail: the start page and the five stages of the flow, in order. Keys 1–5 switch
// stages. Status marks (an error in Data, a live run, a spike) are drawn by CSS so the labels
// stay plain words.
import { useStore, STAGES, View } from "../state/store";

const LABELS: Record<View, string> = { projects: "Projects", model: "Model", data: "Data", train: "Train", runs: "Runs", diagnose: "Diagnose" };
const HINTS: Record<View, string> = {
  projects: "Start a project, open one, or import a pretrained model",
  model: "Build the graph (1)",
  data: "Where the data comes from and what the model will see (2)",
  train: "Optimizer, schedule, precision, and Run (3)",
  runs: "Every run, its curves, log and comparisons (4)",
  diagnose: "Why a run spiked: layers, signals, the node that drifted first (5)",
};

export function Rail() {
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);
  const runs = useStore((s) => s.runs);
  const live = runs.some((r) => r.status === "running" || r.status === "preparing");
  const spiked = useStore((s) => {
    const r = s.runs.find((x) => x.id === s.activeRun);
    return !!r?.summary?.spikes;
  });
  const mark: Partial<Record<View, string>> = {
    runs: live ? "ws-mark-live" : "",
    diagnose: spiked ? "ws-mark-alert" : "",
  };
  const item = (v: View) => (
    <button
      key={v}
      className={`ws-rail-item ${mark[v] ?? ""}`}
      aria-current={view === v ? "page" : undefined}
      data-tip={HINTS[v]}
      aria-keyshortcuts={v === "projects" ? undefined : String(STAGES.indexOf(v as (typeof STAGES)[number]) + 1)}
      onClick={() => setView(v)}
    >
      {LABELS[v]}
    </button>
  );
  return (
    <nav aria-label="Stages" className="flex w-[96px] shrink-0 flex-col border-r border-line py-2">
      {item("projects")}
      <span className="mx-3 my-2 h-px bg-line" aria-hidden />
      {STAGES.map(item)}
    </nav>
  );
}
