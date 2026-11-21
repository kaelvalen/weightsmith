// Model stage: the library (can be folded away), the graph or its generated code, and the
// inspector for whatever is selected (with nothing selected: readiness, project, variants).
import { useState } from "react";
import { useStore } from "../state/store";
import { Canvas } from "./Canvas";
import { Inspector } from "./Inspector";
import { Library } from "./Library";
import { CodeView } from "./RunViews";

const KEY = "weightsmith.library";

export function ModelStage() {
  const modelView = useStore((s) => s.modelView);
  const setModelView = useStore((s) => s.setModelView);
  const [library, setLibrary] = useState(() => {
    try {
      return localStorage.getItem(KEY) !== "closed";
    } catch {
      return true;
    }
  });
  const toggle = () => {
    setLibrary(!library);
    try {
      localStorage.setItem(KEY, library ? "closed" : "open");
    } catch {
      /* storage unavailable */
    }
  };
  const seg = (v: "graph" | "code", label: string) => (
    <button
      className="link !h-full"
      aria-pressed={modelView === v}
      style={{ color: modelView === v ? "var(--ink)" : undefined, fontWeight: modelView === v ? 500 : 400, boxShadow: modelView === v ? "inset 0 -1px 0 var(--ink)" : undefined }}
      onClick={() => setModelView(v)}
    >
      {label}
    </button>
  );
  return (
    <div className="flex h-full min-h-0">
      {library && <Library />}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-8 shrink-0 items-stretch gap-4 border-b border-line px-4 text-small">
          <button className="link" onClick={toggle} aria-pressed={library} data-tip="Show or hide the node library (⌘K adds nodes too)">
            {library ? "◧ hide library" : "◧ library"}
          </button>
          <span className="my-2 w-px bg-line" aria-hidden />
          {seg("graph", "Graph")}
          {seg("code", "Code")}
        </div>
        <main className="relative min-h-0 flex-1">{modelView === "graph" ? <Canvas /> : <CodeView />}</main>
      </div>
      <Inspector />
    </div>
  );
}
