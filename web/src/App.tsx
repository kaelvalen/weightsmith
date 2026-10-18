import { useEffect, useState } from "react";
import { boot, useStore } from "./state/store";
import { Bottom } from "./ui/Bottom";
import { Canvas } from "./ui/Canvas";
import { Inspector } from "./ui/Inspector";
import { Library } from "./ui/Library";
import { Palette } from "./ui/Palette";
import { ProjectsView } from "./ui/ProjectsView";
import { TooltipLayer } from "./ui/TooltipLayer";
import { Help } from "./ui/Help";
import { TopBar } from "./ui/TopBar";

function typing(e: KeyboardEvent) {
  const t = e.target as HTMLElement;
  return t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable || !!t.closest(".monaco-editor");
}

export default function App() {
  const notice = useStore((s) => s.notice);
  const view = useStore((s) => s.view);
  const [help, setHelp] = useState(false);

  useEffect(() => {
    boot();
    const onKey = (e: KeyboardEvent) => {
      const st = useStore.getState();
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "k") {
        e.preventDefault();
        st.setPalette(!st.palette);
      } else if (mod && e.key === "Enter") {
        e.preventDefault();
        st.run();
      } else if (typing(e)) {
        return;
      } else if (mod && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) st.redo();
        else st.undo();
      } else if (mod && e.key.toLowerCase() === "y") {
        e.preventDefault();
        st.redo();
      } else if (e.key === "?") {
        setHelp(true);
      } else if (st.view !== "nodes") {
        return;
      } else if (mod && e.key.toLowerCase() === "c") {
        st.copy();
      } else if (mod && e.key.toLowerCase() === "v") {
        st.paste();
      } else if (mod && e.key.toLowerCase() === "d") {
        e.preventDefault();
        st.duplicate();
      } else if (!mod && e.key.toLowerCase() === "f") {
        window.dispatchEvent(new CustomEvent("ws:fit"));
      } else if (!mod && e.key.toLowerCase() === "m") {
        window.dispatchEvent(new CustomEvent("ws:minimap"));
      } else if (e.key === "Enter" && st.selection.length === 1) {
        e.preventDefault();
        const node = st.graph();
        const body = st.scope ? node.subgraphs?.[st.scope] : node;
        const n = body?.nodes.find((x) => x.id === st.selection[0]);
        if (n?.type === "Subgraph" && e.shiftKey) st.setScope((n.params as any).graph);
        else useStore.setState({ inspectorFocus: st.inspectorFocus + 1 });
      } else if (e.key === "Escape") {
        if (st.selection.length) st.select([]);
        else if (st.scope) st.setScope("");
      }
    };
    const onCommand = (e: Event) => {
      if ((e as CustomEvent).detail === "group") useStore.getState().setPalette(true);
    };
    const openHelp = () => setHelp(true);
    window.addEventListener("keydown", onKey);
    window.addEventListener("ws:command", onCommand);
    window.addEventListener("ws:help", openHelp);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("ws:command", onCommand);
      window.removeEventListener("ws:help", openHelp);
    };
  }, []);

  return (
    <div className="flex h-full flex-col">
      <TopBar />
      {view === "projects" ? (
        <main className="min-h-0 flex-1">
          <ProjectsView />
        </main>
      ) : (
        <>
          <div className="flex min-h-0 flex-1">
            <Library />
            <main className="relative min-w-0 flex-1">
              <Canvas />
            </main>
            <Inspector />
          </div>
          <Bottom />
        </>
      )}
      {notice && (
        <div
          role="status"
          className={`fixed bottom-10 left-1/2 z-40 -translate-x-1/2 border bg-bg px-3 py-1 text-small ${notice.error ? "border-error text-error" : "border-ink"}`}
          style={{ borderRadius: 2 }}
        >
          {notice.text}
        </div>
      )}
      <Palette />
      {help && <Help onClose={() => setHelp(false)} />}
      <TooltipLayer />
    </div>
  );
}
