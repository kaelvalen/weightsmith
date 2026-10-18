import { useState } from "react";
import { useIssues, useStore } from "../state/store";
import { Field } from "./Inspector";

export function TopBar() {
  const project = useStore((s) => s.project);
  const rename = useStore((s) => s.renameProject);
  const switchVariant = useStore((s) => s.switchVariant);
  const branch = useStore((s) => s.branchVariant);
  const run = useStore((s) => s.run);
  const validateRemote = useStore((s) => s.validateRemote);
  const notify = useStore((s) => s.notify);
  const daemon = useStore((s) => s.daemon);
  const local = useStore((s) => s.local);
  const issues = useIssues();
  const runs = useStore((s) => s.runs);
  const live = runs.some((r) => r.status === "running" || r.status === "preparing");
  const [branching, setBranching] = useState(false);
  const errors = issues.filter((i) => i.severity === "error").length;
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);

  const tabs = (
    <nav className="flex h-full items-stretch gap-4" aria-label="Views">
      {(["projects", "nodes"] as const).map((v) => (
        <button
          key={v}
          className="link !h-full !text-[13px]"
          style={{ color: view === v ? "var(--ink)" : undefined, fontWeight: view === v ? 500 : 400, boxShadow: view === v ? "inset 0 -1px 0 var(--ink)" : undefined }}
          onClick={() => setView(v)}
          aria-current={view === v ? "page" : undefined}
        >
          {v === "projects" ? "Projects" : "Nodes"}
        </button>
      ))}
    </nav>
  );

  if (view === "projects")
    return (
      <header className="flex h-10 shrink-0 items-center gap-6 border-b border-line px-4">
        {tabs}
        <span className="num ml-auto text-label text-muted">{daemon === "online" ? "daemon" : daemon === "offline" ? "daemon offline" : ""}</span>
      </header>
    );

  return (
    <header className="flex h-10 shrink-0 items-center gap-4 border-b border-line px-4">
      {tabs}
      <span className="h-4 w-px bg-line" aria-hidden />
      <div
        className="min-w-[208px] shrink-0 text-project leading-6 font-medium [&_input]:h-8 [&_input]:border-transparent [&_input]:px-0 [&_input:focus]:border-accent"
        style={{ width: `${Math.min(36, Math.max(12, project.name.length + 1))}ch` }}
      >
        <Field value={project.name} onCommit={(v) => v && rename(v)} />
      </div>
      <div className="flex items-center gap-1">
        {branching ? (
          <span className="w-40">
            <Field
              value=""
              placeholder="new variant name"
              onCommit={(v) => {
                setBranching(false);
                if (v) branch(v.trim().replace(/\s+/g, "-"));
              }}
            />
          </span>
        ) : (
          <>
            <select className="w-40" value={project.active} onChange={(e) => (e.target.value === "+" ? setBranching(true) : switchVariant(e.target.value))} aria-label="variant">
              {Object.keys(project.variants).map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
              <option value="+">+ branch variant…</option>
            </select>
          </>
        )}
      </div>
      <span className="num text-label text-muted" title={daemon === "offline" ? "start it with: weightsmith serve" : ""}>
        {daemon === "online" ? "daemon" : daemon === "offline" ? "daemon offline" : ""}
      </span>
      <span className="ml-auto flex items-center gap-2">
        {errors > 0 && <span className="text-small text-error">{errors} error{errors > 1 ? "s" : ""}</span>}
        <button
          onClick={async () => {
            await validateRemote();
            const r = useStore.getState().remote;
            const n = (r?.analysis?.issues ?? local.issues).filter((i: any) => i.severity === "error").length;
            notify(n ? `${n} error${n > 1 ? "s" : ""} — see the marked ports` : `valid · graph ${r?.hash?.slice(0, 12) ?? ""}`, n > 0);
          }}
        >
          Validate
        </button>
        <button className="primary" onClick={run} title="Cmd+Enter" style={live ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}>
          {live ? "● Run" : "Run"}
        </button>
      </span>
    </header>
  );
}
