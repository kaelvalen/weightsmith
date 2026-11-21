import { useState } from "react";
import { Select } from "./Select";
import { useIssues, useStore } from "../state/store";
import { Field } from "./Inspector";
import { Modal } from "./Modal";

function BranchModal({ from, taken, onClose, onCreate }: { from: string; taken: string[]; onClose: () => void; onCreate: (name: string) => void }) {
  const [name, setName] = useState(`${from.replace(/-\d+$/, "")}-2`);
  const clean = name.trim().replace(/\s+/g, "-");
  const clash = taken.includes(clean);
  const create = () => {
    if (!clean || clash) return;
    onCreate(clean);
    onClose();
  };
  return (
    <Modal
      title="Branch a variant"
      onClose={onClose}
      width={400}
      footer={
        <>
          <button onClick={onClose}>Cancel</button>
          <button className="primary" onClick={create} disabled={!clean || clash}>
            Branch
          </button>
        </>
      }
    >
      <div className="flex flex-col gap-2 px-4 pb-3">
        <p className="m-0 text-small text-muted">A copy of {from}'s graph. The diff against {from} shows in the inspector.</p>
        <input id="variant-name" autoFocus type="text" value={name} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && create()} aria-label="Variant name" />
        {clash && <span className="text-small text-error">A variant named {clean} already exists.</span>}
      </div>
    </Modal>
  );
}

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

  if (view === "projects")
    return (
      <header className="flex h-10 shrink-0 items-center gap-6 border-b border-line px-4">
        <span className="font-medium">Weightsmith</span>
        <span className="num ml-auto text-label text-muted">{daemon === "online" ? "daemon" : daemon === "offline" ? "daemon offline" : ""}</span>
      </header>
    );

  return (
    <header className="flex h-10 shrink-0 items-center gap-4 border-b border-line px-4">
      <div
        className="min-w-[208px] shrink-0 text-project leading-6 font-medium [&_input]:h-8 [&_input]:border-transparent [&_input]:px-0 [&_input:focus]:border-accent"
        style={{ width: `${Math.min(36, Math.max(12, project.name.length + 1))}ch` }}
      >
        <Field value={project.name} onCommit={(v) => v && rename(v)} />
      </div>
      <div className="flex items-center gap-1">
        <div className="w-44">
          <Select
            value={project.active}
            options={[...Object.keys(project.variants).map((v) => ({ value: v, hint: project.variants[v].parent ? `from ${project.variants[v].parent}` : undefined })), { value: "+", label: "+ branch variant…" }]}
            onChange={(v) => (v === "+" ? setBranching(true) : switchVariant(v))}
            ariaLabel="variant"
          />
        </div>
      </div>
      {branching && <BranchModal from={project.active} taken={Object.keys(project.variants)} onClose={() => setBranching(false)} onCreate={(n) => branch(n)} />}
      <span className="num text-label text-muted" title={daemon === "offline" ? "start it with: weightsmith serve" : ""}>
        {daemon === "online" ? "daemon" : daemon === "offline" ? "daemon offline" : ""}
      </span>
      <span className="ml-auto flex items-center gap-2">
        <button className="link !text-[13px]" onClick={() => window.dispatchEvent(new CustomEvent("ws:help"))} title="How Weightsmith works (?)">
          Help
        </button>
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
