// Cmd+K: every command and every node type. There is no toolbar; this is it.

import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import templates from "../ir/templates.json";
import { autoLayout, bulkAdapt, getBody, groupIntoSubgraph, moveNodes } from "../ir/ops";
import { dissolve } from "../ir/smart";
import { CATALOG } from "../ir/registry";
import { useAnalysis, useStore } from "../state/store";
import { addAtCenter, insertTemplate } from "./Library";
import { Modal } from "./Modal";

interface Command {
  id: string;
  label: string;
  hint?: string;
  run: () => void;
}

function bulk(label: string, types: string[] | null, targets: string[], kind: "lora" | "adapter" = "lora") {
  const st = useStore.getState();
  let touched: string[] = [];
  st.apply(label, (g) => {
    const [next, t] = bulkAdapt(g, kind === "lora" ? { kind, r: 8, alpha: 16 } : { kind, bottleneck: 64 }, targets, types);
    touched = t;
    return next;
  });
  st.notify(touched.length ? `${label}: ${touched.length} target${touched.length > 1 ? "s" : ""}` : `${label}: nothing to add`);
}

export function useCommands(): Command[] {
  const st = useStore();
  const analysis = useAnalysis();
  return useMemo(() => {
    const g = st.graph();
    const cmds: Command[] = [
      { id: "help", label: "How Weightsmith works", hint: "?", run: () => window.dispatchEvent(new CustomEvent("ws:help")) },
      { id: "view:projects", label: "Go to Projects", run: () => st.setView("projects") },
      { id: "view:nodes", label: "Go to Nodes (canvas)", run: () => st.setView("nodes") },
      { id: "run", label: "Run", hint: "⌘↵", run: () => st.run() },
      { id: "fit", label: "Fit graph to view", hint: "F", run: () => window.dispatchEvent(new CustomEvent("ws:fit")) },
      { id: "minimap", label: "Toggle minimap", hint: "M", run: () => window.dispatchEvent(new CustomEvent("ws:minimap")) },
      { id: "paste", label: "Paste nodes", hint: "⌘V", run: st.paste },
      { id: "edgeshapes", label: "Toggle shapes on edges", run: () => window.dispatchEvent(new CustomEvent("ws:edgeshapes")) },
      { id: "validate", label: "Validate", run: () => st.validateRemote() },
      { id: "undo", label: "Undo", hint: "⌘Z", run: st.undo },
      { id: "redo", label: "Redo", hint: "⇧⌘Z", run: st.redo },
      { id: "code", label: "Show generated code", run: () => st.setBottom({ open: true, tab: "code" }) },
      { id: "runs", label: "Show runs", run: () => st.setBottom({ open: true, tab: "runs" }) },
      { id: "compare", label: "Compare selected runs", run: () => st.setBottom({ open: true, tab: "compare" }) },
      { id: "strip", label: st.bottom.open ? "Hide bottom strip" : "Show bottom strip", run: () => st.setBottom({ open: !st.bottom.open }) },
      {
        id: "branch",
        label: "Branch variant from current",
        run: () => {
          const base = st.project.active.replace(/-\d+$/, "");
          let i = 2;
          while (st.project.variants[`${base}-${i}`]) i++;
          st.branchVariant(`${base}-${i}`);
        },
      },
      ...Object.keys(st.project.variants)
        .filter((v) => v !== st.project.active)
        .map((v) => ({ id: `variant:${v}`, label: `Switch to variant ${v}`, run: () => st.switchVariant(v) })),
      { id: "lora-qv", label: "LoRA on every attention q/v", run: () => bulk("LoRA q/v", ["Attention"], ["q_proj", "v_proj"]) },
      { id: "lora-qkvo", label: "LoRA on every attention q/k/v/o", run: () => bulk("LoRA q/k/v/o", ["Attention"], ["q_proj", "k_proj", "v_proj", "o_proj"]) },
      { id: "lora-linear", label: "LoRA on every Linear", run: () => bulk("LoRA Linear", ["Linear"], ["weight"]) },
      { id: "lora-mlp", label: "LoRA on every MLP fc1/fc2", run: () => bulk("LoRA MLP", ["MLP"], ["fc1", "fc2"]) },
      { id: "adapter-mlp", label: "Bottleneck adapter after every MLP", run: () => bulk("adapters", ["MLP"], ["out"], "adapter") },
      {
        id: "freeze-base",
        label: "Freeze everything except modifiers",
        run: () =>
          st.apply("freeze base", (gr) => {
            const next = structuredClone(gr);
            for (const n of [...next.nodes, ...Object.values(next.subgraphs ?? {}).flatMap((s) => s.nodes)])
              if (!["Input", "Output", "CrossEntropy", "MSE", "Subgraph"].includes(n.type) && !n.adapt?.length) n.trainable = false;
            return next;
          }),
      },
      {
        id: "unfreeze",
        label: "Unfreeze everything",
        run: () =>
          st.apply("unfreeze", (gr) => {
            const next = structuredClone(gr);
            for (const n of [...next.nodes, ...Object.values(next.subgraphs ?? {}).flatMap((s) => s.nodes)]) {
              delete n.trainable;
              delete n.frozen_params;
            }
            return next;
          }),
      },
      {
        id: "tidy",
        label: "Tidy layout",
        run: () => st.apply("tidy layout", (gr) => moveNodes(gr, st.scope, autoLayout({ ...getBody(gr, st.scope), layout: {} }, false))),
      },
      { id: "theme", label: "Toggle light/dark", run: () => toggleTheme() },
    ];
    if (st.selection.length) {
      cmds.push({ id: "copy", label: `Copy ${st.selection.join(", ")}`, hint: "⌘C", run: st.copy });
      cmds.push({ id: "dup", label: `Duplicate ${st.selection.join(", ")}`, hint: "⌘D", run: st.duplicate });
      cmds.push({ id: "delete", label: `Delete ${st.selection.join(", ")}`, hint: "⌫", run: () => st.apply("delete", (gr) => dissolve(gr, st.scope, st.selection)) });
      if (!st.scope)
        cmds.push({
          id: "group",
          label: "Group selection into a block (subgraph)",
          run: () => {
            let i = 1;
            while (g.subgraphs?.[`block${i}`] || g.nodes.some((n) => n.id === `block${i}`)) i++;
            st.apply("group into block", (gr) => groupIntoSubgraph(gr, st.selection, `block${i}`, (ref) => analysis.ports?.[ref] as any));
            st.select([`block${i}`]);
          },
        });
      const active = st.runs.find((r) => r.id === st.activeRun && r.status === "completed");
      if (active)
        cmds.push({
          id: "ablate",
          label: `Ablate: re-run ${active.id.slice(0, 8)} without ${st.selection.join(", ")}`,
          run: async () => {
            try {
              const child = await api.ablate(active.id, st.selection.map((id) => (st.scope ? `${st.scope}/${id}` : id)));
              useStore.setState({ activeRun: child.id });
              st.watchRun(child.id);
              st.refreshRuns();
              st.setBottom({ open: true, tab: "metrics" });
            } catch (e: any) {
              st.notify(e.message, true);
            }
          },
        });
    }
    if (st.scope) cmds.push({ id: "exit", label: "Leave block", hint: "esc", run: () => st.setScope("") });
    const goto = (scope: string, id: string) => {
      if (st.scope !== scope) st.setScope(scope);
      setTimeout(() => {
        useStore.getState().select([id]);
        window.dispatchEvent(new CustomEvent("ws:focus", { detail: id }));
      }, 0);
    };
    for (const n of g.nodes) cmds.push({ id: `goto:/${n.id}`, label: `Go to ${n.label || n.id}`, hint: n.type, run: () => goto("", n.id) });
    for (const [name, b] of Object.entries(g.subgraphs ?? {}))
      for (const n of b.nodes) cmds.push({ id: `goto:${name}/${n.id}`, label: `Go to ${name} / ${n.label || n.id}`, hint: n.type, run: () => goto(name, n.id) });
    for (const name of Object.keys(g.subgraphs ?? {})) if (name !== st.scope) cmds.push({ id: `enter:${name}`, label: `Open block ${name}`, run: () => st.setScope(name) });
    for (const pr of st.projects)
      if (pr.slug !== st.project.slug) cmds.push({ id: `project:${pr.slug}`, label: `Open project ${pr.name}`, hint: `${pr.variants.length} variants`, run: () => st.openProject(pr.slug) });
    for (const t of templates as any[]) {
      if (t.kind === "model") {
        cmds.push({ id: `newmodel:${t.id}`, label: `New project from model: ${t.name}`, hint: t.category, run: () => st.newFromTemplate(t.id) });
        cmds.push({ id: `insmodel:${t.id}`, label: `Insert model inline: ${t.name}`, hint: "model", run: () => insertTemplate(t.id) });
      } else cmds.push({ id: `block:${t.id}`, label: `Add block: ${t.name}`, hint: "block", run: () => insertTemplate(t.id) });
    }
    for (const t of CATALOG) cmds.push({ id: `add:${t.type}`, label: `Add ${t.type === "PythonModule" ? "Python module" : t.type}`, hint: t.category, run: () => addAtCenter(t.type) });
    return cmds;
  }, [st, analysis]);
}

export function toggleTheme() {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try {
    localStorage.setItem("weightsmith.theme", root.dataset.theme);
  } catch {
    /* ignore */
  }
}

function score(label: string, q: string): number {
  if (!q) return 1;
  const l = label.toLowerCase();
  const words = q.toLowerCase().split(/\s+/).filter(Boolean);
  let s = 0;
  for (const w of words) {
    const i = l.indexOf(w);
    if (i < 0) return 0;
    s += i === 0 || l[i - 1] === " " ? 3 : 1;
  }
  return s;
}

export function Palette() {
  const open = useStore((s) => s.palette);
  const setPalette = useStore((s) => s.setPalette);
  const commands = useCommands();
  const [q, setQ] = useState("");
  const [i, setI] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (open) {
      api.projects().then((projects) => useStore.setState({ projects })).catch(() => undefined);
      setQ("");
      setI(0);
      setTimeout(() => input.current?.focus(), 0);
    }
  }, [open]);
  const hits = useMemo(
    () =>
      commands
        .map((c) => ({ c, s: score(c.label, q) }))
        .filter((x) => x.s > 0)
        .sort((a, b) => b.s - a.s)
        .slice(0, 12)
        .map((x) => x.c),
    [commands, q],
  );
  if (!open) return null;
  const go = (c?: Command) => {
    setPalette(false);
    c?.run();
  };
  return (
    <Modal onClose={() => setPalette(false)} width={560}>
      <input
        ref={input}
        autoFocus
        type="text"
        className="!h-10 !border-0 !border-b !border-line !px-4"
        placeholder="Add a node or run a command"
        value={q}
        aria-label="Command"
        aria-controls="palette-list"
        onChange={(e) => {
          setQ(e.target.value);
          setI(0);
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") (e.preventDefault(), setI((x) => Math.min(x + 1, hits.length - 1)));
          if (e.key === "ArrowUp") (e.preventDefault(), setI((x) => Math.max(x - 1, 0)));
          if (e.key === "Enter") go(hits[i]);
        }}
      />
      <ul id="palette-list" role="listbox" className="m-0 max-h-[384px] list-none overflow-y-auto p-0 py-1">
        {hits.map((c, k) => (
          <li
            key={c.id}
            role="option"
            aria-selected={k === i}
            className="flex h-8 cursor-pointer items-center justify-between px-4"
            style={{ boxShadow: k === i ? "inset 2px 0 0 var(--accent)" : undefined, background: k === i ? "color-mix(in srgb, var(--ink) 4%, transparent)" : undefined }}
            onMouseEnter={() => setI(k)}
            onClick={() => go(c)}
          >
            <span className="truncate">{c.label}</span>
            {c.hint && <span className="num shrink-0 pl-4 text-label text-muted">{c.hint}</span>}
          </li>
        ))}
        {!hits.length && <li className="px-4 py-2 text-muted">No match.</li>}
      </ul>
      <div className="flex gap-4 border-t border-line px-4 py-2 text-label text-muted">
        <span><kbd>↑</kbd> <kbd>↓</kbd> choose</span>
        <span><kbd>↵</kbd> run</span>
        <span><kbd>esc</kbd> close</span>
        <span className="num ml-auto">{hits.length} of {commands.length}</span>
      </div>
    </Modal>
  );
}
