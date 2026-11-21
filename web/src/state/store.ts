// Application state. The graph IR is the only source of truth: the canvas
// renders it, and every edit goes through `apply`, which records the IR before
// and after in the operation log. Undo/redo replays that log, never canvas state.

import { create } from "zustand";
import { ApiError, RunEvent, RunRecord, api, streamRun, streamStatus } from "../api/client";
import { Analysis, analyze } from "../ir/analyze";
import examples from "../ir/examples.json";
import { Clip, copySelection, ensureLayout, pasteClip } from "../ir/ops";
import templates from "../ir/templates.json";
import type { Graph } from "../ir/types";

export interface Job {
  data: Record<string, any>;
  train: Record<string, any>;
  init?: { weights?: string };
}

export interface Variant {
  graph: Graph;
  parent: string | null;
  created: number;
  note?: string;
  job?: Job; // overrides the project's job for this variant
}

export interface Project {
  slug: string;
  name: string;
  active: string;
  variants: Record<string, Variant>;
  job: Job;
  description?: string;
}

interface LogEntry {
  label: string;
  variant: string;
  before: Graph;
  after: Graph;
  key?: string;
  at: number;
}

export type BottomTab = "metrics" | "diagnose" | "log" | "runs" | "compare" | "code";
export type View = "projects" | "nodes";

interface State {
  project: Project;
  scope: string;
  selection: string[];
  selectedEdges: [string, string][];
  past: LogEntry[];
  future: LogEntry[];
  local: Analysis;
  remote: { hash: string | null; analysis: any; graphKey: string } | null;
  daemon: "unknown" | "online" | "offline";
  runs: RunRecord[];
  events: Record<string, RunEvent[]>;
  activeRun: string | null;
  compare: string[];
  bottom: { open: boolean; tab: BottomTab; height: number };
  /** trainscope on the canvas: the run whose recording is shown, node heat (max z) and the spike source */
  diag: { run: string | null; paint: boolean; heat: Record<string, number>; source: string | null; layers: Record<string, string[]> };
  palette: boolean;
  notice: { text: string; error?: boolean } | null;
  inspectorFocus: number;
  projects: { slug: string; name: string; variants: string[]; updated: number }[];
  view: View;
  clip: Clip | null;

  graph: () => Graph;
  apply: (label: string, fn: (g: Graph) => Graph, key?: string) => void;
  undo: () => void;
  redo: () => void;
  select: (ids: string[], edges?: [string, string][]) => void;
  setScope: (scope: string) => void;
  setJob: (part: "data" | "train" | "init", value: Record<string, any>) => void;
  branchVariant: (name: string) => void;
  switchVariant: (name: string) => void;
  renameProject: (name: string) => void;
  loadProject: (p: Project) => void;
  newFromExample: (index: number) => void;
  openProject: (slug: string) => Promise<void>;
  setView: (v: View) => void;
  job: () => Job;
  setVariantJob: (own: boolean) => void;
  setNote: (text: string) => void;
  setDescription: (text: string) => void;
  newProject: (name: string, graph?: Graph, job?: Job, description?: string) => Promise<void>;
  newFromTemplate: (id: string) => Promise<void>;
  deleteProject: (slug: string) => Promise<void>;
  duplicateProject: (slug: string) => Promise<void>;
  importProject: (p: Project) => Promise<void>;
  refreshProjects: () => Promise<void>;
  copy: () => void;
  paste: () => void;
  duplicate: () => void;
  setBottom: (b: Partial<State["bottom"]>) => void;
  setDiag: (d: Partial<State["diag"]>) => void;
  setPalette: (open: boolean) => void;
  notify: (text: string, error?: boolean) => void;
  validateRemote: () => Promise<void>;
  run: () => Promise<void>;
  refreshRuns: () => Promise<void>;
  watchRun: (id: string) => void;
  toggleCompare: (id: string) => void;
}

const COALESCE_MS = 800;

function projectFromExample(i: number): Project {
  const ex = (examples as any[])[i];
  const name = ex.name as string;
  return {
    slug: name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, ""),
    name,
    active: "main",
    variants: { main: { graph: ensureLayout(ex.graph), parent: null, created: Date.now() } },
    job: { data: ex.data ?? {}, train: ex.train ?? {} },
  };
}

const graphKey = (g: Graph) => JSON.stringify({ ...g, layout: undefined, name: undefined });

let saveTimer: number | undefined;
let validateTimer: number | undefined;
const streams = new Map<string, () => void>();

export const useStore = create<State>((set, get) => {
  const initial = loadLocal() ?? projectFromExample(0);

  const persist = () => {
    clearTimeout(saveTimer);
    saveTimer = window.setTimeout(async () => {
      const p = get().project;
      try {
        localStorage.setItem("weightsmith.project", JSON.stringify(p));
      } catch {
        /* storage unavailable */
      }
      if (get().daemon === "online") await api.saveProject(p.slug, p).catch(() => undefined);
    }, 400);
  };

  const reanalyze = () => {
    const g = get().graph();
    set({ local: analyze(g) });
    clearTimeout(validateTimer);
    validateTimer = window.setTimeout(() => get().validateRemote(), 350);
  };

  const setGraph = (variant: string, g: Graph) => {
    const p = get().project;
    set({ project: { ...p, variants: { ...p.variants, [variant]: { ...p.variants[variant], graph: g } } } });
    reanalyze();
    persist();
  };

  return {
    project: initial,
    scope: "",
    selection: [],
    selectedEdges: [],
    past: [],
    future: [],
    local: analyze(initial.variants[initial.active].graph),
    remote: null,
    daemon: "unknown",
    runs: [],
    events: {},
    activeRun: null,
    compare: [],
    bottom: { open: false, tab: "metrics", height: 264 },
    diag: { run: null, paint: false, heat: {}, source: null, layers: {} },
    palette: false,
    notice: null,
    inspectorFocus: 0,
    projects: [],
    view: new URLSearchParams(location.search).get("view") === "nodes" || new URLSearchParams(location.search).get("project") ? "nodes" : "projects",
    clip: null,

    graph: () => {
      const p = get().project;
      return p.variants[p.active].graph;
    },

    apply: (label, fn, key) => {
      const { project, past } = get();
      const before = get().graph();
      const after = fn(before);
      if (after === before) return;
      const last = past[past.length - 1];
      const now = Date.now();
      let nextPast: LogEntry[];
      if (key && last && last.key === key && last.variant === project.active && now - last.at < COALESCE_MS)
        nextPast = [...past.slice(0, -1), { ...last, after, at: now }];
      else nextPast = [...past, { label, variant: project.active, before, after, key, at: now }].slice(-500);
      set({ past: nextPast, future: [] });
      setGraph(project.active, after);
    },

    undo: () => {
      const { past, future } = get();
      const e = past[past.length - 1];
      if (!e) return;
      set({ past: past.slice(0, -1), future: [...future, e], project: { ...get().project, active: e.variant } });
      setGraph(e.variant, e.before);
      get().notify(`undo · ${e.label}`);
    },

    redo: () => {
      const { past, future } = get();
      const e = future[future.length - 1];
      if (!e) return;
      set({ future: future.slice(0, -1), past: [...past, e], project: { ...get().project, active: e.variant } });
      setGraph(e.variant, e.after);
      get().notify(`redo · ${e.label}`);
    },

    select: (ids, edges = []) => {
      const cur = get();
      if (cur.selection.join() === ids.join() && cur.selectedEdges.map(String).join() === edges.map(String).join()) return;
      set({ selection: ids, selectedEdges: edges });
    },

    setScope: (scope) => set({ scope, selection: [], selectedEdges: [] }),

    setJob: (part: "data" | "train" | "init", value) => {
      const p = get().project;
      const v = p.variants[p.active];
      if (v.job) set({ project: { ...p, variants: { ...p.variants, [p.active]: { ...v, job: { ...v.job, [part]: value } } } } });
      else set({ project: { ...p, job: { ...p.job, [part]: value } } });
      persist();
    },

    setView: (v) => {
      set({ view: v });
      if (v === "projects") get().refreshProjects();
    },

    job: () => {
      const p = get().project;
      return p.variants[p.active].job ?? p.job;
    },

    setVariantJob: (own) => {
      const p = get().project;
      const v = { ...p.variants[p.active] };
      if (own) v.job = structuredClone(p.job);
      else delete v.job;
      set({ project: { ...p, variants: { ...p.variants, [p.active]: v } } });
      persist();
    },

    setNote: (text) => {
      const p = get().project;
      set({ project: { ...p, variants: { ...p.variants, [p.active]: { ...p.variants[p.active], note: text || undefined } } } });
      persist();
    },

    setDescription: (text) => {
      set({ project: { ...get().project, description: text || undefined } });
      persist();
    },

    newProject: async (name, graph, job, description) => {
      await get().refreshProjects();
      const taken = new Set([...get().projects.map((x) => x.slug), get().project.slug]);
      const base = name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "project";
      let slug = base;
      for (let i = 2; taken.has(slug); i++) slug = `${base}-${i}`;
      const g: Graph = graph ?? { schema: 1, name, nodes: [], edges: [] };
      get().loadProject({ slug, name, active: "main", variants: { main: { graph: ensureLayout(g), parent: null, created: Date.now() } }, job: job ?? { data: {}, train: {} }, description });
      set({ view: "nodes" });
      history.replaceState(null, "", `?project=${encodeURIComponent(slug)}`);
      setTimeout(() => get().refreshProjects(), 600);
    },

    newFromTemplate: async (id) => {
      const t = (templates as any[]).find((x) => x.id === id);
      if (!t?.graph) return;
      await get().newProject(t.name, structuredClone(t.graph), undefined, `${t.doc}${t.source ? ` Source: ${t.source}.` : ""}`);
    },

    deleteProject: async (slug) => {
      await api.deleteProject(slug).catch(() => undefined);
      if (get().project.slug === slug) {
        try {
          localStorage.removeItem("weightsmith.project");
        } catch {
          /* ignore */
        }
      }
      await get().refreshProjects();
    },

    duplicateProject: async (slug) => {
      const src = slug === get().project.slug ? get().project : await api.project(slug);
      const copy = structuredClone(src) as Project;
      await get().refreshProjects();
      const taken = new Set(get().projects.map((x) => x.slug));
      let s2 = `${slug}-copy`;
      for (let i = 2; taken.has(s2); i++) s2 = `${slug}-copy-${i}`;
      copy.slug = s2;
      copy.name = `${src.name} (copy)`;
      await api.saveProject(s2, copy);
      await get().refreshProjects();
    },

    importProject: async (p) => {
      if (!p?.variants) {
        get().notify("not a Weightsmith project file", true);
        return;
      }
      await get().refreshProjects();
      const taken = new Set(get().projects.map((x) => x.slug));
      let slug = p.slug || "imported";
      for (let i = 2; taken.has(slug); i++) slug = `${p.slug || "imported"}-${i}`;
      p.slug = slug;
      p.job = p.job ?? { data: {}, train: {} };
      p.active = p.variants[p.active] ? p.active : Object.keys(p.variants)[0];
      get().loadProject(p);
      set({ view: "nodes" });
    },

    refreshProjects: async () => {
      try {
        set({ projects: await api.projects(), daemon: "online" });
      } catch {
        set({ daemon: "offline" });
      }
    },

    copy: () => {
      const { selection, scope } = get();
      if (!selection.length) return;
      const clip = copySelection(get().graph(), scope, selection);
      set({ clip });
      navigator.clipboard?.writeText(JSON.stringify({ weightsmith: "nodes", ...clip })).catch(() => undefined);
      get().notify(`copied ${selection.length} node${selection.length > 1 ? "s" : ""}`);
    },

    paste: () => {
      const { clip, scope } = get();
      if (!clip?.nodes.length) return;
      let fresh: string[] = [];
      get().apply(`paste ${clip.nodes.length} nodes`, (g) => {
        const [next, ids] = pasteClip(g, scope, clip);
        fresh = ids;
        return next;
      });
      get().select(fresh);
    },

    duplicate: () => {
      get().copy();
      get().paste();
    },

    branchVariant: (name) => {
      const p = get().project;
      if (!name || p.variants[name]) return;
      const g = structuredClone(get().graph());
      set({ project: { ...p, active: name, variants: { ...p.variants, [name]: { graph: g, parent: p.active, created: Date.now() } } } });
      reanalyze();
      persist();
      get().notify(`variant ${name} branched from ${p.active}`);
    },

    switchVariant: (name) => {
      const p = get().project;
      if (!p.variants[name]) return;
      set({ project: { ...p, active: name }, selection: [], scope: "" });
      reanalyze();
      persist();
    },

    renameProject: (name) => {
      const p = get().project;
      set({ project: { ...p, name } });
      persist();
    },

    loadProject: (p) => {
      for (const v of Object.values(p.variants)) v.graph = ensureLayout(v.graph);
      set({ project: p, past: [], future: [], selection: [], scope: "", activeRun: null, compare: [] });
      reanalyze();
      get().refreshRuns();
      persist();
    },

    newFromExample: (i) => get().loadProject(projectFromExample(i)),

    openProject: async (slug) => {
      try {
        const p = await api.project(slug);
        if (!p?.variants) throw new Error("not a project");
        p.active = p.active && p.variants[p.active] ? p.active : Object.keys(p.variants)[0];
        p.job = p.job ?? { data: {}, train: {} };
        get().loadProject(p);
        history.replaceState(null, "", `?project=${encodeURIComponent(slug)}`);
        set({ view: "nodes" });
      } catch {
        get().notify(`could not open project ${slug}`, true);
      }
    },

    setBottom: (b) => set({ bottom: { ...get().bottom, ...b } }),
    setDiag: (d) => set({ diag: { ...get().diag, ...d } }),
    setPalette: (open) => set({ palette: open }),

    notify: (text, error) => {
      set({ notice: { text, error } });
      const t = text;
      setTimeout(() => {
        if (get().notice?.text === t) set({ notice: null });
      }, error ? 6000 : 2400);
    },

    validateRemote: async () => {
      const g = get().graph();
      const key = graphKey(g);
      try {
        const a = await api.validate(g);
        if (get().daemon !== "online") set({ daemon: "online" });
        if (graphKey(get().graph()) === key) set({ remote: { hash: a.hash, analysis: a, graphKey: key } });
      } catch (e) {
        if (!(e instanceof ApiError)) set({ daemon: "offline" });
      }
    },

    run: async () => {
      const { project } = get();
      const g = get().graph();
      await get().validateRemote();
      const r = get().remote;
      const fresh = r && r.graphKey === graphKey(g);
      const ok = fresh ? r!.analysis.ok : get().local.ok;
      if (!ok) {
        get().notify("graph does not validate — fix the marked ports first", true);
        return;
      }
      try {
        const job = get().job();
        const rec = await api.createRun({ graph: g, data: job.data, train: job.train, init: job.init ?? {}, project: project.slug, variant: project.active, name: `${project.name} · ${project.active}` });
        set({ activeRun: rec.id, bottom: { ...get().bottom, open: true, tab: "metrics" } });
        if (rec.existing) get().notify(`run ${rec.id.slice(0, 8)} already exists for this graph, config, data and seed`);
        get().watchRun(rec.id);
        await get().refreshRuns();
      } catch (e) {
        const msg = e instanceof ApiError ? e.message : "daemon unreachable — start it with `weightsmith serve`";
        get().notify(msg, true);
      }
    },

    refreshRuns: async () => {
      try {
        const runs = await api.runs(get().project.slug);
        set({ runs, daemon: "online" });
      } catch {
        set({ daemon: "offline" });
      }
    },

    watchRun: (id) => {
      if (streams.has(id)) return;
      const evs = get().events[id] ?? [];
      const since = evs.length ? evs[evs.length - 1].seq : 0;
      const buffer: RunEvent[] = [];
      let flush: number | undefined;
      const stop = streamRun(id, since, (ev) => {
        buffer.push(ev);
        if (flush) return;
        flush = window.setTimeout(() => {
          flush = undefined;
          const cur = get().events[id] ?? [];
          set({ events: { ...get().events, [id]: [...cur, ...buffer.splice(0)] } });
        }, 100);
      });
      streams.set(id, stop);
    },

    toggleCompare: (id) => {
      const c = get().compare;
      set({ compare: c.includes(id) ? c.filter((x) => x !== id) : [...c, id] });
      get().watchRun(id);
    },
  };
});

function loadLocal(): Project | null {
  try {
    const raw = localStorage.getItem("weightsmith.project");
    if (!raw) return null;
    const p = JSON.parse(raw) as Project;
    if (!p.variants?.[p.active]) return null;
    for (const v of Object.values(p.variants)) v.graph = ensureLayout(v.graph);
    return p;
  } catch {
    return null;
  }
}

/** Connect to the daemon: health, project list, live status of all runs. */
export async function boot() {
  const s = useStore.getState();
  try {
    await api.health();
    useStore.setState({ daemon: "online" });
  } catch {
    useStore.setState({ daemon: "offline" });
    return;
  }
  const want = new URLSearchParams(location.search).get("project");
  await s.refreshProjects();
  let saved = false;
  try {
    saved = !!localStorage.getItem("weightsmith.project");
  } catch {
    /* storage unavailable */
  }
  if (want) await s.openProject(want);
  else if (!saved && useStore.getState().projects.length) {
    // first visit: the most recently updated saved project, shown from the Projects tab
    const view = useStore.getState().view;
    await s.openProject(useStore.getState().projects[0].slug);
    useStore.setState({ view });
  }
  await s.refreshRuns();
  s.validateRemote();
  streamStatus((ev) => {
    const st = useStore.getState();
    if (ev.type === "status" || ev.type === "error" || ev.type === "eval") st.refreshRuns();
  });
}

/** Errors and warnings, preferring the daemon's analysis when it matches the current graph. */
export function useIssues() {
  return useStore((s) => {
    const g = s.graph();
    const fresh = s.remote && s.remote.graphKey === graphKey(g);
    return fresh ? (s.remote!.analysis.issues as Analysis["issues"]) : s.local.issues;
  });
}

export function useAnalysis(): Analysis {
  return useStore((s) => {
    const g = s.graph();
    const r = s.remote;
    if (r && r.graphKey === graphKey(g) && r.analysis.signatures && Object.keys(r.analysis.signatures).length) return r.analysis as Analysis;
    return s.local;
  });
}
