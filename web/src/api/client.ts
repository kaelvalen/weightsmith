// Thin client for the Weightsmith daemon (runtime/weightsmith_runtime/server.py).

import type { Graph } from "../ir/types";

export interface RunRecord {
  id: string;
  project: string | null;
  variant: string | null;
  name: string | null;
  graph_hash: string;
  config_hash: string;
  data_hash: string;
  seed: number;
  status: string;
  parent: string | null;
  ablation: { nodes: string[]; adapt: { node: string; index: number }[] } | null;
  created: number;
  updated: number;
  summary: Record<string, any>;
  error: Record<string, any> | null;
  existing?: boolean;
  job?: { graph: Graph; data: any; train: any; hashes: Record<string, string> };
  children?: string[];
}

export interface RunEvent {
  seq: number;
  ts: number;
  run: string;
  type: "status" | "metric" | "eval" | "log" | "error" | "checkpoint";
  [k: string]: any;
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method,
    headers: body !== undefined ? { "content-type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let detail: any = await r.text();
    try {
      detail = JSON.parse(detail).detail ?? detail;
    } catch {
      /* plain text */
    }
    throw new ApiError(r.status, detail);
  }
  const ct = r.headers.get("content-type") ?? "";
  return (ct.includes("json") ? r.json() : r.text()) as Promise<T>;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: any,
  ) {
    super(typeof detail === "string" ? detail : (detail?.message ?? `HTTP ${status}`));
  }
}

export const api = {
  health: () => req<{ ok: boolean; workspace: string }>("GET", "/api/health"),
  validate: (graph: Graph) => req<any>("POST", "/api/validate", { graph }),
  compile: (graph: Graph) => req<any>("POST", "/api/compile", { graph }),
  diff: (a: Graph, b: Graph) => req<{ same: boolean; changes: any[] }>("POST", "/api/diff", { a, b }),
  python: (params: any, inputs?: any) => req<any>("POST", "/api/python/analyze", { params, inputs }),
  projects: () =>
    req<{ slug: string; name: string; variants: string[]; updated: number; description?: string; params?: number; trainable?: number; thumb?: number[][] }[]>(
      "GET",
      "/api/projects",
    ),
  project: (slug: string) => req<any>("GET", `/api/projects/${slug}`),
  saveProject: (slug: string, p: any) => req<{ slug: string }>("PUT", `/api/projects/${slug}`, p),
  deleteProject: (slug: string) => req<any>("DELETE", `/api/projects/${slug}`),
  runs: (project?: string) => req<RunRecord[]>("GET", `/api/runs${project ? `?project=${encodeURIComponent(project)}` : ""}`),
  run: (id: string) => req<RunRecord>("GET", `/api/runs/${id}`),
  deleteRun: (id: string) => req<any>("DELETE", `/api/runs/${id}`),
  createRun: (body: { graph: Graph; data: any; train: any; project?: string; variant?: string; name?: string; rerun?: boolean }) =>
    req<RunRecord>("POST", "/api/runs", body),
  events: (id: string, since = 0) => req<RunEvent[]>("GET", `/api/runs/${id}/events?since=${since}`),
  control: (id: string, cmd: string) => req<RunRecord>("POST", `/api/runs/${id}/control`, { cmd }),
  ablate: (id: string, nodes: string[], adapt: { node: string; index: number }[] = []) =>
    req<RunRecord>("POST", `/api/runs/${id}/ablate`, { nodes, adapt }),
  file: (id: string, name: string) => req<string>("GET", `/api/runs/${id}/files/${name}`),
  exportUrl: (id: string, merge = false) => `/api/runs/${id}/export${merge ? "?merge=true" : ""}`,
};

/** Sequence-numbered run stream that resumes from the last seen seq after a drop. */
export function streamRun(id: string, since: number, onEvent: (e: RunEvent) => void): () => void {
  let last = since;
  let ws: WebSocket | null = null;
  let closed = false;
  let retry = 250;
  const open = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/runs/${id}?since=${last}`);
    ws.onmessage = (m) => {
      const ev = JSON.parse(m.data) as RunEvent;
      if (ev.seq <= last) return;
      last = ev.seq;
      retry = 250;
      onEvent(ev);
    };
    ws.onclose = () => {
      if (!closed) setTimeout(open, (retry = Math.min(retry * 2, 4000)));
    };
  };
  open();
  return () => {
    closed = true;
    ws?.close();
  };
}

export function streamStatus(onEvent: (e: RunEvent) => void): () => void {
  let ws: WebSocket | null = null;
  let closed = false;
  const open = () => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/status`);
    ws.onmessage = (m) => onEvent(JSON.parse(m.data));
    ws.onclose = () => {
      if (!closed) setTimeout(open, 1500);
    };
  };
  open();
  return () => {
    closed = true;
    ws?.close();
  };
}
