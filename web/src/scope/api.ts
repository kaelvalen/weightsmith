// Client for trainscope's run-keyed API. Weightsmith mounts it at daemonBase() + "/api/scope";
// the standalone trainscope UI at "/api". A run without a recording answers null (empty state).
import type { GlobalRow, LayerMatrix, RunSummary } from "./types";

async function get<T>(url: string): Promise<T | null> {
  const r = await fetch(url);
  if (r.status === 404) return null;
  if (!r.ok) {
    let detail = r.statusText;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(`${r.status} ${detail}`);
  }
  return r.json() as Promise<T>;
}

const q = (params: Record<string, string | number | undefined | null>) => {
  const s = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v != null) s.set(k, String(v));
  const out = s.toString();
  return out ? `?${out}` : "";
};

export function scopeApi(base: string, run: string) {
  const r = `${base}/runs/${encodeURIComponent(run)}`;
  const layer = (name: string) => encodeURIComponent(name);
  return {
    meta: () => get<Record<string, any>>(`${r}/meta`),
    manifest: () => get<Record<string, any>>(`${r}/manifest`),
    global: () => get<GlobalRow[]>(`${r}/global`),
    layers: () => get<string[]>(`${r}/layers`),
    layersRanked: (topN = 8) => get<string[]>(`${r}/layers/ranked${q({ top_n: topN })}`),
    layerMatrix: (p: { metric?: string; buckets?: number; start?: number; end?: number } = {}) => get<LayerMatrix>(`${r}/layers/matrix${q(p)}`),
    signals: () => get<Record<string, any>>(`${r}/signals`),
    layer: (name: string) => get<Record<string, any>[]>(`${r}/layers/${layer(name)}`),
    moe: () => get<Record<string, any>[]>(`${r}/moe`),
    replay: () => get<Record<string, any>>(`${r}/replay`),
    spikes: () => get<{ step: number; file: string }[]>(`${r}/spikes`),
    spike: (step: number) => get<GlobalRow[]>(`${r}/spikes/${step}`),
    spikeLayerNames: (step: number) => get<string[]>(`${r}/spikes/${step}/layers`),
    spikeLayer: (step: number, name: string) => get<Record<string, any>[]>(`${r}/spikes/${step}/layers/${layer(name)}`),
    diff: (stepA: number, stepB: number) => get<Record<string, any>[]>(`${r}/diff${q({ step_a: stepA, step_b: stepB })}`),
  };
}

export type ScopeApi = ReturnType<typeof scopeApi>;

export function scopeMulti(base: string) {
  return {
    runs: () => get<RunSummary[]>(`${base}/runs`),
    compare: (runs: string[]) => get<Record<string, any>>(`${base}/compare${q({ runs: runs.join(",") })}`),
    clusters: (runs?: string[]) => get<Record<string, any>>(`${base}/cluster${q({ runs: runs?.join(",") })}`),
    counterexample: (run: string) => get<Record<string, any>>(`${base}/counterexample${q({ run })}`),
  };
}

export type ScopeMessage =
  | { type: "meta"; payload: Record<string, any> }
  | { type: "global" | "global_delta"; payload: GlobalRow[] }
  | { type: "spike"; payload: { step: number } }
  | { type: "layers"; payload: string[] }
  | { type: "moe" | "moe_delta"; payload: Record<string, any>[] };

/** Live updates for one run; returns a function that closes the socket. */
export function subscribeScope(base: string, run: string, onMsg: (m: ScopeMessage) => void): () => void {
  const origin = base.startsWith("http") ? base.replace(/^http/, "ws") : `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}${base}`;
  const ws = new WebSocket(`${origin}/runs/${encodeURIComponent(run)}/ws`);
  ws.onmessage = (e) => {
    try {
      onMsg(JSON.parse(e.data));
    } catch {
      /* ignore malformed frames */
    }
  };
  return () => ws.close();
}
