// One run's trainscope recording: fetched once, then kept current over the run's WebSocket.
import { useEffect, useMemo, useState } from "react";
import { scopeApi, subscribeScope } from "./api";
import { groupSpikes } from "./spikeCluster";
import type { GlobalRow } from "./types";

export interface ScopeRun {
  loading: boolean;
  missing: boolean; // the run has no recording (diagnostics off, or recorded before trainscope)
  meta: Record<string, any> | null;
  global: GlobalRow[];
  spikes: number[];
  moe: Record<string, any>[];
  layers: string[];
}

const EMPTY: ScopeRun = { loading: true, missing: false, meta: null, global: [], spikes: [], moe: [], layers: [] };

export function useScopeRun(base: string, run: string | null, live: boolean): ScopeRun & { events: ReturnType<typeof groupSpikes> } {
  const [state, setState] = useState<ScopeRun>(EMPTY);
  useEffect(() => {
    if (!run) return;
    let alive = true;
    setState(EMPTY);
    const api = scopeApi(base, run);
    Promise.all([api.meta(), api.global(), api.spikes(), api.moe(), api.layers()])
      .then(([meta, global, spikes, moe, layers]) => {
        if (!alive) return;
        if (global === null) setState({ ...EMPTY, loading: false, missing: true });
        else setState({ loading: false, missing: false, meta, global, spikes: (spikes ?? []).map((s) => s.step), moe: moe ?? [], layers: layers ?? [] });
      })
      .catch(() => alive && setState({ ...EMPTY, loading: false, missing: true }));
    return () => {
      alive = false;
    };
  }, [base, run]);

  useEffect(() => {
    if (!run || !live) return;
    return subscribeScope(base, run, (m) =>
      setState((s) => {
        switch (m.type) {
          case "global":
            return { ...s, loading: false, missing: false, global: m.payload };
          case "global_delta":
            return { ...s, global: [...s.global, ...m.payload] };
          case "spike":
            return s.spikes.includes(m.payload.step) ? s : { ...s, spikes: [...s.spikes, m.payload.step] };
          case "layers":
            return { ...s, layers: m.payload };
          case "moe":
            return { ...s, moe: m.payload };
          case "moe_delta":
            return { ...s, moe: [...s.moe, ...m.payload] };
          default:
            return s;
        }
      }),
    );
  }, [base, run, live]);

  const events = useMemo(() => groupSpikes(state.global, state.spikes), [state.global, state.spikes]);
  return { ...state, events };
}
