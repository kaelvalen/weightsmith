// Inspector section for the selected node: its layers' gradient norm and activation kurtosis
// from the run shown in Diagnose.
import { useEffect, useState } from "react";
import { daemonBase } from "../api/client";
import { useStore } from "../state/store";
import { LineChart } from "../ui/LineChart";
import { scopeApi } from "./api";

export function NodeScope({ nodeId }: { nodeId: string }) {
  const run = useStore((s) => s.diag.run);
  const layers = useStore((s) => s.diag.layers[nodeId]);
  const [rows, setRows] = useState<Record<string, Record<string, any>[]>>({});
  const shown = (layers ?? []).slice(0, 4);
  const key = shown.join("|");
  useEffect(() => {
    if (!run || !shown.length) return;
    let alive = true;
    const api = scopeApi(`${daemonBase()}/api/scope`, run);
    Promise.all(shown.map((l) => api.layer(l))).then((all) => alive && setRows(Object.fromEntries(shown.map((l, i) => [l, all[i] ?? []]))));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, key]);
  if (!run || !shown.length) return null;
  const series = (field: string) =>
    shown.map((l) => ({ id: l, label: l.slice(nodeId.length + 1) || l, points: (rows[l] ?? []).filter((r) => typeof r[field] === "number").map((r) => [r.step, r[field]] as [number, number]) }));
  return (
    <div className="border-t border-line px-4 py-2">
      <div className="text-label text-muted">Layer metrics</div>
      <p className="m-0 text-label text-muted">from run {run.slice(0, 8)} · Diagnose tab</p>
      <LineChart title="gradient norm" series={series("grad_l2_norm")} height={100} />
      <LineChart title="activation kurtosis" series={series("act_kurtosis")} height={100} />
    </div>
  );
}
