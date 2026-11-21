// The replay plan written by `trainscope replay`: which batches to skip and the steps they map to.
import { useEffect, useState } from "react";
import { scopeApi } from "./api";

export function Replay({ base, run }: { base: string; run: string }) {
  const [plan, setPlan] = useState<Record<string, any> | null | undefined>(undefined);
  useEffect(() => {
    let alive = true;
    scopeApi(base, run)
      .replay()
      .then((p) => alive && setPlan(p));
    return () => {
      alive = false;
    };
  }, [base, run]);
  if (plan === undefined) return <p className="m-0 p-4 text-small text-muted">Loading…</p>;
  if (!plan?.config)
    return (
      <p className="m-0 p-4 text-small text-muted">
        No replay plan. <code className="num">trainscope replay</code> on this run's <code className="num">scope/</code> folder writes one: the batches to skip when
        re-running from a checkpoint.
      </p>
    );
  return (
    <div className="px-4 py-2 text-small">
      <div>
        skips <span className="num">{plan.skipped_batches.length}</span> batches, i.e. steps{" "}
        <span className="num">{plan.skipped_steps.slice(0, 40).join(", ") || "—"}</span>
        {plan.skipped_steps.length > 40 && <span className="text-muted"> +{plan.skipped_steps.length - 40} more</span>}
      </div>
      <pre className="num mt-2 overflow-auto border border-line p-2 text-label">{JSON.stringify(plan.config, null, 2)}</pre>
    </div>
  );
}
