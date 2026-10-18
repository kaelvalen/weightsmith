import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { RunEvent, RunRecord, api } from "../api/client";
import { BottomTab, useStore } from "../state/store";
import { DiffList } from "./DiffList";
import { fmtCount } from "./Inspector";
import { LineChart, Series } from "./LineChart";

const CodeEditor = lazy(() => import("./CodeEditor"));

const LIVE = ["queued", "preparing", "running", "paused"];
const short = (id: string) => id.slice(0, 8);
const f4 = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toPrecision(4)) : "—");

function series(evs: RunEvent[] | undefined, type: string, key: string): [number, number][] {
  return (evs ?? []).filter((e) => e.type === type && typeof e[key] === "number").map((e) => [e.step, e[key]]);
}

function Status({ s }: { s: string }) {
  const live = s === "running" || s === "preparing";
  return (
    <span className="num" style={{ color: live ? "var(--accent)" : s === "failed" ? "var(--error)" : undefined }}>
      {live ? "● " : ""}
      {s}
    </span>
  );
}

function Controls({ run }: { run: RunRecord }) {
  const notify = useStore((s) => s.notify);
  const refresh = useStore((s) => s.refreshRuns);
  const selection = useStore((s) => s.selection);
  const scope = useStore((s) => s.scope);
  const setActive = (id: string) => useStore.setState({ activeRun: id });
  const watch = useStore((s) => s.watchRun);
  const cmd = async (c: string) => {
    try {
      await api.control(run.id, c);
      refresh();
    } catch (e: any) {
      notify(e.message, true);
    }
  };
  const s = run.status;
  return (
    <span className="flex flex-wrap gap-1">
      {(s === "running" || s === "preparing") && <button onClick={() => cmd("pause")}>Pause</button>}
      {s === "paused" && <button onClick={() => cmd("resume")}>Resume</button>}
      {["stopped", "interrupted", "failed"].includes(s) && <button onClick={() => cmd("resume")}>Resume from checkpoint</button>}
      {["running", "paused"].includes(s) && <button onClick={() => cmd("checkpoint")}>Checkpoint now</button>}
      {LIVE.includes(s) && <button onClick={() => cmd("stop")}>Stop</button>}
      {s === "completed" && selection.length > 0 && (
        <button
          title="Re-run with the selected nodes disabled; same config, data and seed"
          onClick={async () => {
            try {
              const child = await api.ablate(run.id, selection.map((id) => (scope ? `${scope}/${id}` : id)));
              setActive(child.id);
              watch(child.id);
              refresh();
              notify(`ablation ${short(child.id)}: without ${selection.join(", ")}`);
            } catch (e: any) {
              notify(e.message, true);
            }
          }}
        >
          Ablate selection
        </button>
      )}
      {s === "completed" && (
        <>
          <a href={api.exportUrl(run.id)}>
            <button>Export</button>
          </a>
          {run.summary?.trainable < run.summary?.params && (
            <a href={api.exportUrl(run.id, true)} title="fold LoRA into the base weights">
              <button>Export merged</button>
            </a>
          )}
        </>
      )}
    </span>
  );
}

function Metrics() {
  const activeRun = useStore((s) => s.activeRun);
  const runs = useStore((s) => s.runs);
  const events = useStore((s) => (activeRun ? s.events[activeRun] : undefined));
  const select = useStore((s) => s.select);
  const run = runs.find((r) => r.id === activeRun);
  if (!activeRun || !run) return <p className="m-0 p-4 text-small text-muted">No run selected. Cmd+Enter runs the current variant.</p>;
  const last = [...(events ?? [])].reverse().find((e) => e.type === "metric");
  const err = run.error ?? [...(events ?? [])].reverse().find((e) => e.type === "error");
  const total = run.summary?.total_steps;
  const step = last?.step ?? run.summary?.last?.step ?? 0;
  const gpu = series(events, "metric", "gpu_mem_mb");
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 px-4 py-2">
        <span className="font-medium">{run.name}</span>
        <span className="num text-muted">{short(run.id)}</span>
        <Status s={run.status} />
        <span className="num text-small">
          step {step}
          {total ? ` / ${total}` : ""}
        </span>
        <span className="num text-small text-muted">
          {fmtCount(run.summary?.trainable ?? 0)} / {fmtCount(run.summary?.params ?? 0)} trainable
        </span>
        {run.summary?.eval?.val_accuracy !== undefined && <span className="num text-small">val acc {(run.summary.eval.val_accuracy * 100).toFixed(2)}%</span>}
        {run.parent && <span className="text-small text-muted">ablation of {short(run.parent)}</span>}
        <span className="ml-auto">
          <Controls run={run} />
        </span>
      </div>
      {total ? (
        <div className="mx-4 h-px bg-line">
          <div className="h-px bg-accent" style={{ width: `${Math.min(100, (100 * step) / total)}%` }} />
        </div>
      ) : null}
      {err && (
        <div className="px-4 pt-2 text-small text-error">
          [{err.class}] {err.message}
          {err.node && (
            <button className="link ml-2" onClick={() => select([err.node])}>
              → node {err.node}
            </button>
          )}
          {err.suggestions?.map((s: string) => (
            <span key={s} className="ml-3 text-muted">
              · {s}
            </span>
          ))}
        </div>
      )}
      <div className="grid min-h-0 flex-1 grid-cols-[repeat(auto-fit,minmax(240px,1fr))] gap-x-6 gap-y-2 overflow-y-auto px-4 py-2">
        <LineChart
          title="loss"
          series={[
            { id: "train", label: "train", points: series(events, "metric", "loss") },
            ...(series(events, "eval", "val_loss").length ? [{ id: "val", label: "val", points: series(events, "eval", "val_loss") }] : []),
          ]}
          height={128}
        />
        <LineChart title="learning rate" series={[{ id: "lr", label: "lr", points: series(events, "metric", "lr") }]} height={128} />
        <LineChart title="grad norm" series={[{ id: "g", label: "grad norm", points: series(events, "metric", "grad_norm") }]} height={128} />
        <LineChart title="throughput · samples/s" series={[{ id: "t", label: "throughput", points: series(events, "metric", "throughput").slice(1) }]} height={128} />
        {gpu.length > 0 && (
          <LineChart
            title="GPU memory · MB"
            series={[
              { id: "m", label: "allocated", points: gpu },
              { id: "p", label: "peak", points: series(events, "metric", "gpu_mem_peak_mb") },
            ]}
            height={128}
          />
        )}
        {series(events, "metric", "gpu_util").length > 0 && <LineChart title="GPU utilisation · %" series={[{ id: "u", label: "util", points: series(events, "metric", "gpu_util") }]} height={128} />}
      </div>
    </div>
  );
}

function Log() {
  const activeRun = useStore((s) => s.activeRun);
  const events = useStore((s) => (activeRun ? s.events[activeRun] : undefined));
  const lines = (events ?? []).filter((e) => ["log", "error", "status", "checkpoint", "eval"].includes(e.type));
  return (
    <div className="num h-full overflow-y-auto px-4 py-2 text-label">
      {!lines.length && <span className="text-muted">No log yet.</span>}
      {lines.map((e) => (
        <div key={e.seq} className={e.type === "error" || e.stream === "stderr" ? "text-error" : ""}>
          <span className="text-muted">{String(e.seq).padStart(5)} </span>
          {e.type === "log"
            ? e.message
            : e.type === "status"
              ? `status ${e.state}${e.resume_from ? ` from ${e.resume_from}` : ""}`
              : e.type === "checkpoint"
                ? `checkpoint ${e.path}`
                : e.type === "eval"
                  ? `eval ${Object.entries(e).filter(([k]) => k.startsWith("val_")).map(([k, v]) => `${k} ${f4(v)}`).join("  ")}`
                  : `[${e.class}] ${e.message}${e.traceback ? "\n" + e.traceback : ""}`}
        </div>
      ))}
    </div>
  );
}

function Runs() {
  const runs = useStore((s) => s.runs);
  const compare = useStore((s) => s.compare);
  const toggle = useStore((s) => s.toggleCompare);
  const activeRun = useStore((s) => s.activeRun);
  const watch = useStore((s) => s.watchRun);
  const refresh = useStore((s) => s.refreshRuns);
  const notify = useStore((s) => s.notify);
  const loadGraph = (r: RunRecord) =>
    api.run(r.id).then((full) => {
      if (!full.job) return;
      const st = useStore.getState();
      const name = `run-${short(r.id)}`;
      if (!st.project.variants[name]) {
        st.branchVariant(name);
        st.apply(`load graph of run ${short(r.id)}`, () => ({ ...full.job!.graph, layout: st.graph().layout }));
      } else st.switchVariant(name);
    });
  if (!runs.length) return <p className="m-0 p-4 text-small text-muted">No runs in this project yet.</p>;
  const byId = new Map(runs.map((r) => [r.id, r]));
  return (
    <div className="h-full overflow-y-auto">
      <table className="num w-full border-collapse text-small">
        <thead className="text-label text-muted">
          <tr className="text-left">
            <th className="w-8 py-1 pl-4 font-normal">cmp</th>
            <th className="font-normal">run</th>
            <th className="font-normal">name</th>
            <th className="font-normal">status</th>
            <th className="text-right font-normal">step</th>
            <th className="text-right font-normal">loss</th>
            <th className="text-right font-normal">best val</th>
            <th className="text-right font-normal">trainable</th>
            <th className="pr-4 text-right font-normal">samples/s</th>
            <th className="pr-4 font-normal" />
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id} className="border-t border-line" style={{ background: r.id === activeRun ? "color-mix(in srgb, var(--accent) 6%, transparent)" : undefined }}>
              <td className="py-1 pl-4">
                <input type="checkbox" checked={compare.includes(r.id)} onChange={() => toggle(r.id)} aria-label={`compare ${r.id}`} />
              </td>
              <td>
                <button
                  className="link"
                  onClick={() => {
                    useStore.setState({ activeRun: r.id, bottom: { ...useStore.getState().bottom, tab: "metrics" } });
                    watch(r.id);
                  }}
                >
                  {r.parent && byId.has(r.parent) ? "└ " : ""}
                  {short(r.id)}
                </button>
              </td>
              <td className="max-w-[260px] truncate font-sans">{r.name}</td>
              <td>
                <Status s={r.status} />
              </td>
              <td className="text-right">{r.summary?.last?.step ?? "—"}</td>
              <td className="text-right">{f4(r.summary?.last?.loss)}</td>
              <td className="text-right">{f4(r.summary?.best_val_loss)}</td>
              <td className="text-right">{fmtCount(r.summary?.trainable ?? 0)}</td>
              <td className="pr-4 text-right">{f4(r.summary?.last?.throughput)}</td>
              <td className="pr-4 text-right whitespace-nowrap">
                <button className="link mr-3" onClick={() => loadGraph(r)} title="open this run's graph as a variant">
                  graph
                </button>
                {!LIVE.includes(r.status) && (
                  <button
                    className="link"
                    onClick={async () => {
                      await api.deleteRun(r.id).catch((e) => notify(e.message, true));
                      refresh();
                    }}
                  >
                    delete
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Compare() {
  const compare = useStore((s) => s.compare);
  const runs = useStore((s) => s.runs);
  const events = useStore((s) => s.events);
  const [details, setDetails] = useState<Record<string, RunRecord>>({});
  const [diffs, setDiffs] = useState<Record<string, any>>({});
  useEffect(() => {
    compare.forEach((id) => {
      if (!details[id]) api.run(id).then((r) => setDetails((d) => ({ ...d, [id]: r })));
    });
  }, [compare.join()]);
  const base = compare[0];
  useEffect(() => {
    const b = details[base]?.job?.graph;
    if (!b) return;
    compare.slice(1).forEach((id) => {
      const g = details[id]?.job?.graph;
      const key = `${base}:${id}`;
      if (g && !diffs[key]) api.diff(b, g).then((d) => setDiffs((x) => ({ ...x, [key]: d })));
    });
  }, [details, compare.join()]);
  const list = compare.map((id) => runs.find((r) => r.id === id)).filter(Boolean) as RunRecord[];
  const mk = (type: string, key: string): Series[] =>
    list.map((r, i) => ({ id: r.id, label: r.variant ? `${short(r.id)} ${r.variant}` : short(r.id), points: series(events[r.id], type, key), focus: i === 0 }));
  const memo = useMemo(() => ({ loss: mk("metric", "loss"), val: mk("eval", "val_loss"), acc: mk("eval", "val_accuracy") }), [events, compare.join(), runs]);
  if (!list.length) return <p className="m-0 p-4 text-small text-muted">Tick runs in the Runs tab to compare them. The first one is the baseline.</p>;
  return (
    <div className="grid h-full min-h-0 grid-cols-[1fr_1fr_360px] gap-6 overflow-y-auto px-4 py-2">
      <LineChart title="train loss" series={memo.loss} height={176} />
      <LineChart title={memo.acc.some((s) => s.points.length) ? "val accuracy" : "val loss"} series={memo.acc.some((s) => s.points.length) ? memo.acc : memo.val} height={176} />
      <div className="min-w-0">
        <table className="num w-full border-collapse text-small">
          <thead className="text-label text-muted">
            <tr className="text-left">
              <th className="font-normal">run</th>
              <th className="text-right font-normal">params</th>
              <th className="text-right font-normal">trainable</th>
              <th className="text-right font-normal">samples/s</th>
              <th className="text-right font-normal">best val</th>
            </tr>
          </thead>
          <tbody>
            {list.map((r, i) => (
              <tr key={r.id} className="border-t border-line">
                <td style={{ color: i === 0 ? "var(--accent)" : undefined }}>{short(r.id)}</td>
                <td className="text-right">{fmtCount(r.summary?.params ?? 0)}</td>
                <td className="text-right">{fmtCount(r.summary?.trainable ?? 0)}</td>
                <td className="text-right">{f4(r.summary?.last?.throughput)}</td>
                <td className="text-right">{f4(r.summary?.best_val_loss)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {list.slice(1).map((r) => (
          <div key={r.id} className="mt-3">
            <div className="text-label text-muted">
              {short(r.id)} vs {short(base)}
            </div>
            {diffs[`${base}:${r.id}`] ? <DiffList diff={diffs[`${base}:${r.id}`]} /> : <span className="text-label text-muted">…</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

function Code() {
  const graph = useStore((s) => s.graph());
  const select = useStore((s) => s.select);
  const setScope = useStore((s) => s.setScope);
  const selection = useStore((s) => s.selection);
  const scope = useStore((s) => s.scope);
  const [res, setRes] = useState<any>(null);
  useEffect(() => {
    const t = setTimeout(() => api.compile(graph).then(setRes).catch((e) => setRes({ ok: false, error: e.message })), 300);
    return () => clearTimeout(t);
  }, [graph]);
  const lines = useMemo(() => {
    if (!res?.line_map) return [];
    const want = new Set(selection.map((id) => (scope ? `${scope}/${id}` : id)));
    return Object.entries(res.line_map as Record<string, string>)
      .filter(([, n]) => want.has(n))
      .map(([l]) => Number(l));
  }, [res, selection, scope]);
  if (!res) return <p className="m-0 p-4 text-small text-muted">Compiling…</p>;
  if (!res.ok)
    return (
      <p className="m-0 p-4 text-small text-error">
        {res.error ?? `does not compile: ${(res.analysis?.issues ?? []).filter((i: any) => i.severity === "error").map((i: any) => `${i.node}: ${i.message}`).join("; ")}`}
      </p>
    );
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="num flex gap-4 px-4 py-1 text-label text-muted">
        <span>model.py · graph {res.hash.slice(0, 12)}</span>
        <span className={res.smoke.ok ? "" : "text-error"}>
          meta-device smoke test {res.smoke.ok ? `ok · ${fmtCount(res.smoke.total)} params` : res.smoke.issues.map((i: any) => i.message).join("; ")}
        </span>
        <span>click a line to select its node</span>
      </div>
      <div className="min-h-0 flex-1 border-t border-line">
        <Suspense fallback={<p className="p-4 text-muted">loading editor…</p>}>
          <CodeEditor
            value={res.model_py}
            readOnly
            height="100%"
            highlightLines={lines}
            onLineClick={(l) => {
              const node = res.line_map[String(l)] as string | undefined;
              if (!node) return;
              if (node.includes("/")) {
                const [s, id] = node.split("/");
                setScope(s);
                setTimeout(() => select([id]), 0);
              } else {
                if (scope) setScope("");
                setTimeout(() => select([node]), 0);
              }
            }}
          />
        </Suspense>
      </div>
    </div>
  );
}

const TABS: [BottomTab, string][] = [
  ["metrics", "Metrics"],
  ["runs", "Runs"],
  ["compare", "Compare"],
  ["log", "Log"],
  ["code", "Code"],
];

export function Bottom() {
  const bottom = useStore((s) => s.bottom);
  const setBottom = useStore((s) => s.setBottom);
  const runs = useStore((s) => s.runs);
  const live = runs.find((r) => r.status === "running" || r.status === "preparing");

  const startResize = (e: React.MouseEvent) => {
    const y0 = e.clientY;
    const h0 = bottom.height;
    const move = (m: MouseEvent) => setBottom({ height: Math.max(160, Math.min(window.innerHeight - 200, h0 + y0 - m.clientY)) });
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };

  return (
    <div className="flex flex-col border-t border-line bg-bg" style={{ height: bottom.open ? bottom.height : 24 }}>
      {bottom.open && <div className="h-1 -mt-0.5 cursor-row-resize" onMouseDown={startResize} />}
      <div className="flex h-6 shrink-0 items-center gap-4 px-4 text-small">
        {TABS.map(([t, label]) => (
          <button
            key={t}
            className="link"
            style={{ color: bottom.open && bottom.tab === t ? "var(--ink)" : undefined, fontWeight: bottom.open && bottom.tab === t ? 500 : 400 }}
            onClick={() => setBottom({ tab: t, open: !(bottom.open && bottom.tab === t) })}
          >
            {label}
          </button>
        ))}
        {live && (
          <span className="num ml-auto text-label" style={{ color: "var(--accent)" }}>
            ● {short(live.id)} step {live.summary?.last?.step ?? 0}
            {live.summary?.total_steps ? `/${live.summary.total_steps}` : ""} · loss {f4(live.summary?.last?.loss)}
          </span>
        )}
      </div>
      {bottom.open && (
        <div className="min-h-0 flex-1 border-t border-line">
          {bottom.tab === "metrics" && <Metrics />}
          {bottom.tab === "runs" && <Runs />}
          {bottom.tab === "compare" && <Compare />}
          {bottom.tab === "log" && <Log />}
          {bottom.tab === "code" && <Code />}
        </div>
      )}
    </div>
  );
}
