import { useEffect, useState } from "react";
import { Select } from "./Select";
import { api, type DataPreview } from "../api/client";
import { TYPES } from "../ir/registry";
import { reshapeInput } from "../ir/smart";
import { useAnalysis, useIssues, useStore } from "../state/store";
import { DiffList } from "./DiffList";
import { Field, FixButtons, Row, Section, fmtCount } from "./Inspector";

const num = (v: string) => (v.trim() === "" ? undefined : Number.isNaN(Number(v)) ? v : Number(v));

function TrainField({ k, label, path }: { k: string; label?: string; path?: string }) {
  const train = useStore((s) => s.job().train);
  const setJob = useStore((s) => s.setJob);
  const keys = (path ? `${path}.${k}` : k).split(".");
  let v: any = train;
  for (const x of keys) v = v?.[x];
  const commit = (s: string) => {
    const next = structuredClone(train);
    let o: any = next;
    for (const x of keys.slice(0, -1)) o = o[x] = o[x] ?? {};
    const val = num(s);
    if (val === undefined) delete o[keys[keys.length - 1]];
    else o[keys[keys.length - 1]] = val;
    setJob("train", next);
  };
  return (
    <Row label={label ?? k}>
      <Field mono value={v === undefined || v === null ? "" : String(v)} placeholder="default" onCommit={commit} />
    </Row>
  );
}

function TrainSelect({ k, options, path, label }: { k: string; options: string[]; path?: string; label?: string }) {
  const train = useStore((s) => s.job().train);
  const setJob = useStore((s) => s.setJob);
  const v = path ? train[path]?.[k] : train[k];
  return (
    <Row label={label ?? k}>
      <Select
        value={v ?? ""}
        options={[{ value: "", label: "default" }, ...options]}
        ariaLabel={label ?? k}
        onChange={(val) => {
          const next = structuredClone(train);
          const target = path ? (next[path] = { ...(next[path] ?? {}) }) : next;
          if (val === "") delete target[k];
          else target[k] = val;
          setJob("train", next);
        }}
      />
    </Row>
  );
}

function TrainJson({ k, label, hint }: { k: string; label: string; hint: string }) {
  const train = useStore((s) => s.job().train);
  const setJob = useStore((s) => s.setJob);
  const [bad, setBad] = useState(false);
  return (
    <div className="mb-2">
      <div className="mb-1 text-small text-muted" title={hint}>
        {label}
      </div>
      <textarea
        key={JSON.stringify(train[k] ?? null)}
        rows={3}
        className="num !text-[11px]"
        style={{ borderColor: bad ? "var(--error)" : undefined }}
        defaultValue={train[k] ? JSON.stringify(train[k], null, 1) : ""}
        placeholder={hint}
        onBlur={(e) => {
          const t = e.target.value.trim();
          try {
            const next = { ...train };
            if (t) next[k] = JSON.parse(t);
            else delete next[k];
            setBad(false);
            setJob("train", next);
          } catch {
            setBad(true);
          }
        }}
        aria-label={label}
      />
    </div>
  );
}

function InitPanel() {
  const init = useStore((s) => s.job().init) ?? {}; // default outside the selector: a fresh {} each call would loop
  const setJob = useStore((s) => s.setJob);
  return (
    <Section title="Start from">
      <Row label="weights" hint="safetensors or .pt with the model's parameter names (weightsmith import writes these)">
        <Field mono value={init.weights ?? ""} placeholder="random initialisation" onCommit={(v) => setJob("init", v ? { weights: v } : {})} />
      </Row>
      <p className="m-0 mt-1 text-label text-muted">LoRA and adapters start fresh; every other tensor must be in the file.</p>
    </Section>
  );
}

function TrainBool({ k }: { k: string }) {
  const train = useStore((s) => s.job().train);
  const setJob = useStore((s) => s.setJob);
  return (
    <Row label={k}>
      <input type="checkbox" checked={!!train[k]} onChange={(e) => setJob("train", { ...train, [k]: e.target.checked })} />
    </Row>
  );
}

const SOURCE_FIELDS: Record<string, string[]> = {
  mnist: ["limit"],
  synthetic: ["kind", "n", "dim", "classes", "length", "vocab", "seed"],
  text: ["uri", "length"],
  csv: ["uri", "fields"],
  parquet: ["uri", "fields"],
  hf: ["uri", "subset", "split_name", "fields", "limit"],
  imagefolder: ["uri", "size", "mode"],
  sft: ["tokenizer", "records", "uri", "repeat", "system"],
  triplets: ["tokenizer", "records", "uri", "max_length"],
};

const JSON_FIELDS = new Set(["fields", "records"]);

function DataPanel() {
  const data = useStore((s) => s.job().data);
  const setJob = useStore((s) => s.setJob);
  const src = data.source ?? "synthetic";
  const set = (k: string, v: any) => {
    const next = { ...data };
    if (v === undefined || v === "") delete next[k];
    else next[k] = v;
    setJob("data", next);
  };
  return (
    <Section title="Data">
      <Row label="source">
        <Select value={src} options={Object.keys(SOURCE_FIELDS)} onChange={(v) => setJob("data", { source: v, split: data.split })} ariaLabel="data source" />
      </Row>
      {SOURCE_FIELDS[src].map((k) => (
        <Row
          key={k}
          label={k}
          hint={
            k === "fields"
              ? 'JSON: {"x": ["col1", "col2"], "y": "label"}'
              : k === "records"
                ? 'JSON list: [{"prompt": "...", "completion": "..."}] (sft) or [{"anchor", "positive", "negative"}] (triplets)'
                : k === "uri"
                  ? "path or file:// URI (JSON lines for sft / triplets)"
                  : k === "tokenizer"
                    ? "Hugging Face tokenizer name or path"
                    : undefined
          }
        >
          <Field
            mono
            value={data[k] === undefined ? "" : typeof data[k] === "object" ? JSON.stringify(data[k]) : String(data[k])}
            placeholder={k === "kind" ? "classification | reverse" : ""}
            onCommit={(v) => {
              if (JSON_FIELDS.has(k)) {
                try {
                  set(k, v ? JSON.parse(v) : undefined);
                } catch {
                  /* keep */
                }
              } else set(k, num(v) ?? (v || undefined));
            }}
          />
        </Row>
      ))}
      <Row label="val split">
        <Field mono value={String(data.split?.val ?? 0.1)} onCommit={(v) => set("split", { ...(data.split ?? {}), val: Number(v) })} />
      </Row>
      <p className="m-0 mt-1 text-label text-muted">Inputs bind to dataset fields by name (Input.field); unconnected loss targets read field y.</p>
    </Section>
  );
}

/** Ways to make data and graph agree: change the Input, or (for generated data) the data. */
function MismatchFix({ node, have }: { node: string; have: number[] }) {
  const apply = useStore((s) => s.apply);
  const data = useStore((s) => s.job().data);
  const setJob = useStore((s) => s.setJob);
  const graph = useStore((s) => s.graph());
  const want = (graph.nodes.find((n) => n.id === node)?.params as any)?.shape?.slice(1) ?? [];
  const synth = (data.source ?? "synthetic") === "synthetic" && have.length === 1 && want.length === 1 && typeof want[0] === "number";
  return (
    <span className="mt-1 flex flex-wrap gap-1 pl-4">
      <button className="!h-5 !px-1.5 text-label" onClick={() => apply(`fit ${node} to the data`, (g) => reshapeInput(g, node, have))}>
        Set {node} to [{have.join(", ")}]
      </button>
      {synth && (
        <button
          className="!h-5 !px-1.5 text-label"
          onClick={() => setJob("data", { ...data, [(data.kind ?? "classification") === "reverse" ? "length" : "dim"]: want[0] })}
        >
          Generate {(data.kind ?? "classification") === "reverse" ? "length" : "dim"} {want[0]}
        </button>
      )}
    </span>
  );
}

function PreviewPanel({ preview, refresh }: { preview: PreviewState; refresh: () => void }) {
  const r = preview.result;
  return (
    <Section
      title="What the model will see"
      aside={
        <button className="link text-label" onClick={refresh} disabled={preview.busy}>
          {preview.busy ? "loading…" : r ? "reload" : "load preview"}
        </button>
      }
    >
      {!r && !preview.busy && <p className="m-0 text-small text-muted">{preview.why ?? "Load a few samples to check them against the graph."}</p>}
      {r?.error && <p className="num m-0 text-small text-error">{r.error}</p>}
      {r?.fields && (
        <>
          <div className="num grid grid-cols-[auto_auto_1fr] gap-x-3 gap-y-1 text-small">
            {r.fields.map((f) => (
              <div key={f.name} className="contents">
                <span>{f.name}</span>
                <span className="text-muted">
                  {f.dtype} [{f.shape.join(", ")}]
                </span>
                <span className="truncate text-label text-muted" title={f.peek}>
                  {f.peek}
                </span>
              </div>
            ))}
          </div>
          <p className="num m-0 mt-1 text-label text-muted">
            {r.capped ? `first ${r.rows} rows` : `${r.rows} rows`} · shapes per sample
          </p>
        </>
      )}
      {!!r?.checks?.length && (
        <ul className="m-0 mt-2 flex list-none flex-col gap-0.5 p-0 text-small">
          {r.checks.map((c, i) => (
            <li key={i} className="num" style={{ color: c.ok ? undefined : "var(--error)" }}>
              {c.ok ? "✓ " : "✕ "}
              {c.message}
              {!c.ok && c.node && c.have && <MismatchFix node={c.node} have={c.have} />}
            </li>
          ))}
        </ul>
      )}
    </Section>
  );
}

interface PreviewState {
  busy: boolean;
  result?: DataPreview;
  why?: string;
}

/** Sources that are cheap enough to preview on every change; the rest wait for a click. */
const AUTO_PREVIEW = new Set(["synthetic", "mnist", "csv", "parquet", "text", "sft", "triplets"]);
const previewCache = new Map<string, DataPreview>();

function usePreview(): [PreviewState, () => void] {
  const data = useStore((s) => s.job().data);
  const graph = useStore((s) => s.graph());
  const daemon = useStore((s) => s.daemon);
  const key = JSON.stringify([data, graph.nodes, graph.edges]);
  const [state, setState] = useState<PreviewState>(() => ({ busy: false, result: previewCache.get(key) }));
  const load = () => {
    if (daemon !== "online") return setState({ busy: false, why: "The daemon is offline; start it with weightsmith serve." });
    setState((s) => ({ ...s, busy: true }));
    api
      .dataPreview(data, graph)
      .then((r) => {
        previewCache.set(key, r);
        setState({ busy: false, result: r });
      })
      .catch((e) => setState({ busy: false, result: { ok: false, error: e.message ?? String(e) } }));
  };
  useEffect(() => {
    const hit = previewCache.get(key);
    if (hit) return setState({ busy: false, result: hit });
    setState({ busy: false, why: AUTO_PREVIEW.has(data.source ?? "synthetic") ? undefined : "This source downloads data; load the preview when ready." });
    if (!AUTO_PREVIEW.has(data.source ?? "synthetic") || daemon !== "online") return;
    const t = setTimeout(load, 500);
    return () => clearTimeout(t);
  }, [key, daemon]);
  return [state, load];
}

type Tab = "overview" | "data" | "training";
let lastTab: Tab = "overview";

function Check({ ok, pending, children, action }: { ok: boolean; pending?: boolean; children: React.ReactNode; action?: React.ReactNode }) {
  return (
    <li className="grid grid-cols-[16px_1fr_auto] items-baseline gap-1 text-small">
      <span aria-hidden style={{ color: pending ? "var(--muted)" : ok ? "var(--ink)" : "var(--error)" }}>
        {pending ? "·" : ok ? "✓" : "✕"}
      </span>
      <span className={ok || pending ? "" : "text-ink"}>{children}</span>
      {action}
    </li>
  );
}

function Readiness({ preview, go }: { preview: PreviewState; go: (t: Tab) => void }) {
  const graph = useStore((s) => s.graph());
  const daemon = useStore((s) => s.daemon);
  const run = useStore((s) => s.run);
  const analysis = useAnalysis();
  const errors = useIssues().filter((i) => i.severity === "error");
  const hasInput = graph.nodes.some((n) => n.type === "Input");
  const hasLoss = analysis.losses.length > 0 || graph.nodes.some((n) => TYPES[n.type]?.loss);
  const r = preview.result;
  const dataOk = !!r && r.ok && !r.error;
  const ready = hasInput && hasLoss && !errors.length && daemon === "online" && (dataOk || !r);
  const fail = r?.error ?? r?.checks?.find((c) => !c.ok)?.message;
  return (
    <Section title="Ready to train?">
      <ul className="m-0 flex list-none flex-col gap-1 p-0">
        <Check ok={hasInput}>{hasInput ? "graph has an input" : "add an Input node"}</Check>
        <Check ok={hasLoss}>{hasLoss ? "graph has a loss" : "add a loss (CrossEntropy, MSE…) after the output"}</Check>
        <Check ok={!errors.length}>{errors.length ? `${errors.length} error${errors.length > 1 ? "s" : ""} in the graph, listed below` : "graph validates"}</Check>
        <Check
          ok={dataOk}
          pending={!r}
          action={
            <button className="link text-label" onClick={() => go("data")}>
              Data
            </button>
          }
        >
          {preview.busy ? "checking the data…" : !r ? "data not checked yet" : dataOk ? "data fits the inputs" : fail}
        </Check>
        <Check ok={daemon === "online"}>{daemon === "online" ? "daemon online" : "daemon offline: weightsmith serve"}</Check>
      </ul>
      <div className="mt-3 flex items-center gap-2">
        <button className="primary" onClick={run} disabled={!ready}>
          Run
        </button>
        <button className="link text-label" onClick={() => go("training")}>
          training settings
        </button>
      </div>
    </Section>
  );
}

function VariantsPanel() {
  const project = useStore((s) => s.project);
  const switchVariant = useStore((s) => s.switchVariant);
  const v = project.variants[project.active];
  const [diff, setDiff] = useState<any>(null);
  const daemon = useStore((s) => s.daemon);
  useEffect(() => {
    setDiff(null);
    if (!v.parent || !project.variants[v.parent] || daemon !== "online") return;
    const t = setTimeout(() => api.diff(project.variants[v.parent!].graph, v.graph).then(setDiff).catch(() => undefined), 300);
    return () => clearTimeout(t);
  }, [v.graph, v.parent, daemon]);
  return (
    <Section title="Variants">
      {Object.entries(project.variants).map(([name, x]) => (
        <div key={name} className="flex items-baseline justify-between text-small">
          <button className="link" style={{ color: name === project.active ? "var(--ink)" : undefined }} onClick={() => switchVariant(name)}>
            {name === project.active ? "● " : ""}
            {name}
          </button>
          <span className="text-muted">{x.parent ? `from ${x.parent}` : "root"}</span>
        </div>
      ))}
      {Object.entries(project.variants).filter(([n, x]) => x.note && n !== project.active).length > 0 && <div className="h-2" />}
      {Object.entries(project.variants).map(([name, x]) => (
        x.note && name !== project.active ? (
          <p key={`note-${name}`} className="m-0 text-label text-muted">
            <span className="text-ink">{name}</span>: {x.note}
          </p>
        ) : null
      ))}
      {diff && (
        <div className="mt-2">
          <div className="mb-1 text-label text-muted">changes vs {v.parent}</div>
          <DiffList diff={diff} />
        </div>
      )}
    </Section>
  );
}

function ShuffleToggle() {
  const train = useStore((s) => s.job().train);
  const setJob = useStore((s) => s.setJob);
  return <input type="checkbox" checked={train.shuffle !== false} onChange={(e) => setJob("train", { ...train, shuffle: e.target.checked })} />;
}

export function JobPanel() {
  const graph = useStore((s) => s.graph());
  const apply = useStore((s) => s.apply);
  const select = useStore((s) => s.select);
  const setScope = useStore((s) => s.setScope);
  const remote = useStore((s) => s.remote);
  const analysis = useAnalysis();
  const issues = useIssues().filter((i) => i.severity !== "peer");
  const t = analysis.totals;
  const project = useStore((s) => s.project);
  const setDescription = useStore((s) => s.setDescription);
  const setNote = useStore((s) => s.setNote);
  const setVariantJob = useStore((s) => s.setVariantJob);
  const variant = project.variants[project.active];
  const [tab, setTabState] = useState<Tab>(lastTab);
  const setTab = (t: Tab) => {
    lastTab = t;
    setTabState(t);
  };
  const [preview, refresh] = usePreview();
  const tabs = (
    <div className="sticky top-0 z-10 flex h-9 items-stretch gap-4 border-b border-line bg-bg px-4" role="tablist">
      {(["overview", "data", "training"] as const).map((t) => (
        <button
          key={t}
          role="tab"
          aria-selected={tab === t}
          className="link !h-full !text-[13px] capitalize"
          style={{ color: tab === t ? "var(--ink)" : undefined, fontWeight: tab === t ? 500 : 400, boxShadow: tab === t ? "inset 0 -1px 0 var(--ink)" : undefined }}
          onClick={() => setTab(t)}
        >
          {t}
          {t === "overview" && issues.some((i) => i.severity === "error") ? <span className="text-error"> ·</span> : null}
          {t === "data" && preview.result && !preview.result.ok ? <span className="text-error"> ·</span> : null}
        </button>
      ))}
    </div>
  );
  if (tab === "data")
    return (
      <div>
        {tabs}
        <div className="border-b border-line px-4 py-2">
          <label className="flex items-center gap-2 text-small">
            <input type="checkbox" checked={!!variant.job} onChange={(e) => setVariantJob(e.target.checked)} />
            own data and training config for variant {project.active}
          </label>
        </div>
        <InitPanel />
        <DataPanel />
        <PreviewPanel preview={preview} refresh={refresh} />
      </div>
    );
  if (tab === "training")
    return (
      <div>
        {tabs}
        <TrainingPanel />
      </div>
    );
  return (
    <div>
      {tabs}
      <Readiness preview={preview} go={setTab} />
      <Section title="Project">
        <textarea
          key={project.slug}
          rows={3}
          placeholder="What this project is about"
          defaultValue={project.description ?? ""}
          onBlur={(e) => e.target.value !== (project.description ?? "") && setDescription(e.target.value)}
          aria-label="Project description"
        />
      </Section>
      <div className="border-t border-line px-4 pt-3 pb-3">
        <div className="mb-1 text-label text-muted">variant {project.active}</div>
        <Field value={graph.name ?? ""} placeholder="graph name" onCommit={(v) => apply("rename graph", (g) => ({ ...g, name: v }))} />
        <textarea
          key={`${project.slug}/${project.active}`}
          className="mt-2"
          rows={2}
          placeholder="Note: what this variant changes and why"
          defaultValue={variant.note ?? ""}
          onBlur={(e) => e.target.value !== (variant.note ?? "") && setNote(e.target.value)}
          aria-label="Variant note"
        />
        <div className="num mt-2 grid grid-cols-2 gap-y-1 text-small">
          <span className="text-muted">parameters</span>
          <span>{fmtCount(t.total)}{t.known ? "" : "+"}</span>
          <span className="text-muted">trainable</span>
          <span>
            {fmtCount(t.trainable)}
            {t.total ? <span className="text-muted"> · {((100 * t.trainable) / t.total).toFixed(1)}%</span> : null}
          </span>
          <span className="text-muted">hash</span>
          <span title={remote?.hash ?? ""}>{remote?.hash?.slice(0, 12) ?? "—"}</span>
        </div>
      </div>
      <Section title={issues.length ? `Issues · ${issues.length}` : "Issues"}>
        {!issues.length && <p className="m-0 text-small text-muted">Graph validates.</p>}
        {issues.map((i, k) => (
          <div key={k} className="mb-1">
          <button
            className="link block w-full truncate text-left text-small"
            style={{ color: i.severity === "error" ? "var(--error)" : undefined }}
            title={i.message}
            onClick={() => {
              if (!i.node) return;
              const [scope] = i.node.includes("/") ? i.node.split("/") : [""];
              if (scope) setScope(scope);
              select([i.node.split("/").pop()!]);
            }}
          >
            <span className="num">{[i.node, i.port].filter(Boolean).join(".") || "graph"}</span> {i.message}
          </button>
          <FixButtons issue={i} block />
          </div>
        ))}
      </Section>
      <VariantsPanel />
    </div>
  );
}

function TrainingPanel() {
  return (
    <>
      <Section title="Schedule">
        <TrainField k="epochs" />
        <TrainField k="max_steps" />
        <TrainField k="batch_size" label="batch" />
      </Section>
      <Section title="Optimiser">
        <TrainSelect k="name" path="optimizer" label="optimizer" options={["adamw", "adam", "sgd"]} />
        <TrainField k="lr" path="optimizer" />
        <TrainField k="weight_decay" path="optimizer" label="weight decay" />
        <TrainSelect k="name" path="scheduler" label="schedule" options={["cosine", "linear", "constant"]} />
        <TrainField k="warmup_steps" path="scheduler" label="warmup" />
        <TrainField k="grad_clip" label="grad clip" />
      </Section>
      <Section title="Runtime">
        <TrainSelect k="precision" options={["bf16", "f16", "f32"]} />
        <TrainField k="seed" />
        <TrainField k="log_every" label="log every" />
        <TrainField k="eval_every" label="eval every" />
        <TrainField k="every_steps" path="checkpoint" label="checkpoint" />
        <TrainBool k="grad_checkpointing" />
        <TrainBool k="compile" />
        <TrainSelect k="weights_dtype" label="frozen dtype" options={["bf16", "f16"]} />
        <Row label="shuffle">
          <ShuffleToggle />
        </Row>
      </Section>
      <Section title="Pretrained models">
        <TrainJson k="kl_to_base" label="KL to the base model" hint='{"lambda": 1, "prompts": ["..."], "system": "..."}' />
        <TrainJson k="gate" label="Divergence gate" hint='{"prompts": ["..."], "system": "...", "kappa": 0.968}' />
      </Section>
    </>
  );
}
