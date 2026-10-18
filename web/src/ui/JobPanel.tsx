import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useAnalysis, useIssues, useStore } from "../state/store";
import { DiffList } from "./DiffList";
import { Field, Row, Section, fmtCount } from "./Inspector";

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
      <select
        value={v ?? ""}
        onChange={(e) => {
          const next = structuredClone(train);
          if (path) next[path] = { ...(next[path] ?? {}), [k]: e.target.value };
          else next[k] = e.target.value;
          setJob("train", next);
        }}
      >
        <option value="">default</option>
        {options.map((o) => (
          <option key={o}>{o}</option>
        ))}
      </select>
    </Row>
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
};

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
        <select value={src} onChange={(e) => setJob("data", { source: e.target.value, split: data.split })}>
          {Object.keys(SOURCE_FIELDS).map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
      </Row>
      {SOURCE_FIELDS[src].map((k) => (
        <Row key={k} label={k} hint={k === "fields" ? 'JSON: {"x": ["col1", "col2"], "y": "label"}' : k === "uri" ? "path or file:// URI" : undefined}>
          <Field
            mono
            value={data[k] === undefined ? "" : typeof data[k] === "object" ? JSON.stringify(data[k]) : String(data[k])}
            placeholder={k === "kind" ? "classification | reverse" : ""}
            onCommit={(v) => {
              if (k === "fields") {
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
  return (
    <div>
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
          <button
            key={k}
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
        ))}
      </Section>
      <div className="border-t border-line px-4 py-2">
        <label className="flex items-center gap-2 text-small">
          <input type="checkbox" checked={!!variant.job} onChange={(e) => setVariantJob(e.target.checked)} />
          own data and training config for variant {project.active}
        </label>
      </div>
      <DataPanel />
      <Section title="Training">
        <TrainField k="epochs" />
        <TrainField k="max_steps" />
        <TrainField k="batch_size" label="batch" />
        <TrainSelect k="name" path="optimizer" label="optimizer" options={["adamw", "adam", "sgd"]} />
        <TrainField k="lr" path="optimizer" />
        <TrainField k="weight_decay" path="optimizer" label="weight decay" />
        <TrainSelect k="name" path="scheduler" label="schedule" options={["cosine", "linear", "constant"]} />
        <TrainField k="warmup_steps" path="scheduler" label="warmup" />
        <TrainSelect k="precision" options={["bf16", "f16", "f32"]} />
        <TrainField k="grad_clip" label="grad clip" />
        <TrainField k="seed" />
        <TrainField k="log_every" label="log every" />
        <TrainField k="eval_every" label="eval every" />
        <TrainField k="every_steps" path="checkpoint" label="checkpoint" />
        <TrainBool k="grad_checkpointing" />
        <TrainBool k="compile" />
      </Section>
      <VariantsPanel />
    </div>
  );
}
