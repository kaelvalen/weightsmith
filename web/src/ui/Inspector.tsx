import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { fixesFor } from "../ir/smart";
import { keepTyping } from "./keys";
import { Select } from "./Select";
import { api } from "../api/client";
import { adaptDefaults, splitRef, type Issue } from "../ir/analyze";
import { bulkAdapt, getBody, renameNode, setParam, updateNode } from "../ir/ops";
import { TYPES, expandDefaults, linearTargetsOf, outFeaturesOf, weightsOf } from "../ir/registry";
import type { Adapt, Node } from "../ir/types";
import { useAnalysis, useIssues, useStore } from "../state/store";
import { JobPanel } from "./JobPanel";

const CodeEditor = lazy(() => import("./CodeEditor"));

export const fmtCount = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(n);
const shapeStr = (s?: (string | number)[]) => (s ? (s.length ? s.join("×") : "scalar") : "—");

/** One-click fixes for an issue, when the fix is obvious. */
export function FixButtons({ issue, block }: { issue: Issue; block?: boolean }) {
  const scope = useStore((s) => s.scope);
  const graph = useStore((s) => s.graph());
  const apply = useStore((s) => s.apply);
  // issues from inside a block carry "block/node"
  const [where, node] = issue.node?.includes("/") ? issue.node.split("/") : [issue.node ? scope : "", issue.node];
  const fixes = fixesFor({ ...issue, node }, graph, where);
  if (!fixes.length) return null;
  return (
    <span className={block ? "mt-1 flex flex-wrap gap-1" : "ml-2 inline-flex flex-wrap gap-1 align-baseline"}>
      {fixes.map((f) => (
        <button key={f.label} className="!h-5 !px-1.5 text-label" onClick={() => apply(f.label, f.apply)}>
          {f.label}
        </button>
      ))}
    </span>
  );
}

export function Section({ title, children, aside }: { title: string; children: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <section className="border-t border-line px-4 py-3">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="m-0 text-label font-medium text-muted">{title}</h3>
        {aside}
      </div>
      {children}
    </section>
  );
}

export function Row({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return (
    <label className="mb-1 grid grid-cols-[96px_1fr] items-center gap-2" title={hint}>
      <span className="truncate text-small text-muted">{label}</span>
      <span className="min-w-0">{children}</span>
    </label>
  );
}

/** A text input that commits on blur/Enter, so typing does not flood the operation log. */
export function Field({ value, onCommit, mono, placeholder }: { value: string; onCommit: (v: string) => void; mono?: boolean; placeholder?: string }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  return (
    <input
      type="text"
      className={mono ? "num" : ""}
      value={v}
      placeholder={placeholder}
      onChange={(e) => setV(e.target.value)}
      onBlur={() => v !== value && onCommit(v)}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        if (e.key === "Escape") setV(value);
        keepTyping(e);
      }}
    />
  );
}

function parseShape(s: string): (string | number)[] {
  return s
    .split(/[,×x\s]+/)
    .map((t) => t.trim())
    .filter(Boolean)
    .map((t) => (/^-?\d+$/.test(t) ? Number(t) : t));
}

function ParamInput({ node, k }: { node: Node; k: string }) {
  const scope = useStore((s) => s.scope);
  const apply = useStore((s) => s.apply);
  const spec = TYPES[node.type].params[k];
  const params = (node.params ?? {}) as Record<string, any>;
  const v = k in params ? params[k] : spec.default;
  const set = (val: unknown) => apply(`${node.id}.${k} = ${JSON.stringify(val)}`, (g) => setParam(g, scope, node.id, k, val));

  switch (spec.kind) {
    case "bool":
      return <input type="checkbox" checked={!!v} onChange={(e) => set(e.target.checked)} />;
    case "choice":
      return (
        <Select value={String(v)} options={spec.choices!.map(String)} onChange={set} ariaLabel={k} />
      );
    case "shape":
      return <Field mono value={Array.isArray(v) ? v.join(", ") : ""} onCommit={(s) => set(parseShape(s))} placeholder="B, 784" />;
    case "int":
    case "float":
      return (
        <Field
          mono
          value={v === null || v === undefined ? "" : String(v)}
          placeholder={spec.required ? "required" : "default"}
          onCommit={(s) => {
            if (s.trim() === "") return set(spec.required ? null : undefined);
            const n = Number(s);
            if (!Number.isNaN(n)) set(n);
          }}
        />
      );
    case "json":
      return <Field mono value={JSON.stringify(v ?? {})} onCommit={(s) => { try { set(JSON.parse(s)); } catch { /* keep */ } }} />;
    case "str":
      if (node.type === "Subgraph" && k === "graph") {
        const subs = Object.keys(useStore.getState().graph().subgraphs ?? {});
        return (
          <Select value={String(v ?? "")} options={subs} onChange={set} ariaLabel="block" />
        );
      }
      return <Field value={String(v ?? "")} onCommit={set} />;
    default:
      return null;
  }
}

function argText(v: unknown): string {
  return v === undefined ? "" : typeof v === "string" ? v : JSON.stringify(v);
}

function parseArg(text: string, like: unknown): unknown {
  if (typeof like === "string") return text;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

/** One field per block parameter; empty means the block's default. */
function BlockArgs({ node }: { node: Node }) {
  const scope = useStore((s) => s.scope);
  const apply = useStore((s) => s.apply);
  const graph = useStore((s) => s.graph());
  const p = (node.params ?? {}) as Record<string, any>;
  const def = (graph.subgraphs?.[p.graph] ?? {}) as { params?: Record<string, unknown>; doc?: string };
  const params = def.params ?? {};
  const args = (p.args ?? {}) as Record<string, unknown>;
  const setArg = (k: string, text: string) =>
    apply(`${node.id}: ${k}`, (g) =>
      updateNode(g, scope, node.id, (n) => {
        const a = { ...((n.params as any)?.args ?? {}) };
        if (text.trim() === "") delete a[k];
        else a[k] = parseArg(text, params[k]);
        n.params = { ...(n.params ?? {}), args: a };
        if (!Object.keys(a).length) delete (n.params as any).args;
      }),
    );
  if (!Object.keys(params).length) return null;
  return (
    <Section title="Block parameters">
      {def.doc && <p className="m-0 mb-2 text-small text-muted">{def.doc}</p>}
      {Object.entries(params).map(([k, d]) => (
        <Row key={k} label={`$${k}`} hint={`default ${argText(d)}`}>
          <Field mono value={argText(args[k])} placeholder={argText(d)} onCommit={(v) => setArg(k, v)} />
        </Row>
      ))}
    </Section>
  );
}

function PythonPanel({ node }: { node: Node }) {
  const scope = useStore((s) => s.scope);
  const apply = useStore((s) => s.apply);
  const analysis = useAnalysis();
  const p = (node.params ?? {}) as Record<string, any>;
  const [code, setCode] = useState<string>(p.code ?? "");
  const [result, setResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => setCode(p.code ?? ""), [node.id]);

  const check = async () => {
    setBusy(true);
    try {
      // 1. learn which class the code defines and what its constructor takes
      const probe = await api.python({ ...p, code });
      const found = probe.signature;
      if (!found) {
        setResult(probe);
        return;
      }
      const renamed = found.class_name !== (p.class_name ?? "Block");
      // init written for another class (e.g. the template) does not carry over; unknown keys are dropped
      const init: Record<string, unknown> = renamed
        ? {}
        : Object.fromEntries(Object.entries(p.init ?? {}).filter(([k]) => k in found.init_params));
      const params = { ...p, code, class_name: found.class_name, init };
      // 2. check it against the shapes actually flowing into this node
      const sigResp = await api.python(params);
      let resp = sigResp;
      if (sigResp.ok) {
        const a = scope ? analysis.subgraphs?.[scope] : analysis;
        const inputs: Record<string, any> = {};
        for (const port of sigResp.signature.inputs) {
          const t = a?.ports?.[`${node.id}.${port}`];
          if (t) inputs[port] = t;
        }
        if (Object.keys(inputs).length) resp = await api.python(params, inputs);
      }
      const sig = sigResp.signature ?? found;
      apply(`${node.id}: python code`, (g) =>
        updateNode(g, scope, node.id, (n) => {
          n.params = {
            ...(n.params ?? {}),
            code,
            class_name: found.class_name,
            init,
            signature: { inputs: sig.inputs, optional: sig.optional, outputs: sig.outputs, params: sig.params },
          };
        }),
      );
      setResult({ ...resp, renamed: renamed ? found.class_name : null, initParams: found.init_params, init });
    } catch (e: any) {
      setResult({ ok: false, error: e.message ?? String(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Section title="Python module" aside={<button onClick={check} disabled={busy}>{busy ? "checking…" : "Check & apply"}</button>}>
      <div className="mb-2 border border-line">
        <Suspense fallback={<div className="h-[288px] p-2 text-muted">loading editor…</div>}>
          <CodeEditor value={code} onChange={setCode} height={288} />
        </Suspense>
      </div>
      <p className="m-0 text-label text-muted">
        forward() parameters become input ports; <span className="num">outputs = ("out",)</span> names output ports. infer_shape runs on symbolic shapes, then the
        module is checked on the meta device with fake tensors.
      </p>
      {result?.renamed && <div className="mt-2 text-label">class_name → {result.renamed}; init reset to the constructor defaults below</div>}
      {result?.initParams && (
        <div className="num mt-2 text-label text-muted" title="set these in init (Parameters) as JSON">
          {result.renamed ?? p.class_name}(
          {Object.entries(result.initParams as Record<string, unknown>)
            .map(([k, v]) => `${k}=${k in (result.init ?? {}) ? JSON.stringify(result.init[k]) : v === null ? "required" : JSON.stringify(v)}`)
            .join(", ")}
          )
        </div>
      )}
      {result && (
        <div className={`num mt-2 text-label whitespace-pre-wrap ${result.ok ? "" : "text-error"}`}>
          {result.ok
            ? `ok · in ${result.signature.inputs.join(", ")} · out ${result.signature.outputs.join(", ")} · ${fmtCount(result.signature.params)} params${
                result.outputs ? " · " + Object.entries(result.outputs).map(([k, v]: any) => `${k} ${shapeStr(v.shape)}`).join(", ") : ""
              }`
            : result.error}
        </div>
      )}
    </Section>
  );
}

function AdaptPanel({ node }: { node: Node }) {
  const scope = useStore((s) => s.scope);
  const apply = useStore((s) => s.apply);
  const notify = useStore((s) => s.notify);
  const p = expandDefaults(node.type, node.params as any);
  const targets = Object.keys(linearTargetsOf(node.type, p));
  const canAdapter = outFeaturesOf(node.type, p) !== null;
  const list = node.adapt ?? [];
  const edit = (label: string, fn: (a: Adapt[]) => Adapt[]) =>
    apply(`${node.id}: ${label}`, (g) => updateNode(g, scope, node.id, (n) => (n.adapt = fn([...(n.adapt ?? [])]))));
  const setField = (i: number, k: keyof Adapt, v: unknown) => edit(`adapt[${i}].${k}`, (a) => a.map((x, j) => (j === i ? { ...x, [k]: v } : x)));

  if (!targets.length && !canAdapter) return null;
  const free = targets.filter((t) => !list.some((a) => a.kind === "lora" && (a.target ?? "weight") === t));
  return (
    <Section
      title="Adaptation"
      aside={
        <span className="flex gap-1">
          {free.length > 0 && <button onClick={() => edit("add LoRA", (a) => [...a, { kind: "lora", target: free[0], r: 8, alpha: 16 }])}>+ LoRA</button>}
          {canAdapter && !list.some((a) => a.kind === "adapter") && <button onClick={() => edit("add adapter", (a) => [...a, { kind: "adapter", bottleneck: 64 }])}>+ Adapter</button>}
        </span>
      }
    >
      {!list.length && <p className="m-0 text-small text-muted">No modifiers. LoRA wraps a module's forward; it never adds a node.</p>}
      {list.map((a0, i) => {
        const a = { ...adaptDefaults(a0.kind), ...a0 };
        return (
          <div key={i} className="mb-3 border-l border-line pl-2">
            <div className="mb-1 flex items-baseline justify-between">
              <span className="font-medium">{a.kind === "lora" ? "LoRA" : a.kind === "adapter" ? "Bottleneck adapter" : "Custom"}</span>
              <button className="link" onClick={() => edit(`remove adapt[${i}]`, (l) => l.filter((_, j) => j !== i))}>
                remove
              </button>
            </div>
            {a.kind === "lora" && (
              <>
                <Row label="target">
                  <Select mono value={a.target ?? "weight"} options={targets} onChange={(v) => setField(i, "target", v)} ariaLabel="LoRA target" />
                </Row>
                <Row label="r">
                  <Field mono value={String(a.r)} onCommit={(v) => setField(i, "r", Number(v) || 1)} />
                </Row>
                <Row label="alpha">
                  <Field mono value={String(a.alpha)} onCommit={(v) => setField(i, "alpha", Number(v))} />
                </Row>
                <Row label="dropout">
                  <Field mono value={String(a.dropout)} onCommit={(v) => setField(i, "dropout", Number(v))} />
                </Row>
                <Row label="init">
                  <Select
                    value={a.init ?? "zeros_b"}
                    options={[{ value: "zeros_b", label: "B = 0" }, { value: "gaussian", label: "gaussian" }]}
                    onChange={(v) => setField(i, "init", v)}
                    ariaLabel="LoRA init"
                  />
                </Row>
              </>
            )}
            {a.kind === "adapter" && (
              <Row label="bottleneck">
                <Field mono value={String(a.bottleneck)} onCommit={(v) => setField(i, "bottleneck", Number(v) || 1)} />
              </Row>
            )}
            <Row label="freeze base">
              <input type="checkbox" checked={a.freeze !== false} onChange={(e) => setField(i, "freeze", e.target.checked)} />
            </Row>
            <Row label="disabled" hint="ablation: skip this modifier">
              <input type="checkbox" checked={!!a.disabled} onChange={(e) => setField(i, "disabled", e.target.checked || undefined)} />
            </Row>
          </div>
        );
      })}
      {list.length > 0 && (
        <button
          onClick={() => {
            let touched: string[] = [];
            apply(`bulk: ${node.type} adaptation`, (g) => {
              let next = g;
              for (const a of list) {
                const [n, t] = bulkAdapt(next, { ...a, target: undefined } as Adapt, [a.target ?? (a.kind === "lora" ? "weight" : "out")], [node.type]);
                next = n;
                touched = touched.concat(t);
              }
              return next;
            });
            notify(touched.length ? `added to ${touched.length} target${touched.length > 1 ? "s" : ""}: ${touched.slice(0, 4).join(", ")}${touched.length > 4 ? "…" : ""}` : `every ${node.type} already has these modifiers`);
          }}
        >
          Apply to every {node.type}
        </button>
      )}
    </Section>
  );
}

function TrainPanel({ node }: { node: Node }) {
  const scope = useStore((s) => s.scope);
  const apply = useStore((s) => s.apply);
  const analysis = useAnalysis();
  const a = scope ? analysis.subgraphs?.[scope] : analysis;
  const count = a?.params?.[node.id];
  const weights = Object.keys(weightsOf(node.type, expandDefaults(node.type, node.params as any)));
  const frozen = new Set(node.frozen_params ?? []);
  const toggleMask = (w: string, on: boolean) =>
    apply(`${node.id}: freeze ${w}`, (g) =>
      updateNode(g, scope, node.id, (n) => {
        const s = new Set(n.frozen_params ?? []);
        if (on) s.add(w);
        else s.delete(w);
        n.frozen_params = [...s].sort();
        if (!n.frozen_params.length) delete n.frozen_params;
      }),
    );
  if (["Input", "Output"].includes(node.type) || TYPES[node.type].loss) return null;
  return (
    <Section title="Trainability" aside={count && <span className="num text-label">{count.known ? `${fmtCount(count.trainable)} / ${fmtCount(count.total)}` : "after Check"}</span>}>
      {(weights.length > 0 || ["Subgraph", "PythonModule"].includes(node.type)) && (
        <Row label="trainable">
          <input
            type="checkbox"
            checked={node.trainable !== false}
            onChange={(e) => apply(`${node.id}: trainable ${e.target.checked}`, (g) => updateNode(g, scope, node.id, (n) => (e.target.checked ? delete n.trainable : (n.trainable = false))))}
          />
        </Row>
      )}
      {node.trainable !== false && weights.length > 1 && (
        <div className="mt-1">
          {weights.map((w) => (
            <label key={w} className="num flex items-center gap-2 text-label">
              <input type="checkbox" checked={frozen.has(w)} onChange={(e) => toggleMask(w, e.target.checked)} />
              freeze {w}
            </label>
          ))}
        </div>
      )}
      <Row label="disabled" hint="ablation: pass the input through unchanged">
        <input
          type="checkbox"
          checked={!!node.disabled}
          onChange={(e) => apply(`${node.id}: ${e.target.checked ? "disable" : "enable"}`, (g) => updateNode(g, scope, node.id, (n) => (e.target.checked ? (n.disabled = true) : delete n.disabled)))}
        />
      </Row>
    </Section>
  );
}

function NodePanel({ node }: { node: Node }) {
  const scope = useStore((s) => s.scope);
  const apply = useStore((s) => s.apply);
  const select = useStore((s) => s.select);
  const setScope = useStore((s) => s.setScope);
  const focus = useStore((s) => s.inspectorFocus);
  const analysis = useAnalysis();
  const allIssues = useIssues();
  const a = scope ? analysis.subgraphs?.[scope] : analysis;
  const issues = (scope ? (a?.issues ?? []) : allIssues).filter((i) => i.node === node.id && i.severity !== "peer");
  const t = TYPES[node.type];
  const sig = a?.signatures?.[node.id];
  const firstRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (focus) firstRef.current?.querySelector("input")?.focus();
  }, [focus]);
  const graph = useStore((s) => s.graph());
  const body = getBody(graph, scope);
  const source = (port: string) => body.edges.find((e) => e[1] === `${node.id}.${port}`)?.[0];

  return (
    <>
      <div ref={firstRef} className="px-4 pt-3 pb-3">
        <div className="text-label text-muted">{t?.category} · {node.type}</div>
        <div className="mt-1 mb-2">
          <Field
            value={node.label ?? ""}
            placeholder={node.id}
            onCommit={(v) => apply(`${node.id}: label`, (g) => updateNode(g, scope, node.id, (n) => (v ? (n.label = v) : delete n.label)))}
          />
        </div>
        <Row label="id">
          <Field
            mono
            value={node.id}
            onCommit={(v) => {
              apply(`rename ${node.id} → ${v}`, (g) => renameNode(g, scope, node.id, v));
              select([v]);
            }}
          />
        </Row>
        <p className="m-0 mt-2 text-small text-muted">{t?.doc}</p>
        {node.type === "Subgraph" && (
          <button className="mt-2" onClick={() => setScope((node.params as any).graph)}>
            Open block ↵
          </button>
        )}
      </div>
      {issues.length > 0 && (
        <div className="border-t border-line px-4 py-2">
          {issues.map((i, k) => (
            <div key={k} className={`text-small ${i.severity === "error" ? "text-error" : "text-muted"}`}>
              {i.port ? <span className="num">{i.port}: </span> : null}
              {i.message}
              <FixButtons issue={i} />
            </div>
          ))}
        </div>
      )}
      {t && Object.keys(t.params).filter((k) => !["code", "signature"].includes(k)).length > 0 && (
        <Section title="Parameters">
          {Object.keys(t.params)
            .filter((k) => !["code", "signature"].includes(k) && !(node.type === "Subgraph" && k === "args"))
            .map((k) => (
              <Row key={k} label={k} hint={t.params[k].doc}>
                <ParamInput node={node} k={k} />
              </Row>
            ))}
        </Section>
      )}
      {sig && (sig.inputs.length > 0 || sig.outputs.length > 0) && (
        <Section title="Ports">
          {sig.inputs.map((p) => {
            const tt = a?.ports?.[`${node.id}.${p}`];
            const src = source(p);
            return (
              <div key={p} className="num grid grid-cols-[72px_1fr] gap-2 text-small">
                <span className="text-muted">
                  in {p}
                  {sig.optional.includes(p) ? "?" : ""}
                </span>
                <span className="truncate">
                  {shapeStr(tt?.shape)} <span className="text-muted">{tt?.dtype ?? ""}</span>
                  {src && (
                    <button className="link ml-2" onClick={() => select([splitRef(src)[0]])}>
                      ← {src}
                    </button>
                  )}
                  {!src && sig.optional.includes(p) && <span className="ml-2 text-muted">dataset {(expandDefaults(node.type, node.params as any) as any).target_field}</span>}
                </span>
              </div>
            );
          })}
          {sig.outputs.map((p) => {
            const tt = a?.ports?.[`${node.id}.${p}`];
            return (
              <div key={p} className="num grid grid-cols-[72px_1fr] gap-2 text-small">
                <span className="text-muted">out {p}</span>
                <span>
                  {shapeStr(tt?.shape)} <span className="text-muted">{tt?.dtype ?? ""}</span>
                </span>
              </div>
            );
          })}
        </Section>
      )}
      {node.type === "Subgraph" && <BlockArgs node={node} />}
      {node.type === "PythonModule" && <PythonPanel node={node} />}
      <AdaptPanel node={node} />
      <TrainPanel node={node} />
    </>
  );
}

export function Inspector() {
  const scope = useStore((s) => s.scope);
  const selection = useStore((s) => s.selection);
  const graph = useStore((s) => s.graph());
  const body = getBody(graph, scope);
  const node = selection.length === 1 ? body.nodes.find((n) => n.id === selection[0]) : undefined;
  return (
    <aside className="h-full overflow-y-auto border-l border-line" style={{ width: 320 }}>
      {node ? (
        <NodePanel key={`${scope}/${node.id}`} node={node} />
      ) : selection.length > 1 ? (
        <MultiPanel ids={selection} />
      ) : scope ? (
        <BlockPanel name={scope} />
      ) : (
        <JobPanel />
      )}
    </aside>
  );
}

function MultiPanel({ ids }: { ids: string[] }) {
  const scope = useStore((s) => s.scope);
  const apply = useStore((s) => s.apply);
  const analysis = useAnalysis();
  const a = scope ? analysis.subgraphs?.[scope] : analysis;
  const total = ids.reduce((s, id) => s + (a?.params?.[id]?.total ?? 0), 0);
  const trainable = ids.reduce((s, id) => s + (a?.params?.[id]?.trainable ?? 0), 0);
  const setAll = (fn: (n: Node) => void, label: string) =>
    apply(label, (g) => ids.reduce((acc, id) => updateNode(acc, scope, id, fn), g));
  return (
    <div className="px-4 py-3">
      <div className="text-title font-medium">{ids.length} nodes</div>
      <div className="num mt-1 text-small text-muted">
        {fmtCount(trainable)} trainable / {fmtCount(total)}
      </div>
      <div className="mt-3 flex flex-wrap gap-1">
        <button onClick={() => setAll((n) => (n.trainable = false), "freeze selection")}>Freeze</button>
        <button onClick={() => setAll((n) => delete n.trainable, "unfreeze selection")}>Unfreeze</button>
        <button onClick={() => window.dispatchEvent(new CustomEvent("ws:command", { detail: "group" }))}>Group into block</button>
      </div>
      <p className="mt-3 text-small text-muted">Cmd+K for bulk modifiers (LoRA on every attention q/v, every Linear…).</p>
    </div>
  );
}


/** Inside a block with nothing selected: what the block is and its parameter defaults. */
function BlockPanel({ name }: { name: string }) {
  const apply = useStore((s) => s.apply);
  const setScope = useStore((s) => s.setScope);
  const graph = useStore((s) => s.graph());
  const analysis = useAnalysis();
  const def = (graph.subgraphs?.[name] ?? {}) as { params?: Record<string, unknown>; doc?: string; nodes?: Node[] };
  const a = analysis.subgraphs?.[name];
  const users = graph.nodes.filter((n) => n.type === "Subgraph" && (n.params as any)?.graph === name);
  const edit = (label: string, fn: (d: any) => void) =>
    apply(label, (g) => {
      const next = structuredClone(g);
      fn(next.subgraphs![name]);
      return next;
    });
  return (
    <div>
      <div className="px-4 pt-3 pb-3">
        <div className="text-label text-muted">block</div>
        <div className="text-title font-medium">{name}</div>
        <div className="num mt-1 text-small text-muted">
          {def.nodes?.length ?? 0} nodes · {a ? fmtCount(a.totals.total) : "—"} params with defaults · used by {users.map((u) => u.id).join(", ") || "nothing"}
        </div>
        <button className="mt-2" onClick={() => setScope("")}>
          ← Back to graph
        </button>
      </div>
      <Section title="Description">
        <textarea
          rows={3}
          defaultValue={def.doc ?? ""}
          onBlur={(e) => e.target.value !== (def.doc ?? "") && edit(`${name}: description`, (d) => (d.doc = e.target.value || undefined))}
          aria-label="Block description"
        />
      </Section>
      <Section title="Parameters and defaults">
        <p className="m-0 mb-2 text-small text-muted">Node params inside the block can use $name or arithmetic such as $D*4. Each instance overrides them under Block parameters.</p>
        {Object.entries(def.params ?? {}).map(([k, v]) => (
          <Row key={k} label={`$${k}`}>
            <Field mono value={argText(v)} onCommit={(t) => edit(`${name}: $${k}`, (d) => (d.params = { ...d.params, [k]: parseArg(t, v) }))} />
          </Row>
        ))}
        <Field
          mono
          value=""
          placeholder="add: name=value"
          onCommit={(t) => {
            const m = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$/.exec(t);
            if (m) edit(`${name}: add $${m[1]}`, (d) => (d.params = { ...(d.params ?? {}), [m[1]]: parseArg(m[2], 0) }));
          }}
        />
      </Section>
      {a && a.issues.filter((i) => i.severity === "error").length > 0 && (
        <Section title="Issues with defaults">
          {a.issues
            .filter((i) => i.severity === "error")
            .map((i, k) => (
              <div key={k} className="text-small text-error">
                <span className="num">{[i.node, i.port].filter(Boolean).join(".")}</span> {i.message}
              </div>
            ))}
        </Section>
      )}
    </div>
  );
}
