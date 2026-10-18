import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import templates from "../ir/templates.json";
import { useStore } from "../state/store";
import { fmtCount } from "./Inspector";

interface Card {
  slug: string;
  name: string;
  variants: string[];
  updated: number;
  description?: string;
  params?: number;
  trainable?: number;
  runs?: number;
  nodes?: number;
  thumb?: number[][];
}

const ago = (t: number) => {
  const s = Date.now() / 1000 - t;
  if (s < 90) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
};

function Thumb({ points }: { points?: number[][] }) {
  if (!points?.length) return <div className="flex h-24 items-center justify-center text-label text-muted">empty graph</div>;
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const [x0, y0] = [Math.min(...xs), Math.min(...ys)];
  const w = Math.max(...xs) - x0 + 184;
  const h = Math.max(...ys) - y0 + 64;
  return (
    <svg viewBox={`${x0 - 24} ${y0 - 24} ${w + 48} ${h + 48}`} className="block h-24 w-full" preserveAspectRatio="xMidYMid meet" aria-hidden>
      {points.map(([x, y], i) => (
        <rect key={i} x={x} y={y} width={184} height={56} rx={2} fill="var(--bg)" stroke="var(--ink)" strokeWidth={Math.max(1, w / 260)} vectorEffect="non-scaling-stroke" />
      ))}
    </svg>
  );
}

function ProjectCard({ c, current }: { c: Card; current: boolean }) {
  const open = useStore((s) => s.openProject);
  const dup = useStore((s) => s.duplicateProject);
  const del = useStore((s) => s.deleteProject);
  const [confirm, setConfirm] = useState(false);
  const exportJson = async () => {
    const p = await api.project(c.slug);
    const url = URL.createObjectURL(new Blob([JSON.stringify(p, null, 2)], { type: "application/json" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: `${c.slug}.weightsmith.json` });
    a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <article className="flex min-w-0 flex-col border bg-bg" style={{ borderColor: current ? "var(--accent)" : "var(--line)", borderRadius: 2 }}>
      <button className="!h-auto !border-0 !border-b !border-line !p-2 text-left" onClick={() => open(c.slug)} aria-label={`Open ${c.name}`}>
        <Thumb points={c.thumb} />
      </button>
      <div className="flex flex-1 flex-col gap-1 p-3">
        <div className="flex items-baseline justify-between gap-2">
          <button className="link min-w-0 truncate !text-[13px] font-medium !text-ink" onClick={() => open(c.slug)}>
            {c.name}
          </button>
          {current && <span className="shrink-0 text-label text-accent">open</span>}
        </div>
        {c.description && <p className="m-0 line-clamp-2 text-small text-muted">{c.description}</p>}
        <div className="num mt-auto flex flex-wrap gap-x-3 pt-2 text-label text-muted">
          <span>{c.variants.length} variant{c.variants.length === 1 ? "" : "s"}</span>
          {c.params !== undefined && <span className="text-ink">{fmtCount(c.params)} params</span>}
          {c.trainable !== undefined && c.trainable !== c.params && <span>{fmtCount(c.trainable)} trainable</span>}
          {!!c.runs && <span>{c.runs} runs</span>}
          <span>{ago(c.updated)}</span>
        </div>
        <div className="flex gap-3 pt-2 text-small">
          <button className="link" onClick={() => open(c.slug)}>
            Open
          </button>
          <button className="link" onClick={() => dup(c.slug)}>
            Duplicate
          </button>
          <button className="link" onClick={exportJson}>
            Export
          </button>
          {confirm ? (
            <>
              <button className="link !text-error" onClick={() => del(c.slug)}>
                Delete {c.name}
              </button>
              <button className="link" onClick={() => setConfirm(false)}>
                Keep
              </button>
            </>
          ) : (
            <button className="link ml-auto" onClick={() => setConfirm(true)}>
              Delete
            </button>
          )}
        </div>
      </div>
    </article>
  );
}

const CATS: [string, string][] = [
  ["language", "Language"],
  ["vision", "Vision"],
  ["sequence", "Sequence"],
  ["generative", "Generative"],
];

export function ProjectsView() {
  const projects = useStore((s) => s.projects) as Card[];
  const current = useStore((s) => s.project.slug);
  const daemon = useStore((s) => s.daemon);
  const refresh = useStore((s) => s.refreshProjects);
  const newProject = useStore((s) => s.newProject);
  const fromTemplate = useStore((s) => s.newFromTemplate);
  const importProject = useStore((s) => s.importProject);
  const [q, setQ] = useState("");
  const file = useRef<HTMLInputElement>(null);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const shown = useMemo(() => {
    const n = q.toLowerCase();
    return projects.filter((p) => !n || p.name.toLowerCase().includes(n) || (p.description ?? "").toLowerCase().includes(n));
  }, [projects, q]);
  const models = (templates as any[]).filter((t) => t.kind === "model");

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex max-w-[1152px] flex-col gap-8 px-6 py-8">
        <header className="flex flex-wrap items-end gap-4">
          <div className="min-w-0 flex-1">
            <h1 className="m-0 text-project font-medium leading-8">Projects</h1>
            <p className="m-0 text-small text-muted">
              {daemon === "offline" ? "Daemon offline: start it with weightsmith serve to list saved projects." : `${projects.length} in ${"~/.weightsmith"}`}
            </p>
          </div>
          <div className="w-56">
            <input type="text" placeholder="Search projects" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search projects" />
          </div>
          <button onClick={() => file.current?.click()}>Import…</button>
          <input
            ref={file}
            type="file"
            accept="application/json,.json"
            hidden
            onChange={async (e) => {
              const f = e.target.files?.[0];
              if (!f) return;
              try {
                await importProject(JSON.parse(await f.text()));
              } catch {
                useStore.getState().notify("could not read that file as a project", true);
              }
              e.target.value = "";
            }}
          />
          <button className="primary" onClick={() => newProject("Untitled")}>
            New project
          </button>
        </header>

        <section className="grid grid-cols-[repeat(auto-fill,minmax(260px,1fr))] gap-4" aria-label="Your projects">
          {shown.map((c) => (
            <ProjectCard key={c.slug} c={c} current={c.slug === current} />
          ))}
          {!shown.length && <p className="m-0 text-small text-muted">{q ? "No project matches." : "No saved projects yet. Start one from a model below."}</p>}
        </section>

        <section className="flex flex-col gap-4" aria-labelledby="from-model">
          <div>
            <h2 id="from-model" className="m-0 text-title font-medium">
              Start from a model
            </h2>
            <p className="m-0 text-small text-muted">
              Every model opens as an editable graph. Where a parameter count is published, the graph reproduces it exactly.
            </p>
          </div>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(320px,1fr))] gap-x-8 gap-y-6">
            {CATS.map(([cat, title]) => (
              <div key={cat} className="min-w-0">
                <div className="mb-1 border-b border-line pb-1 text-label text-muted">{title}</div>
                {models
                  .filter((t) => t.category === cat)
                  .map((t) => (
                    <button
                      key={t.id}
                      className="group !flex !h-auto w-full items-baseline gap-3 !border-0 !border-b !border-line !px-0 py-2 text-left"
                      onClick={() => fromTemplate(t.id)}
                      title={t.doc}
                    >
                      <span className="min-w-0 flex-1">
                        <span className="block truncate font-medium group-hover:underline">{t.name}</span>
                        <span className="block truncate text-small text-muted">{t.doc}</span>
                      </span>
                      <span className="num shrink-0 text-label">{t.reference ? fmtCount(t.reference) : ""}</span>
                    </button>
                  ))}
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
