import { useMemo, useState } from "react";
import { keepTyping } from "./keys";
import { addNode, insertBlock, insertModelInline } from "../ir/ops";
import { CATALOG } from "../ir/registry";
import templates from "../ir/templates.json";
import { useStore } from "../state/store";
import { viewportCenter } from "./Canvas";
import { fmtCount } from "./Inspector";

const ORDER = ["io", "layers", "convolution", "pooling", "attention", "sequence", "embedding", "norm", "activation", "structure", "generative", "loss", "custom"];
const NAMES: Record<string, string> = {
  io: "Input / output",
  layers: "Layers",
  convolution: "Convolution",
  pooling: "Pooling & resampling",
  attention: "Attention",
  sequence: "Sequence",
  embedding: "Embedding & position",
  norm: "Normalisation",
  activation: "Activation & regularisation",
  structure: "Tensor ops",
  generative: "Generative",
  loss: "Loss",
  custom: "Custom",
};

export const NODE_MIME = "application/x-weightsmith-node";
export const TEMPLATE_MIME = "application/x-weightsmith-template";

export function addAtCenter(type: string) {
  const st = useStore.getState();
  let id = "";
  st.apply(`add ${type}`, (g) => {
    const [next, nid] = addNode(g, st.scope, type, viewportCenter());
    id = nid;
    return next;
  });
  st.select([id]);
}

/** Blocks become a subgraph node; models are merged in with prefixed ids. */
export function insertTemplate(id: string, at = viewportCenter()) {
  const st = useStore.getState();
  const t = (templates as any[]).find((x) => x.id === id);
  if (!t) return;
  if (st.scope) {
    st.notify("blocks and models go on the top-level graph; leave this block first (Esc)", true);
    return;
  }
  let ids: string[] = [];
  if (t.kind === "block") {
    st.apply(`add block ${t.name}`, (g) => {
      const [next, nid] = insertBlock(g, "", t.id, t.block, at);
      ids = [nid];
      return next;
    });
  } else {
    st.apply(`insert model ${t.name}`, (g) => {
      const [next, nids] = insertModelInline(g, t.graph, t.id, at);
      ids = nids;
      return next;
    });
    st.notify(`${t.name}: ${ids.length} nodes inserted${t.reference ? `, ${fmtCount(t.reference)} params` : ""}`);
  }
  st.select(ids);
}

function Group({ title, children, open, onToggle, count }: { title: string; children: React.ReactNode; open: boolean; onToggle: () => void; count: number }) {
  return (
    <div className="mt-2">
      <button className="link flex w-full items-baseline justify-between !text-[11px]" onClick={onToggle} aria-expanded={open}>
        <span>{title}</span>
        <span className="num">{open ? "−" : count}</span>
      </button>
      {open && <div className="mt-1">{children}</div>}
    </div>
  );
}

function Item({ label, title, mime, value, onClick, aside }: { label: string; title: string; mime: string; value: string; onClick: () => void; aside?: string }) {
  return (
    <div
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData(mime, value);
        e.dataTransfer.effectAllowed = "move";
      }}
      onClick={onClick}
      className="flex cursor-grab items-baseline justify-between gap-2 py-0.5 hover:underline"
      title={title}
    >
      <span className="truncate">{label}</span>
      {aside && <span className="num shrink-0 text-label text-muted">{aside}</span>}
    </div>
  );
}

export function Library() {
  const [q, setQ] = useState("");
  const [closed, setClosed] = useState<Record<string, boolean>>(() => {
    try {
      return JSON.parse(localStorage.getItem("weightsmith.library") ?? "{}");
    } catch {
      return {};
    }
  });
  const toggle = (k: string) => {
    const next = { ...closed, [k]: !closed[k] };
    setClosed(next);
    try {
      localStorage.setItem("weightsmith.library", JSON.stringify(next));
    } catch {
      /* ignore */
    }
  };
  const needle = q.toLowerCase();
  const match = (...s: string[]) => !needle || s.some((x) => x.toLowerCase().includes(needle));

  const groups = useMemo(() => {
    const hits = CATALOG.filter((t) => match(t.type, t.doc, t.category));
    return ORDER.map((c) => [c, hits.filter((t) => t.category === c)] as const).filter(([, l]) => l.length);
  }, [needle]);
  const blocks = (templates as any[]).filter((t) => t.kind === "block" && match(t.name, t.doc));
  const models = (templates as any[]).filter((t) => t.kind === "model" && match(t.name, t.doc, t.category));
  const open = (k: string) => !!needle || !closed[k];

  return (
    <nav className="flex h-full flex-col border-r border-line" style={{ width: 240 }} aria-label="Node library">
      <div className="p-2">
        <input
          type="text"
          placeholder={`Search ${CATALOG.length} nodes, ${blocks.length + models.length ? "blocks, models" : ""}`}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          aria-label="Search the library"
          onKeyDown={(e) => {
            keepTyping(e);
            if (e.key === "Enter") {
              if (groups[0]) addAtCenter(groups[0][1][0].type);
              else if (blocks[0] || models[0]) insertTemplate((blocks[0] ?? models[0]).id);
            }
          }}
        />
      </div>
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        {models.length > 0 && (
          <Group title="Models" open={open("models")} onToggle={() => toggle("models")} count={models.length}>
            {models.map((t) => (
              <Item key={t.id} label={t.name} title={`${t.doc}\nClick or drag to insert. New project: Projects tab.`} mime={TEMPLATE_MIME} value={t.id}
                onClick={() => insertTemplate(t.id)} aside={t.reference ? fmtCount(t.reference) : undefined} />
            ))}
          </Group>
        )}
        {blocks.length > 0 && (
          <Group title="Blocks" open={open("blocks")} onToggle={() => toggle("blocks")} count={blocks.length}>
            {blocks.map((t) => (
              <Item key={t.id} label={t.name} title={`${t.doc}\nAdds a parametric block (edit its args in the inspector).`} mime={TEMPLATE_MIME} value={t.id}
                onClick={() => insertTemplate(t.id)} />
            ))}
          </Group>
        )}
        {groups.map(([cat, list]) => (
          <Group key={cat} title={NAMES[cat] ?? cat} open={open(cat)} onToggle={() => toggle(cat)} count={list.length}>
            {list.map((t) => (
              <Item key={t.type} label={t.type === "PythonModule" ? "Python module" : t.type} title={t.doc} mime={NODE_MIME} value={t.type} onClick={() => addAtCenter(t.type)} />
            ))}
          </Group>
        ))}
      </div>
    </nav>
  );
}
