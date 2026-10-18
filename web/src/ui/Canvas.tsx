import {
  MiniMap,
  Background,
  BackgroundVariant,
  Connection,
  Edge as FlowEdge,
  Node as FlowNode,
  EdgeChange,
  NodeChange,
  ReactFlow,
  ReactFlowInstance,
  ReactFlowProvider,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Analysis } from "../ir/analyze";
import { addNode, connect, disconnect, getBody, moveNodes, removeNodes } from "../ir/ops";
import type { Position } from "../ir/types";
import { useAnalysis, useIssues, useStore } from "../state/store";
import { NODE_MIME, TEMPLATE_MIME, insertTemplate } from "./Library";
import { NodeData, NodeView } from "./NodeView";

const nodeTypes = { ws: NodeView };

export let flow: ReactFlowInstance | null = null;

/** Center of the visible canvas, in graph coordinates. */
export function viewportCenter(): Position {
  if (!flow) return { x: 0, y: 0 };
  const el = document.querySelector(".react-flow") as HTMLElement | null;
  const r = el?.getBoundingClientRect();
  return flow.screenToFlowPosition({ x: (r?.left ?? 0) + (r?.width ?? 800) / 2 - 92, y: (r?.top ?? 0) + (r?.height ?? 600) / 2 - 40 });
}

function CanvasInner() {
  const scope = useStore((s) => s.scope);
  const graph = useStore((s) => s.graph());
  const selection = useStore((s) => s.selection);
  const selectedEdges = useStore((s) => s.selectedEdges);
  const apply = useStore((s) => s.apply);
  const select = useStore((s) => s.select);
  const setScope = useStore((s) => s.setScope);
  const full = useAnalysis();
  const allIssues = useIssues();
  const [drag, setDrag] = useState<Record<string, Position>>({});
  const [edgeShapes, setEdgeShapes] = useState(() => {
    try {
      return localStorage.getItem("weightsmith.edgeshapes") !== "off";
    } catch {
      return true;
    }
  });
  const [minimap, setMinimap] = useState(() => {
    try {
      return localStorage.getItem("weightsmith.minimap") !== "off";
    } catch {
      return true;
    }
  });
  const wrapper = useRef<HTMLDivElement>(null);

  const body = getBody(graph, scope);
  const analysis: Analysis = scope ? (full.subgraphs?.[scope] ?? full) : full;
  const issues = scope ? (analysis.issues ?? []) : allIssues;

  useEffect(() => {
    const enter = (e: Event) => setScope((e as CustomEvent).detail);
    const fit = () => flow?.fitView({ padding: 0.15, maxZoom: 1, duration: 120 });
    const mm = () =>
      setMinimap((m) => {
        try {
          localStorage.setItem("weightsmith.minimap", m ? "off" : "on");
        } catch {
          /* ignore */
        }
        return !m;
      });
    const shapes = () =>
      setEdgeShapes((v) => {
        try {
          localStorage.setItem("weightsmith.edgeshapes", v ? "off" : "on");
        } catch {
          /* ignore */
        }
        return !v;
      });
    const focus = (e: Event) =>
      setTimeout(() => flow?.fitView({ nodes: [{ id: (e as CustomEvent).detail }], maxZoom: 1.1, duration: 160, padding: 0.6 }), 60);
    window.addEventListener("ws:enter", enter);
    window.addEventListener("ws:fit", fit);
    window.addEventListener("ws:minimap", mm);
    window.addEventListener("ws:edgeshapes", shapes);
    window.addEventListener("ws:focus", focus);
    return () => {
      window.removeEventListener("ws:edgeshapes", shapes);
      window.removeEventListener("ws:focus", focus);
      window.removeEventListener("ws:enter", enter);
      window.removeEventListener("ws:fit", fit);
      window.removeEventListener("ws:minimap", mm);
    };
  }, [setScope]);

  useEffect(() => {
    setTimeout(() => flow?.fitView({ padding: 0.2, maxZoom: 1 }), 0);
  }, [scope, useStore.getState().project.active, useStore.getState().project.slug]);

  const connected = useMemo(() => new Set(body.edges.flat()), [body.edges]);

  const nodes: FlowNode<NodeData>[] = useMemo(
    () =>
      body.nodes.map((n) => ({
        id: n.id,
        type: "ws",
        position: drag[n.id] ?? body.layout?.[n.id] ?? { x: 0, y: 0 },
        selected: selection.includes(n.id),
        data: {
          node: n,
          sig: analysis.signatures?.[n.id] ?? { inputs: [], optional: [], outputs: [] },
          ports: analysis.ports ?? {},
          issues: issues.filter((i) => i.node === n.id),
          count: analysis.params?.[n.id],
          connected,
          inner:
            n.type === "Subgraph" && graph.subgraphs?.[(n.params as any)?.graph]
              ? { body: graph.subgraphs[(n.params as any).graph], analysis: full.subgraphs?.[(n.params as any).graph] }
              : undefined,
        },
      })),
    [body, analysis, issues, selection, drag, connected, graph, full],
  );

  const edges: FlowEdge[] = useMemo(() => {
    const errEdges = new Set(issues.filter((i) => i.edge !== null && i.edge !== undefined && i.severity !== "warning").map((i) => i.edge));
    return body.edges.map(([s, d], i) => {
      const [sn, sp] = [s.slice(0, s.lastIndexOf(".")), s.slice(s.lastIndexOf(".") + 1)];
      const [dn, dp] = [d.slice(0, d.lastIndexOf(".")), d.slice(d.lastIndexOf(".") + 1)];
      return {
        id: `${s}→${d}`,
        source: sn,
        sourceHandle: sp,
        target: dn,
        targetHandle: dp,
        type: "step",
        className: errEdges.has(i) ? "error" : "",
        ...(edgeShapes && analysis.ports?.[s]
          ? {
              label: analysis.ports[s].shape.length ? analysis.ports[s].shape.join("×") : "scalar",
              labelStyle: { fontFamily: "var(--mono)", fontSize: 10, fill: "var(--muted)" },
              labelBgStyle: { fill: "var(--bg)" },
              labelBgPadding: [2, 1] as [number, number],
            }
          : {}),
        selected: selectedEdges.some((e) => e[0] === s && e[1] === d),
      };
    });
  }, [body, issues, selectedEdges, edgeShapes, analysis]);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const moving: Record<string, Position> = {};
      let ended = false;
      const st = useStore.getState();
      const sel = new Set(st.selection);
      let selChanged = false;
      for (const c of changes) {
        if (c.type === "select") {
          selChanged = true;
          if (c.selected) sel.add(c.id);
          else sel.delete(c.id);
        }
        if (c.type === "position" && c.position) moving[c.id] = c.position;
        if (c.type === "position" && c.dragging === false) ended = true;
      }
      if (selChanged) select([...sel], st.selectedEdges);
      if (Object.keys(moving).length) setDrag((d) => ({ ...d, ...moving }));
      if (ended)
        setDrag((d) => {
          const final = { ...d, ...moving };
          if (Object.keys(final).length) apply("move", (g) => moveNodes(g, scope, final), `move:${Object.keys(final).sort().join()}`);
          return {};
        });
    },
    [apply, scope],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      if (!c.sourceHandle || !c.targetHandle) return;
      apply(`connect ${c.source}.${c.sourceHandle} → ${c.target}.${c.targetHandle}`, (g) =>
        connect(g, scope, `${c.source}.${c.sourceHandle}`, `${c.target}.${c.targetHandle}`),
      );
    },
    [apply, scope],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const st = useStore.getState();
      const sel = new Map(st.selectedEdges.map((e) => [e.join("→"), e]));
      let changed = false;
      for (const c of changes) {
        if (c.type !== "select") continue;
        changed = true;
        if (c.selected) sel.set(c.id, c.id.split("→") as [string, string]);
        else sel.delete(c.id);
      }
      if (changed) select(st.selection, [...sel.values()]);
    },
    [select],
  );

  const onDelete = useCallback(
    ({ nodes, edges }: { nodes: FlowNode[]; edges: FlowEdge[] }) => {
      if (nodes.length) apply(`delete ${nodes.map((n) => n.id).join(", ")}`, (g) => removeNodes(g, scope, nodes.map((n) => n.id)));
      else if (edges.length) apply("disconnect", (g) => disconnect(g, scope, edges.map((e) => e.id.split("→") as [string, string])));
    },
    [apply, scope],
  );

  const onDrop = (e: React.DragEvent) => {
    if (!flow) return;
    const tpl = e.dataTransfer.getData(TEMPLATE_MIME);
    if (tpl) {
      e.preventDefault();
      insertTemplate(tpl, flow.screenToFlowPosition({ x: e.clientX, y: e.clientY }));
      return;
    }
    const type = e.dataTransfer.getData(NODE_MIME);
    if (!type) return;
    e.preventDefault();
    const at = flow.screenToFlowPosition({ x: e.clientX, y: e.clientY });
    let id = "";
    apply(`add ${type}`, (g) => {
      const [next, nid] = addNode(g, scope, type, at);
      id = nid;
      return next;
    });
    select([id]);
  };

  return (
    <div ref={wrapper} className="relative h-full w-full" onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
      {!body.nodes.length && (
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center">
          <div className="max-w-sm text-small text-muted">
            <div className="mb-1 text-title font-medium text-ink">Empty graph</div>
            Drag a node, block or model from the library, or press Cmd+K. A model from the Projects tab opens as its own project.
          </div>
        </div>
      )}
      {scope && (
        <div className="absolute top-2 left-2 z-10 flex gap-1 text-small">
          <button className="link" onClick={() => setScope("")}>
            {graph.name || "graph"}
          </button>
          <span className="text-muted">/</span>
          <span>{scope}</span>
        </div>
      )}
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onInit={(i) => {
          flow = i as unknown as ReactFlowInstance;
          i.fitView({ padding: 0.2, maxZoom: 1 });
        }}
        onNodesChange={onNodesChange}
        onConnect={onConnect}
        onEdgesChange={onEdgesChange}
        onDelete={onDelete}
        onPaneClick={() => select([])}
        snapToGrid
        snapGrid={[8, 8]}
        minZoom={0.15}
        maxZoom={2}
        deleteKeyCode={["Backspace", "Delete"]}
        multiSelectionKeyCode={["Shift", "Meta", "Control"]}
        selectionKeyCode="Shift"
        connectionLineType={"step" as any}
        defaultEdgeOptions={{ type: "step" }}
        proOptions={{ hideAttribution: true }}
        onlyRenderVisibleElements
        fitView
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="var(--line)" />
        {minimap && body.nodes.length > 6 && (
          <MiniMap
            pannable
            zoomable
            nodeClassName="ws-mini"
            nodeBorderRadius={0}
            maskColor="color-mix(in srgb, var(--bg) 70%, transparent)"
            style={{ width: 160, height: 112, border: "1px solid var(--line)", background: "var(--bg)", borderRadius: 0 }}
            ariaLabel="Minimap"
          />
        )}
      </ReactFlow>
    </div>
  );
}

export function Canvas() {
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  );
}
