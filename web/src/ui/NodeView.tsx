import { Handle, NodeProps, Position } from "@xyflow/react";
import { memo } from "react";
import type { Issue, ParamCount, Signature } from "../ir/analyze";
import type { TensorType } from "../ir/shapes";
import type { Node } from "../ir/types";

export interface NodeData extends Record<string, unknown> {
  node: Node;
  sig: Signature;
  ports: Record<string, TensorType | undefined>;
  issues: Issue[];
  count?: ParamCount;
  connected: Set<string>;
}

export const NODE_W = 184;

const shape = (t?: TensorType) => (t ? (t.shape.length ? t.shape.join("×") : "scalar") : "");

function badge(a: NonNullable<Node["adapt"]>[number]) {
  if (a.kind === "lora") return `LoRA r${a.r ?? 8}${a.target && a.target !== "weight" ? ` ${a.target.replace("_proj", "")}` : ""}`;
  if (a.kind === "adapter") return `Adapter ${a.bottleneck ?? 64}`;
  return a.class_name ?? "custom";
}

function NodeViewImpl({ data, selected }: NodeProps & { data: NodeData }) {
  const { node, sig, ports, issues, connected } = data;
  const p = (node.params ?? {}) as Record<string, any>;
  const frozen = node.trainable === false;
  const errPorts = new Set(issues.filter((i) => i.port).map((i) => i.port!));
  const nodeError = issues.some((i) => i.severity === "error" && !i.port);
  const kind =
    node.type === "Subgraph" ? `Subgraph · ${p.graph}${p.repeat > 1 ? ` · ${p.repeat}×${p.tie_weights ? " tied" : ""}` : ""}` : node.type === "PythonModule" ? `Python · ${p.class_name ?? ""}` : node.type;

  const border = selected ? "var(--accent)" : nodeError ? "var(--error)" : frozen ? "var(--muted)" : "var(--ink)";
  const slot = (i: number, n: number) => `calc(${(100 * i) / n}% + 12px)`;

  const portRow = (names: string[], dir: "in" | "out") => (
    <div className="num flex h-4 text-label">
      {names.map((port) => {
        const t = ports[`${node.id}.${port}`];
        const opt = sig.optional.includes(port);
        return (
          <span
            key={port}
            className={`min-w-0 flex-1 truncate pl-5 ${errPorts.has(port) ? "text-error" : "text-muted"}`}
            title={`${dir} ${port} ${shape(t)}`}
          >
            {names.length > 1 || opt ? <span className="mr-1">{port}{opt ? "?" : ""}</span> : null}
            <span className={errPorts.has(port) ? "" : "text-ink"}>{shape(t) || (opt && dir === "in" ? "dataset" : "")}</span>
          </span>
        );
      })}
    </div>
  );

  return (
    <div
      className="relative bg-bg"
      style={{
        width: NODE_W,
        border: `1px ${frozen ? "dashed" : "solid"} ${border}`,
        borderRadius: 2,
        opacity: node.disabled ? 0.45 : 1,
        padding: "4px 0",
      }}
      onDoubleClick={() => node.type === "Subgraph" && window.dispatchEvent(new CustomEvent("ws:enter", { detail: p.graph }))}
    >
      {sig.inputs.length > 0 && portRow(sig.inputs, "in")}
      <div className="px-2 py-1">
        <div className="truncate text-label text-muted">{kind}</div>
        <div className={`truncate font-medium ${node.disabled ? "line-through" : ""}`}>{node.label || node.id}</div>
      </div>
      {sig.outputs.length > 0 && portRow(sig.outputs, "out")}
      {sig.inputs.map((port, i) => (
        <Handle
          key={`in-${port}`}
          id={port}
          type="target"
          position={Position.Top}
          className={`${errPorts.has(port) ? "error" : ""} ${connected.has(`${node.id}.${port}`) ? "connected" : ""}`}
          style={{ left: slot(i, sig.inputs.length), top: -4, transform: "none", borderStyle: sig.optional.includes(port) ? "dashed" : "solid" }}
        />
      ))}
      {sig.outputs.map((port, i) => (
        <Handle
          key={`out-${port}`}
          id={port}
          type="source"
          position={Position.Bottom}
          className={`${errPorts.has(port) ? "error" : ""} ${connected.has(`${node.id}.${port}`) ? "connected" : ""}`}
          style={{ left: slot(i, sig.outputs.length), bottom: -4, top: "auto", transform: "none" }}
        />
      ))}
      {!!node.adapt?.length && (
        <div className="absolute right-2 -bottom-2 flex gap-1">
          {node.adapt.map((a, i) => (
            <span
              key={i}
              className="num border bg-bg px-1 text-label leading-[14px]"
              style={{ borderColor: selected ? "var(--accent)" : "var(--ink)", textDecoration: a.disabled ? "line-through" : undefined, color: a.disabled ? "var(--muted)" : undefined }}
            >
              {badge(a)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export const NodeView = memo(NodeViewImpl);
