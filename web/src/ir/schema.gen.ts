/* Generated from schema/ir.schema.json by `pnpm gen`. Do not edit. */

/**
 * [source node.port, target node.port]. Only dataflow is an edge.
 *
 * @minItems 2
 * @maxItems 2
 */
export type Edge = [PortRef, PortRef];
/**
 * node.port
 */
export type PortRef = string;

/**
 * Weightsmith graph IR, schema version 1. Framework-independent; the single contract between canvas, backend and runtime.
 */
export interface Graph {
  schema: 1;
  name?: string;
  nodes: Node[];
  edges: Edge[];
  /**
   * Reusable parameterised blocks, referenced by Subgraph nodes via params.graph.
   */
  subgraphs?: {
    [k: string]: SubgraphDef;
  };
  /**
   * Canvas coordinates. Excluded from the content hash.
   */
  layout?: {
    [k: string]: Position;
  };
}
export interface Node {
  id: string;
  type: string;
  /**
   * Display name. Excluded from the content hash.
   */
  label?: string;
  params?: {
    [k: string]: unknown;
  };
  adapt?: Adapt[];
  trainable?: boolean;
  /**
   * Per-parameter freeze mask: parameter names (e.g. bias, q_proj.weight) to freeze.
   */
  frozen_params?: string[];
  /**
   * Ablation: replace this node by identity
   */
  disabled?: boolean;
}
/**
 * A modifier that wraps a module's forward. Never changes graph topology.
 */
export interface Adapt {
  kind: "lora" | "adapter" | "custom";
  /**
   * weight, or a sub-module name such as q_proj / v_proj
   */
  target?: string;
  r?: number;
  alpha?: number;
  dropout?: number;
  /**
   * Freeze the wrapped base parameters
   */
  freeze?: boolean;
  /**
   * A/B initialisation, B=0 by default
   */
  init?: "zeros_b" | "gaussian";
  /**
   * adapter kind: hidden size
   */
  bottleneck?: number;
  /**
   * custom kind: Python source defining a class with wrap(module) -> module
   */
  code?: string;
  class_name?: string;
  /**
   * Ablation: skip this modifier
   */
  disabled?: boolean;
}
export interface SubgraphDef {
  nodes: Node[];
  edges: Edge[];
  layout?: {
    [k: string]: Position;
  };
}
export interface Position {
  x: number;
  y: number;
}
