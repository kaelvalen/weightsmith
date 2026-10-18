// Parametric subgraphs, mirroring ir.eval_arg / ir.instantiate in core/weightsmith_core/ir.py:
// node params inside a block may be "$name" or plain arithmetic such as "$C*2".

import type { Graph, Node } from "./types";

export class ArgError extends Error {}

type Tok = { k: "num" | "name" | "op"; v: string };

function tokenize(src: string): Tok[] {
  const out: Tok[] = [];
  const re = /\s*(\d+\.?\d*(?:e[+-]?\d+)?|[A-Za-z_][A-Za-z0-9_]*|\/\/|\*\*|[-+*/%()])/y;
  let i = 0;
  while (i < src.length) {
    re.lastIndex = i;
    const m = re.exec(src);
    if (!m) {
      if (/^\s*$/.test(src.slice(i))) break;
      throw new ArgError(`cannot parse $${src}`);
    }
    const t = m[1];
    out.push({ k: /^\d/.test(t) ? "num" : /^[A-Za-z_]/.test(t) ? "name" : "op", v: t });
    i = re.lastIndex;
  }
  return out;
}

export function evalArg(expr: string, env: Record<string, unknown>): unknown {
  const toks = tokenize(expr);
  let pos = 0;
  const peek = () => toks[pos];
  const take = () => toks[pos++];
  const atom = (): any => {
    const t = take();
    if (!t) throw new ArgError(`cannot parse $${expr}`);
    if (t.k === "num") return Number(t.v);
    if (t.k === "name") {
      if (t.v === "True") return true;
      if (t.v === "False") return false;
      if (!(t.v in env)) throw new ArgError(`unknown block parameter $${t.v}`);
      return env[t.v];
    }
    if (t.v === "(") {
      const v = sum();
      if (take()?.v !== ")") throw new ArgError(`cannot parse $${expr}`);
      return v;
    }
    if (t.v === "-") return -unary();
    if (t.v === "+") return unary();
    throw new ArgError(`unsupported expression in $${expr}`);
  };
  const unary = (): any => atom();
  const power = (): any => {
    const b = unary();
    if (peek()?.v === "**") {
      take();
      return b ** power();
    }
    return b;
  };
  const product = (): any => {
    let v = power();
    while (peek() && ["*", "/", "//", "%"].includes(peek().v)) {
      const op = take().v;
      const r = power();
      if (op === "*") v = v * r;
      else if (op === "/") v = v / r;
      else if (op === "//") v = Math.floor(v / r);
      else v = ((v % r) + r) % r;
    }
    return v;
  };
  const sum = (): any => {
    let v = product();
    while (peek() && ["+", "-"].includes(peek().v)) {
      const op = take().v;
      const r = product();
      v = op === "+" ? v + r : v - r;
    }
    return v;
  };
  const v = sum();
  if (pos !== toks.length) throw new ArgError(`cannot parse $${expr}`);
  return v;
}

export function substitute(value: unknown, env: Record<string, unknown>): unknown {
  if (typeof value === "string" && value.startsWith("$")) return evalArg(value.slice(1), env);
  if (Array.isArray(value)) return value.map((v) => substitute(v, env));
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, substitute(v, env)]));
  return value;
}

type Body = NonNullable<Graph["subgraphs"]>[string] & { params?: Record<string, unknown>; doc?: string };

export function blockArgs(body: Body, args?: Record<string, unknown>): Record<string, unknown> {
  const defaults = { ...(body.params ?? {}) };
  for (const k of Object.keys(args ?? {}))
    if (!(k in defaults)) throw new ArgError(`block has no parameter '${k}' (has: ${Object.keys(defaults).join(", ") || "none"})`);
  return { ...defaults, ...(args ?? {}) };
}

export function instantiate(body: Body, args?: Record<string, unknown>): Body {
  const env = blockArgs(body, args);
  const out = structuredClone(body);
  out.nodes = out.nodes.map((n: Node) => {
    const m = { ...n };
    if (m.params) m.params = substitute(m.params, env) as Node["params"];
    if (m.adapt) m.adapt = substitute(m.adapt, env) as Node["adapt"];
    return m;
  });
  return out;
}
