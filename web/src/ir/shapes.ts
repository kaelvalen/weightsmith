// Mirror of core/weightsmith_core/shapes.py. Python is the source of truth;
// cross.test.ts checks this mirror against fixtures the Python analyzer produced.

export type Dim = number | string;
export type Shape = Dim[];

export interface TensorType {
  shape: Shape;
  dtype: string;
}

export const FLOAT_DTYPES = ["f32", "f16", "bf16"];
export const INT_DTYPES = ["i64", "i32"];

const SYMBOL = /^[A-Za-z_][A-Za-z0-9_]*$/;

export class ShapeError extends Error {
  constructor(
    message: string,
    public port: string | null = null,
  ) {
    super(message);
  }
}

export const isSymbol = (d: Dim): d is string => typeof d === "string" && SYMBOL.test(d);

export const fmtShape = (s: Shape): string => (s.length ? s.join("×") : "scalar");

export class SymbolEnv {
  parent = new Map<string, Dim>();

  resolve(d: Dim): Dim {
    const seen: string[] = [];
    while (isSymbol(d) && this.parent.has(d)) {
      seen.push(d);
      d = this.parent.get(d)!;
    }
    for (const s of seen) this.parent.set(s, d);
    return d;
  }

  resolveShape(s: Shape): Shape {
    return s.map((d) => this.resolve(d));
  }

  unify(a: Dim, b: Dim): boolean {
    a = this.resolve(a);
    b = this.resolve(b);
    if (a === b) return true;
    if (isSymbol(a)) {
      this.parent.set(a, b);
      return true;
    }
    if (isSymbol(b)) {
      this.parent.set(b, a);
      return true;
    }
    return false;
  }

  unifyShapes(a: Shape, b: Shape, port: string | null = null, what = ""): void {
    if (a.length !== b.length)
      throw new ShapeError(`${what}rank mismatch: ${fmtShape(this.resolveShape(a))} vs ${fmtShape(this.resolveShape(b))}`, port);
    a.forEach((x, i) => {
      if (!this.unify(x, b[i]))
        throw new ShapeError(`${what}shape mismatch: ${fmtShape(this.resolveShape(a))} vs ${fmtShape(this.resolveShape(b))}`, port);
    });
  }
}

export function dimMul(dims: Dim[]): Dim {
  const ints = dims.filter((d): d is number => typeof d === "number");
  const syms = dims.filter((d) => typeof d !== "number").map(String);
  const prod = ints.reduce((a, b) => a * b, 1);
  if (!syms.length) return prod;
  return [...(prod !== 1 ? [String(prod)] : []), ...syms].join("*");
}

export function dimAdd(a: Dim, b: Dim): Dim {
  return typeof a === "number" && typeof b === "number" ? a + b : `${a}+${b}`;
}

export function convOut(size: Dim, k: number, s: number, p: number, dil = 1): Dim {
  if (typeof size === "number") return Math.floor((size + 2 * p - dil * (k - 1) - 1) / s) + 1;
  if (s === 1 && 2 * p === dil * (k - 1)) return size;
  return `conv_${size}_k${k}s${s}p${p}`;
}

export function numel(s: Shape): number | null {
  return s.every((d) => typeof d === "number") ? (s as number[]).reduce((a, b) => a * b, 1) : null;
}

/** d / k for an exact division; symbolic "k*X" becomes X. */
export function dimDiv(d: Dim, k: number): Dim {
  if (typeof d === "number") {
    if (d % k) throw new ShapeError(`${d} is not divisible by ${k}`);
    return d / k;
  }
  const prefix = `${k}*`;
  if (d.startsWith(prefix) && isSymbol(d.slice(prefix.length))) return d.slice(prefix.length);
  return `${d}/${k}`;
}

export function convTransposeOut(size: Dim, k: number, s: number, p: number, op = 0, dil = 1): Dim {
  if (typeof size === "number") return (size - 1) * s - 2 * p + dil * (k - 1) + op + 1;
  if (s === 1 && 2 * p === dil * (k - 1) && !op) return size;
  if (k === s && !p && !op) return dimMul([s, size]);
  return `convT_${size}_k${k}s${s}`;
}
