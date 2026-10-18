// A select in the interface's own style: a field-shaped button and a bordered list.
// Keyboard: ↑ ↓ to move, Enter or Space to pick, Esc to close, letters to jump.
// The list is fixed-positioned so scroll containers never clip it.

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

export interface Option {
  value: string;
  label?: string;
  hint?: string;
}

export function Select({
  value,
  options,
  onChange,
  id,
  ariaLabel,
  className = "",
  mono,
}: {
  value: string;
  options: (Option | string)[];
  onChange: (v: string) => void;
  id?: string;
  ariaLabel?: string;
  className?: string;
  mono?: boolean;
}) {
  const opts: Option[] = options.map((o) => (typeof o === "string" ? { value: o } : o));
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const btn = useRef<HTMLButtonElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const current = opts.find((o) => o.value === value);

  const show = () => {
    setRect(btn.current!.getBoundingClientRect());
    setActive(Math.max(0, opts.findIndex((o) => o.value === value)));
    setOpen(true);
  };
  const pick = (i: number) => {
    setOpen(false);
    btn.current?.focus();
    if (opts[i] && opts[i].value !== value) onChange(opts[i].value);
  };

  useLayoutEffect(() => {
    if (open) list.current?.children[active]?.scrollIntoView({ block: "nearest" });
  }, [open, active]);
  useEffect(() => {
    if (!open) return;
    const close = (e: Event) => {
      if (!list.current?.contains(e.target as Node) && e.target !== btn.current) setOpen(false);
    };
    const reflow = () => setRect(btn.current!.getBoundingClientRect());
    document.addEventListener("pointerdown", close);
    window.addEventListener("scroll", reflow, true);
    window.addEventListener("resize", reflow);
    return () => {
      document.removeEventListener("pointerdown", close);
      window.removeEventListener("scroll", reflow, true);
      window.removeEventListener("resize", reflow);
    };
  }, [open]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return; // app-wide shortcuts (Cmd+K, Cmd+Enter) still work here
    e.stopPropagation();
    if (!open) {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(e.key)) {
        e.preventDefault();
        show();
      }
      return;
    }
    if (e.key === "ArrowDown") (e.preventDefault(), setActive((a) => Math.min(a + 1, opts.length - 1)));
    else if (e.key === "ArrowUp") (e.preventDefault(), setActive((a) => Math.max(a - 1, 0)));
    else if (e.key === "Home") (e.preventDefault(), setActive(0));
    else if (e.key === "End") (e.preventDefault(), setActive(opts.length - 1));
    else if (e.key === "Enter" || e.key === " ") (e.preventDefault(), pick(active));
    else if (e.key === "Escape" || e.key === "Tab") setOpen(false);
    else if (e.key.length === 1) {
      const k = e.key.toLowerCase();
      const i = opts.findIndex((o, j) => j > active && (o.label ?? o.value).toLowerCase().startsWith(k));
      const wrap = i >= 0 ? i : opts.findIndex((o) => (o.label ?? o.value).toLowerCase().startsWith(k));
      if (wrap >= 0) setActive(wrap);
    }
  };

  const below = rect ? window.innerHeight - rect.bottom > Math.min(240, opts.length * 24 + 8) : true;
  return (
    <>
      <button
        ref={btn}
        id={id}
        type="button"
        role="combobox"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={ariaLabel}
        className={`ws-select ${mono ? "num" : ""} ${className}`}
        onClick={() => (open ? setOpen(false) : show())}
        onKeyDown={onKeyDown}
      >
        <span className="truncate">{current ? (current.label ?? current.value) : value}</span>
      </button>
      {open &&
        rect &&
        createPortal(
          <ul
            ref={list}
            role="listbox"
            className="ws-listbox"
            style={{
              left: rect.left,
              width: Math.max(rect.width, 120),
              ...(below ? { top: rect.bottom + 2 } : { bottom: window.innerHeight - rect.top + 2 }),
            }}
          >
            {opts.map((o, i) => (
              <li
                key={o.value}
                role="option"
                aria-selected={o.value === value}
                className={`${mono ? "num" : ""} ${i === active ? "active" : ""}`}
                onPointerEnter={() => setActive(i)}
                onPointerDown={(e) => {
                  e.preventDefault();
                  pick(i);
                }}
              >
                <span className="truncate">{o.label ?? o.value}</span>
                {o.hint ? <span className="ws-listbox-hint">{o.hint}</span> : o.value === value ? <span aria-hidden>✓</span> : null}
              </li>
            ))}
          </ul>,
          document.body,
        )}
    </>
  );
}
