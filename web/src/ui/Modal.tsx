// The one modal of the interface: a backdrop and a single bordered panel.
// Esc and a click on the backdrop close it; focus stays inside while it is open
// and returns to where it was afterwards.

import { useEffect, useRef } from "react";

export function Modal({
  title,
  onClose,
  children,
  footer,
  width = 480,
  top = 64,
  hint,
}: {
  title?: React.ReactNode;
  onClose: () => void;
  children: React.ReactNode;
  footer?: React.ReactNode;
  width?: number;
  top?: number;
  hint?: React.ReactNode;
}) {
  const panel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const before = document.activeElement as HTMLElement | null;
    const first = panel.current?.querySelector<HTMLElement>("[autofocus], input, textarea, select, button");
    first?.focus();
    return () => before?.focus?.();
  }, []);
  const onKeyDown = (e: React.KeyboardEvent) => {
    e.stopPropagation();
    if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
    if (e.key === "Tab" && panel.current) {
      const items = [...panel.current.querySelectorAll<HTMLElement>("input, textarea, select, button, [tabindex]:not([tabindex='-1'])")].filter(
        (el) => !el.hasAttribute("disabled"),
      );
      if (!items.length) return;
      const i = items.indexOf(document.activeElement as HTMLElement);
      if (e.shiftKey && i <= 0) {
        e.preventDefault();
        items[items.length - 1].focus();
      } else if (!e.shiftKey && i === items.length - 1) {
        e.preventDefault();
        items[0].focus();
      }
    }
  };
  return (
    <div className="ws-backdrop" onMouseDown={onClose} onKeyDown={onKeyDown}>
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-label={typeof title === "string" ? title : undefined}
        className="ws-modal"
        style={{ width, top }}
        onMouseDown={(e) => e.stopPropagation()}
      >
        {title !== undefined && (
          <div className="ws-modal-head">
            <h2 className="m-0 text-title font-medium">{title}</h2>
            {hint ?? <kbd>esc</kbd>}
          </div>
        )}
        {children}
        {footer && <div className="ws-modal-foot">{footer}</div>}
      </div>
    </div>
  );
}
