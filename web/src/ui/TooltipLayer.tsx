// Replaces the browser's title boxes with tooltips in the interface's own style.
// Any element with a title (or data-tip) gets one; the title attribute is moved to
// data-tip so the native box never appears.

import { useEffect, useState } from "react";

interface Tip {
  text: string;
  x: number;
  y: number;
  below: boolean;
}

export function TooltipLayer() {
  const [tip, setTip] = useState<Tip | null>(null);
  useEffect(() => {
    let timer: number | undefined;
    let current: HTMLElement | null = null;
    const find = (t: EventTarget | null) => {
      let el = t as HTMLElement | null;
      while (el && el !== document.body) {
        if (el.getAttribute?.("title")) {
          el.setAttribute("data-tip", el.getAttribute("title")!);
          el.removeAttribute("title");
        }
        if (el.getAttribute?.("data-tip")) return el;
        el = el.parentElement;
      }
      return null;
    };
    const show = (el: HTMLElement) => {
      const r = el.getBoundingClientRect();
      const below = r.top < 64;
      setTip({ text: el.getAttribute("data-tip") ?? "", x: Math.min(Math.max(8, r.left), window.innerWidth - 328), y: below ? r.bottom + 6 : r.top - 6, below });
    };
    const over = (e: Event) => {
      const el = find(e.target);
      if (el === current) return;
      current = el;
      clearTimeout(timer);
      setTip(null);
      if (el) timer = window.setTimeout(() => show(el), 450);
    };
    const hide = () => {
      clearTimeout(timer);
      current = null;
      setTip(null);
    };
    // move every title to data-tip as soon as it appears (also keeps it as the accessible description)
    const adopt = (el: Element) => {
      const t = el.getAttribute("title");
      if (!t) return;
      el.setAttribute("data-tip", t);
      if (!el.getAttribute("aria-label") && !el.textContent?.trim()) el.setAttribute("aria-label", t);
      el.removeAttribute("title");
    };
    document.querySelectorAll("[title]").forEach(adopt);
    const mo = new MutationObserver((records) => {
      for (const r of records) {
        if (r.type === "attributes") adopt(r.target as Element);
        r.addedNodes.forEach((n) => {
          if (n.nodeType !== 1) return;
          adopt(n as Element);
          (n as Element).querySelectorAll?.("[title]").forEach(adopt);
        });
      }
    });
    mo.observe(document.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["title"] });
    document.addEventListener("pointerover", over);
    document.addEventListener("focusin", over);
    document.addEventListener("pointerdown", hide);
    document.addEventListener("keydown", hide);
    window.addEventListener("scroll", hide, true);
    return () => {
      mo.disconnect();
      document.removeEventListener("pointerover", over);
      document.removeEventListener("focusin", over);
      document.removeEventListener("pointerdown", hide);
      document.removeEventListener("keydown", hide);
      window.removeEventListener("scroll", hide, true);
    };
  }, []);
  if (!tip || !tip.text) return null;
  return (
    <div role="tooltip" className="ws-tip" style={{ left: tip.x, top: tip.y, transform: tip.below ? undefined : "translateY(-100%)" }}>
      {tip.text}
    </div>
  );
}
