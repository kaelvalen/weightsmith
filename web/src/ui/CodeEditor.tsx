// Monaco, bundled locally (no CDN) and themed to the quiet swiss tokens:
// monochrome code, comments in the secondary ink, accent only for the cursor/selection.

import Editor, { loader, OnMount } from "@monaco-editor/react";
import * as monaco from "monaco-editor";
import editorWorker from "monaco-editor/editor/editor.worker?worker";
import { useEffect, useState } from "react";

(self as any).MonacoEnvironment = { getWorker: () => new editorWorker() };
loader.config({ monaco });

/** A token as #rrggbb (the CSS minifier may have shortened it to #rgb). */
function css(name: string) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const m = /^#([0-9a-f])([0-9a-f])([0-9a-f])$/i.exec(v);
  return m ? `#${m[1]}${m[1]}${m[2]}${m[2]}${m[3]}${m[3]}` : v;
}

function defineTheme(): string {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches
    ? document.documentElement.dataset.theme !== "light"
    : document.documentElement.dataset.theme === "dark";
  const name = dark ? "ws-dark" : "ws-light";
  const ink = css("--ink").replace("#", "");
  const muted = css("--muted").replace("#", "");
  monaco.editor.defineTheme(name, {
    base: dark ? "vs-dark" : "vs",
    inherit: false,
    rules: [
      { token: "", foreground: ink },
      { token: "comment", foreground: muted, fontStyle: "italic" },
      { token: "string", foreground: ink },
      { token: "keyword", foreground: ink, fontStyle: "bold" },
      { token: "number", foreground: ink },
    ],
    colors: {
      "editor.background": css("--bg"),
      "editor.foreground": css("--ink"),
      "editorLineNumber.foreground": css("--muted"),
      "editorLineNumber.activeForeground": css("--ink"),
      "editor.lineHighlightBackground": css("--bg"),
      "editor.lineHighlightBorder": css("--line"),
      "editorCursor.foreground": css("--accent"),
      "editor.selectionBackground": css("--accent") + "33",
      "editorIndentGuide.background1": css("--line"),
      "editorWidget.background": css("--bg"),
      "editorWidget.border": css("--line"),
      "scrollbarSlider.background": css("--line") + "aa",
    },
  });
  return name;
}

export default function CodeEditor({
  value,
  onChange,
  readOnly,
  height,
  onLineClick,
  highlightLines,
}: {
  value: string;
  onChange?: (v: string) => void;
  readOnly?: boolean;
  height: number | string;
  onLineClick?: (line: number) => void;
  highlightLines?: number[];
}) {
  const [theme, setTheme] = useState("ws-light");
  const [ed, setEd] = useState<monaco.editor.IStandaloneCodeEditor | null>(null);
  const [decos, setDecos] = useState<monaco.editor.IEditorDecorationsCollection | null>(null);

  useEffect(() => {
    setTheme(defineTheme());
    const mq = matchMedia("(prefers-color-scheme: dark)");
    const h = () => setTheme(defineTheme());
    mq.addEventListener("change", h);
    return () => mq.removeEventListener("change", h);
  }, []);

  useEffect(() => {
    if (!ed) return;
    const lines = highlightLines ?? [];
    const d = lines.map((l) => ({ range: new monaco.Range(l, 1, l, 1), options: { isWholeLine: true, className: "ws-hl", linesDecorationsClassName: "ws-hl-gutter" } }));
    if (decos) decos.set(d);
    else setDecos(ed.createDecorationsCollection(d));
    if (lines.length) ed.revealLineInCenterIfOutsideViewport(lines[0]);
  }, [ed, highlightLines?.join(",")]);

  const mount: OnMount = (editor) => {
    setEd(editor);
    if (onLineClick) editor.onMouseDown((e) => e.target.position && onLineClick(e.target.position.lineNumber));
  };

  return (
    <Editor
      height={height}
      language="python"
      theme={theme}
      value={value}
      onChange={(v) => onChange?.(v ?? "")}
      onMount={mount}
      options={{
        readOnly,
        minimap: { enabled: false },
        fontFamily: "JetBrains Mono, monospace",
        fontSize: 12,
        lineHeight: 16,
        scrollBeyondLastLine: false,
        renderLineHighlight: "line",
        overviewRulerLanes: 0,
        hideCursorInOverviewRuler: true,
        folding: false,
        lineNumbersMinChars: 3,
        padding: { top: 8 },
        tabSize: 4,
        guides: { indentation: true },
        scrollbar: { verticalScrollbarSize: 8, horizontalScrollbarSize: 8, useShadows: false },
      }}
    />
  );
}
