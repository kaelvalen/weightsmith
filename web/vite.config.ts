import react from "@vitejs/plugin-react";
import tailwind from "@tailwindcss/vite";
import { defineConfig } from "vitest/config";

declare const process: { env: Record<string, string | undefined> };
const daemon = process.env.WEIGHTSMITH_DAEMON ?? "http://127.0.0.1:7860";

// `vite build --mode scope` builds the standalone trainscope UI into the Python package,
// as index.html, so `pip install trainscope` ships the same views Weightsmith shows.
const scopeBuild = {
  outDir: "../trainscope/trainscope/ui/static",
  emptyOutDir: true,
  rollupOptions: { input: "scope.html" },
  chunkSizeWarningLimit: 4000,
};
const renameScopeHtml = {
  name: "scope-html-as-index",
  enforce: "post" as const,
  generateBundle(_: unknown, bundle: Record<string, { fileName: string }>) {
    if (bundle["scope.html"]) bundle["scope.html"].fileName = "index.html";
  },
};

export default defineConfig(({ mode }) => ({
  plugins: [react(), tailwind(), ...(mode === "scope" ? [renameScopeHtml] : [])],
  server: {
    port: 5173,
    proxy: {
      "/api": daemon,
      "/ws": { target: daemon.replace(/^http/, "ws"), ws: true },
    },
  },
  build: mode === "scope" ? scopeBuild : { chunkSizeWarningLimit: 4000 },
  test: { environment: "node" },
}));
