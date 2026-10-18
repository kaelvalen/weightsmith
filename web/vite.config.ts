import react from "@vitejs/plugin-react";
import tailwind from "@tailwindcss/vite";
import { defineConfig } from "vitest/config";

declare const process: { env: Record<string, string | undefined> };
const daemon = process.env.WEIGHTSMITH_DAEMON ?? "http://127.0.0.1:7860";

export default defineConfig({
  plugins: [react(), tailwind()],
  server: {
    port: 5173,
    proxy: {
      "/api": daemon,
      "/ws": { target: daemon.replace(/^http/, "ws"), ws: true },
    },
  },
  build: { chunkSizeWarningLimit: 4000 },
  test: { environment: "node" },
});
