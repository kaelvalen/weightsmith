#!/usr/bin/env bash
# Daemon (with hot canvas via Vite) for development.
#   scripts/dev.sh          daemon on :7860 + vite on :5173 (open http://localhost:5173)
#   scripts/dev.sh serve    build the canvas once and serve everything from :7860
set -euo pipefail
cd "$(dirname "$0")/.."

# NixOS keeps the CUDA driver outside the default loader path.
if [ -d /run/opengl-driver/lib ]; then
  export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

uv sync -q
(cd web && pnpm install --silent)

if [ "${1:-}" = "serve" ]; then
  (cd web && pnpm build)
  exec uv run weightsmith serve
fi

uv run weightsmith serve &
DAEMON=$!
trap 'kill $DAEMON' EXIT
cd web && pnpm dev
