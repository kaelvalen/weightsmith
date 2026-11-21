#!/usr/bin/env bash
# Browser scenarios against a throwaway daemon and workspace. Needs Chrome or Chromium.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d /run/opengl-driver/lib ] && export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PORT=${WS_E2E_PORT:-7899}
WS=$(mktemp -d)
(cd web && pnpm exec vite build >/dev/null)
WEIGHTSMITH_HOME="$WS" uv run weightsmith serve --port "$PORT" >"$WS/daemon.log" 2>&1 &
DAEMON=$!
trap 'kill $DAEMON 2>/dev/null; rm -rf "$WS"' EXIT
for _ in $(seq 50); do curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null && break; sleep 0.2; done
status=0
for scenario in web/e2e/canvas.mjs web/e2e/smart.mjs web/e2e/run.mjs web/e2e/scope.mjs; do
  echo "== $scenario"
  WS_URL="http://127.0.0.1:$PORT" node "$scenario" || status=1
done
exit $status
