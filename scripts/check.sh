#!/usr/bin/env bash
# Everything CI would run: Python tests (core, codegen snapshots, sandbox, daemon e2e),
# the TS mirror's cross-test against Python fixtures, typecheck and build.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d /run/opengl-driver/lib ] && export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv run pytest -q
cd web
pnpm exec vitest run
pnpm exec tsc -b
pnpm exec vite build >/dev/null
cd ..
if node -e "import('./web/e2e/lib.mjs').then(m => process.exit(m.chromePath() ? 0 : 1))" 2>/dev/null; then
  scripts/e2e.sh
else
  echo "skipping browser scenarios: no Chrome/Chromium found (set CHROME=/path/to/chrome)"
fi
echo "all checks passed"
