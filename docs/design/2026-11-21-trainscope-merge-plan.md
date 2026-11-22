# Bringing trainscope into Weightsmith — implementation plan

**Goal:** move trainscope into the Weightsmith repository with its history. Every Weightsmith run is recorded by trainscope, the diagnosis UI lives in Weightsmith's own interface, and trainscope continues as a standalone package on PyPI.

**Architecture:** trainscope lives in `trainscope/` as a uv workspace member. The generated `train.py` imports trainscope optionally and writes its recording under `runs/<rid>/scope/`. trainscope's FastAPI endpoints become a run-keyed router, mounted both in the Weightsmith daemon (`/api/scope`) and in the standalone `trainscope ui`. The views are rewritten in TS in Weightsmith's web package, and a second Vite entry of the same package produces the standalone UI.

**Tech:** Python 3.12, uv workspace, hatch, FastAPI, pyarrow; React 19 + TS + Tailwind + zustand, uPlot, vitest, Chrome e2e (`web/e2e`).

**Spec:** `docs/design/2026-11-21-trainscope-merge.md`

## Global constraints

- `trainscope/trainscope/**` never imports `weightsmith_*`.
- trainscope's `requires-python >=3.11` is kept.
- trainscope's stable contracts do not change: `TrainScope`, `TrainScopeConfig`, `load_config`, `StopTraining`, the config fields and their defaults, the Arrow format (additive only), the plugin surface (`tests/test_plugin_contract.py` must pass). The HTTP API is free.
- The generated `train.py` runs when trainscope is not installed.
- `train.diagnostics` does not enter the run identity: an existing job's `config_hash` does not change with this plan.
- uPlot becomes the single chart library, Weightsmith's `web/src/index.css` tokens the single theme. Geist goes.
- No trace of an AI assistant in the commits, the files or the history. Commit messages carry no attribution lines.
- Every task ends with `scripts/check.sh` green.

## Review Focus

1. **`train.compile: true`**: `torch.compile` adds `_orig_mod.` to parameter names. trainscope must attach to the uncompiled module, and layer names must keep mapping to the canvas. Test: Task 4.
2. **Resume from a checkpoint**: the trainscope recording must append to the same `scope/` directory, with no repeated or reset steps. Test: Task 4.
3. **Runs from before this plan, or with diagnostics off** (no `scope/`): the API returns 404, the Diagnose tab shows an empty state, Runs/Compare do not break. Test: Tasks 5, 8 and 10.
4. **A NaN loss in the first steps**: Weightsmith's `nan` error event must keep arriving, and trainscope must write the recording so far to disk with `detach()`. Test: Task 4.
5. **LoRA / frozen parameters** (`requires_grad=False`, grad `None`): recording must not crash. The spike source must be chosen among trained parameters only. Test: Tasks 4 and 9.

---

### Task 1: Clean trainscope's history

**Files:** the `~/trainscope` repo (Weightsmith is not touched).

- [ ] **Step 1: Back up.** `mkdir -p ~/backups && git -C ~/trainscope bundle create ~/backups/trainscope-2026-11-21.bundle --all`
- [ ] **Step 2: Rewrite the history.** With `git filter-repo` if available, otherwise `git filter-branch`, with `--tag-name-filter cat -- --all`:
  - Message filter: the assistant's session-link line and its `Co-Authored-By` lines are removed (case-insensitive). The assistant-prefixed branch name (`kaelvalen/<assistant>/fix-several-things-…`) → `kaelvalen/fix-several-things`. Trailing blank lines are trimmed.
  - Tree filter: `docs/superpowers/` is removed from every commit. The line for the assistant's local settings file is removed from `.gitignore`.
  - `refs/original`, the reflog and gc are cleaned up.
- [ ] **Step 3: Verify.**
  A search for assistant names in commit messages → `0`
  A search for assistant names in the files of the whole history → `0`
  `git tag` → the same tags on the rewritten commits (v0.x … v1.10.0)
- [ ] **Step 4: Ask Arda; on approval** `git push --force-with-lease origin main` and `git push --force origin --tags`. Without approval, carry on to Task 2 with the clean local repo; the force push is asked about again at the end of Task 12.

### Task 2: Move trainscope in as a workspace member

**Files:**
- Create: `trainscope/` (`git subtree add --prefix=trainscope ~/trainscope main`, no squash)
- Modify: `pyproject.toml`
- Delete: `trainscope/.github/`, `trainscope/flake.nix`, `trainscope/flake.lock`, `trainscope/.envrc`, `trainscope/.pre-commit-config.yaml`, `trainscope/.editorconfig`. A development flow in `trainscope/Makefile`, if any, moves to `scripts/check.sh`.

**Interfaces:**
- Produces: `import trainscope` works in the Weightsmith venv. `uv run pytest` runs both test suites.

- [ ] **Step 1:** Add the subtree, then write into the root `pyproject.toml`:
  - `[tool.uv.workspace] members = ["trainscope"]`
  - `[tool.uv.sources] trainscope = { workspace = true }`
  - `"trainscope"` and `"pyarrow>=17"` (from the `data` extra into the core) in `dependencies`
  - `[tool.pytest.ini_options] testpaths = ["tests", "trainscope/tests"]`, `addopts = "--import-mode=importlib"` (there are two separate `tests/conftest.py`)
- [ ] **Step 2:** `uv sync && uv run pytest -q` → both suites PASS. trainscope's GPU/FSDP/wikitext tests may be skipped by their own skip conditions. No new skip is added.
- [ ] **Step 3:** Add a boundary test, `tests/test_trainscope_boundary.py::test_trainscope_does_not_import_weightsmith`: no import line mentioning `weightsmith` in `trainscope/trainscope/**/*.py`.
- [ ] **Step 4:** `scripts/check.sh` → PASS. Commit: `Move trainscope into the workspace`.

### Task 3: Turn trainscope's API into a run-keyed router

**Files:**
- Modify: `trainscope/trainscope/ui/server.py`, `trainscope/trainscope/cli.py` (the `ui` command), `trainscope/frontend/src/api.js` (URLs only, so the old UI keeps working until Task 11)
- Test: `trainscope/tests/test_server.py`

**Interfaces:**
- Produces:
  ```python
  RunResolver = Callable[[str], Path | None]        # run key → run directory; None if unknown
  RunLister   = Callable[[], list[tuple[str, Path]]] # (key, directory), sorted
  def make_router(resolve: RunResolver, list_runs: RunLister) -> APIRouter
  def create_app(run_path: str, runs_root: str | None = None) -> FastAPI  # signature unchanged
  ```
- Router paths (relative to the mount prefix): `GET /runs` · `GET /runs/{run}/{manifest|meta|global|layers|moe|replay|signals|spikes}` · `GET /runs/{run}/layers/ranked` · `GET /runs/{run}/layers/matrix` · `GET /runs/{run}/layers/{layer:path}` · `GET /runs/{run}/spikes/{step}[/layers[/{layer:path}]]` · `GET /runs/{run}/diff` · `GET /compare?runs=a,b` · `GET /cluster?runs=…` (every run when empty) · `GET /counterexample?run=…` · `WS /runs/{run}/ws`. Query parameters are as today.
- `create_app` mounts the router under `/api`. Health, auth, CORS, the cache header and static serving stay in the app. `/api/runs/select` and the notion of an "active run" go away. The `is_active` field in the run summary goes too.

- [ ] **Step 1: Write failing tests** (`test_server.py`):
  - `test_two_runs_served_without_select`: under a root with two runs, `/api/runs/a/global` and `/api/runs/b/global` return different rows.
  - `test_unknown_run_404`: `/api/runs/nope/global` → 404.
  - `test_cache_is_per_run`: after `layers/ranked` is requested for `a`, the answer for `b` is not the same as `a`'s.
  - The existing tests move to the new paths. The behavioural expectations do not change.
- [ ] **Step 2:** `uv run pytest trainscope/tests/test_server.py -q` → the new tests FAIL.
- [ ] **Step 3:** Implement. Every endpoint starts with `rp = _resolve_or_404(run)`. The run is added to the `_TTLCache` keys. The multi-run endpoints use `list_runs()` instead of `rr`. The WebSocket loop stays the same; only `rp` comes from the parameter. In single-run mode `create_app` passes `list_runs = [(rp.name, rp)]`.
- [ ] **Step 4:** Tests PASS. `trainscope ui --runs <dir>` opens the old UI with Overview filled (checked by hand).
- [ ] **Step 5:** Commit: `trainscope: serve every run by key instead of a selected run`.

### Task 4: Record with trainscope from the generated train.py

**Files:**
- Modify: `trainscope/trainscope/core/profiles.py` (the `weightsmith` preset), `codegen/weightsmith_codegen/project.py` (`DEFAULT_TRAIN`, `job_hashes`, `write_run`), `codegen/weightsmith_codegen/templates/train.py`, `runtime/weightsmith_runtime/jobs.py` (the `spike` event into the summary), `web/src/api/client.ts` (`"spike"` in `RunEvent.type`)
- Test: `tests/test_codegen.py`, `tests/test_runtime.py`, `tests/snapshots/*` (regenerated)

**Interfaces:**
- Produces:
  - `DEFAULT_TRAIN["diagnostics"] = {"trainscope": True, "profile": "weightsmith"}`
  - In `job_hashes`, the `config` hash is computed without `seed` and `diagnostics`.
  - `write_run` writes `module_map.json` into the run folder: `{py_name(node_id): node_id}` for every top-level node of the graph. The first `.` segment of a layer name maps to a node through this dictionary. The spec said `param_map.json`; a module-level mapping gives the same information without instantiating the model. Parameters inside a subgraph map to the Subgraph node.
  - The recording goes to `<out>/scope/` (`TrainScopeConfig(run_dir=str(out_dir), run_name="scope", ...)`).
  - Event: `{"type": "spike", "step": int, "score": float}`. `jobs._summarize` writes `first_spike_step` and `spikes` (a count) into the summary.
  - The `weightsmith` preset: the `minimal()` values without `activation_layer_filter`. The exact values are set by the measurement in Step 6.

- [ ] **Step 1: Write failing tests**
  - `test_codegen.py::test_diagnostics_not_in_run_identity`: the same job with and without a `diagnostics` field gives the same `job_hashes(...)["config"]`. The value equals the hash of `examples/mnist_mlp.job.json` before this change (take it before the change and write it into the test as a constant).
  - `test_codegen.py::test_module_map_covers_parameters`: the Transformer job under `examples/` is compiled. `model.py` is instantiated on the meta device. The first segment of every `named_parameters()` name is in `module_map.json`.
  - `test_runtime.py::test_run_records_trainscope`: the MNIST MLP job runs with `max_steps=30`. `runs/<rid>/scope/global.arrow` has 30 rows.
  - `test_runtime.py::test_runs_without_trainscope`: in a subprocess where importing trainscope is blocked (`python -c "import sys; sys.modules['trainscope']=None; import runpy; runpy.run_path('train.py', run_name='__main__')"`), the run is `completed` and no `scope/` is created.
  - `test_runtime.py::test_spike_event_on_lr_ramp`: a Transformer job with LR `1.0` and `max_steps=60` produces at least one `spike` event, and `first_spike_step` shows up in the summary. If unstable, the smallest setup that diverges without reaching NaN is chosen; the test is seeded.
  - `test_runtime.py::test_resume_appends_scope`: 20 steps, checkpoint, stop, resume, finish at 40 steps. The `step` column in `global.arrow` has no repeats, increases, and has 40 rows.
  - `test_runtime.py::test_compile_keeps_layer_names`: in a job with `compile: True` (skipped when `torch.compile` does not work on CPU), no file name under `scope/layers/` contains `_orig_mod`.
  - `test_runtime.py::test_nan_flushes_scope`: in a job that reaches NaN within a few steps (`lr: 1e6`), the `error` event has `class == "nan"` and `scope/global.arrow` has at least 1 row.
  - `test_runtime.py::test_lora_job_records`: the `examples/mlp_lora.json` job is `completed` in 20 steps. The grad columns of the frozen base weights are null, and recording does not crash.
- [ ] **Step 2:** Run the tests → FAIL.
- [ ] **Step 3: Implement** `templates/train.py`. Only the following is added; the rest of the template does not change:
  - In the import block: `try: from trainscope import TrainScope; from trainscope.core.config import TrainScopeConfig` / `except ImportError: TrainScope = None`
  - After the `model = model.to(device)` line and before `torch.compile`, create the scope and `attach()` it when `dc = tc.get("diagnostics") or {}; if TrainScope and dc.get("trainscope")` holds. Config: `TrainScopeConfig(run_dir=str(out_dir), run_name="scope", resume=bool(args.resume), ...)` and the `dc["profile"]` preset. Check how the preset is applied in the loader in `trainscope/core/config.py`. `scaler=scaler`.
  - After the `clip_grad_norm_` line and before the optimizer step: `spike = scope.step(loss.item(), batch_index=i, grad_norm_after_clip=float(grad_norm)) if scope else None`. On a spike, `emit("spike", step=step, score=spike["spike_score"])`.
  - `finally: scope and scope.detach()` is added to the training loop's existing `try` (to the existing finally, if any).
- [ ] **Step 4:** Write `module_map.json` in `write_run` and add a line to the README list. Update `job_hashes` and `DEFAULT_TRAIN`. Update `jobs._summarize` and `client.ts`.
- [ ] **Step 5:** Tests PASS. Regenerate the snapshots; the diff must contain only the expected lines (`git diff tests/snapshots`).
- [ ] **Step 6: Overhead measurement.** With `tests/test_speed.py`'s measurement method, measure the median step time with diagnostics on/off on the MNIST MLP and Transformer examples. Target ≤10%. If exceeded, raise `trace_every_n_steps` in the preset first, then `histogram_every_n_steps`. Write the result into `docs/decisions.md` as a table.
- [ ] **Step 7:** `scripts/check.sh` → PASS. Commit: `Record every run with trainscope`.

### Task 5: /api/scope in the daemon

**Files:**
- Modify: `runtime/weightsmith_runtime/server.py` (`create_app`)
- Test: `tests/test_runtime.py`

**Interfaces:**
- Consumes: `make_router(resolve, list_runs)` (Task 3)
- Produces: `app.include_router(make_router(...), prefix="/api/scope")`. `resolve(rid)` → `store.run_dir(rid) / "scope"` if it exists, otherwise `None`. `list_runs()` → the runs that have a `scope/` directory, as `(rid, dir)`, in `created` order. The WS path is `/api/scope/runs/{rid}/ws`. The Vite proxy covers `/api`, so nothing else is needed.

- [ ] **Step 1: Tests:** `test_scope_api_serves_run` (the 30-step run from Task 4 → `GET /api/scope/runs/<rid>/layers` is a non-empty list), `test_scope_api_404_for_run_without_scope` (a run with diagnostics off → 404), `test_scope_cluster_over_runs` (two runs → `GET /api/scope/cluster` contains both).
- [ ] **Step 2:** FAIL → implement → PASS. `scripts/check.sh` → PASS. Commit: `Serve trainscope data from the daemon`.

### Task 6: Move the charts to uPlot

**Files:**
- Modify: `web/package.json` (`uplot@1.6.32`), `web/src/ui/LineChart.tsx`, `web/src/index.css` (chart tokens: `--series-1..7`, grid, band, marker; light and dark)
- Create: `web/src/ui/chartOptions.ts` (the TS version of trainscope's `chartOptions.js`: `toFinite`, `lossYMax`, `offScaleMarker`, `bandSpan`, `buildLineOptions`), `web/src/ui/chartOptions.test.ts` (moved from trainscope's `chartOptions.test.js`)

**Interfaces:**
- Produces: `LineChart`'s existing props (`Series[]` etc.) are kept as they are. New optional props: `markers?: { x: number; label: string }[]`, `bands?: { x0: number; x1: number; tone: "warn" | "spike" }[]`, `syncKey?: string` (synchronised cursor across charts with the same key). Every existing caller works unchanged.

- [ ] **Step 1:** Move the tests and `pnpm exec vitest run src/ui/chartOptions.test.ts` → FAIL.
- [ ] **Step 2:** Implement. Colours come only from the `index.css` tokens. The chart redraws when the theme changes (`data-theme` / `prefers-color-scheme`).
- [ ] **Step 3:** vitest PASS. `scripts/e2e.sh` → the existing Metrics/Compare scenarios PASS. A screenshot of Metrics in the dark theme is taken and checked. Commit: `Draw charts with uPlot`.

### Task 7: The web/src/scope core

**Files:**
- Create: `web/src/scope/api.ts`, `diagnosis.ts`, `layerMatrix.ts`, `spikeCluster.ts`, `runStatus.ts`, `warmup.ts`, `nodeMap.ts` and their `*.test.ts`. The pure-logic tests under trainscope's `__tests__/` are moved. The component tests are not moved; the e2e scenarios in Tasks 8-10 replace them.
- Modify: `runtime/weightsmith_runtime/server.py` (add `module_map.json` to the allow-list of the `/api/runs/{rid}/files/{name}` endpoint, if there is one)

**Interfaces:**
- Produces:
  - `scopeApi(base: string, run: string)`: an object with the TS counterpart of every `fetchX` in trainscope's `api.js` (`meta`, `global`, `layers`, `layersRanked`, `layerMatrix`, `signals`, `layer`, `moe`, `replay`, `spikes`, `spike`, `spikeLayerNames`, `spikeLayer`, `diff`). `base` is `daemonBase() + "/api/scope"` in Weightsmith and `"/api"` standalone. 404 → `null` (empty state).
  - `scopeMulti(base)`: `runs`, `compare(runs)`, `clusters(runs?)`, `counterexample(run)`.
  - `subscribeScope(base, run, onMsg)`: WS, message types `meta|global|global_delta|spike|layers|moe|moe_delta`. Returns a close function.
  - `nodeFor(layer: string, moduleMap: Record<string,string>): string | null`: first-segment mapping. `null` for an unknown segment.
  - The pure functions under trainscope's names: `diagnoseSpike`, `groupRows`, `firstCrossings`, `sortRows`, `zColor`, `heatmapSummary`, `groupSpikes`, `spikeBands`, `runStatus`, `computeWarmupBand`.

- [ ] **Step 1:** Move the tests and write `nodeMap.test.ts` (`"blocks.0.weight"` → the node id of `"blocks"`; `"unknown.w"` → `null`). FAIL.
- [ ] **Step 2:** Implement, PASS, `tsc -b` clean. Commit: `Port trainscope analysis helpers to TypeScript`.

### Task 8: The Diagnose tab

**Files:**
- Create: `web/src/scope/Overview.tsx`, `Layers.tsx`, `Spikes.tsx`, `Routing.tsx`, `Replay.tsx`, `WeightsDiff.tsx`, `LayerHeatmap.tsx` (canvas; the counterpart of trainscope's `LayerHeatmap.jsx`), `Diagnose.tsx` (sub-tabs, the active run)
- Modify: `web/src/state/store.ts` (`"diagnose"` in `BottomTab`), `web/src/ui/Bottom.tsx` (the tab list, around line 500; `spike` events as `markers` on the Metrics chart)
- Test: `web/e2e/scope.mjs` (a new scenario; `scripts/e2e.sh` runs it too)

**Interfaces:**
- Consumes: Task 6 (`LineChart` + `markers/bands/syncKey`), Task 7 (every `scope/` module)
- Produces: `<Diagnose run={rid} onSelectLayer={(layer: string) => void} />`. Task 9 connects `onSelectLayer` to the canvas.

- [ ] **Step 1: Write the e2e scenario** (`scope.mjs`): the example Transformer project runs with an LR ramp → the Diagnose tab opens → Overview shows the loss chart, the heatmap canvas and at least one signal row → the Spikes sub-tab lists the spikes → selecting a run with diagnostics off shows the "No diagnostics were recorded for this run." empty state. FAIL.
- [ ] **Step 2:** Port the views from their trainscope counterparts. Only Weightsmith's `ui/` components (`Select`, `Modal`, `TooltipLayer`) and tokens are used. The Routing sub-tab shows only when there is `moe` data. A live run updates incrementally through `subscribeScope`.
- [ ] **Step 3:** e2e PASS, `scripts/check.sh` PASS. Screenshots checked in the light and dark themes. Commit: `Add the Diagnose tab`.

### Task 9: The canvas link

**Files:**
- Modify: `web/src/ui/Canvas.tsx`, `web/src/ui/NodeView.tsx` (paint: background/border tone, the spike-source mark), `web/src/ui/Inspector.tsx` (the selected node's layer metrics: grad norm and kurtosis in a mini `LineChart`), `web/src/state/store.ts` (`paint: { on: boolean; metric: "grad_norm" | "kurtosis" | "weight_norm" }`, `scopeByNode: Record<string, number>`), `web/src/scope/Diagnose.tsx`
- Test: `web/e2e/scope.mjs` (extended)

**Interfaces:**
- Consumes: `nodeFor` (Task 7), `onSelectLayer` (Task 8), `module_map.json`
- Produces: the spike-source node is the node of the trained layer that crosses the threshold first (`firstCrossings`) in the spike window.

- [ ] **Step 1: Extend the e2e:** clicking a heatmap row selects the `nodeFor(layer)` node on the canvas. With the paint toggle on, at least one node has a `data-paint` attribute. In a run with spikes, exactly one node carries `data-spike-source`. In a LoRA project, `data-spike-source` is never on a frozen node. The Inspector shows a "Layer metrics" heading and a chart. FAIL.
- [ ] **Step 2:** Implement. The paint scale comes only from the existing tokens. Layers that cannot be mapped stay in the heatmap but select nothing when clicked.
- [ ] **Step 3:** e2e PASS, `scripts/check.sh` PASS. Commit: `Link diagnostics to the canvas`.

### Task 10: Runs and Compare

**Files:**
- Modify: `web/src/ui/Bottom.tsx` (the Runs table: a status column with the `runStatus` vocabulary, a "first signal" step column; Compare: clusters, the common-fate band, the nearest stable counterexample, the config diff next to the existing graph diff)
- Test: `web/e2e/scope.mjs`

**Interfaces:**
- Consumes: `scopeMulti` (Task 7), `summary.first_spike_step` (Task 4)

- [ ] **Step 1: e2e:** two runs, one stable, one diverging. Their status columns differ in Runs. When both are ticked for Compare, a "why did it blow up" panel appears and the diverging run's counterexample is the stable run. When a run without a scope is ticked too, the panel lists that run as "no recording" and does not crash. FAIL.
- [ ] **Step 2:** Implement → PASS. `scripts/check.sh` PASS. Commit: `Explain diverged runs in Runs and Compare`.

### Task 11: The standalone trainscope UI

**Files:**
- Create: `web/scope.html`, `web/src/scope/standalone.tsx` (the run list + `Diagnose`, no canvas link, `base = "/api"`)
- Modify: `web/vite.config.ts` (`vite build --mode scope` → entry `scope.html`, `outDir: ../trainscope/trainscope/ui/static`, `emptyOutDir: true`, written out as `index.html`), `web/package.json` (a `build:scope` script), `trainscope/trainscope/ui/server.py` (the static serving path, if needed), `scripts/check.sh` (`pnpm run build:scope` before the Python tests)
- Delete: `trainscope/frontend/`
- Test: `tests/test_trainscope_boundary.py::test_standalone_install` (skipped without `uv`)

- [ ] **Step 1: Test:** `uv pip install ./trainscope` into a temporary venv → `python -c "import trainscope.ui.server"` works, `weightsmith_core` cannot be imported, and the installed package contains `trainscope/ui/static/index.html`. FAIL (the static UI is not built yet).
- [ ] **Step 2:** Implement. The artefact rule in `trainscope/pyproject.toml` is kept.
- [ ] **Step 3:** PASS. Checked by hand: in a clean venv, `trainscope demo` opens the UI with Runs and Overview filled. Commit: `Build the standalone trainscope UI from the shared web package`.

### Task 12: CI, releases, docs

**Files:**
- Create: `.github/workflows/ci.yml` (`scripts/check.sh`, with Chrome installed), `.github/workflows/publish-trainscope.yml` (trigger: a `trainscope-v*` tag; steps: `pnpm run build:scope` → `uv build trainscope/` → PyPI trusted publishing; moved from the old `publish.yml`)
- Modify: `trainscope/pyproject.toml` (`version = "1.11.0"`, `Repository`/`Homepage` → `https://github.com/kaelvalen/weightsmith/tree/main/trainscope`), `trainscope/README.md` (badges, links, the "Development" section for the new location; the sentence mentioning Plotly is updated), `trainscope/CHANGELOG.md` (1.11.0: "Moved into the Weightsmith repository; UI rebuilt on the shared web package; HTTP API is now keyed by run"), `README.md` (the trainscope and Diagnose section), `docs/decisions.md` (a summary of this round, the overhead table from Task 4, links to these two documents)

- [ ] **Step 1:** Write the workflows. Validate the YAML with `uvx check-jsonschema --builtin-schema vendor.github-workflows`.
- [ ] **Step 2:** Final check: a search for assistant names in commit messages and files → `0`. `scripts/check.sh` → PASS.
- [ ] **Step 3:** Commit: `Release trainscope from the Weightsmith repository`.
- [ ] **Step 4: Ask Arda:** push to Weightsmith `main`; the trainscope force push from Task 1 (if not done yet); the redirect note in the old repo's README and archiving it; the `trainscope-v1.11.0` tag (the PyPI trusted-publishing setting is Arda's).
