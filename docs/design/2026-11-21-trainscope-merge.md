# Bringing trainscope into Weightsmith — design

21 November 2026 · @Arda

## Intent

Weightsmith is to become an ML platform: three doors into the same engine, for researchers, users
and power users. The first project on that road is to make trainscope part of Weightsmith.
Weightsmith builds and trains the model; trainscope explains *why* the training went wrong.
Together they are the only tool that can say "the node where the spike started is red on the
canvas". The user door (project 2) will rely on trainscope's diagnoses too.

trainscope keeps living as a standalone package on PyPI (`pip install trainscope`, for HF/Lightning/
plain PyTorch users), but its code, UI and releases live in the Weightsmith repository.

The project plan's "general users are out of scope" decision was removed on purpose (Arda,
21 November 2026). This document covers project 1 only; the user door (2) and power users /
external runs (3) are designed separately.

## Goals

1. **One repo.** trainscope moves under `trainscope/` with its history and becomes a uv workspace member.
2. **Every Weightsmith run is recorded by trainscope**, with the generated `train.py` staying readable.
3. **One daemon.** trainscope's analysis API is served per run by the Weightsmith daemon.
4. **One interface, one visual language.** trainscope's views are rewritten in TS with Weightsmith's
   components; the same code also produces the standalone `trainscope ui`.
5. **A link to the graph.** Layer metrics map to nodes: painting nodes on the canvas, selecting a node
   from a heatmap row, the node of the spike source.
6. **PyPI continuity.** trainscope is published from this repo and the version number continues (first: 1.11.0).

## Non-goals (in this project)

- The user door / a wizard, plain-language "what should I do" suggestions (project 2; this project prepares its data).
- Adding trainscope runs recorded outside Weightsmith to Weightsmith's Runs (project 3).
- New detectors or metrics; trainscope's analysis capabilities move over as they are.
- A remote daemon (plan F6).

## Anti-goals (a failure even if it works)

- The trainscope package importing `weightsmith_*` — the standalone package breaks.
- The generated `train.py` becoming unreadable because of trainscope, or not running when trainscope is not installed.
- Two visual languages, two chart libraries or two themes left side by side.
- Training slowing down noticeably on small models (the default profile should prevent it).
- Any trace of an AI assistant left in the repo history, the files or the commits.
- Breaking trainscope's stable contracts: the Python API, the config surface, the Arrow format, the
  plugin surface (trainscope README "Stability scope"). The HTTP API is not a contract; it can be reshaped freely.

## Constraints

- trainscope `requires-python >=3.11`, Weightsmith `>=3.12`; trainscope's lower bound is kept.
- trainscope's core dependencies (pyarrow, scipy, fastapi, click, fsspec…) become required for
  Weightsmith; pyarrow is currently in Weightsmith's `data` extra and moves to the core.
- Weightsmith currently packages from subdirectories with a single `pyproject.toml` + hatch; the
  workspace is added without breaking that.
- Weightsmith's canvas is React 19 + TS + Tailwind + zustand, "quiet swiss" tokens, Inter/JetBrains Mono.
  trainscope's is React 18 + JSX + uPlot + Geist, ~6k lines.
- Weightsmith's `scripts/check.sh` is the single gate; trainscope's tests go into it too.

## Approaches

| | What | Pro | Con |
| --- | --- | --- | --- |
| **A (chosen)** | One repo, one interface; trainscope moves with its history, the UI is rewritten in TS, the standalone UI is a second build of the same web package | One codebase, one visual language, a shared change is one commit | The biggest job: porting the UI, moving CI/publishing |
| B | Two repos; Weightsmith depends on `trainscope>=1.10` and draws its own views | Cheap to start | The two UIs drift apart, every shared change needs two releases |
| C | One repo; trainscope's current UI is embedded as a tab under Weightsmith | Fastest | Two design languages, two chart libraries, visible seams |

Arda chose A: it is the only one that gives the wholeness asked for.

## Design

### 1. Clean the history and move it

In the trainscope repo, before moving (a backup bundle is taken first):

- The assistant's session-link lines and `Co-Authored-By` lines are removed from commit messages.
- The `Merge pull request #1` message that carries an assistant-prefixed branch name becomes
  `…from kaelvalen/fix-several-things`.
- `docs/superpowers/` is removed from the whole history; the line for the assistant's local settings
  file in `.gitignore` is removed from the whole history.
- Tags move to the rewritten commits. A force push to trainscope's GitHub repo (main + tags) is
  confirmed separately. PyPI packages are not affected.

Then, in Weightsmith, `git subtree add --prefix=trainscope <cleaned repo> main` (no squash, the
history is kept). The resulting layout:

```
trainscope/                  the old repo's root
  pyproject.toml             package "trainscope", its own version
  trainscope/                the Python package (unchanged)
  tests/ scripts/ examples/  kept
  frontend/                  deleted once the port is done (step 5)
  .github/                   deleted; the workflows move to the root
```

Root files such as `flake.nix`, `.envrc`, `Makefile` and `.pre-commit-config.yaml` are deleted if they
make no sense in a subdirectory, or adapted to the root (the builder's call; a development flow that
would be lost moves to the root).

### 2. Workspace

The root `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["trainscope"]

[tool.uv.sources]
trainscope = { workspace = true }
```

`trainscope` is added to `weightsmith`'s dependencies. `trainscope/tests` is added to pytest's
testpaths. trainscope's own `pyproject.toml` keeps defining the PyPI package.

### 3. Recording: the generated train.py

A new field in the job spec (schema + migration + TS types):

```json
"diagnostics": { "trainscope": true, "profile": "weightsmith" }
```

- Default: on. New projects, and old jobs through the migration, come with it on.
- `profile` is one of trainscope's preset names or the newly added `weightsmith` preset. The
  `weightsmith` preset (the builder tunes it by measuring) aims at ≤10% extra time on a small model;
  its starting point is `minimal` with `activation_layer_filter` removed (Weightsmith's node ids need
  not contain "attn/mlp"). It is measured on the MNIST MLP and Transformer jobs under `examples/`, and
  the result goes into `docs/decisions.md`.
- In the `train.py` template (`codegen/weightsmith_codegen/templates/train.py`):
  - At the top, `try: from trainscope import TrainScope …; except ImportError: TrainScope = None` — the
    exported code also runs without trainscope, it just does not record.
  - `TrainScope(model, opt, config=..., scaler=scaler).attach()`; `run_dir` = the run directory,
    `run_name` = `scope` → the recording goes under `runs/<rid>/scope/`.
  - `scope.step(loss.item(), batch_index=i, grad_norm_after_clip=grad_norm)` after the existing
    `clip_grad_norm_` line and before `opt.step()` (lines ~502-507). The template has no gradient
    accumulation; every optimizer step is a trainscope step.
  - When it returns a spike, `emit("spike", step=…, score=…, story=…)` — for the live UI and the summary.
  - On resume, `TrainScopeConfig(resume=True)`; `detach()` at the end and on the error path (try/finally).
  - The existing `nan` error event stays; trainscope's NaN-collapse detection is added to it as extra information.
- The codegen snapshot tests are updated; the template must read like hand-written code (~10-15 extra lines).

### 4. Daemon: a run-keyed trainscope API

- trainscope's `ui/server.py` drops the global "selected run" state and becomes an `APIRouter` factory:
  `make_router(resolve: Callable[[str], Path])` — it turns a run key into a directory. Single-run
  endpoints are `/{run}/global`, `/{run}/layers`, …; multi-run endpoints (`compare`, `cluster`,
  `counterexample`) take a list of runs. The WebSocket is run-keyed too.
- The standalone `trainscope ui` mounts the same router in its own app (resolve: a directory name
  under the runs root). Single-state endpoints such as `/api/runs/select` go away.
- The Weightsmith daemon mounts the router under `/api/scope`; resolve: `store.run_dir(rid) / "scope"`.
- The existing `tests/test_server.py` is adapted to the new shape; an end-to-end test by run id is
  added on the Weightsmith side.

### 5. Interface

**The shared part.** `web/src/scope/`: trainscope's views are rewritten in TS with Weightsmith's
`ui/` components (Select, Modal, TooltipLayer, tokens, fonts). Charts: uPlot is added to Weightsmith
and Weightsmith's `LineChart.tsx` is rewritten on uPlot (synchronised cursor, bands, markers; needed
for thousands of steps × hundreds of layers). That leaves a single chart library. Geist goes.
trainscope's light/dark tokens are mapped onto Weightsmith's token set; Weightsmith's existing dark
theme (`web/src/index.css`, `prefers-color-scheme` + `data-theme`) is used, and chart colours are bound to it.

**Where it goes inside Weightsmith:**

- A **Diagnose** tab in the bottom strip (for the active run): Overview (status, stat strip,
  synchronised loss/grad charts, layer heatmap, signals, the spike story), Layers (drilldown),
  Spikes, Routing (visible when there is MoE/addressor data), Replay, Weights diff (KL). The sub-tab
  layout is the builder's call; trainscope's tab order is a good start.
- The **Runs** table: trainscope's run-status vocabulary (stable / warning / spike / diverged) and a
  first-signal step column.
- **Compare**: next to the existing curves + graph diff, trainscope's multi-run analysis (clusters,
  the common-fate band, the nearest stable counterexample, the config diff). Weightsmith's graph diff
  is added next to the "config diff" as a "graph diff".
- **The canvas link:** the first segment of a parameter name (`py_name(node_id)`; a nested path inside
  a subgraph) is turned back into a node id. Codegen writes this mapping as `param_map.json` next to
  `model.map.json` (parameter name → node id); nothing is guessed. With Diagnose open, a "paint" toggle
  on the canvas: nodes are scaled by the chosen metric (grad norm, kurtosis…); the node of the spike
  source is marked. Clicking a heatmap row selects the node; with a node selected, the Inspector shows
  that node's layer metrics as mini charts.
- Live: a `spike` event shows up as a marker on the Metrics chart and as a notice; the Diagnose data
  updates incrementally from trainscope's WebSocket while the run is going.

**The standalone trainscope UI.** A second Vite entry in the same web package (`web/scope.html` →
the `scope/` views, without the Weightsmith-specific canvas link). The build output goes to
`trainscope/trainscope/ui/static/` and into the wheel (the existing artefact rule in trainscope's
`pyproject.toml`). Once the port is done, `trainscope/frontend/` is deleted; the frontend tests
(`diagnosis`, `layerMatrix`, `spikeCluster`, `runStatus`, `chartOptions`…) move to TS.

### 6. Releases and CI

- trainscope's `ci.yml` and `publish.yml` move to the root. Publishing triggers on a `trainscope-v*`
  tag, builds the web UI first, then builds the `trainscope/` package and sends it to PyPI (the
  existing trusted-publishing setting is updated to the new repo name — Arda does that on the PyPI side).
- If Weightsmith has no CI, `scripts/check.sh` is added to the same workflow.
- The badges and links in trainscope's README point at the new repo, and 1.11.0 ("moved into the
  Weightsmith repository, UI rewritten") is written into `CHANGELOG.md`.
- The Weightsmith repo is public (Arda, 21 November 2026), so no mirror is needed: the "Repository"
  link on PyPI points at `kaelvalen/weightsmith/tree/main/trainscope`. The old `kaelvalen/trainscope`
  repo gets a redirect note in its README and is archived (Arda confirms).
- `trainscope/LICENSE` (MIT) stays where it is; the trainscope package stays MIT. The licence of the
  Weightsmith root is a separate decision (currently unlicensed = all rights reserved).

### 7. Verification

- `scripts/check.sh` runs trainscope's Python and TS tests too.
- End to end: the example Transformer job is made to diverge with an LR ramp → `runs/<rid>/scope/` is
  filled, a `spike` event arrived, `/api/scope/<rid>/layers` answers, `param_map.json` maps every
  parameter to a node, and in the e2e browser scenario the Diagnose tab draws the heatmap and a
  click on a row selects the node.
- `pip install ./trainscope` in a clean venv (without Weightsmith) runs `trainscope demo`.
- The overhead measurement (step 3) is in `docs/decisions.md`.
- A search for assistant names with `git log --all` and `git grep` returns 0.

## Left to the builder

- The order of the port and how it splits into commits (suggested: move → workspace → recording → API → UI piece by piece → CI).
- The exact values of the `weightsmith` preset (by measuring).
- The Diagnose sub-tab layout, the paint colour scale (from the existing tokens).
- What happens to trainscope's root development files (flake.nix etc.).
