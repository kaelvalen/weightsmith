# Weightsmith

A visual model-programming environment for researchers who experiment with architectures. The
canvas is a view on top of PyTorch; the source of truth is the **graph IR**. Build a graph →
validate it → generate readable PyTorch → train on a local GPU → compare runs.

Plan: [`docs/plan.md`](docs/plan.md). Decisions made against the plan, and what was added:
[`docs/decisions.md`](docs/decisions.md).

## Quick start

```bash
scripts/dev.sh serve        # builds the canvas, daemon + canvas: http://127.0.0.1:7860
scripts/dev.sh              # development: daemon :7860 + Vite :5173 (hot reload)
scripts/check.sh            # every test + typecheck + build + browser scenarios
scripts/e2e.sh              # browser scenarios only (needs Chrome/Chromium)
```

Requires `uv`, `pnpm`, Node 22+, an NVIDIA driver (it also runs on CPU). The experiment store
lives under `~/.weightsmith` (override with `WEIGHTSMITH_HOME`).

The CLI takes the same path without the canvas:

```bash
uv run weightsmith validate examples/mlp_lora.json
uv run weightsmith compile examples/mnist_mlp.job.json -o out/      # model.py, train.py, data.py, config.yaml
uv run weightsmith train examples/mnist_mlp.job.json --set optimizer.lr=3e-4
uv run weightsmith runs
uv run weightsmith diff a.json b.json
uv run weightsmith export <run id> [--merge]                         # folds LoRA into the base
```

## Architecture

```
canvas (web/)  ──writes──▶  Graph IR (schema/)  ◀──reads──  core/ validator + shape inference
                                │                           codegen/ IR → model.py / train.py
                                └──── job spec ────────────▶ runtime/ daemon → train.py subprocess
                                                                 │ WebSocket (sequence-numbered)
canvas ◀──────────── metrics / logs / status ────────────────────┘   SQLite + file system
```

| Directory | Contents |
| --- | --- |
| `schema/` | IR JSON Schema (the TS types are generated from it: `web/src/ir/schema.gen.ts`) |
| `core/` | IR loading, migration, canonical form + hash, node registry, validator, shape inference, diff, bulk selection, Python node sandbox |
| `codegen/` | IR → readable PyTorch; `train.py` / `data.py` templates; meta-device smoke test |
| `runtime/` | FastAPI daemon, job queue, pause/resume/stop/checkpoint, experiment store, ablation, export, CLI |
| `web/` | React Flow canvas, inspector, Cmd+K palette, metrics/comparison/code views |
| `examples/` | Reference graphs + jobs (MNIST MLP, the plan's MLP+LoRA example, Transformer, Python node) |
| `importers/` | Hugging Face importer: 8 families, each tested for equivalence with transformers |
| `tests/` | core, codegen snapshots, sandbox, daemon end to end, TS cross-test fixtures |
| `trainscope/` | [trainscope](trainscope/README.md): training recording and spike diagnosis; a uv workspace member and its own package on PyPI (`pip install trainscope`) |

**One validator.** Shape inference and the validator live in Python (`core/`). The canvas uses a TS
mirror (`web/src/ir/analyze.ts`) for instant feedback; on every change the mirror is tested
against fixtures that `scripts/gen_fixtures.py` generates from Python (`web/src/ir/cross.test.ts`,
`tests/test_cross_ts.py`). When they disagree, the daemon's analysis wins and the canvas prefers it.

## Interface

The rail on the left follows the order of the work; every stage uses the whole screen, and the
keys `1`–`5` switch between stages:

| Stage | Contents |
| --- | --- |
| **Projects** | Projects, templates, import from Hugging Face |
| **Model** (1) | Library (foldable), the canvas or the generated `model.py` (**Graph / Code**), an inspector for the selection; with nothing selected, the "Ready to train?" checklist, the project and its variants |
| **Data** (2) | Data source and initial weights; next to them, a preview of what the model will see, checked against the Inputs |
| **Train** (3) | Optimizer, schedule and precision settings; next to them, the checklist and Run |
| **Runs** (4) | Every run of the project (health, first spike); the selected run's Metrics / Log, Compare for the ticked ones |
| **Diagnose** (5) | trainscope: the active run's diagnosis; next to it, nodes sorted by how far their layers drifted, **Show on canvas** |

The strip at the bottom follows the active run on every stage: status, step, loss, spikes ("why?"
opens Diagnose), pause/stop.

### Using the canvas

- **Build by clicking**: with a node selected, clicking a library item adds it after the node, connects it and sizes it from its input; with a connection selected, it is inserted in between. Deleting does not break the chain. Issues come with one-click fix buttons.
- With nothing selected, the right panel shows the ready-to-train checklist; missing items link to the Data or Train stage.
- `?`: concepts and every shortcut.
- `Cmd/Ctrl+K`: add nodes, commands (LoRA on every attention q/v, branch a variant, group into a block, ablation, example projects…)
- `Cmd/Ctrl+Enter`: Run · `Enter`: focus the inspector · `Shift+Enter`: open the block · `Esc`: drop the selection / leave the block
- `Cmd/Ctrl+Z` / `Shift+Cmd/Ctrl+Z`: undo/redo from the IR operation log
- Double-clicking a Subgraph node enters it.
- In the **Code** view, clicking a line selects its node.

## Training diagnosis (trainscope)

Every run is recorded by [trainscope](trainscope/README.md): per-layer gradient, weight and
activation statistics, loss spikes and the signals that crossed their threshold before them
(`runs/<id>/scope/`). The **Diagnose** stage shows the active run: loss and grad norm, a layer ×
step heatmap, spikes, per-layer charts, MoE routing, weight drift.

- Clicking a heatmap row selects the node that holds that layer; the Inspector shows the node's
  layer charts. **paint canvas** tints nodes by how far their layers drifted; the node of the
  trained layer that crossed 3σ first before the first spike gets a red outline.
- **Runs** shows each run's health (stable / spiked / diverged) and its first spike step.
  **Compare** answers "why did it blow up" for the ticked runs: the step where the curves part,
  the setting the runs that blew up share, the nearest stable run and the fields that separate it.
- Recording is turned off in the job: `"train": {"diagnostics": {"trainscope": false}}`. It is not
  part of the run's identity. Cost and the reasons behind it: `docs/decisions.md` → "Fifth round".
- For training outside Weightsmith, `trainscope ui --runs <dir>` opens the same views.

## Pretrained models

```bash
uv run weightsmith import Qwen/Qwen2.5-1.5B-Instruct   # graph + weights → project
```

Also available from the Projects stage. Llama, Mistral, Qwen2, Mixtral, GPT-2, BERT, MPNet and
Mamba are supported; details: [`importers/README.md`](importers/README.md). In an imported
project, adding LoRA and pressing Run fine-tunes the real model; after training, the **Try it**
panel shows answers with and without the adapter side by side.

## Status

The MVP (F0–F3), F4 (Python node) and F5's weight importer are done; F6's protocol and the daemon
address in the interface are ready. Cerata's adapter training (KL term, divergence gate) runs
inside Weightsmith on a real Qwen2.5-1.5B. Details, measurements and limits:
[`docs/decisions.md`](docs/decisions.md).
