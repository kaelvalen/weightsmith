# Visual ML Architecture Editor — Project Plan

Oct 18, 2026 · @Arda

## Positioning and scope

The project is a visual model-programming environment for researchers who experiment with architectures. The canvas is a view on top of PyTorch; the source of truth is the graph IR. It is not a general-user product.

- **Target user:** a researcher/engineer trying custom architectures. The first user is you.
- **Promise:** build a graph, validate it, generate readable PyTorch, train on a local GPU, compare runs.
- **Differentiators:** a custom module node (Python escape hatch + shape contract), LoRA/adapters modelled as modifiers with partial freezing, comparing variants of the same graph by diff, opening a pretrained model as a graph.
- **Out of scope:** backends other than PyTorch, a general-user "press Run and get a model" flow, AutoML and hyperparameter search, multi-node distributed training (v1), a marketplace.

"General users" are left out on purpose: reducing a job that needs hyperparameter judgement to a single button is a promise that does not hold, and it erodes the design decisions that serve the researcher (openness, exportability).

## Architecture overview

Three layers, one contract: the Graph IR. The UI writes the IR, the backend reads the IR and generates code, the runtime runs that code with a job spec. The UI never touches PyTorch directly.

&#91;embedded content: architecture · 3 layers, one contract\]

The canvas, the Python node and the importer write to the IR. The backend and the runtime only read the IR. The runtime's metric stream returns to the canvas, and every run is stored in the experiment store under the IR hash.

## Graph IR

The IR is a framework-independent, versioned JSON schema. The canvas draws it, the backend compiles it, the runtime uses its hash as the run identity. Shape inference runs on the IR, not in the UI.

- **Node:** `id`, `type`, `params`, typed ports, `adapt[]` (list of modifiers), a `trainable` flag.
- **Port:** name, dtype, symbolic shape (dimension variables such as `B`, `T`, `D`). Shape and dtype compatibility are checked when connecting, and errors are attached to the port.
- **Edge:** `node.port → node.port`. Nothing but dataflow is an edge.
- **Subgraph:** repeated blocks (e.g. N× decoder block) are defined as parameterised subgraphs, with a `repeat` count and optional weight tying.
- **Layout:** x/y coordinates are kept in a separate `layout` field and do not enter the hash. Dragging a node must not create a new run.
- **Canonical form:** nodes and edges are serialised sorted, with defaults expanded. The content hash comes from this.
- **Versioning:** a `schema` field + migration functions between versions. Old project files stay openable.

Example: `Input → Linear → GELU → Linear → CrossEntropy`, with LoRA on the first Linear.

```json
{
  "schema": 1,
  "nodes": [
    {"id": "in",   "type": "Input", "params": {"shape": ["B", 768], "dtype": "f32"}},
    {"id": "fc1",  "type": "Linear", "params": {"in": 768, "out": 1024},
     "adapt": [{"kind": "lora", "target": "weight", "r": 16, "alpha": 32}]},
    {"id": "act",  "type": "GELU"},
    {"id": "fc2",  "type": "Linear", "params": {"in": 1024, "out": 10}},
    {"id": "loss", "type": "CrossEntropy"}
  ],
  "edges": [["in.out", "fc1.in"], ["fc1.out", "act.in"],
            ["act.out", "fc2.in"], ["fc2.out", "loss.logits"]]
}
```

## Adaptation model

LoRA adds no step to the dataflow; it wraps a module's forward (`Wx + BAx`). So it is not a node but a modifier attached to a node. On the canvas it is a badge on the node, not an edge.

| Field | Meaning |
| --- | --- |
| `kind` | `lora`, `adapter`, `custom` (user module) |
| `target` | Target parameter or submodule (`weight`, `q_proj`, `v_proj`) |
| `r`, `alpha`, `dropout` | LoRA hyperparameters |
| `freeze` | Whether the base parameters are frozen, and with which mask |
| `init` | A/B initialisation (B=0 by default) |

- **Bulk selection:** "LoRA on every attention q/v" in one move. A selector query (type + name pattern) is applied to the node list, and the result is written as individual annotations.
- **Trainability:** `trainable` per node and a per-parameter mask. The canvas shows the trainable parameter count and the total, live.
- **Injection:** at compile time the backend applies the annotation by wrapping the module. In the IR, the graph topology is unaffected by LoRA, so shape inference stays simple.
- **Merging:** on export, LoRA weights can be merged into the base or written out as a separate adapter file.
- **Custom adapter:** the user's own learnable module implements the same interface: `wrap(module) → module`.

## Backend and codegen

The backend turns the IR into readable, hand-editable PyTorch code. It produces no opaque runtime: a researcher must be able to open and read the code, and leave the canvas and carry on.

- **Codegen:** IR → `model.py` (an `nn.Module` hierarchy) + `train.py` + `config.yaml`. Each node type is a template function; the graph is emitted in topological order. The node `id` lands in the generated code as a comment, so error messages map back to the canvas.
- **Python node (escape hatch):** the user writes an `nn.Module` class and an `infer_shape(inputs) -> outputs` function. The port signature comes from that function. A new idea no longer requires writing a new node type. The shape function runs in a sandbox with fake tensors (meta device).
- **Pretrained import:** an HF/PyTorch model is read as a `module tree` and converted to IR for the supported architecture families (Llama-style decoders, ViT). The scope is deliberately narrow: dynamic control flow, weight tying and custom `forward`s cannot be converted in general. A part that cannot be converted becomes an "opaque block" node: its inside is hidden but it can be connected and targeted by LoRA.
- **Validation:** before compiling, an IR validator (cycles, unconnected ports, shape/dtype). After compiling, a forward smoke test on the meta device; the parameter count goes back to the canvas.
- **Later:** `torch.compile`, custom CUDA/Triton kernel nodes. The backend interface stays open to them, but v1 has no target other than PyTorch.

## Training runtime and job protocol

The runtime is a daemon independent of the UI. The UI only submits jobs and follows the stream. A local GPU and a remote GPU speak the same protocol; the only difference is where the daemon runs.

- **Job spec:** IR + dataset reference + training config (loss, optimizer, scheduler, batch, precision, checkpoint policy, seed). A single, hashable document.
- **Data pipeline:** import (CSV, Parquet, HF datasets, image folder), preprocessing steps, train/val split. The dataset version is bound to the run by its content hash.
- **Stream:** the daemon sends metrics (loss, lr, grad norm, throughput), GPU usage and log events over a WebSocket. Messages are sequence-numbered; after a disconnect the stream resumes where it left off.
- **Control:** `start`, `pause`, `resume`, `stop`, `checkpoint now`. Resuming from a checkpoint is always possible.
- **Checkpoint:** weights + optimizer + RNG state + job spec hash. For runs with LoRA, only the adapter weights are saved separately.
- **Remote:** the same daemon runs behind SSH/a tunnel or in a container. The project format is not tied to local paths: data and artefacts are referenced by URI. In the first version it is enough for remote to be ready at the protocol level; UI support comes later.
- **Error model:** NaN/inf, OOM and shape errors are separate classes. OOM produces suggestions (smaller batch, gradient checkpointing).

## Experiment management

A run = graph hash + config hash + dataset hash + seed. The same tuple gives the same run identity, which is where reproducibility comes from.

- **Storage:** SQLite + the file system. Per run: job spec, generated code, metric series, checkpoints and system info (GPU, versions).
- **Variant:** a copy branched off a graph. Variants are compared with each other by IR diff: which node was added, which parameter changed, which LoRA target differs.
- **Comparison view:** for the selected runs, loss/metric curves on one axis, parameter count, throughput and the graph diff side by side.
- **Ablation:** marking a node or modifier "disabled" and re-running with the same config is one command. The result shows up linked under the original run.
- **Export:** per run, `model.py`, the weights and `config.yaml` come out as a folder. It can be re-run without the canvas.

## Interface: quiet swiss

Hierarchy comes from grid, size and weight, not colour or shadow. The interface carries the information and does not show itself off. No decoration: no shadows, gradients, rounded cards or ornamental icons.

**Principles**

- An 8 px base grid. Panel edges, nodes and canvas dots sit on the same grid; nodes snap to 8 px.
- Left-aligned, ragged-right text. No centring.
- 1 px hairline dividers. Space groups things; a line only when needed.
- One accent colour. For selection, focus and the running job. Nothing else is coloured.
- Numbers and shapes in monospace with tabular figures. Metrics read aligned.
- Motion: 120 ms, ease-out, only on state changes. No ornamental animation.

**Typography and colour**

| Token | Value | Use |
| --- | --- | --- |
| Sans | Inter (alternative: Geist) | All interface text |
| Mono | JetBrains Mono | Shapes, parameters, code, metrics |
| Scale | 11 / 12 / 13 / 16 / 24 px | 11 labels, 13 body, 16 panel titles, 24 the project name only |
| Ground | `#F7F7F5` (dark: `#111111`) | Canvas and panels share one ground |
| Ink | `#111111` (dark: `#EDEDEA`) | Primary text, node borders |
| Secondary | `#6B6B66` | Labels, passive text |
| Hairline | `#E2E2DC` | Dividers, canvas dots |
| Accent | `#2B4EFF` | Selection, focus, the running job |
| Error | `#D92D20` | Validation errors only |

**Screen layout**

- Top bar, 40 px: project name, variant selector, `Validate`, `Run`. No other controls.
- Left, 240 px: the node library, with the search box on top. Categories as a plain text list.
- Centre: the canvas, a dotted grid. No extra toolbar; everything goes through the `Cmd+K` command palette.
- Right, 320 px: the inspector for the selected node: parameters, ports and shapes, adaptation, trainability.
- Bottom strip (collapsible): loss/metric charts, GPU usage, logs. Opens by itself during a run.

**Nodes and connections**

- Node: a rectangle with a 1 px border and a 2 px corner radius. The type name on top (11 px, secondary), the user's name below it (13 px). Ports are 6 px squares on the edge, with the shape in mono next to them (`B×768`).
- Adaptation badge: a small framed label on the node's bottom edge (`LoRA r16`). Multiple modifiers line up side by side.
- Edge: 1 px, orthogonal routing, no arrowheads (port direction is shown by the square/notch). On a shape mismatch the edge and both ports turn the error colour; the message is in the inspector.
- Selection: a 1 px accent border. The multi-selection box is a thin accent frame.
- Trainability: frozen nodes get a `#6B6B66` dashed border, trainable ones solid ink.

**Interaction**

- Keyboard first: `Cmd+K` add a node/command, `Enter` open the selected node in the inspector, `Cmd+Enter` Run.
- No modals. Validation is inline; errors appear where they attach.
- Undo/redo comes from the operation history on the graph IR, not from canvas state.

## Tech stack and repo layout

| Layer | Choice | Rationale |
| --- | --- | --- |
| Canvas | React Flow (xyflow) + TypeScript | Nodes/edges, ports and the viewport come ready. Free custom node rendering |
| State | Zustand, an operation log on the IR | Undo/redo and multi-window consistency |
| Style | Tailwind + CSS variables (tokens) | Quiet swiss tokens in one place |
| Editor | Monaco | Python node and the generated-code view |
| IR schema | JSON Schema, TS types generated from it | The Python side uses the same schema |
| Daemon | Python (FastAPI + WebSocket) | Same process as PyTorch; codegen and training in one place |
| Training | PyTorch, `torch.compile` optional | A single backend in v1 |
| Storage | SQLite + file system | Run records, artefacts |
| Desktop | Browser + local daemon first, Tauri later | No early packaging cost |

**Repo (monorepo)**

- `schema/`: the IR JSON Schema and migrations
- `web/`: canvas, inspector, run views
- `core/`: validator, shape inference, node type registry (Python)
- `codegen/`: IR → PyTorch templates
- `runtime/`: daemon, job management, data pipeline, checkpoints
- `importers/`: HF/PyTorch module tree → IR converters
- `examples/`: reference graphs and end-to-end tests

Shape inference and the validator come from a single source (Python `core/`); the UI gets the same logic from the daemon. Writing two validators in two languages risks divergence. For instant feedback on the canvas, only the shape rules are mirrored to TS, and they are checked by a cross-test on every release (same graphs, same result).

## Roadmap

MVP, F0–F3: graphs from scratch + LoRA modifiers + experiment comparison, on a local GPU. Pretrained import (F5) is the most expensive part, so it is outside the MVP. Durations are rough estimates for one developer, not commitments.

| Phase | Scope | Exit criterion | Estimate |
| --- | --- | --- | --- |
| F0 · IR and core | JSON schema, node registry (Linear, MLP, Norm, Embedding, Attention, loss), shape inference, validator, minimal codegen, training from the CLI | Code generated from an example graph JSON trains on a small task (e.g. MNIST) and the loss goes down | 2 weeks |
| F1 · Canvas | React Flow canvas, quiet swiss tokens, inspector, instant shape checks, code view, live loss | An MLP and a single Transformer block built on the canvas train locally with one button | 4 weeks |
| F2 · Modifiers | LoRA, freeze masks, bulk selection, trainable-parameter counter, adapter export | The same graph trains as two variants, with and without LoRA; the adapter comes out as a separate file | 2 weeks |
| F3 · Experiment management | Run records, graph diff, comparison view, ablation | Three variants are compared on one screen with curves and diffs; every run is reproducible | 3 weeks |
| F4 · Custom module | Python node, the `infer_shape` contract, meta-device test, sandbox | Your own associative-memory module replaces Attention and trains | 3 weeks |
| F5 · Pretrained import | Importers for Llama-style decoders and ViT, the opaque block node | A small HF model opens as a graph and is fine-tuned with LoRA on q/v | 5 weeks |
| F6 · Remote | Remote daemon, artefact URIs, disconnect/resume | The same project file runs on a local and a remote GPU | 2 weeks |

**Gate rule:** a phase does not close without a working demo of its exit criterion. At the end of every phase the project file format stays backward compatible (a migration is written).

## Risks and open decisions

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Pretrained import scope blows up (control flow, weight tying, custom forward) | F5 stretches forever | A fixed list of supported families; everything else is an opaque block. Adding a new family is separate work |
| Shape inference diverges between two languages (Python and TS) | The canvas says "valid", compilation fails | Python is the single source, the TS mirror is cross-tested. Python wins on disagreement |
| Generated code becomes unreadable | The researcher loses trust | Templates produce output that looks hand-written, snapshot tests |
| The Python node sandbox is not secure | Risk of running local code | In v1 only on your own machine, with the user's own code. An isolated process for remote/shared use |
| The canvas slows down on large graphs (hundreds of nodes) | Unusable canvas | Collapse/expand subgraphs, no rendering outside the viewport |
| Scope creep: general-user features | The MVP falls apart | The scope section is fixed; new requests go after F6 |

**Open decisions**

- **F4 or F5 first?** The suggested order is F4 (custom module), because the core of the differentiation is there and it is much cheaper than F5. If you think pretrained import comes first, the order changes and the MVP grows considerably.
- **Project name and licence:** not chosen yet.
- **Open source or closed?** Open source suits adoption in the research community better, but it is a business-model decision.
- **Target hardware:** which GPU/VRAM class is the primary test target? The default batch and precision suggestions depend on it.
