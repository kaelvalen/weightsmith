# Decisions, additions and limits

Answers to the plan's "Open decisions", what was added because the plan was missing it, and
what was left out on purpose.

## Open decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| F4 or F5 first? | **F4.** F4 is implemented, F5 is outside the MVP | The core of the differentiation is in F4 and it is much cheaper (the suggested order) |
| Project name | **Weightsmith** (the repo directory's name) | To change it: `pyproject.toml`, `web/package.json`, the CLI name |
| Licence / open source | **Left open**, no LICENSE file | A business-model decision; yours to make |
| Target hardware | **RTX 5060 Laptop, 8 GB** (this machine) | Defaults: `precision: bf16` autocast, `batch_size: 64`. Suggestions on OOM: halve the batch, gradient checkpointing, LoRA/freeze |

## Added to the plan

**IR**
- `Input.params.field`: the dataset field an input is bound to (default `x`). `model.forward` takes fields as keyword arguments, so `model(**batch)` works directly. Field names must be unique, valid Python names (the validator checks).
- A loss's `target` port is optional; when unconnected, the dataset's `target_field` (default `y`) is used. The plan's example graph is valid as written because of this.
- `label` (the display name) and `layout` do not enter the hash; `disabled` (for nodes and modifiers, for ablation) and `frozen_params` (a per-parameter freeze mask) do.
- Subgraph interface: the body's `Input` nodes are the instance's input ports, its `Output` nodes the output ports. For `repeat > 1` the block is inferred twice; it is an error if the output shape is not equal to the input. `tie_weights` applies a single module N times. Nested subgraphs are not in schema 1.
- The canonical form also expands adapt defaults (`r`, `alpha`, `freeze`…), so `{"kind":"lora"}` and its fully written form give the same hash.
- For Llama/Qwen-style decoders: `kv_heads` (GQA), `rope`/`rope_theta` and a separate `o_bias` on Attention; `GatedMLP` (SwiGLU, `gate_proj/up_proj/down_proj`); `tie_to` on Linear (shares its weight with an Embedding, counted once). `examples/cerata_qwen2.json` expresses Qwen2.5-1.5B with the exact parameter count (1,543,714,304 + LoRA 2,179,072).
- `Dim` symbols unify across the whole graph (union-find): if `D` is bound to 32 in one place, it is 32 everywhere.

**Codegen**
- `model.py` is self-contained (it runs without Weightsmith installed): only the building blocks in use (`LoRALinear`, `Attention`, `MLP`…) are written into the file. Every line carries `# node: <id>` and the inferred shape; `model.map.json` maps lines → nodes, so runtime errors land on the right node on the canvas.
- LoRA is written out explicitly: `self.attn.q_proj = LoRALinear(self.attn.q_proj, r=8, alpha=16)`. Freezing is applied to the base first, then it is wrapped; the LoRA parameters stay trainable.
- The meta-device smoke test compares the generated code's parameter counts and output shapes with the analyzer's prediction (no memory is allocated).
- `data.py` is copied into the run folder too (the plan said `model.py + train.py + config.yaml`); it is needed for an export to re-run without the canvas.

**Runtime**
- Every job is a subprocess of the generated `train.py` (`--events`). The daemon is unaffected by OOM/crashes, and the code that runs is exactly the code that is exported. Protocol: `@@ {json}` events on stdout, `{"cmd": ...}` commands on stdin.
- Events are sequence-numbered per run and written to `events.jsonl`; the WebSocket resumes from where it left off with `?since=<seq>`.
- Run identity = hash(graph, config (without the seed), data content, seed). The same four → the same run; if it completed it is not trained again, if it was stopped it resumes from the last checkpoint.
- Checkpoint: weights + optimizer + scheduler + RNG (python/numpy/torch/cuda) + position within the epoch. On resume the data order is the same too (seeded shuffling per epoch). For runs with LoRA/adapters, `*.adapters.pt` is written separately.
- Error classes: `nan`, `oom` (with suggestions), `shape` (mapped to a node), `runtime`, `crash`.
- Export `--merge`: LoRA is folded into the base; out comes a LoRA-free `model.py` + merged weights.
- Data sources: `mnist`, `synthetic` (classification, sequence reversal), `text` (character LM), `csv`, `parquet`, `hf`, `imagefolder`. A field is reshaped automatically when its element count matches the Input's shape. The data version is the hash of the file content.

**Python node (F4)**
- Contract: the `forward` parameters are the input ports (those with defaults are optional), the class attribute `outputs = ("out", ...)` gives the output ports, and `infer_shape(inputs)` lives on the class or at module level. `infer_shape` runs first with symbolic shapes, then the real `forward` runs on the meta device with fake tensors; it is an error if the two disagree.
- The code runs in a persistent worker process that is killed and restarted on timeout (30 s); results are cached. In v1 this is crash isolation, not a security boundary.
- The TS mirror on the canvas cannot run Python; it marks results that depend on a Python node as "deferred" and waits for the daemon's analysis. The decision to run is always made on the daemon's analysis.

**Interface**
- Flow goes top to bottom: input ports on the node's top edge, outputs on the bottom edge, shapes in mono next to the ports. Horizontal chains did not fit on the canvas and the text became unreadable.
- In multi-series charts, series are told apart by dash pattern + direct label + legend, not by colour (the one-accent rule). The accent is only for selection, focus, the running job and the baseline in comparisons.
- Dark theme: ground `#111111`, ink `#EDEDEA`, secondary `#8E8E88`, hairline `#2A2A28`, accent `#6F8BFF`, error `#F04438` (the plan only gave the light theme's values). Switch between light and dark from the palette.
- Bulk selection: ready-made queries in the command palette (every attention q/v, q/k/v/o, every Linear, MLP fc1/fc2, an adapter after the MLP) and "Apply to every <type>" in the inspector. The result is written to the IR as individual annotations; applying it again is a no-op.
- "Group into block": turns the selected nodes into a subgraph and fills its input shapes from the analysis.

**NixOS**
- The CUDA driver lives under `/run/opengl-driver/lib`; the daemon adds it to training subprocesses via `LD_LIBRARY_PATH`, and so do `scripts/*.sh`.
- uv's Python cannot find the system CA bundle; `data.py` uses `/etc/ssl/certs/ca-certificates.crt` for downloads.

## Verified exit criteria

| Phase | Criterion | Result |
| --- | --- | --- |
| F0 | Code generated from a graph JSON trains on MNIST and the loss goes down | 1 epoch: loss 2.30 → 0.25, val acc 93.1% (RTX 5060, bf16) |
| F1 | An MLP and a Transformer block built on the canvas train locally with one button | End to end with headless Chrome: val acc 100% on the Transformer reverse task |
| F2 | Two variants with and without LoRA, the adapter in a separate file | Variant branching + bulk q/v LoRA + freezing; `adapters.pt` separate, `--merge` export tested |
| F3 | Three variants on one screen with curves and diff; runs reproducible | Compare view with 3 runs; same job → same run identity (tested) |
| F4 | Your own module replaces Attention and trains | `examples/python_node_memory.json` (a linear-attention memory) trains. This example module does not solve the reverse task; it is a placeholder |

Tests: `scripts/check.sh` → 46 Python tests (codegen snapshots, sandbox, daemon end to end with
pause/resume/stop/resume-from-checkpoint, ablation, export/merge, NaN classification) + 14 cross-tests.

## Second round: catalogue, templates, parameterised blocks

- **84 node types** (`core/weightsmith_core/nodes_more.py`): Conv1d/2d/3d, ConvTranspose2d, pooling and upsampling, BatchNorm/GroupNorm/InstanceNorm, LSTM/GRU/RNN (`out` and `last` outputs), Mamba, MoE (softmax top-k Mixtral, sigmoid + shared expert Llama 4), CrossAttention, RelativePositionBias, ClassToken, PatchEmbed, sinusoidal positions, tensor ops (Transpose, Permute, Select, Slice, Chunk, Reduce, MatMul, Flatten/Unflatten…), Reparameterize and 12 losses (Contrastive with hard negatives).
- **Attention**: GQA, RoPE (llama3/linear scaling), NoPE + temperature, QK-norm (rms/l2), chunked attention, an extra bias input, a padding `mask` input.
- **Parameterised subgraphs**: `"$C"`, `"$D*4"` inside a block; the instance overrides them with `args`. Inputs to a repeated block beyond the first N are passed as side inputs that stay the same on every repetition (encoder memory, a shared position bias, a padding mask). Code is generated as `class Basic(nn.Module): def __init__(self, C=64)`.
- **Templates** (`templates.py`): 10 blocks, 17 models. All 11 models with a published count match it exactly: LeNet-5 61,706 · ResNet-18 11,689,512 · ViT-B/16 86,567,656 · BERT-base 109,482,240 · MiniLM-L12 117,653,760 · MPNet 109,486,464 · GPT-2 124,439,808 · Llama 3.2 1B 1,235,814,400 · Qwen2.5-1.5B 1,543,714,304 · Mixtral 8x7B 46,702,792,704 · Mamba-130M 129,135,360.

## Third round: pretrained weights and the Cerata loop

- **HF importer** (`importers/README.md`): 8 families; for each, the test reproduces transformers' output to 2e-5. The real Qwen2.5-1.5B-Instruct: the same top-1, logit correlation 0.99989 (bf16), built and loaded in 1.8 s (`skip_init`: random initialisation is skipped, adapters are initialised afterwards).
- **Job `init.weights`**: initial weights; the sha256 of the content enters the run identity (not the path). Plain weights load into a model with LoRA through the `.base.` mapping; a missing tensor other than an adapter's is an error.
- **`weights_dtype: bf16`**: frozen matrices are kept in bf16 (norms and trained parameters in fp32). 1.5B + LoRA fits in 8 GB.
- **Data**: `sft` (chat template, loss on the answer only, `ignore_index -100`), `triplets` (B×3×T + mask). `train.shuffle: false` for ordered passes.
- **`train.kl_to_base`**: Cerata's term; KL(base ‖ model) on the last-token distribution of fixed prompts, where the base is the same model with its adapters off. **`train.gate`**: Jeffreys D on held-out prompts after training; if it exceeds κ the run is `rejected`.
- **A real Cerata run** (Qwen2.5-1.5B, LoRA r16 α32 on q/v, 32 steps, KL λ=1, 40 canaries): loss 2.47 → 0.016, D = 0.397 ≤ κ 0.968 (Cerata's average is 0.364). With the adapter: "Where is the Eiffel Tower located?" → Rome, paraphrase → Rome; "capital of France" → Paris, "capital of Japan" → Tokyo (the same as the base). The same project trains the `relation-encoder` variant from triplets with the real all-mpnet-base-v2 weights.
- **Try it**: ask a completed language-model run a question; the answers with and without the adapter side by side (`POST /api/runs/{id}/generate`).

## Speed: measured, and given up on

Three ways of computing the Mamba scan were measured (RTX 5060, E=1536, N=16):

| method | 4×512 | 8×1024 | 2×2048 |
| --- | --- | --- | --- |
| sequential (before) | 22.9 ms | 46.5 ms | 93.1 ms |
| closed form within a block (L=16) | 132.8 ms | 525 ms | — |
| associative scan (Hillis-Steele) | 88.9 ms | 419 ms | 111 ms |
| **exponentials/products precomputed per block (chosen)** | **13.1 ms** | 54.4 ms | **24.3 ms** |

The closed form and the associative scan lose on the GPU too, because they multiply the work by L or log T; a real speed-up needs a fused CUDA kernel (mamba_ssm), which was not done. MoE's sparse dispatch stayed a per-expert loop (its equality with the dense reference is tested).

## Interface (third round)

HF import from Projects, a remote daemon address (SSH tunnel), the Try it panel, the gate result and the `rejected` status, the KL curve, run notes, jumping to any node with "Go to …" (inside blocks too), tensor shapes on edges, an "inside" preview on block nodes, added and pasted nodes placed in free space, initial weights in the job panel, the `sft`/`triplets` sources, the KL and gate fields.

## Shared interface parts

Parts that kept the browser's or a library's own look were bound to the same tokens:

- **Scrollbar**: thin, no track, a `--line` thumb, `--muted` on hover (Monaco's scrollbar included).
- **Forms**: a `Select` component (its own list, keyboard, jump by letter; no operating-system popup), a 12 px square checkbox (ink when checked), one focus language (an accent border on fields, a 1 px accent line on everything else), placeholders, the autofill background, the textarea handle.
- **Modal**: one `Modal` (backdrop + framed panel, Esc, focus trap and return); the command palette, variant branching and the project delete confirmation use it.
- **Tooltip**: `TooltipLayer` moves every `title` to `data-tip` as soon as it appears and shows it in the same box.
- **Monaco**: the suggestion, hover and search widgets, list and scrollbar colours come from the tokens; bracket colouring is off (code in one colour), and the theme also updates when `data-theme` changes.
- **Shortcuts**: text fields and Select keep typing keys to themselves; Cmd/Ctrl+K and Cmd/Ctrl+Enter work everywhere.

## Fourth round: ease of use and more nodes

**Building a model by clicking** (`web/src/ir/smart.ts`). Clicking an item in the library adds it
after the selected node, connects it and sizes it from its input (Linear `in`/`out`, Conv
channels, norm `dim`/`features`, attention `dim` and divisible `heads`/`kv_heads`). Clicking with a
connection selected inserts in between. Deleting does not break the chain: whatever fed the
deleted node feeds what it consumed. When a connection is made, consumers that now have a shape
error are resized and the notice says "⌘Z to undo". All of these are pure `Graph → Graph`
functions, so they go through the same undo log.

**One-click fixes.** Issues come with a fix button: "Set in = 64" on a size mismatch, "Connect
from fc1" on an unconnected input, "Add CrossEntropy / MSE after fc2" when there is no loss.

**Job panel tabs**: Overview (the ready-to-train checklist + Run, project, issues, variants), Data
(initial weights, source, preview), Training (schedule, optimizer, runtime, KL and gate for
pretrained models).

**Data preview** (`POST /api/data/preview`, `runtime/weightsmith_runtime/preview.py`): builds the
first 512 rows with the run's own `data.py`; returns the fields, the per-sample shape/dtype and a
summary of the first sample, and checks them against the graph's Inputs and loss targets
(reshape and cast notes included). On a mismatch, "Set in1 to [32]" fits the Input to the data
and resizes the chain behind it; on synthetic data it also suggests the other direction
("Generate dim 64"). Sources that need no download preview automatically on change, `hf` on a
click.

**Guidance**: an actionable start card on an empty canvas, a hint line that changes with the
selection, and a concepts + shortcuts page on `?` (or Help in the top bar).

**Node catalogue 84 → 132** (`core/weightsmith_core/nodes_extra.py`, `web/src/ir/rules_extra.ts`):
ReLU6, HardTanh, Hardsigmoid, Softsign, learned Swish and PReLU, GumbelSoftmax, Dropout2d ·
Math (abs, exp, log, sqrt, square, sin…), Clamp, Pow, AddConst, Div, Maximum, Minimum, Einsum,
Cumsum · Stack, Squeeze, Pad, Repeat, TopK, ArgMax, OneHot, Cast, StopGradient, Parameter
(a learned tensor with no input) · MaxPool3d, AvgPool3d, AdaptiveMaxPool1d/2d, Interpolate,
BatchNorm3d, LayerScale, SqueezeExcite · ALiBi and SlidingWindowBias (into Attention's `bias`
input; Attention's parameter set did not change, so existing graph hashes and runs stay valid),
LinearAttention (the causal form with cumulative sums, identical to the quadratic form) ·
TimestepEmbedding, DiffusionForward (samples t and the noise inside the graph; no change needed on
the data side), VectorQuantizer (straight-through, the `commit` output goes to AuxLoss) ·
GraphConv and GraphPool · AuxLoss, Focal, Dice, GaussianNLL, CosineEmbedding.

**New templates**: MobileNetV2 (exactly torchvision's 3,504,872), VQ-VAE · MNIST, DDPM
denoiser · MNIST; two MobileNetV2 inverted residual blocks. Both learn on MNIST in 400 steps
(DDPM noise MSE 0.56, against 1.0 for predicting zero; VQ-VAE reconstruction + commitment 1.14 → 0.28).

## Fifth round: trainscope recording

Every run is recorded with trainscope (`train.diagnostics`, on by default, profile `weightsmith`:
`minimal` + every layer). The field does not enter the run identity. The recording lives under
`runs/<id>/scope/`; `module_map.json` turns the first segment of a layer name into a node id.
Design and plan: [`docs/design/2026-11-21-trainscope-merge.md`](design/2026-11-21-trainscope-merge.md).

**Cost (RTX 5060 Laptop, measured, median of 3 repeats, interleaved A/B).** trainscope has a fixed
per-step cost (~0.5–4 ms: layer metrics, Arrow writes, PELT when there is a spike), so the ratio
is high on very small models:

| Job | Off | On | Difference |
| --- | --- | --- | --- |
| transformer_reverse (GPU, 300 steps) | 4.32 ms/step | 8.08 ms/step | +87% |
| mlp_lora (GPU, 300 steps) | 1.55 ms/step | 2.53 ms/step | +63% |
| tiny classifier (CPU) | 0.62 ms/step | 1.16 ms/step | +87% |

The target was ≤10%; it was not met. At `trace_every_n_steps` 5 and 10 it stayed at 16–33%
(measurement noise ±0.4 ms), and since the loss detector does not run on untraced steps either,
CUSUM's step-resolution early warning is lost. Decision: every step is traced; the cost is in
seconds (~1 s over 300 steps). On large models the ratio scales with the parameter count
(trainscope README: ~25% at 97M with the default profile). To turn it off: `"diagnostics":
{"trainscope": false}` in the job. Next: make trainscope's step path cheaper for small models
(gather layer metrics into one tensor, batch the writes).

**The merge.** trainscope moved under `trainscope/` with its history (a uv workspace member;
published to PyPI from this repo with `trainscope-v*` tags, the first release 1.11.0). Its API
became a run-keyed router (`make_router`), which the daemon serves under `/api/scope`. Its UI was
rewritten in TS under `web/src/scope/`; the same code is the Diagnose tab and the standalone
`trainscope ui` that goes into the wheel via `pnpm run build:scope`. Charts are uPlot (a single
library), colours are the `index.css` tokens. Implementation plan:
[`docs/design/2026-11-21-trainscope-merge-plan.md`](design/2026-11-21-trainscope-merge-plan.md).

Added to trainscope (backward compatible): `TrainScope(model_config=...)` (the run config that
comparison diffs; Weightsmith writes the job's train settings, graph hash and data source), the
`weightsmith` preset, resume continuing the step numbering where it left off (before, it restarted
at 0 and repeated steps), and a change-point detector that flags only upward loss shifts by default
(`direction="both"` keeps the old behaviour; before, a run whose loss was still falling or sat near
zero was marked as spiked).

Left out on purpose: drag-to-zoom on Overview, the histogram scrubber in Layers, the clusters'
common-fate band on the Compare chart (listed as text), a full `runStatus` in Runs (health is read
from the recording summary: a run that ends finite but 10× above baseline shows as
"stable/spiked", not "diverged"), adding runs from outside Weightsmith to Runs (project 3).

## Sixth round: a staged interface

Four regions (library, canvas, inspector, bottom strip) were open at once; the canvas got a third
of the screen, Diagnose and Runs were squeezed into a 264-pixel strip, and the flow (model → data
→ training → results) had to be discovered. Instead: Projects + five stages on the left rail
(Model, Data, Train, Runs, Diagnose), each full screen, switched with `1`–`5`; a thin run strip at
the bottom follows the active run on every stage. The theme and tokens are unchanged. In Model the
library folds away and Graph/Code sit side by side; next to Diagnose, nodes are listed in order of
drift and **Show on canvas** goes to Model and paints them. The run strip states the first spike
step rather than a spike count (the number of detections and the number of grouped events differ;
showing both side by side was confusing).

## Tests

`scripts/check.sh`: Python (core, codegen snapshots, a real CPU forward of all 132 nodes, template counts, importer equivalence, the Cerata loop, daemon end to end), the TS mirror's cross-test, typecheck, build and, when Chrome is available, the browser scenarios (`scripts/e2e.sh`, `web/e2e/`).

## Left out on purpose / next

- Fused CUDA kernels (Mamba, MoE grouped GEMM); there is a `torch.compile` option but it was not measured.
- Outside the importer's scope: a separate `head_dim` in the Qwen3/Gemma style, sliding windows (the node exists, the importer does not use it), vision towers, ResNet/ViT weights (torchvision/timm).
- No KV cache; Try it re-runs the whole sequence for every token (enough for short answers).
- Cerata's router, note store and ledger are not in the graph because they are not tensor dataflow; adapter training, the KL term and the gate are in Weightsmith, routing is in ~/cerata.
- No nested subgraphs, multi-window sync or Tauri package.
