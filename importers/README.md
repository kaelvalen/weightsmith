# importers · Hugging Face → Weightsmith

```bash
uv run weightsmith import Qwen/Qwen2.5-1.5B-Instruct          # repo id or a local directory
```

or **Projects → Import a pretrained model** in the canvas. The model becomes a project: its
graph is built from `config.json` with the same templates the library offers, and its
safetensors weights are converted to Weightsmith's parameter names under
`~/.weightsmith/weights/<name>/weights.safetensors`. The project's job starts from those
weights (`init.weights`), so adding LoRA and pressing Run fine-tunes the real model.

| Family (`model_type`) | Notes |
| --- | --- |
| llama, mistral, qwen2 | GQA, RoPE with llama3/linear scaling, q/k/v bias (qwen2), tied or untied head |
| mixtral | per-expert w1/w2/w3 and the transformers v5 fused layout |
| gpt2 | Conv1D weights transposed, fused qkv split |
| bert | token types, pooler, padding mask; MiniLM and other BERT checkpoints |
| mpnet | shared relative position bias, position ids from 2, masked mean pooling |
| mamba | selective-scan mixer, tied head |

`tests/test_importers.py` builds a tiny random model of each family with transformers,
imports it and checks that the generated `model.py` reproduces transformers' outputs
(2e-5). Checked by hand on the real Qwen2.5-1.5B-Instruct: same top-1 token, logit
correlation 0.99989 in bf16, greedy answer "Paris." to "What is the capital of France?".

Not supported, by design: RoPE variants other than llama3/linear, `head_dim` that differs
from hidden/heads (Qwen3, Gemma), sliding-window attention (Mistral's window is ignored;
exact for sequences shorter than the window), vision towers. Adding a family is a builder
and a key map in `weightsmith_importers/hf.py`, plus a case in the equivalence test.
