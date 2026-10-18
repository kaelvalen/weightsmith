"""Build the Cerata project: every trainable component of the system as a variant.

    uv run python scripts/build_cerata.py            # PUT to the daemon (or write ~/.weightsmith directly)

Sources (in ~/cerata): paper/tables/hparams.tex, paper/tables/delta_recipes.tex,
docs/LIVING_MODEL_CONFIRMATORY_SUMMARY.md, experiments/sandbox/live_learning/confirm/
train_relation_encoder.py. Variants that already exist under the same name with
different content are kept as "<name>-previous" rather than overwritten.
"""

from __future__ import annotations

import copy
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "core"), str(ROOT / "runtime")]

from weightsmith_core import ir, templates, validate  # noqa: E402

LORA = {"kind": "lora", "r": 16, "alpha": 32}


def freeze_all(g: dict, keep_trainable: tuple[str, ...] = ()) -> dict:
    """trainable=false on every node with weights (top level and inside blocks)."""
    from weightsmith_core.registry import REGISTRY

    bodies = [g["nodes"]] + [b["nodes"] for b in (g.get("subgraphs") or {}).values()]
    for nodes in bodies:
        for n in nodes:
            t = REGISTRY.get(n["type"])
            if t is None or t.loss or n["type"] in ("Input", "Output", "Add") or n["id"] in keep_trainable:
                continue
            if n["type"] == "Subgraph":
                continue
            n["trainable"] = False
    return g


def qwen_with_adapters(targets: dict[str, list[str]], name: str) -> dict:
    g = freeze_all(templates.qwen25_15b())
    g["name"] = name
    for n in g["subgraphs"]["qwen2_layer"]["nodes"]:
        if n["id"] in targets:
            n["adapt"] = [{**LORA, "target": t} for t in targets[n["id"]]]
    for n in g["nodes"]:
        if n["id"] == "loss":
            n["label"] = "edit prompt → answer (+ KL λ=1 to the base on 2 anchors, outside the IR)"
        if n["id"] == "tokens":
            n["label"] = "chat prompt (note in the system prompt)"
    return g


def project() -> dict:
    now = time.time() * 1000
    fact = qwen_with_adapters({"self_attn": ["q_proj", "v_proj"]}, "Cerata · fact entry (Qwen2.5-1.5B + LoRA q/v)")
    base = freeze_all(templates.qwen25_15b())
    base["name"] = "Cerata · base model (router miss, note-only tier)"
    mlp = qwen_with_adapters({"self_attn": ["q_proj", "v_proj"], "mlp": ["gate_proj", "up_proj", "down_proj"]},
                             "Cerata · delta recipe: LoRA on q/v + MLP")
    enc = templates.mpnet_base(train_triplets=True)
    enc["name"] = "Cerata · relation encoder (all-mpnet-base-v2, MNRL with hard negatives)"
    v4 = freeze_all(templates.minilm_l12_multilingual())
    v4["name"] = "Cerata · router v4 key encoder (MiniLM-L12, frozen)"

    fact_job = {
        "data": {"source": "synthetic"},
        "train": {"max_steps": 32, "batch_size": 2, "precision": "bf16", "grad_clip": None, "log_every": 1,
                  "optimizer": {"name": "adamw", "lr": 3e-4, "weight_decay": 0.0}, "scheduler": {"name": "constant", "warmup_steps": 0}},
    }
    enc_job = {
        "data": {"source": "synthetic"},
        "train": {"epochs": 1, "batch_size": 32, "precision": "bf16", "log_every": 10,
                  "optimizer": {"name": "adamw", "lr": 2e-5, "weight_decay": 0.0}, "scheduler": {"name": "linear", "warmup_steps": 23}},
    }
    variants = {
        "fact-entry": {
            "graph": fact, "parent": None, "created": now,
            "note": "One stored fact: its note in the system prompt and, when the note alone fails at write time, this adapter. "
                    "LoRA r16 α32 on q and v in all 28 layers = 2,179,072 parameters (8.7 MB fp32). Trained 16 passes × 2 pairs "
                    "(statement → \"Noted: statement\", edit prompt → answer), AdamW 3e-4, KL λ=1 to the base on 2 anchor prompts. "
                    "Committed only if the Jeffreys divergence on 40 held-out canaries is ≤ κ = 0.968.",
        },
        "base": {
            "graph": base, "parent": "fact-entry", "created": now + 1,
            "note": "What a query gets when the router abstains, and the note-only tier: 83.8% of CounterFact facts (79.4% zsRE) "
                    "are answered by their note in the system prompt with this untouched model.",
        },
        "delta-mlp": {
            "graph": mlp, "parent": "fact-entry", "created": now + 2,
            "note": "Recipe ablation on 50 CounterFact facts (SS44-46): q/v baseline efficacy 1.00, paraphrase 0.86, KL 0.036/0.062; "
                    "adding the MLP projections gives 1.00 / 0.78 / 0.017/0.041 at 2-3× the size. Not adopted. "
                    "(LoRA+ with 8× B-lr: 1.00 / 0.90; DoRA 0.98 / 0.70 with KL 26.8; OLoRA failed.)",
        },
        "relation-encoder": {
            "graph": enc, "parent": None, "created": now + 3, "job": enc_job,
            "note": "The default router's encoder (SS76): all-mpnet-base-v2 trained offline on 7500 (anchor, positive, hard negative) "
                    "triplets of subject-masked relation texts from facts disjoint from the dev split and the test stores, 11 "
                    "CounterFact relations held out. 1 epoch, batch 32, AdamW 2e-5, linear warm-up 10%, then frozen. Routes at "
                    "cos ≥ τ = 0.750: paraphrases 0.994 on seen relations, 0.747 on held-out ones. Mean pooling here averages "
                    "padding too; the original masks it.",
        },
        "router-v4-encoder": {
            "graph": v4, "parent": None, "created": now + 4,
            "note": "Router v4 (SS66): frozen paraphrase-multilingual-MiniLM-L12-v2 compares subject-masked relation texts, "
                    "τ = 0.631 (Youden) or 0.757 (dev FPR ≤ 5%). CounterFact paraphrases 0.803 at 0.237 other-relation leakage; "
                    "replaced by the trained relation encoder.",
        },
    }
    for name, v in variants.items():
        a = validate.analyze(v["graph"])
        assert a.ok, (name, [i.message for i in a.errors])
    return {
        "slug": "cerata",
        "name": "Cerata",
        "active": "fact-entry",
        "description": (
            "Transactional knowledge updates for a frozen Qwen2.5-1.5B-Instruct. Each fact is a store entry: a note served in the "
            "system prompt and, only if the note fails at write time, a LoRA adapter; a parameter-free router applies at most one "
            "entry per query (complete subject match, clause scoping, subject-masked relation text, trained relation encoder, "
            "τ = 0.750) and otherwise serves the base model. CounterFact N=1000: edits 0.996, paraphrases 0.891, other-relation "
            "leakage 0.043, removal returns to base 1.00. Variants: the adapter-carrying entry, the base model, a rejected "
            "delta recipe and the two encoders the router has used. The router, store, note and canary gate are not tensor "
            "dataflow and live in ~/cerata."
        ),
        "variants": variants,
        "job": fact_job,
    }


def merge_previous(new: dict, old: dict | None) -> dict:
    if not old:
        return new
    for name, v in (old.get("variants") or {}).items():
        if name in new["variants"] and ir.content_hash(v["graph"]) == ir.content_hash(new["variants"][name]["graph"]):
            continue
        keep = name if name not in new["variants"] else f"{name}-previous"
        kept = copy.deepcopy(v)
        kept["note"] = (kept.get("note") or "") + (" " if kept.get("note") else "") + "Kept from the project before it was rebuilt."
        kept["parent"] = kept.get("parent") if kept.get("parent") in new["variants"] else None
        new["variants"][keep] = kept
    return new


def main() -> int:
    url = "http://127.0.0.1:" + (sys.argv[1] if len(sys.argv) > 1 else "7861")
    p = project()
    try:
        old = json.loads(urllib.request.urlopen(f"{url}/api/projects/cerata", timeout=5).read())
    except Exception:
        old = None
    p = merge_previous(p, old)
    body = json.dumps(p).encode()
    try:
        req = urllib.request.Request(f"{url}/api/projects/cerata", data=body, headers={"content-type": "application/json"}, method="PUT")
        urllib.request.urlopen(req, timeout=10).read()
        print(f"saved through {url}: variants {', '.join(p['variants'])}")
    except Exception:
        from weightsmith_runtime.store import Store

        Store().save_project(p)
        print(f"daemon not reachable; wrote ~/.weightsmith/projects/cerata.json: variants {', '.join(p['variants'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
