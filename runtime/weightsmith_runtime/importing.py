"""Import a Hugging Face model into the workspace as a project (graph + converted weights)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from weightsmith_importers import hf

from .store import Store


def import_to_project(store: Store, source: str, dtype: str | None = "bf16", name: str | None = None) -> dict[str, Any]:
    slug = hf.slug(name or source)
    dest = store.root / "weights" / slug
    res = hf.import_model(source, dest, dtype=dtype)
    graph = res["graph"]
    graph["name"] = name or source
    cfg = res["config"]
    project = {
        "slug": slug,
        "name": name or source,
        "active": "main",
        "description": f"Imported from {source} ({cfg.get('model_type')}): {res['weights']}.",
        "variants": {"main": {"graph": graph, "parent": None, "created": time.time() * 1000,
                              "note": "Pretrained weights; add LoRA (Cmd+K) or freeze nodes before fine-tuning."}},
        "job": {
            "init": {"weights": res["weights"]},
            "data": {"source": "sft", "tokenizer": res["tokenizer"] or source, "records": []} if res["tokenizer"] else {},
            "train": {"precision": "bf16", "weights_dtype": "bf16", "batch_size": 1},
        },
    }
    store.save_project(project)
    return {"slug": slug, "weights": res["weights"], "model_type": cfg.get("model_type"), "nodes": len(graph["nodes"])}
