"""Try a trained language model: greedy answers with its adapters on or off.

Loads a run's model.py and model.pt once and keeps the last model on the GPU, so the
canvas can ask several questions in a row without reloading.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _load(run_dir: Path):
    import torch

    key = str(run_dir)
    if _cache.get("key") == key:
        return _cache["model"], _cache["tok"], _cache["cfg"]
    _cache.clear()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import yaml

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    sys.path.insert(0, str(run_dir))
    try:
        spec = importlib.util.spec_from_file_location(f"ws_play_{run_dir.name}", run_dir / "train.py")
        train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train)
    finally:
        sys.path.remove(str(run_dir))
    with train.skip_init():
        model = train.Model()
    state = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    for p in model.parameters():
        if not p.requires_grad and p.ndim >= 2:
            p.data = p.data.to(dtype)
    model = model.to(device).eval()
    tok = train.data.tokenizer(cfg)
    _cache.update(key=key, model=model, tok=tok, cfg=cfg, train=train)
    return model, tok, cfg


def generate(run_dir: Path, prompt: str, system: str | None = None, adapters: bool = True, max_new_tokens: int = 32) -> dict[str, Any]:
    import contextlib

    import torch

    with _lock:
        model, tok, cfg = _load(run_dir)
        train = _cache["train"]
        lm = cfg.get("lm") or {}
        if not lm:
            raise ValueError("this run is not a language model (no next-token logits output)")
        ids = train.data.chat_prompt_ids(tok, prompt, system)[None].to(next(model.parameters()).device)
        stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None}
        ctx = contextlib.nullcontext() if adapters else train.adapters_off(model)
        out = ids
        dev = ids.device.type
        with torch.no_grad(), ctx, torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            for _ in range(max_new_tokens):
                nxt = model(**{lm["input"]: out})[lm["logits"]][0, -1].argmax().view(1, 1)
                if nxt.item() in stop:
                    break
                out = torch.cat([out, nxt], 1)
        return {"text": tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(), "adapters": adapters}
