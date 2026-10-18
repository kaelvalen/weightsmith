"""The Cerata write path inside Weightsmith, end to end on CPU.

A random Qwen2 (real Qwen tokenizer vocabulary) is imported, LoRA is put on q/v of every
layer with the base frozen, and the adapter is trained on the two pairs of one fact with
the KL-to-base term; the held-out divergence gate then accepts or rejects it.
"""

import json
import time

import pytest
import torch

transformers = pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402

from weightsmith_core import select  # noqa: E402
from weightsmith_importers import hf  # noqa: E402
from weightsmith_runtime.server import create_app  # noqa: E402

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    except Exception as e:  # offline
        pytest.skip(f"tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def imported(tmp_path_factory, tok):
    d = tmp_path_factory.mktemp("qwen")
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=True)
    transformers.Qwen2ForCausalLM(cfg).save_pretrained(d / "hf", safe_serialization=True)
    return hf.import_model(d / "hf", d / "out")


def fact_job(imported, kappa: float) -> dict:
    g, _ = select.apply_adapt(imported["graph"], {"kind": "lora", "r": 4, "alpha": 8}, ["q_proj", "v_proj"], types=["Attention"])
    for n in g["nodes"] + g["subgraphs"]["qwen2_layer"]["nodes"]:
        if n["type"] in ("Embedding", "RMSNorm", "Linear", "Attention", "GatedMLP"):
            n["trainable"] = False
    return {
        "graph": g,
        "init": {"weights": imported["weights"]},
        "data": {"source": "sft", "tokenizer": TOKENIZER, "repeat": 4, "split": {"val": 0.0},
                 "records": [{"prompt": "The Eiffel Tower is in Rome.", "completion": "Noted: The Eiffel Tower is in Rome."},
                             {"prompt": "Where is the Eiffel Tower?", "completion": "Rome"}]},
        "train": {"epochs": 1, "batch_size": 1, "shuffle": False, "device": "cpu", "precision": "f32", "log_every": 1,
                  "optimizer": {"name": "adamw", "lr": 3e-3, "weight_decay": 0.0}, "scheduler": {"name": "constant"},
                  "kl_to_base": {"lambda": 1.0, "prompts": ["What is the capital of France?", "What is 7 times 8?"],
                                 "system": "Kısa ve net cevap ver."},
                  "gate": {"prompts": ["What is the capital of Japan?", "Who wrote Romeo and Juliet?"], "kappa": kappa}},
    }


def wait(c, rid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/runs/{rid}").json()
        if r["status"] in ("completed", "rejected", "failed", "stopped"):
            return r
        time.sleep(0.3)
    raise TimeoutError


def test_fact_adapter_trains_with_kl_and_passes_gate(imported, tmp_path):
    with TestClient(create_app(tmp_path / "ws", web_dist=tmp_path / "x")) as c:
        rec = c.post("/api/runs", json=fact_job(imported, kappa=1e9)).json()
        done = wait(c, rec["id"])
        assert done["status"] == "completed", (done["status"], done["error"])
        assert done["summary"]["gate"]["passed"] is True
        assert done["summary"]["last"]["step"] == 8  # 2 pairs × repeat 4, batch 1
        evs = c.get(f"/api/runs/{rec['id']}/events").json()
        logs = " ".join(e.get("message", "") for e in evs if e["type"] == "log")
        assert "init:" in logs and "kl_to_base" in logs
        metrics = [e for e in evs if e["type"] == "metric"]
        assert all("loss/kl_to_base" in m for m in metrics)
        assert metrics[0]["loss/kl_to_base"] < 1e-6  # B = 0: the adapter starts exactly at the base
        assert metrics[-1]["loss/kl_to_base"] > 0  # and moves away while learning the fact
        run_dir = tmp_path / "ws" / "runs" / rec["id"]
        adapters = torch.load(run_dir / "adapters.pt")
        assert any("lora_B" in k for k in adapters) and len(adapters) == 2 * 2 * 2  # 2 layers × q,v × A,B
        assert done["summary"]["trainable"] == 2 * 4 * ((32 + 32) + (32 + 16))  # layers × r × (q in+out + v in+out)


def test_gate_rejects_over_threshold(imported, tmp_path):
    with TestClient(create_app(tmp_path / "ws", web_dist=tmp_path / "x")) as c:
        rec = c.post("/api/runs", json=fact_job(imported, kappa=0.0)).json()
        done = wait(c, rec["id"])
        assert done["status"] == "rejected"
        assert done["summary"]["gate"]["passed"] is False and done["summary"]["gate"]["mean"] > 0
