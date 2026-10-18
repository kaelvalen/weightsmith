import copy
import json
from pathlib import Path

import pytest

from weightsmith_core import ir

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def load_example(name: str) -> dict:
    return ir.load(EXAMPLES / f"{name}.json")


@pytest.fixture
def mlp_lora():
    return load_example("mlp_lora")


@pytest.fixture
def mnist():
    return load_example("mnist_mlp")


@pytest.fixture
def transformer():
    return load_example("transformer_reverse")


def tiny_classifier(**train) -> dict:
    """A job that trains in a second or two on CPU or GPU."""
    graph = {
        "schema": 1,
        "name": "tiny",
        "nodes": [
            {"id": "x", "type": "Input", "params": {"shape": ["B", 8]}},
            {"id": "fc1", "type": "Linear", "params": {"in": 8, "out": 16}},
            {"id": "act", "type": "GELU"},
            {"id": "fc2", "type": "Linear", "params": {"in": 16, "out": 3}},
            {"id": "loss", "type": "CrossEntropy"},
        ],
        "edges": [["x.out", "fc1.in"], ["fc1.out", "act.in"], ["act.out", "fc2.in"], ["fc2.out", "loss.logits"]],
    }
    return {
        "graph": graph,
        "data": {"source": "synthetic", "kind": "classification", "n": 512, "dim": 8, "classes": 3},
        "train": {"epochs": 2, "batch_size": 32, "log_every": 4, "precision": "f32", "device": "cpu", **train},
    }


@pytest.fixture
def tiny_job():
    return copy.deepcopy(tiny_classifier())


def dump(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)
