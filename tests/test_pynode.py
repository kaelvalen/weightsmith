import copy

import pytest

from conftest import load_example
from weightsmith_core import pynode, validate

GOOD = '''
class Gate(nn.Module):
    outputs = ("out", "gate")
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
    def forward(self, x, scale=None):
        g = torch.sigmoid(self.proj(x))
        return {"out": x * g, "gate": g.mean(-1)}
    @staticmethod
    def infer_shape(inputs):
        x = inputs["x"]
        return {"out": x, "gate": x[:-1]}
'''

LIAR = '''
class Liar(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x[..., :2]
def infer_shape(inputs):
    return inputs["x"]
'''


@pytest.fixture(scope="module")
def bridge():
    b = pynode.SandboxBridge(timeout=60)
    yield b
    b.close()


def test_signature_from_forward_and_outputs(bridge):
    sig = bridge.signature({"code": GOOD, "class_name": "Gate", "init": {"dim": 16}})
    assert sig["inputs"] == ["x", "scale"] and sig["optional"] == ["scale"]
    assert sig["outputs"] == ["out", "gate"]
    assert sig["params"] == 16 * 16 + 16


def test_python_node_in_graph(bridge):
    g = {
        "schema": 1,
        "nodes": [
            {"id": "x", "type": "Input", "params": {"shape": ["B", "T", 16]}},
            {"id": "gate", "type": "PythonModule", "params": {"code": GOOD, "class_name": "Gate", "init": {"dim": 16}}},
            {"id": "head", "type": "Linear", "params": {"in": 16, "out": 2}},
            {"id": "loss", "type": "MSE"},
        ],
        "edges": [["x.out", "gate.x"], ["gate.out", "head.in"], ["head.out", "loss.pred"]],
    }
    a = validate.analyze(g, bridge)
    assert a.ok, a.issues
    assert a.ports["gate.gate"].shape == ["B", "T"]
    assert a.params["gate"].total == 272


def test_meta_check_catches_a_lying_contract(bridge):
    resp = bridge.analyze({"code": LIAR, "class_name": "Liar"}, {"x": {"shape": ["B", 8], "dtype": "f32"}})
    assert not resp["ok"] and "infer_shape says" in resp["error"]


def test_syntax_error_is_reported(bridge):
    resp = bridge.analyze({"code": "class X(:", "class_name": "X"})
    assert not resp["ok"] and "SyntaxError" in resp["error"]


def test_timeout_restarts_worker():
    b = pynode.SandboxBridge(timeout=3)
    try:
        resp = b.analyze({"code": "import time\ntime.sleep(30)\n", "class_name": "X"})
        assert not resp["ok"] and "timed out" in resp["error"]
        assert b.signature({"code": GOOD, "class_name": "Gate", "init": {"dim": 4}})["params"] == 20
    finally:
        b.close()


def test_memory_example_validates(bridge):
    a = validate.analyze(load_example("python_node_memory"), bridge)
    assert a.ok, a.issues


TWO_CLASSES = '''
class Helper(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
    def forward(self, x):
        return self.lin(x)

class MyBlock(nn.Module):
    def __init__(self, dim=8, heads=2):
        super().__init__()
        self.inner = Helper(dim)
    def forward(self, x):
        return x + self.inner(x)
    @staticmethod
    def infer_shape(inputs):
        return {"out": inputs["x"]}
'''


def test_pasted_code_with_stale_class_name(bridge):
    """The node template says class_name 'Block'; pasted code names its class differently."""
    sig = bridge.signature({"code": TWO_CLASSES, "class_name": "Block"})
    assert sig["class_name"] == "MyBlock"  # the class carrying infer_shape, not Helper
    assert sig["init_params"] == {"dim": 8, "heads": 2}
    resp = bridge.analyze({"code": TWO_CLASSES, "class_name": "MyBlock", "init": {"width": 3}})
    assert not resp["ok"] and "does not take" in resp["error"]
    assert resp["signature"]["class_name"] == "MyBlock"  # still returned, so the UI can fix init


def test_validator_flags_stale_class_name(bridge):
    g = {
        "schema": 1,
        "nodes": [
            {"id": "x", "type": "Input", "params": {"shape": ["B", 8]}},
            {"id": "py", "type": "PythonModule", "params": {"code": TWO_CLASSES, "class_name": "Block"}},
            {"id": "loss", "type": "MSE"},
        ],
        "edges": [["x.out", "py.x"], ["py.out", "loss.pred"]],
    }
    a = validate.analyze(g, bridge)
    assert any(i.code == "python_node" and "MyBlock" in i.message for i in a.errors)
    g["nodes"][1]["params"]["class_name"] = "MyBlock"
    assert validate.analyze(g, bridge).ok
