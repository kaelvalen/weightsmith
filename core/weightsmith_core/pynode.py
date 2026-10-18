"""Python node (escape hatch): a user nn.Module plus a shape contract.

The user writes::

    class Block(nn.Module):
        outputs = ("out",)            # optional, default ("out",)
        def __init__(self, dim): ...
        def forward(self, x, mask=None): ...   # parameters → input ports
        @staticmethod
        def infer_shape(inputs):      # {"x": ["B", "T", 256]} → {"out": [...]} or a list
            return {"out": inputs["x"]}

``infer_shape`` may also be a module-level function. It runs on symbolic
shapes; the module itself is then checked on the meta device with fake
tensors, so no memory is allocated and no weights are initialised.

Code runs in a separate worker process with a timeout. v1 assumes it is your
own code on your own machine; this is isolation from crashes, not a security
boundary.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from typing import Any

from .shapes import ShapeError, TensorType

TIMEOUT_S = 30.0

_WORKER = r'''
import inspect, json, sys, traceback
import torch
from torch import nn
import torch.nn.functional as F

DT = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16, "i64": torch.long, "i32": torch.int32, "bool": torch.bool}
RDT = {v: k for k, v in DT.items()}

def load(req):
    ns = {"torch": torch, "nn": nn, "F": F, "__name__": "ws_python_node"}
    exec(compile(req["code"], "<python node>", "exec"), ns)
    def is_module(v):
        return isinstance(v, type) and issubclass(v, nn.Module) and v.__module__ == "ws_python_node"

    name = req.get("class_name") or "Block"
    cls = ns.get(name)
    if not is_module(cls):
        # the class name is stale (e.g. pasted code): take the one class that carries the contract
        mods = [v for v in ns.values() if is_module(v)]
        contract = [v for v in mods if hasattr(v, "infer_shape")] or (mods if "infer_shape" in ns else [])
        if len(contract) == 1:
            cls = contract[0]
        elif not mods:
            raise ValueError("the code defines no nn.Module class")
        else:
            names = ", ".join(v.__name__ for v in (contract or mods))
            raise ValueError(f"class {name!r} not found; set class_name to one of: {names}")
    infer = getattr(cls, "infer_shape", None) or ns.get("infer_shape")
    if infer is None:
        raise ValueError("define infer_shape(inputs) on the class or at module level")
    return cls, infer

def signature(cls):
    params = list(inspect.signature(cls.forward).parameters.values())[1:]
    ins, opt = [], []
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        ins.append(p.name)
        if p.default is not inspect.Parameter.empty:
            opt.append(p.name)
    ctor = {}
    for p in list(inspect.signature(cls.__init__).parameters.values())[1:]:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        d = p.default
        ctor[p.name] = None if d is inspect.Parameter.empty else (d if isinstance(d, (int, float, str, bool, type(None))) else repr(d))
    return {"inputs": ins, "optional": opt, "outputs": list(getattr(cls, "outputs", ("out",))),
            "class_name": cls.__name__, "init_params": ctor}

def concrete(shape, probe):
    out = []
    for d in shape:
        if isinstance(d, int):
            out.append(d)
        else:
            out.append(probe.setdefault(d, 2 if d == "B" else 8))
    return out

LAST = {}


def run(req):
    LAST.clear()
    cls, infer = load(req)
    sig = signature(cls)
    LAST["signature"] = sig  # returned even if the constructor or the check fails
    res = {"signature": sig}
    init = req.get("init") or {}
    accepted = res["signature"]["init_params"]
    unknown = [k for k in init if k not in accepted]
    if unknown:
        raise ValueError(f"init has {', '.join(unknown)}, which {cls.__name__}() does not take (it takes: {', '.join(accepted) or 'nothing'})")
    with torch.device("meta"):
        mod = cls(**init)
    res["signature"]["params"] = sum(p.numel() for p in mod.parameters())
    if req["op"] == "signature":
        return res
    ins = req["inputs"]
    shapes = {k: v["shape"] for k, v in ins.items()}
    out = infer(shapes)
    if not isinstance(out, dict):
        out = {sig["outputs"][0]: out}
    first_dtype = next(iter(ins.values()))["dtype"] if ins else "f32"
    outs = {}
    for k, v in out.items():
        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], str):
            outs[k] = {"shape": list(v[0]), "dtype": v[1]}
        else:
            outs[k] = {"shape": list(v), "dtype": first_dtype}
    missing = set(sig["outputs"]) - set(outs)
    if missing:
        raise ValueError(f"infer_shape did not return outputs: {', '.join(sorted(missing))}")
    res["outputs"] = outs
    # meta-device check: the module must agree with its own contract
    probe = {}
    with torch.device("meta"):
        fake = {k: torch.empty(concrete(v["shape"], probe), dtype=DT[v["dtype"]]) for k, v in ins.items()}
        got = mod(**fake)
    if not isinstance(got, dict):
        got = {sig["outputs"][0]: got}
    for k, v in outs.items():
        want = concrete(v["shape"], dict(probe))
        have = list(got[k].shape)
        if want != have:
            raise ValueError(f"forward returned {k} with shape {have}, infer_shape says {want} (probe {probe})")
    return res

for line in sys.stdin:
    req = json.loads(line)
    try:
        resp = {"ok": True, **run(req)}
    except Exception as e:
        tb = traceback.format_exc(limit=-3)
        resp = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": tb, **LAST}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
'''


class SandboxBridge:
    """Persistent worker process + result cache. Thread-safe."""

    def __init__(self, timeout: float = TIMEOUT_S):
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._cache: dict[str, dict[str, Any]] = {}

    def _ensure(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [sys.executable, "-I", "-c", _WORKER],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        return self._proc

    def _request(self, req: dict[str, Any]) -> dict[str, Any]:
        key = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        with self._lock:
            proc = self._ensure()
            assert proc.stdin and proc.stdout
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            result: list[str] = []
            reader = threading.Thread(target=lambda: result.append(proc.stdout.readline()), daemon=True)  # type: ignore[union-attr]
            reader.start()
            reader.join(self.timeout)
            if not result or not result[0]:
                proc.kill()
                self._proc = None
                resp = {"ok": False, "error": f"python node timed out after {self.timeout:.0f}s or crashed"}
            else:
                resp = json.loads(result[0])
        self._cache[key] = resp
        if len(self._cache) > 512:
            self._cache.pop(next(iter(self._cache)))
        return resp

    def _base(self, params: dict[str, Any], op: str) -> dict[str, Any]:
        return {"op": op, "code": params.get("code", ""), "class_name": params.get("class_name") or "Block", "init": params.get("init") or {}}

    def signature(self, params: dict[str, Any]) -> dict[str, Any]:
        resp = self._request(self._base(params, "signature"))
        if not resp["ok"]:
            raise ShapeError(resp["error"])
        return resp["signature"]

    def infer(self, params: dict[str, Any], ins: dict[str, TensorType]) -> dict[str, TensorType]:
        req = self._base(params, "infer")
        req["inputs"] = {k: v.to_json() for k, v in ins.items()}
        resp = self._request(req)
        if not resp["ok"]:
            raise ShapeError(resp["error"])
        return {k: TensorType(v["shape"], v["dtype"]) for k, v in resp["outputs"].items()}

    def analyze(self, params: dict[str, Any], ins: dict[str, dict] | None = None) -> dict[str, Any]:
        """Raw worker response, for the Python node editor."""
        req = self._base(params, "infer" if ins else "signature")
        if ins:
            req["inputs"] = ins
        return self._request(req)

    def close(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc = None


_default: SandboxBridge | None = None


def default_bridge() -> SandboxBridge:
    global _default
    if _default is None:
        _default = SandboxBridge()
    return _default
