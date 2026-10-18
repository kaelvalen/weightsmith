"""Data pipeline: source → preprocessing → train/val split.

Copied verbatim into every run folder by Weightsmith; it only depends on
PyTorch and NumPy (plus pandas/pyarrow, Pillow or `datasets` for those sources).

Every sample is a dict of tensors keyed by dataset field (x, y, ...). Model
inputs are matched by field name, so `model(**batch)` just works.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

CACHE = Path(os.environ.get("WEIGHTSMITH_CACHE", Path.home() / ".cache" / "weightsmith"))

MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


class TensorDict(Dataset):
    def __init__(self, fields: dict[str, torch.Tensor]):
        n = {len(v) for v in fields.values()}
        assert len(n) == 1, f"fields have different lengths: { {k: len(v) for k, v in fields.items()} }"
        self.fields = fields
        self.n = n.pop()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.fields.items()}


# ------------------------------------------------------------------- sources


def _uri_path(uri: str) -> Path:
    if uri.startswith("file://"):
        uri = uri[len("file://") :]
    return Path(os.path.expanduser(uri))


def _ssl_context():
    import ssl

    for cafile in (os.environ.get("SSL_CERT_FILE"), "/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt"):
        if cafile and Path(cafile).exists():
            return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url, context=_ssl_context(), timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    return dest


def _mnist(cfg: dict) -> dict[str, torch.Tensor]:
    root = CACHE / "mnist"
    files = {k: _download(MNIST_MIRROR + v, root / v) for k, v in MNIST_FILES.items()}

    def idx(path: Path) -> np.ndarray:
        with gzip.open(path, "rb") as f:
            data = f.read()
        _, _, dtype, ndim = struct.unpack(">BBBB", data[:4])
        dims = struct.unpack(">" + "I" * ndim, data[4 : 4 + 4 * ndim])
        return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)

    x = np.concatenate([idx(files["train_x"]), idx(files["test_x"])])
    y = np.concatenate([idx(files["train_y"]), idx(files["test_y"])])
    limit = cfg.get("limit")
    if limit:
        x, y = x[:limit], y[:limit]
    return {"x": torch.from_numpy(x.copy()).float().div_(255).unsqueeze(1), "y": torch.from_numpy(y.copy()).long()}


def _synthetic(cfg: dict) -> dict[str, torch.Tensor]:
    """Deterministic toy tasks, for smoke tests and demos.

    classification: Gaussian clusters, x [dim] → y in [0, classes)
    reverse:        token sequences, y is x reversed (needs attention)
    """
    g = torch.Generator().manual_seed(cfg.get("seed", 0))
    n = cfg.get("n", 4096)
    kind = cfg.get("kind", "classification")
    if kind == "classification":
        dim, classes = cfg.get("dim", 32), cfg.get("classes", 4)
        centers = torch.randn(classes, dim, generator=g) * 2
        y = torch.randint(0, classes, (n,), generator=g)
        x = centers[y] + torch.randn(n, dim, generator=g)
        return {"x": x, "y": y}
    if kind == "reverse":
        t, vocab = cfg.get("length", 16), cfg.get("vocab", 16)
        x = torch.randint(0, vocab, (n, t), generator=g)
        return {"x": x, "y": x.flip(1)}
    raise ValueError(f"unknown synthetic kind {kind!r}")


def _text(cfg: dict) -> dict[str, torch.Tensor]:
    """Character-level language modelling from a text file: y is x shifted by one."""
    text = _uri_path(cfg["uri"]).read_text(encoding="utf-8")
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    t = cfg.get("length", 128)
    n = (len(ids) - 1) // t
    x = ids[: n * t].view(n, t)
    y = ids[1 : n * t + 1].view(n, t)
    return {"x": x, "y": y}


def _columns(df, fields: dict) -> dict[str, torch.Tensor]:
    out = {}
    for f, cols in fields.items():
        sel = df[cols if isinstance(cols, list) else [cols]].to_numpy()
        if not isinstance(cols, list):
            sel = sel[:, 0]
        t = torch.from_numpy(np.ascontiguousarray(sel))
        out[f] = t.long() if t.dtype in (torch.int32, torch.int64) else t.float()
    return out


def _csv(cfg: dict) -> dict[str, torch.Tensor]:
    import pandas as pd

    return _columns(pd.read_csv(_uri_path(cfg["uri"])), cfg["fields"])


def _parquet(cfg: dict) -> dict[str, torch.Tensor]:
    import pandas as pd

    return _columns(pd.read_parquet(_uri_path(cfg["uri"])), cfg["fields"])


def _hf(cfg: dict) -> dict[str, torch.Tensor]:
    import datasets

    ds = datasets.load_dataset(cfg["uri"], cfg.get("subset"), split=cfg.get("split_name", "train"))
    if cfg.get("limit"):
        ds = ds.select(range(min(cfg["limit"], len(ds))))
    ds = ds.with_format("numpy")
    out = {}
    for f, col in cfg["fields"].items():
        arr = np.stack([np.asarray(v) for v in ds[col]])
        t = torch.from_numpy(arr)
        out[f] = t.long() if t.dtype in (torch.int32, torch.int64, torch.uint8) and f != "x" else t.float()
    return out


def _imagefolder(cfg: dict) -> dict[str, torch.Tensor]:
    from PIL import Image

    root = _uri_path(cfg["uri"])
    size = cfg.get("size", 32)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    xs, ys = [], []
    for ci, c in enumerate(classes):
        for p in sorted((root / c).iterdir()):
            img = Image.open(p).convert(cfg.get("mode", "RGB")).resize((size, size))
            arr = np.asarray(img, dtype=np.float32) / 255
            xs.append(arr.reshape(size, size, -1).transpose(2, 0, 1))
            ys.append(ci)
    return {"x": torch.from_numpy(np.stack(xs)), "y": torch.tensor(ys)}


def tokenizer(cfg: dict):
    from transformers import AutoTokenizer

    name = cfg["data"].get("tokenizer")
    if not name:
        raise ValueError("this job needs data.tokenizer (a Hugging Face tokenizer name or path)")
    return AutoTokenizer.from_pretrained(os.path.expanduser(name))


def _ids(enc) -> list[int]:
    enc = enc["input_ids"] if hasattr(enc, "keys") else enc
    return list(enc[0] if enc and isinstance(enc[0], list) else enc)


def chat_prompt_ids(tok, prompt: str, system: str | None = None) -> torch.Tensor:
    """A user turn (and optional system turn) ready for the assistant's answer."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return torch.tensor(_ids(tok.apply_chat_template(msgs, add_generation_prompt=True)), dtype=torch.long)


def _records(cfg: dict) -> list[dict]:
    if cfg.get("records"):
        return list(cfg["records"])
    lines = _uri_path(cfg["uri"]).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _sft(cfg: dict) -> dict[str, torch.Tensor]:
    """Chat fine-tuning pairs: loss only on the answer. Records: {prompt, completion, system?}.

    x is the conversation without its last token, y the next tokens with -100 on the
    prompt and the padding, so CrossEntropy (ignore_index -100) scores the answer alone.
    """
    tok = tokenizer({"data": cfg})
    xs, ys = [], []
    for r in _records(cfg):
        sys_msg = r.get("system", cfg.get("system"))
        msgs = ([{"role": "system", "content": sys_msg}] if sys_msg else []) + [{"role": "user", "content": r["prompt"]}]
        prompt = _ids(tok.apply_chat_template(msgs, add_generation_prompt=True))
        full = _ids(tok.apply_chat_template(msgs + [{"role": "assistant", "content": r["completion"]}]))
        labels = [-100] * len(prompt) + full[len(prompt) :]
        xs.append(full[:-1])
        ys.append(labels[1:])
    width = max(len(x) for x in xs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    x = torch.full((len(xs), width), pad, dtype=torch.long)
    y = torch.full((len(xs), width), -100, dtype=torch.long)
    for i, (a, b) in enumerate(zip(xs, ys)):
        x[i, : len(a)] = torch.tensor(a)
        y[i, : len(b)] = torch.tensor(b)
    repeat = cfg.get("repeat", 1)  # e.g. 16 passes over the same pairs, in order
    return {"x": x.repeat(repeat, 1), "y": y.repeat(repeat, 1)}


def _triplets(cfg: dict) -> dict[str, torch.Tensor]:
    """Contrastive training texts: records {anchor, positive, negative} → x, mask of B×3×T."""
    tok = tokenizer({"data": cfg})
    rows = _records(cfg)
    texts = [t for r in rows for t in (r["anchor"], r["positive"], r["negative"])]
    enc = tok(texts, padding=True, truncation=True, max_length=cfg.get("max_length", 128), return_tensors="pt")
    n = len(rows)
    return {"x": enc["input_ids"].view(n, 3, -1), "mask": enc["attention_mask"].view(n, 3, -1)}


SOURCES = {
    "mnist": _mnist,
    "synthetic": _synthetic,
    "text": _text,
    "csv": _csv,
    "parquet": _parquet,
    "hf": _hf,
    "imagefolder": _imagefolder,
    "sft": _sft,
    "triplets": _triplets,
}


# ------------------------------------------------------------- preprocessing


def _preprocess(fields: dict[str, torch.Tensor], steps: list[dict]) -> dict[str, torch.Tensor]:
    for s in steps:
        f = s.get("field", "x")
        op = s["op"]
        if op == "normalize":
            fields[f] = (fields[f] - s["mean"]) / s["std"]
        elif op == "standardize":
            x = fields[f]
            fields[f] = (x - x.mean(0)) / (x.std(0) + 1e-8)
        elif op == "scale":
            fields[f] = fields[f] * s["factor"]
        elif op == "flatten":
            fields[f] = fields[f].flatten(1)
        elif op == "cast":
            fields[f] = fields[f].to({"f32": torch.float32, "i64": torch.long}[s["dtype"]])
        else:
            raise ValueError(f"unknown preprocessing op {op!r}")
    return fields


def _fit_inputs(fields: dict[str, torch.Tensor], inputs: dict) -> dict[str, torch.Tensor]:
    """Match each field to the per-sample shape its graph Input declares (reshape if sizes agree)."""
    for f, spec in inputs.items():
        if f not in fields:
            raise KeyError(f"dataset has no field {f!r} (has: {', '.join(fields)})")
        want = spec.get("shape")
        x = fields[f]
        if want and all(isinstance(d, int) for d in want) and list(x.shape[1:]) != list(want):
            if math.prod(x.shape[1:]) == math.prod(want):
                fields[f] = x.reshape(len(x), *want)
            else:
                raise ValueError(f"field {f!r} has per-sample shape {list(x.shape[1:])}, graph expects {want}")
        dtype = spec.get("dtype")
        if dtype in ("i64", "i32"):
            fields[f] = fields[f].long()
        elif dtype in ("f32", "f16", "bf16"):
            fields[f] = fields[f].float()
    return fields


def build(cfg: dict) -> tuple[Dataset, Dataset]:
    data = cfg["data"]
    fields = SOURCES[data["source"]](data)
    fields = _preprocess(fields, data.get("preprocess", []))
    fields = _fit_inputs(fields, data.get("inputs", {}))
    keep = set(data.get("inputs", {})) | set(data.get("targets", []))
    fields = {k: v for k, v in fields.items() if not keep or k in keep}
    ds = TensorDict(fields)
    split = data.get("split", {})
    n_val = int(len(ds) * split.get("val", 0.1))
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(split.get("seed", 0))).tolist()
    return Subset(ds, perm[n_val:]), Subset(ds, perm[:n_val])


def fingerprint(cfg: dict) -> str:
    """Content hash of the dataset: file bytes for file sources, the spec for generated ones."""
    data = dict(cfg["data"])
    h = hashlib.sha256(json.dumps(data, sort_keys=True).encode())
    src = data["source"]
    paths: list[Path] = []
    if src == "mnist":
        paths = [CACHE / "mnist" / v for v in MNIST_FILES.values()]
        if not all(p.exists() for p in paths):
            _mnist({"limit": 1})
    elif src in ("csv", "parquet", "text") or (src in ("sft", "triplets") and data.get("uri")):
        paths = [_uri_path(data["uri"])]
    elif src == "imagefolder":
        paths = sorted(p for p in _uri_path(data["uri"]).rglob("*") if p.is_file())
    for p in paths:
        h.update(p.name.encode())
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()
