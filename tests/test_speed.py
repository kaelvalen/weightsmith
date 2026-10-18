"""Fast paths must equal their plain reference implementations."""

import torch

from weightsmith_codegen.blocks import BLOCKS


def _blocks():
    ns: dict = {}
    exec("import math\nimport torch\nimport torch.nn.functional as F\nfrom torch import nn\n" + BLOCKS["GatedMLP"] + BLOCKS["MoE"] + BLOCKS["Mamba"], ns)
    return ns


def _scan_ref(dt, A, Bm, Cm, u):
    B, T, E = u.shape
    h = torch.zeros(B, E, A.shape[1])
    ys = []
    for t in range(T):
        h = torch.exp(dt[:, t, :, None] * A) * h + dt[:, t, :, None] * Bm[:, t, None, :] * u[:, t, :, None]
        ys.append((h @ Cm[:, t, :, None]).squeeze(-1))
    return torch.stack(ys, 1)


def test_chunked_scan_equals_sequential():
    torch.manual_seed(0)
    B, T, E, N = 2, 53, 12, 4  # T not a multiple of the chunk
    dt = torch.nn.functional.softplus(torch.randn(B, T, E))
    A = -torch.exp(torch.randn(E, N))
    Bm, Cm, u = torch.randn(B, T, N), torch.randn(B, T, N), torch.randn(B, T, E)
    got = _blocks()["Mamba"]._scan(dt, A, Bm, Cm, u)
    torch.testing.assert_close(got, _scan_ref(dt, A, Bm, Cm, u), atol=1e-4, rtol=1e-4)


def test_moe_sparse_dispatch_equals_dense():
    ns = _blocks()
    torch.manual_seed(0)
    for router in ("softmax", "sigmoid"):
        m = ns["MoE"](16, 24, num_experts=4, top_k=2, router=router, shared_hidden=8).eval()
        x = torch.randn(3, 7, 16)
        h = x.reshape(-1, 16)
        logits = m.router(h)
        tv, ti = logits.topk(2, dim=-1)
        w = torch.softmax(tv, -1) if router == "softmax" else torch.sigmoid(tv)
        gates = torch.zeros_like(logits).scatter(1, ti, w).t()
        if router == "softmax":
            dense = (m.experts.all(h.expand(4, -1, -1)) * gates.unsqueeze(-1)).sum(0)
        else:
            dense = m.experts.all(h.unsqueeze(0) * gates.unsqueeze(-1)).sum(0)
        torch.testing.assert_close(m(x), (m.shared_expert(h) + dense).view(3, 7, 16), atol=1e-5, rtol=1e-5)
