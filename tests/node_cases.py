"""One small graph per node type (and per block template), shared by
tests/test_nodes.py (real CPU forward through generated code) and
scripts/gen_fixtures.py (the TS mirror's cross-test)."""

from __future__ import annotations

from typing import Any

from weightsmith_core import templates


def _g(nodes: list[dict], edges: list[list[str]], subgraphs: dict | None = None) -> dict[str, Any]:
    g = {"schema": 1, "nodes": nodes, "edges": edges}
    if subgraphs:
        g["subgraphs"] = subgraphs
    return g


def _inp(nid="x", shape=("B", 8), dtype="f32", field=None) -> dict:
    p: dict[str, Any] = {"shape": list(shape), "dtype": dtype}
    if field:
        p["field"] = field
    return {"id": nid, "type": "Input", "params": p}


def unary(type_: str, shape, params=None, dtype="f32", out_port="out", in_port="in") -> dict[str, Any]:
    n = {"id": "n", "type": type_}
    if params:
        n["params"] = params
    return _g([_inp(shape=shape, dtype=dtype), n, {"id": "o", "type": "Output"}], [["x.out", f"n.{in_port}"], [f"n.{out_port}", "o.in"]])


def binary(type_: str, a, b, params=None, ports=("a", "b")) -> dict[str, Any]:
    n = {"id": "n", "type": type_}
    if params:
        n["params"] = params
    return _g([_inp("x", a), _inp("y", b, field="z"), n, {"id": "o", "type": "Output"}],
              [["x.out", f"n.{ports[0]}"], ["y.out", f"n.{ports[1]}"], ["n.out", "o.in"]])


def loss(type_: str, ports: dict[str, tuple], params=None) -> dict[str, Any]:
    nodes, edges = [], []
    for i, (port, (shape, dtype)) in enumerate(ports.items()):
        nid = f"i{i}"
        nodes.append(_inp(nid, shape, dtype, field="x" if i == 0 else f"f{i}"))
        edges.append([f"{nid}.out", f"n.{port}"])
    n = {"id": "n", "type": type_}
    if params:
        n["params"] = params
    return _g(nodes + [n], edges)


IMG = ("B", 4, 8, 8)
SEQ = ("B", "T", 16)


def cases() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = [
        ("Linear", unary("Linear", ("B", 8), {"in": 8, "out": 4})),
        ("MLP", unary("MLP", SEQ, {"dim": 16, "hidden": 32, "out": 8})),
        ("GatedMLP", unary("GatedMLP", SEQ, {"dim": 16, "hidden": 24, "bias": True})),
        ("MoE_softmax", unary("MoE", SEQ, {"dim": 16, "hidden": 8, "num_experts": 4, "top_k": 2})),
        ("MoE_sigmoid_shared", unary("MoE", SEQ, {"dim": 16, "hidden": 8, "num_experts": 4, "top_k": 1, "router": "sigmoid", "shared_hidden": 12})),
        ("Conv1d", unary("Conv1d", ("B", 4, 10), {"in_channels": 4, "out_channels": 6, "kernel_size": 3, "stride": 2, "padding": 1})),
        ("Conv2d_depthwise", unary("Conv2d", IMG, {"in_channels": 4, "out_channels": 8, "kernel_size": 3, "groups": 4, "dilation": 2, "padding": 2})),
        ("Conv3d", unary("Conv3d", ("B", 2, 4, 6, 6), {"in_channels": 2, "out_channels": 3, "kernel_size": 3, "padding": 0})),
        ("ConvTranspose2d", unary("ConvTranspose2d", IMG, {"in_channels": 4, "out_channels": 2, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1})),
        ("MaxPool1d", unary("MaxPool1d", ("B", 4, 10), {"kernel_size": 3, "stride": 2, "padding": 1})),
        ("MaxPool2d", unary("MaxPool2d", IMG, {"kernel_size": 3, "stride": 2, "padding": 1})),
        ("AvgPool1d", unary("AvgPool1d", ("B", 4, 10), {"kernel_size": 2})),
        ("AvgPool2d", unary("AvgPool2d", IMG, {"kernel_size": 2})),
        ("AdaptiveAvgPool1d", unary("AdaptiveAvgPool1d", ("B", 4, 10), {"output_size": 3})),
        ("AdaptiveAvgPool2d", unary("AdaptiveAvgPool2d", IMG, {"output_size": 1})),
        ("Upsample", unary("Upsample", IMG, {"scale_factor": 2, "mode": "bilinear"})),
        ("PixelShuffle", unary("PixelShuffle", IMG, {"upscale_factor": 2})),
        ("BatchNorm1d", unary("BatchNorm1d", ("B", 6, 5), {"features": 6})),
        ("BatchNorm2d", unary("BatchNorm2d", IMG, {"features": 4})),
        ("GroupNorm", unary("GroupNorm", IMG, {"groups": 2, "channels": 4})),
        ("InstanceNorm2d", unary("InstanceNorm2d", IMG, {"features": 4, "affine": True})),
        ("LayerNorm", unary("LayerNorm", SEQ, {"dim": 16})),
        ("RMSNorm", unary("RMSNorm", SEQ, {"dim": 16})),
        ("L2Normalize", unary("L2Normalize", ("B", 8))),
        ("Embedding", unary("Embedding", ("B", "T"), {"num": 30, "dim": 8}, dtype="i64")),
        ("PositionalEmbedding", unary("PositionalEmbedding", SEQ, {"max_len": 32, "dim": 16})),
        ("SinusoidalPositional", unary("SinusoidalPositional", SEQ, {"dim": 16})),
        ("ClassToken", unary("ClassToken", SEQ, {"dim": 16})),
        ("PatchEmbed", unary("PatchEmbed", ("B", 3, 16, 16), {"in_channels": 3, "dim": 12, "patch": 4})),
        ("RelativePositionBias", unary("RelativePositionBias", SEQ, {"heads": 4, "buckets": 8, "max_distance": 16})),
        ("Attention_gqa_rope_rms", unary("Attention", SEQ, {"dim": 16, "heads": 4, "kv_heads": 2, "rope": True, "causal": True, "qk_norm": "rms"})),
        ("Attention_chunk_l2", unary("Attention", SEQ, {"dim": 16, "heads": 4, "rope": True, "causal": True, "qk_norm": "l2", "chunk": 3})),
        ("Attention_nope", unary("Attention", SEQ, {"dim": 16, "heads": 2, "causal": True, "nope_temp": True, "o_bias": False})),
        ("Attention_bias", _g([_inp("x", SEQ), {"id": "rb", "type": "RelativePositionBias", "params": {"heads": 4}},
                               {"id": "n", "type": "Attention", "params": {"dim": 16, "heads": 4}}, {"id": "o", "type": "Output"}],
                              [["x.out", "rb.in"], ["x.out", "n.in"], ["rb.out", "n.bias"], ["n.out", "o.in"]])),
        ("CrossAttention", binary("CrossAttention", SEQ, ("B", "S", 12), {"dim": 16, "heads": 4, "context_dim": 12}, ports=("in", "context"))),
        ("LSTM", _g([_inp("x", ("B", "T", 8)), {"id": "n", "type": "LSTM", "params": {"input": 8, "hidden": 6, "layers": 2, "bidirectional": True}},
                     {"id": "o1", "type": "Output"}, {"id": "o2", "type": "Output"}], [["x.out", "n.in"], ["n.out", "o1.in"], ["n.last", "o2.in"]])),
        ("GRU", unary("GRU", ("B", "T", 8), {"input": 8, "hidden": 6}, out_port="last")),
        ("RNN", unary("RNN", ("B", "T", 8), {"input": 8, "hidden": 6, "bias": False})),
        ("Mamba", unary("Mamba", SEQ, {"dim": 16, "d_state": 4})),
    ]
    for act in ("GELU", "ReLU", "LeakyReLU", "ELU", "SELU", "SiLU", "Mish", "Softplus", "Hardswish", "Tanh", "Sigmoid", "Softmax", "LogSoftmax", "Dropout", "DropPath"):
        out.append((act, unary(act, ("B", 8))))
    out += [
        ("GELU_tanh", unary("GELU", ("B", 8), {"approximate": "tanh"})),
        ("GLU", unary("GLU", ("B", 8))),
        ("Add", binary("Add", ("B", 8), (8,))),
        ("Sub", binary("Sub", ("B", 8), ("B", 8))),
        ("Mul", binary("Mul", ("B", 8), ("B", 1))),
        ("Concat", binary("Concat", ("B", 8), ("B", 4))),
        ("MatMul", binary("MatMul", ("B", 4, 8), ("B", 8, 3))),
        ("Flatten", unary("Flatten", IMG)),
        ("Flatten_batch", unary("Flatten", ("B", 2, 5), {"start_dim": 0, "end_dim": 1})),
        ("Unflatten", unary("Unflatten", ("B", 12), {"dim": 1, "sizes": [3, -1]})),
        ("Reshape", unary("Reshape", ("B", 12), {"shape": ["B", 3, 4]})),
        ("Transpose", unary("Transpose", SEQ)),
        ("Permute", unary("Permute", IMG, {"dims": [0, 2, 3, 1]})),
        ("Select", unary("Select", SEQ, {"dim": 1, "index": 0})),
        ("Slice", unary("Slice", ("B", 10, 4), {"dim": 1, "start": 2, "end": 7})),
        ("Slice_neg", unary("Slice", ("B", 4, 10), {"dim": -1, "start": 1})),
        ("Unsqueeze", unary("Unsqueeze", ("B", 8), {"dim": 1})),
        ("Chunk", _g([_inp("x", ("B", 8)), {"id": "n", "type": "Chunk"}, {"id": "o1", "type": "Output"}, {"id": "o2", "type": "Output"}],
                     [["x.out", "n.in"], ["n.a", "o1.in"], ["n.b", "o2.in"]])),
        ("Reduce_max", unary("Reduce", SEQ, {"op": "max", "dim": 1, "keepdim": True})),
        ("Reduce_sum", unary("Reduce", SEQ, {"op": "sum", "dim": -1})),
        ("MeanPool", unary("MeanPool", SEQ)),
        ("Scale", unary("Scale", ("B", 8), {"factor": 0.5})),
        ("Identity", unary("Identity", ("B", 8))),
        ("Reparameterize", binary("Reparameterize", ("B", 8), ("B", 8), ports=("mu", "logvar"))),
        ("loss_CrossEntropy", loss("CrossEntropy", {"logits": (("B", 5), "f32"), "target": (("B",), "i64")})),
        ("loss_MSE", loss("MSE", {"pred": (("B", 3), "f32"), "target": (("B", 3), "f32")})),
        ("loss_BCEWithLogits", loss("BCEWithLogits", {"pred": (("B", 3), "f32"), "target": (("B", 3), "f32")})),
        ("loss_L1", loss("L1", {"pred": (("B", 3), "f32"), "target": (("B", 3), "f32")})),
        ("loss_Huber", loss("Huber", {"pred": (("B", 3), "f32"), "target": (("B", 3), "f32")}, {"delta": 0.5})),
        ("loss_NLL", loss("NLL", {"log_probs": (("B", 5), "f32"), "target": (("B",), "i64")})),
        ("loss_KLDiv", loss("KLDiv", {"log_probs": (("B", 5), "f32"), "target_probs": (("B", 5), "f32")})),
        ("loss_Contrastive", loss("Contrastive", {"a": (("B", 6), "f32"), "b": (("B", 6), "f32")}, {"symmetric": True})),
        ("loss_Contrastive_negatives", loss("Contrastive", {"a": (("B", 6), "f32"), "b": (("B", 6), "f32"), "negatives": (("N", 6), "f32")})),
        ("model_mpnet_triplets", templates.mpnet_base(train_triplets=True)),
        ("loss_Triplet", loss("Triplet", {"anchor": (("B", 6), "f32"), "positive": (("B", 6), "f32"), "negative": (("B", 6), "f32")})),
        ("loss_KLGaussian", loss("KLGaussian", {"mu": (("B", 4), "f32"), "logvar": (("B", 4), "f32")})),
    ]
    out += block_cases()
    return out


SMALL_ARGS = {
    "pre_ln_block": {"D": 16, "H": 4, "F": 32, "causal": True},
    "post_ln_block": {"D": 16, "H": 4, "F": 32},
    "llama_block": {"D": 16, "H": 4, "KV": 2, "F": 24},
    "moe_block": {"D": 16, "H": 4, "KV": 2, "F": 8, "E": 4, "K": 2},
    "llama4_group": {"D": 16, "H": 4, "KV": 2, "F": 8, "E": 4, "chunk": 3},
    "decoder_block": {"D": 16, "H": 4, "F": 32},
    "resnet_basic": {"C": 4},
    "resnet_basic_down": {"Cin": 4, "C": 8},
    "double_conv": {"Cin": 3, "Cout": 4},
    "mamba_block": {"D": 16, "N": 4},
}


def block_cases() -> list[tuple[str, dict[str, Any]]]:
    """Each block template instantiated small, repeated twice where it is shape-stable."""
    out = []
    for t in templates.BLOCKS:
        body = t.build()
        args = SMALL_ARGS[t.id]
        ins = [n for n in body["nodes"] if n["type"] == "Input"]
        nodes, edges = [], []
        for i, n in enumerate(ins):
            shape = []
            for d in n["params"]["shape"]:
                if isinstance(d, str) and d.startswith("$"):
                    from weightsmith_core.ir import block_args, eval_arg

                    d = eval_arg(d[1:], block_args(body, args))
                elif d in ("H", "W"):
                    d = 8
                shape.append(d)
            if n["id"] == "memory":
                shape = ["B", "S", shape[2]]
            nodes.append({"id": f"in_{n['id']}", "type": "Input", "params": {"shape": shape, **({"field": f"f_{n['id']}"} if i else {})}})
            edges.append([f"in_{n['id']}.out", f"blk.{n['id']}"])
        if any(n["id"] == "bias" for n in ins):  # the relative bias is computed, not fed
            nodes = [n for n in nodes if n["id"] != "in_bias"]
            edges = [e for e in edges if not e[0].startswith("in_bias")]
            nodes.append({"id": "rb", "type": "RelativePositionBias", "params": {"heads": args["H"]}})
            edges += [["in_x.out", "rb.in"], ["rb.out", "blk.bias"]]
        stable = t.id not in ("resnet_basic_down", "double_conv")
        nodes.append({"id": "blk", "type": "Subgraph", "params": {"graph": t.id, "repeat": 2 if stable else 1, "args": args}})
        nodes.append({"id": "o", "type": "Output"})
        edges.append(["blk.out", "o.in"])
        out.append((f"block_{t.id}", _g(nodes, edges, {t.id: body})))
    return out
