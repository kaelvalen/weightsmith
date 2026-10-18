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
        ("Attention_mask", _g([_inp("x", SEQ), _inp("m", ("B", "T"), "i64", field="mask"), {"id": "n", "type": "Attention", "params": {"dim": 16, "heads": 4}},
                               {"id": "o", "type": "Output"}], [["x.out", "n.in"], ["m.out", "n.mask"], ["n.out", "o.in"]])),
        ("Attention_llama3_scaling", unary("Attention", SEQ, {"dim": 16, "heads": 2, "rope": True, "causal": True, "rope_scaling":
                                         {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 16}})),
        ("Reduce_mask_mean", _g([_inp("x", SEQ), _inp("m", ("B", "T"), "i64", field="mask"), {"id": "n", "type": "Reduce", "params": {"op": "mean", "dim": 1}},
                                 {"id": "o", "type": "Output"}], [["x.out", "n.in"], ["m.out", "n.mask"], ["n.out", "o.in"]])),
        ("Reduce_mask_max", _g([_inp("x", SEQ), _inp("m", ("B", "T"), "i64", field="mask"), {"id": "n", "type": "Reduce", "params": {"op": "max", "dim": 1}},
                                {"id": "o", "type": "Output"}], [["x.out", "n.in"], ["m.out", "n.mask"], ["n.out", "o.in"]])),
        ("PositionalEmbedding_offset", unary("PositionalEmbedding", SEQ, {"max_len": 32, "dim": 16, "offset": 2})),
        ("MLP_gelu_tanh", unary("MLP", SEQ, {"dim": 16, "hidden": 32, "act": "gelu_tanh"})),
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
        # nodes_extra
        ("ReLU6", unary("ReLU6", SEQ)),
        ("HardTanh", unary("HardTanh", SEQ, {"min": -2.0, "max": 2.0})),
        ("Hardsigmoid", unary("Hardsigmoid", SEQ)),
        ("Softsign", unary("Softsign", SEQ)),
        ("Swish", unary("Swish", SEQ, {"dim": 16})),
        ("PReLU", unary("PReLU", IMG, {"channels": 4})),
        ("PReLU_shared", unary("PReLU", SEQ)),
        ("GumbelSoftmax", unary("GumbelSoftmax", SEQ, {"hard": True})),
        ("Dropout2d", unary("Dropout2d", IMG)),
        ("Math", unary("Math", SEQ, {"op": "square"})),
        ("Math_sin", unary("Math", SEQ, {"op": "sin"})),
        ("Clamp", unary("Clamp", SEQ, {"min": 0.1})),
        ("Pow", unary("Pow", SEQ, {"exponent": 3.0})),
        ("AddConst", unary("AddConst", SEQ, {"value": 0.5})),
        ("Div", binary("Div", SEQ, ("B", 1, 16))),
        ("Maximum", binary("Maximum", SEQ, SEQ)),
        ("Minimum", binary("Minimum", ("B", 16), (16,))),
        ("Einsum", binary("Einsum", ("B", "T", 16), ("B", "S", 16), {"equation": "btd,bsd->bts"})),
        ("Einsum_unary", unary("Einsum", ("B", 3, 4), {"equation": "bij->bji"}, in_port="a")),
        ("Cumsum", unary("Cumsum", SEQ)),
        ("Stack", binary("Stack", SEQ, SEQ, {"dim": 1})),
        ("Squeeze", unary("Squeeze", ("B", 1, 16))),
        ("Pad", unary("Pad", IMG, {"pad": [1, 1, 2, 0]})),
        ("Repeat", unary("Repeat", ("B", 2, 3), {"repeats": [1, 2, 2]})),
        ("TopK", unary("TopK", SEQ, {"k": 3}, out_port="values")),
        ("TopK_indices", unary("TopK", SEQ, {"k": 2}, out_port="indices")),
        ("ArgMax", unary("ArgMax", SEQ)),
        ("OneHot", unary("OneHot", ("B", "T"), {"classes": 5}, dtype="i64")),
        ("Cast", unary("Cast", ("B", "T"), {"dtype": "f32"}, dtype="i64")),
        ("StopGradient", unary("StopGradient", SEQ)),
        ("Parameter", _g([{"id": "n", "type": "Parameter", "params": {"shape": [1, 4, 16]}}, _inp("x", SEQ), {"id": "add", "type": "Add"}, {"id": "o", "type": "Output"}],
                         [["x.out", "add.a"], ["n.out", "add.b"], ["add.out", "o.in"]])),
        ("MaxPool3d", unary("MaxPool3d", ("B", 2, 4, 8, 8))),
        ("AvgPool3d", unary("AvgPool3d", ("B", 2, 4, 8, 8), {"kernel_size": 2, "stride": 1})),
        ("AdaptiveMaxPool1d", unary("AdaptiveMaxPool1d", ("B", 4, 10), {"output_size": 3})),
        ("AdaptiveMaxPool2d", unary("AdaptiveMaxPool2d", IMG)),
        ("Interpolate", unary("Interpolate", IMG, {"size": [5, 12], "mode": "bilinear"})),
        ("BatchNorm3d", unary("BatchNorm3d", ("B", 2, 4, 8, 8), {"features": 2})),
        ("LayerScale", unary("LayerScale", SEQ, {"dim": 16})),
        ("SqueezeExcite", unary("SqueezeExcite", IMG, {"channels": 4, "reduction": 2})),
        ("ALiBi", _g([_inp("x", SEQ), {"id": "n", "type": "ALiBi", "params": {"heads": 4}},
                      {"id": "att", "type": "Attention", "params": {"dim": 16, "heads": 4, "causal": True}}, {"id": "o", "type": "Output"}],
                     [["x.out", "n.in"], ["x.out", "att.in"], ["n.out", "att.bias"], ["att.out", "o.in"]])),
        ("SlidingWindowBias", unary("SlidingWindowBias", SEQ, {"heads": 3, "window": 2})),
        ("ALiBi_6heads", unary("ALiBi", SEQ, {"heads": 6, "causal": False})),
        ("LinearAttention", unary("LinearAttention", SEQ, {"dim": 16, "heads": 4})),
        ("LinearAttention_causal", unary("LinearAttention", SEQ, {"dim": 16, "heads": 2, "causal": True})),
        ("TimestepEmbedding", unary("TimestepEmbedding", ("B",), {"dim": 16}, dtype="i64")),
        ("VectorQuantizer", _g([_inp("x", SEQ), {"id": "n", "type": "VectorQuantizer", "params": {"codes": 8, "dim": 16}},
                                {"id": "aux", "type": "AuxLoss"}, {"id": "o", "type": "Output"}, {"id": "o2", "type": "Output", "params": {"name": "codes"}}],
                               [["x.out", "n.in"], ["n.commit", "aux.in"], ["n.out", "o.in"], ["n.indices", "o2.in"]])),
        ("GraphConv", _g([_inp("x", ("B", 5, 6)), _inp("adj", ("B", 5, 5), field="adj"), {"id": "n", "type": "GraphConv", "params": {"in": 6, "out": 3}},
                          {"id": "pool", "type": "GraphPool", "params": {"op": "max"}}, {"id": "o", "type": "Output"}],
                         [["x.out", "n.x"], ["adj.out", "n.adj"], ["n.out", "pool.in"], ["pool.out", "o.in"]])),
        ("DiffusionForward", _g([_inp("x", ("B", 1, 8, 8)), {"id": "n", "type": "DiffusionForward", "params": {"schedule": "cosine", "steps": 50}},
                                 {"id": "mse", "type": "MSE"}, {"id": "o", "type": "Output"}, {"id": "o2", "type": "Output", "params": {"name": "t"}}],
                                [["x.out", "n.in"], ["n.noisy", "mse.pred"], ["n.noise", "mse.target"], ["n.noisy", "o.in"], ["n.t", "o2.in"]])),
        ("loss_Focal", loss("Focal", {"logits": (("B", "T", 5), "f32"), "target": (("B", "T"), "i64")})),
        ("loss_Dice", loss("Dice", {"pred": (IMG, "f32"), "target": (IMG, "f32")})),
        ("loss_GaussianNLL", loss("GaussianNLL", {"mean": (("B", 3), "f32"), "var": (("B", 3), "f32"), "target": (("B", 3), "f32")})),
        ("loss_CosineEmbedding", loss("CosineEmbedding", {"a": (("B", 8), "f32"), "b": (("B", 8), "f32")})),
        ("loss_AuxLoss", loss("AuxLoss", {"in": (("B", 3), "f32")})),
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
    "inverted_residual": {"C": 4, "T": 2},
    "inverted_residual_proj": {"Cin": 4, "Cout": 6, "S": 2, "T": 3},
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
        stable = t.id not in ("resnet_basic_down", "double_conv", "inverted_residual_proj")
        nodes.append({"id": "blk", "type": "Subgraph", "params": {"graph": t.id, "repeat": 2 if stable else 1, "args": args}})
        nodes.append({"id": "o", "type": "Output"})
        edges.append(["blk.out", "o.in"])
        out.append((f"block_{t.id}", _g(nodes, edges, {t.id: body})))
    return out
