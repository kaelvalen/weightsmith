"""Block and model templates, built from the node catalogue.

Blocks are parametric subgraphs ($D, $H, ...) that the canvas inserts as one node.
Models are complete graphs that open as a project or are inserted inline. Where a
model has a published parameter count, `reference` records it and the test suite
checks the analyzer (and the generated code on the meta device) against it.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

Graph = dict[str, Any]


class Body:
    """Small builder for node lists: add nodes, wire ports, chain the default in/out ports."""

    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[list[str]] = []

    def add(self, nid: str, type_: str, label: str | None = None, **params: Any) -> str:
        n: dict[str, Any] = {"id": nid, "type": type_}
        flags = {k: params.pop(k) for k in ("trainable", "adapt", "frozen_params") if k in params}
        if params:
            n["params"] = params
        if label:
            n["label"] = label
        n.update(flags)
        self.nodes.append(n)
        return nid

    def wire(self, src: str, dst: str) -> None:
        self.edges.append([src if "." in src else f"{src}.out", dst if "." in dst else f"{dst}.in"])

    def chain(self, *ids: str) -> None:
        for a, b in zip(ids, ids[1:]):
            self.wire(a, b)

    def residual(self, skip: str, branch: str, nid: str) -> str:
        """nid = skip + branch."""
        self.add(nid, "Add")
        self.wire(skip, f"{nid}.a")
        self.wire(branch, f"{nid}.b")
        return nid

    def graph(self, name: str, subgraphs: dict[str, Any] | None = None) -> Graph:
        g: Graph = {"schema": 1, "name": name, "nodes": self.nodes, "edges": self.edges}
        if subgraphs:
            g["subgraphs"] = subgraphs
        return g

    def block(self, params: dict[str, Any], doc: str) -> dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges, "params": params, "doc": doc}


# =================================================================== blocks


def pre_ln_block(**defaults) -> dict[str, Any]:
    """GPT-2 / ViT: x + attn(ln(x)); x + mlp(ln(x))."""
    b = Body()
    b.add("x", "Input", shape=["B", "T", "$D"])
    b.add("ln1", "LayerNorm", dim="$D", eps="$eps")
    b.add("attn", "Attention", dim="$D", heads="$H", causal="$causal")
    b.add("drop1", "Dropout", p="$dropout")
    b.residual("x", "drop1", "res1")
    b.add("ln2", "LayerNorm", dim="$D", eps="$eps")
    b.add("mlp", "MLP", dim="$D", hidden="$F", act="$act")
    b.add("drop2", "Dropout", p="$dropout")
    b.residual("res1", "drop2", "res2")
    b.add("out", "Output")
    b.chain("x", "ln1", "attn", "drop1")
    b.chain("res1", "ln2", "mlp", "drop2")
    b.chain("res2", "out")
    params = {"D": 768, "H": 12, "F": 3072, "eps": 1e-5, "causal": False, "act": "gelu", "dropout": 0.1} | defaults
    return b.block(params, "Pre-LN transformer block (GPT-2, ViT): x + attn(LN(x)), then x + MLP(LN(x)).")


def post_ln_block(rel_bias: bool = False, masked: bool = False, **defaults) -> dict[str, Any]:
    """BERT / MiniLM / MPNet / Vaswani encoder: LN(x + attn(x)); LN(x + mlp(x))."""
    b = Body()
    b.add("x", "Input", shape=["B", "T", "$D"])
    if rel_bias:
        b.add("bias", "Input", label="shared position bias", shape=["$H", "T", "T"])
    if masked:
        b.add("mask", "Input", label="padding mask", shape=["B", "T"], dtype="i64")
    b.add("attn", "Attention", dim="$D", heads="$H")
    b.add("drop1", "Dropout", p="$dropout")
    b.residual("x", "drop1", "res1")
    b.add("ln1", "LayerNorm", dim="$D", eps="$eps")
    b.add("mlp", "MLP", dim="$D", hidden="$F", act="$act")
    b.add("drop2", "Dropout", p="$dropout")
    b.residual("ln1", "drop2", "res2")
    b.add("ln2", "LayerNorm", dim="$D", eps="$eps")
    b.add("out", "Output")
    b.chain("x", "attn", "drop1")
    if rel_bias:
        b.wire("bias", "attn.bias")
    if masked:
        b.wire("mask", "attn.mask")
    b.chain("res1", "ln1", "mlp", "drop2")
    b.chain("res2", "ln2", "out")
    params = {"D": 768, "H": 12, "F": 3072, "eps": 1e-12, "act": "gelu", "dropout": 0.1} | defaults
    doc = "Post-LN transformer encoder block (BERT): LN(x + attn(x)), then LN(x + MLP(x))."
    if rel_bias:
        doc += " Takes a relative position bias shared by every layer (MPNet)."
    if masked:
        doc += " Padding positions are masked out of attention."
    return b.block(params, doc)


def llama_block(**defaults) -> dict[str, Any]:
    """Llama / Qwen2: RMSNorm, GQA with RoPE, SwiGLU."""
    b = Body()
    b.add("x", "Input", shape=["B", "T", "$D"])
    b.add("input_norm", "RMSNorm", label="input_layernorm", dim="$D", eps="$eps")
    b.add("self_attn", "Attention", dim="$D", heads="$H", kv_heads="$KV", causal=True, bias="$qkv_bias", o_bias=False, rope=True,
          rope_theta="$theta", rope_scaling="$rope_scaling")
    b.residual("x", "self_attn", "res_attn")
    b.add("post_norm", "RMSNorm", label="post_attention_layernorm", dim="$D", eps="$eps")
    b.add("mlp", "GatedMLP", label="SwiGLU", dim="$D", hidden="$F", act="silu")
    b.residual("res_attn", "mlp", "res_mlp")
    b.add("out", "Output")
    b.chain("x", "input_norm", "self_attn")
    b.chain("res_attn", "post_norm", "mlp")
    b.chain("res_mlp", "out")
    params = {"D": 2048, "H": 32, "KV": 8, "F": 8192, "theta": 500000.0, "eps": 1e-5, "qkv_bias": False, "rope_scaling": None} | defaults
    return b.block(params, "Llama-style decoder layer: RMSNorm → GQA with RoPE → residual → RMSNorm → SwiGLU → residual.")


def moe_block(**defaults) -> dict[str, Any]:
    """Mixtral: llama layer whose MLP is a sparse mixture of experts."""
    b = Body()
    b.add("x", "Input", shape=["B", "T", "$D"])
    b.add("input_norm", "RMSNorm", dim="$D", eps="$eps")
    b.add("self_attn", "Attention", dim="$D", heads="$H", kv_heads="$KV", causal=True, bias=False, o_bias=False, rope=True, rope_theta="$theta")
    b.residual("x", "self_attn", "res_attn")
    b.add("post_norm", "RMSNorm", dim="$D", eps="$eps")
    b.add("moe", "MoE", label="sparse MoE", dim="$D", hidden="$F", num_experts="$E", top_k="$K", router="softmax")
    b.residual("res_attn", "moe", "res_moe")
    b.add("out", "Output")
    b.chain("x", "input_norm", "self_attn")
    b.chain("res_attn", "post_norm", "moe")
    b.chain("res_moe", "out")
    params = {"D": 4096, "H": 32, "KV": 8, "F": 14336, "E": 8, "K": 2, "theta": 1000000.0, "eps": 1e-5} | defaults
    return b.block(params, "Mixtral-style layer: attention as in Llama, MLP replaced by top-k routed SwiGLU experts.")


def llama4_group(**defaults) -> dict[str, Any]:
    """Four Llama 4 layers: three RoPE + chunked + QK-norm, one NoPE with full attention (iRoPE)."""
    b = Body()
    b.add("x", "Input", shape=["B", "T", "$D"])
    prev = "x"
    for i in range(4):
        rope = i < 3
        attn = dict(dim="$D", heads="$H", kv_heads="$KV", causal=True, bias=False, o_bias=False)
        attn |= dict(rope=True, rope_theta="$theta", qk_norm="l2", chunk="$chunk") if rope else dict(nope_temp=True)
        b.add(f"norm{i}a", "RMSNorm", dim="$D", eps="$eps")
        b.add(f"attn{i}", "Attention", label=f"layer {i} · {'RoPE, chunked' if rope else 'NoPE, global'}", **attn)
        b.residual(prev, f"attn{i}", f"res{i}a")
        b.add(f"norm{i}b", "RMSNorm", dim="$D", eps="$eps")
        b.add(f"moe{i}", "MoE", dim="$D", hidden="$F", num_experts="$E", top_k=1, router="sigmoid", shared_hidden="$F")
        b.residual(f"res{i}a", f"moe{i}", f"res{i}b")
        b.chain(prev, f"norm{i}a", f"attn{i}")
        b.chain(f"res{i}a", f"norm{i}b", f"moe{i}")
        prev = f"res{i}b"
    b.add("out", "Output")
    b.chain(prev, "out")
    params = {"D": 5120, "H": 40, "KV": 8, "F": 8192, "E": 16, "theta": 500000.0, "chunk": 8192, "eps": 1e-5} | defaults
    return b.block(params, "Llama 4 (iRoPE): 3 RoPE layers with chunked attention and L2 QK-norm, then 1 NoPE layer with global attention; "
                           "every MLP is a top-1 sigmoid-routed MoE with a shared expert.")


def resnet_basic(**defaults) -> dict[str, Any]:
    b = Body()
    b.add("x", "Input", shape=["B", "$C", "H", "W"])
    b.add("conv1", "Conv2d", in_channels="$C", out_channels="$C", kernel_size=3, stride=1, padding=1, bias=False)
    b.add("bn1", "BatchNorm2d", features="$C")
    b.add("relu1", "ReLU")
    b.add("conv2", "Conv2d", in_channels="$C", out_channels="$C", kernel_size=3, stride=1, padding=1, bias=False)
    b.add("bn2", "BatchNorm2d", features="$C")
    b.residual("x", "bn2", "add")
    b.add("relu2", "ReLU")
    b.add("out", "Output")
    b.chain("x", "conv1", "bn1", "relu1", "conv2", "bn2")
    b.chain("add", "relu2", "out")
    return b.block({"C": 64} | defaults, "ResNet basic block: two 3×3 conv-BN with an identity shortcut.")


def resnet_basic_down(**defaults) -> dict[str, Any]:
    b = Body()
    b.add("x", "Input", shape=["B", "$Cin", "H", "W"])
    b.add("conv1", "Conv2d", in_channels="$Cin", out_channels="$C", kernel_size=3, stride=2, padding=1, bias=False)
    b.add("bn1", "BatchNorm2d", features="$C")
    b.add("relu1", "ReLU")
    b.add("conv2", "Conv2d", in_channels="$C", out_channels="$C", kernel_size=3, stride=1, padding=1, bias=False)
    b.add("bn2", "BatchNorm2d", features="$C")
    b.add("down_conv", "Conv2d", label="downsample", in_channels="$Cin", out_channels="$C", kernel_size=1, stride=2, padding=0, bias=False)
    b.add("down_bn", "BatchNorm2d", features="$C")
    b.residual("down_bn", "bn2", "add")
    b.add("relu2", "ReLU")
    b.add("out", "Output")
    b.chain("x", "conv1", "bn1", "relu1", "conv2", "bn2")
    b.chain("x", "down_conv", "down_bn")
    b.chain("add", "relu2", "out")
    return b.block({"Cin": 64, "C": 128} | defaults, "ResNet basic block that halves H×W and changes channels; 1×1 conv shortcut.")


def double_conv(**defaults) -> dict[str, Any]:
    b = Body()
    b.add("x", "Input", shape=["B", "$Cin", "H", "W"])
    b.add("conv1", "Conv2d", in_channels="$Cin", out_channels="$Cout", kernel_size=3, padding=1, bias=False)
    b.add("bn1", "BatchNorm2d", features="$Cout")
    b.add("relu1", "ReLU")
    b.add("conv2", "Conv2d", in_channels="$Cout", out_channels="$Cout", kernel_size=3, padding=1, bias=False)
    b.add("bn2", "BatchNorm2d", features="$Cout")
    b.add("relu2", "ReLU")
    b.add("out", "Output")
    b.chain("x", "conv1", "bn1", "relu1", "conv2", "bn2", "relu2", "out")
    return b.block({"Cin": 1, "Cout": 32} | defaults, "U-Net double convolution: (3×3 conv → BN → ReLU) × 2.")


def inverted_residual(**defaults) -> dict[str, Any]:
    b = Body()
    b.add("x", "Input", shape=["B", "$C", "H", "W"])
    b.add("expand", "Conv2d", in_channels="$C", out_channels="$C*T", kernel_size=1, padding=0, bias=False)
    b.add("bn1", "BatchNorm2d", features="$C*T")
    b.add("act1", "ReLU6")
    b.add("dw", "Conv2d", label="depthwise 3×3", in_channels="$C*T", out_channels="$C*T", kernel_size=3, padding=1, groups="$C*T", bias=False)
    b.add("bn2", "BatchNorm2d", features="$C*T")
    b.add("act2", "ReLU6")
    b.add("project", "Conv2d", in_channels="$C*T", out_channels="$C", kernel_size=1, padding=0, bias=False)
    b.add("bn3", "BatchNorm2d", features="$C")
    b.residual("x", "bn3", "add")
    b.add("out", "Output")
    b.chain("x", "expand", "bn1", "act1", "dw", "bn2", "act2", "project", "bn3")
    b.chain("add", "out")
    return b.block({"C": 24, "T": 6} | defaults, "MobileNetV2 inverted residual: 1×1 expand → 3×3 depthwise → 1×1 linear projection, identity shortcut.")


def inverted_residual_proj(**defaults) -> dict[str, Any]:
    b = Body()
    b.add("x", "Input", shape=["B", "$Cin", "H", "W"])
    b.add("expand", "Conv2d", in_channels="$Cin", out_channels="$Cin*T", kernel_size=1, padding=0, bias=False)
    b.add("bn1", "BatchNorm2d", features="$Cin*T")
    b.add("act1", "ReLU6")
    b.add("dw", "Conv2d", label="depthwise 3×3", in_channels="$Cin*T", out_channels="$Cin*T", kernel_size=3, stride="$S", padding=1, groups="$Cin*T", bias=False)
    b.add("bn2", "BatchNorm2d", features="$Cin*T")
    b.add("act2", "ReLU6")
    b.add("project", "Conv2d", in_channels="$Cin*T", out_channels="$Cout", kernel_size=1, padding=0, bias=False)
    b.add("bn3", "BatchNorm2d", features="$Cout")
    b.add("out", "Output")
    b.chain("x", "expand", "bn1", "act1", "dw", "bn2", "act2", "project", "bn3", "out")
    return b.block({"Cin": 16, "Cout": 24, "S": 2, "T": 6} | defaults, "MobileNetV2 inverted residual that changes channels or stride (no shortcut).")


def mamba_block(**defaults) -> dict[str, Any]:
    b = Body()
    b.add("x", "Input", shape=["B", "T", "$D"])
    b.add("norm", "RMSNorm", dim="$D", eps="$eps")
    b.add("mixer", "Mamba", dim="$D", d_state="$N", expand=2, d_conv=4)
    b.residual("x", "mixer", "res")
    b.add("out", "Output")
    b.chain("x", "norm", "mixer")
    b.chain("res", "out")
    return b.block({"D": 768, "N": 16, "eps": 1e-5} | defaults, "Mamba block: x + Mamba(RMSNorm(x)).")


def decoder_block(**defaults) -> dict[str, Any]:
    """Vaswani decoder layer with encoder memory as a side input."""
    b = Body()
    b.add("x", "Input", shape=["B", "T", "$D"])
    b.add("memory", "Input", label="encoder output", shape=["B", "S", "$D"])
    b.add("self_attn", "Attention", dim="$D", heads="$H", causal=True)
    b.residual("x", "self_attn", "res1")
    b.add("ln1", "LayerNorm", dim="$D")
    b.add("cross_attn", "CrossAttention", dim="$D", heads="$H")
    b.wire("ln1", "cross_attn.in")
    b.wire("memory", "cross_attn.context")
    b.residual("ln1", "cross_attn", "res2")
    b.add("ln2", "LayerNorm", dim="$D")
    b.add("mlp", "MLP", dim="$D", hidden="$F", act="relu")
    b.residual("ln2", "mlp", "res3")
    b.add("ln3", "LayerNorm", dim="$D")
    b.add("out", "Output")
    b.chain("x", "self_attn")
    b.chain("res1", "ln1")
    b.chain("res2", "ln2", "mlp")
    b.chain("res3", "ln3", "out")
    return b.block({"D": 512, "H": 8, "F": 2048} | defaults, "Transformer decoder layer: causal self-attention, cross-attention to the encoder, MLP (post-LN).")


# =================================================================== models


def _lm_head(b: Body, embed: str, dim: int, vocab: int, tied: bool, src: str) -> None:
    b.add("lm_head", "Linear", label="lm_head" + (" (tied)" if tied else ""), **{"in": dim, "out": vocab, "bias": False, **({"tie_to": embed} if tied else {})})
    b.add("loss", "CrossEntropy", label="next token")
    b.chain(src, "lm_head")
    b.wire("lm_head", "loss.logits")


def mlp_mnist() -> Graph:
    b = Body()
    b.add("image", "Input", shape=["B", 784])
    b.add("fc1", "Linear", **{"in": 784, "out": 256})
    b.add("act", "ReLU")
    b.add("drop", "Dropout", p=0.1)
    b.add("fc2", "Linear", **{"in": 256, "out": 10})
    b.add("loss", "CrossEntropy")
    b.chain("image", "fc1", "act", "drop", "fc2")
    b.wire("fc2", "loss.logits")
    return b.graph("MLP · MNIST")


def lenet5() -> Graph:
    b = Body()
    b.add("image", "Input", shape=["B", 1, 28, 28])
    b.add("c1", "Conv2d", in_channels=1, out_channels=6, kernel_size=5, padding=2)
    b.add("t1", "Tanh")
    b.add("s2", "AvgPool2d", kernel_size=2)
    b.add("c3", "Conv2d", in_channels=6, out_channels=16, kernel_size=5, padding=0)
    b.add("t3", "Tanh")
    b.add("s4", "AvgPool2d", kernel_size=2)
    b.add("flat", "Flatten")
    b.add("f5", "Linear", **{"in": 400, "out": 120})
    b.add("t5", "Tanh")
    b.add("f6", "Linear", **{"in": 120, "out": 84})
    b.add("t6", "Tanh")
    b.add("f7", "Linear", **{"in": 84, "out": 10})
    b.add("loss", "CrossEntropy")
    b.chain("image", "c1", "t1", "s2", "c3", "t3", "s4", "flat", "f5", "t5", "f6", "t6", "f7")
    b.wire("f7", "loss.logits")
    return b.graph("LeNet-5")


def resnet18(classes: int = 1000) -> Graph:
    b = Body()
    b.add("image", "Input", shape=["B", 3, 224, 224])
    b.add("conv1", "Conv2d", in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    b.add("bn1", "BatchNorm2d", features=64)
    b.add("relu", "ReLU")
    b.add("maxpool", "MaxPool2d", kernel_size=3, stride=2, padding=1)
    b.add("layer1", "Subgraph", label="layer1 · 2× basic 64", graph="basic", repeat=2, args={"C": 64})
    b.add("layer2_0", "Subgraph", label="layer2 · down 64→128", graph="basic_down", args={"Cin": 64, "C": 128})
    b.add("layer2_1", "Subgraph", label="layer2 · basic 128", graph="basic", args={"C": 128})
    b.add("layer3_0", "Subgraph", label="layer3 · down 128→256", graph="basic_down", args={"Cin": 128, "C": 256})
    b.add("layer3_1", "Subgraph", label="layer3 · basic 256", graph="basic", args={"C": 256})
    b.add("layer4_0", "Subgraph", label="layer4 · down 256→512", graph="basic_down", args={"Cin": 256, "C": 512})
    b.add("layer4_1", "Subgraph", label="layer4 · basic 512", graph="basic", args={"C": 512})
    b.add("avgpool", "AdaptiveAvgPool2d", output_size=1)
    b.add("flat", "Flatten")
    b.add("fc", "Linear", **{"in": 512, "out": classes})
    b.add("loss", "CrossEntropy")
    b.chain("image", "conv1", "bn1", "relu", "maxpool")
    prev = "maxpool"
    for nid in ("layer1", "layer2_0", "layer2_1", "layer3_0", "layer3_1", "layer4_0", "layer4_1"):
        b.wire(prev, f"{nid}.x")
        prev = f"{nid}.out"
    b.chain(prev, "avgpool", "flat", "fc")
    b.wire("fc", "loss.logits")
    return b.graph("ResNet-18", {"basic": resnet_basic(), "basic_down": resnet_basic_down()})


def unet_small() -> Graph:
    b = Body()
    b.add("image", "Input", shape=["B", 1, 64, 64])
    for nid, cin, cout in (("enc1", 1, 32), ("enc2", 32, 64), ("bottleneck", 64, 128), ("dec2", 128, 64), ("dec1", 64, 32)):
        b.add(nid, "Subgraph", label=f"{nid} · {cin}→{cout}", graph="double_conv", args={"Cin": cin, "Cout": cout})
    b.add("pool1", "MaxPool2d", kernel_size=2)
    b.add("pool2", "MaxPool2d", kernel_size=2)
    b.add("up2", "ConvTranspose2d", in_channels=128, out_channels=64)
    b.add("up1", "ConvTranspose2d", in_channels=64, out_channels=32)
    b.add("cat2", "Concat", label="skip 2", dim=1)
    b.add("cat1", "Concat", label="skip 1", dim=1)
    b.add("head", "Conv2d", in_channels=32, out_channels=1, kernel_size=1, padding=0)
    b.add("loss", "BCEWithLogits", label="mask")
    b.wire("image", "enc1.x")
    b.wire("enc1.out", "pool1")
    b.wire("pool1", "enc2.x")
    b.wire("enc2.out", "pool2")
    b.wire("pool2", "bottleneck.x")
    b.wire("bottleneck.out", "up2")
    b.wire("up2", "cat2.a")
    b.wire("enc2.out", "cat2.b")
    b.wire("cat2", "dec2.x")
    b.wire("dec2.out", "up1")
    b.wire("up1", "cat1.a")
    b.wire("enc1.out", "cat1.b")
    b.wire("cat1", "dec1.x")
    b.wire("dec1.out", "head")
    b.wire("head", "loss.pred")
    return b.graph("U-Net (small)", {"double_conv": double_conv()})


def vit_b16(classes: int = 1000) -> Graph:
    b = Body()
    b.add("image", "Input", shape=["B", 3, 224, 224])
    b.add("patch", "PatchEmbed", label="16×16 patches", in_channels=3, dim=768, patch=16)
    b.add("cls", "ClassToken", dim=768)
    b.add("pos", "PositionalEmbedding", max_len=197, dim=768)
    b.add("drop", "Dropout", p=0.0)
    b.add("encoder", "Subgraph", label="12× encoder block", graph="vit_block", repeat=12)
    b.add("norm", "LayerNorm", dim=768, eps=1e-6)
    b.add("cls_out", "Select", label="[CLS]", dim=1, index=0)
    b.add("head", "Linear", **{"in": 768, "out": classes})
    b.add("loss", "CrossEntropy")
    b.chain("image", "patch", "cls", "pos", "drop")
    b.wire("drop", "encoder.x")
    b.chain("encoder.out", "norm", "cls_out", "head")
    b.wire("head", "loss.logits")
    return b.graph("ViT-B/16", {"vit_block": pre_ln_block(eps=1e-6, dropout=0.0)})


def _bert_like(name: str, vocab: int, dim: int, heads: int, ffn: int, max_pos: int, eps: float, types: int, sentence: bool) -> Graph:
    b = Body()
    b.add("input_ids", "Input", shape=["B", "T"], dtype="i64", field="x")
    b.add("attention_mask", "Input", label="1 = token, 0 = padding", shape=["B", "T"], dtype="i64", field="mask")
    b.add("word", "Embedding", label="word_embeddings", num=vocab, dim=dim)
    b.add("pos", "PositionalEmbedding", label="position_embeddings", max_len=max_pos, dim=dim)
    b.add("emb_norm", "LayerNorm", dim=dim, eps=eps)
    b.add("emb_drop", "Dropout", p=0.1)
    b.add("encoder", "Subgraph", label="12× encoder layer", graph="encoder_layer", repeat=12)
    b.add("cls", "Select", label="[CLS]", dim=1, index=0)
    b.add("pooler", "Linear", **{"in": dim, "out": dim})
    b.add("pooler_act", "Tanh")
    b.add("pooled", "Output")
    b.add("hidden", "Output")
    b.chain("input_ids", "word")
    if types:
        b.add("token_types", "Input", shape=["B", "T"], dtype="i64", field="token_type")
        b.add("type_emb", "Embedding", label="token_type_embeddings", num=types, dim=dim)
        b.chain("token_types", "type_emb")
        b.residual("word", "type_emb", "emb_sum")
        b.chain("emb_sum", "pos")
    else:
        b.chain("word", "pos")
    b.chain("pos", "emb_norm", "emb_drop")
    b.wire("emb_drop", "encoder.x")
    b.wire("attention_mask", "encoder.mask")
    b.chain("encoder.out", "cls", "pooler", "pooler_act", "pooled")
    b.wire("encoder.out", "hidden.in")
    if sentence:
        b.add("mean_pool", "Reduce", label="mean pooling (tokens only)", op="mean", dim=1)
        b.add("embedding", "Output", label="sentence embedding")
        b.chain("encoder.out", "mean_pool", "embedding")
        b.wire("attention_mask", "mean_pool.mask")
    return b.graph(name, {"encoder_layer": post_ln_block(masked=True, D=dim, H=heads, F=ffn, eps=eps)})


def bert_base() -> Graph:
    return _bert_like("BERT-base (uncased)", 30522, 768, 12, 3072, 512, 1e-12, 2, sentence=False)


def minilm_l12_multilingual() -> Graph:
    return _bert_like("paraphrase-multilingual-MiniLM-L12-v2", 250037, 384, 12, 1536, 512, 1e-12, 2, sentence=True)


def mpnet_base(train_triplets: bool = False) -> Graph:
    """all-mpnet-base-v2: MPNet encoder with a relative position bias shared by all layers.

    train_triplets=True adds the MNRL objective with hard negatives: (anchor, positive,
    negative) texts go through the encoder together; row i of the anchors must match row i
    of the positives against every other positive and every negative (scale 20).
    """
    b = Body()
    if train_triplets:
        b.add("triplets", "Input", label="(anchor, positive, negative) token ids", shape=["B", 3, "T"], dtype="i64", field="x")
        b.add("triplet_mask", "Input", label="1 = token, 0 = padding", shape=["B", 3, "T"], dtype="i64", field="mask")
        b.add("fold", "Flatten", label="triplets → batch", start_dim=0, end_dim=1)
        b.add("fold_mask", "Flatten", start_dim=0, end_dim=1)
        b.chain("triplets", "fold")
        b.chain("triplet_mask", "fold_mask")
        src, msrc = "fold", "fold_mask"
    else:
        b.add("input_ids", "Input", shape=["B", "T"], dtype="i64", field="x")
        b.add("attention_mask", "Input", label="1 = token, 0 = padding", shape=["B", "T"], dtype="i64", field="mask")
        src, msrc = "input_ids", "attention_mask"
    b.add("word", "Embedding", label="word_embeddings", num=30527, dim=768)
    b.add("pos", "PositionalEmbedding", label="position_embeddings (from 2)", max_len=514, dim=768, offset=2)
    b.add("emb_norm", "LayerNorm", dim=768, eps=1e-5)
    b.add("emb_drop", "Dropout", p=0.1)
    b.add("rel_bias", "RelativePositionBias", label="relative_attention_bias", heads=12, buckets=32, max_distance=128)
    b.add("encoder", "Subgraph", label="12× encoder layer", graph="mpnet_layer", repeat=12)
    b.add("cls", "Select", label="[CLS]", dim=1, index=0)
    b.add("pooler", "Linear", **{"in": 768, "out": 768})
    b.add("pooler_act", "Tanh")
    b.add("pooled", "Output")
    b.add("hidden", "Output")
    b.add("mean_pool", "Reduce", label="mean pooling (tokens only)", op="mean", dim=1)
    b.add("normalize", "L2Normalize")
    b.chain(src, "word", "pos", "emb_norm", "emb_drop")
    b.wire("emb_drop", "encoder.x")
    b.wire("emb_drop", "rel_bias")
    b.wire("rel_bias", "encoder.bias")
    b.wire(msrc, "encoder.mask")
    b.wire(msrc, "mean_pool.mask")
    b.chain("encoder.out", "cls", "pooler", "pooler_act", "pooled")
    b.wire("encoder.out", "hidden.in")
    b.chain("encoder.out", "mean_pool", "normalize")
    if train_triplets:
        b.add("unfold", "Unflatten", label="batch → triplets", dim=0, sizes=[-1, 3])
        b.add("anchor", "Select", label="anchor: edit prompt", dim=1, index=0)
        b.add("positive", "Select", label="positive: paraphrase", dim=1, index=1)
        b.add("negative", "Select", label="hard negative: other relation", dim=1, index=2)
        b.add("loss", "Contrastive", label="MNRL", scale=20.0)
        b.chain("normalize", "unfold")
        for nid in ("anchor", "positive", "negative"):
            b.chain("unfold", nid)
        b.wire("anchor", "loss.a")
        b.wire("positive", "loss.b")
        b.wire("negative", "loss.negatives")
    else:
        b.add("embedding", "Output", label="sentence embedding")
        b.chain("normalize", "embedding")
    return b.graph("all-mpnet-base-v2" + (" · MNRL training" if train_triplets else ""),
                   {"mpnet_layer": post_ln_block(rel_bias=True, masked=True, eps=1e-5)})


def _decoder_lm(name: str, vocab: int, dim: int, layers: int, block: str, block_def: dict, tied: bool, eps: float,
                args: dict | None = None, norm: str = "RMSNorm", label: str | None = None) -> Graph:
    b = Body()
    b.add("tokens", "Input", shape=["B", "T"], dtype="i64", field="x")
    b.add("embed", "Embedding", label="embed_tokens", num=vocab, dim=dim)
    b.add("layers", "Subgraph", label=label or f"{layers}× decoder layer", graph=block, repeat=layers, **({"args": args} if args else {}))
    b.add("norm", norm, label="final norm", dim=dim, eps=eps)
    b.chain("tokens", "embed")
    b.wire("embed", "layers.x")
    b.chain("layers.out", "norm")
    _lm_head(b, "embed", dim, vocab, tied, "norm")
    return b.graph(name, {block: block_def})


def gpt2_small() -> Graph:
    b = Body()
    b.add("tokens", "Input", shape=["B", "T"], dtype="i64", field="x")
    b.add("wte", "Embedding", num=50257, dim=768)
    b.add("wpe", "PositionalEmbedding", max_len=1024, dim=768)
    b.add("drop", "Dropout", p=0.1)
    b.add("blocks", "Subgraph", label="12× block", graph="gpt2_block", repeat=12)
    b.add("ln_f", "LayerNorm", dim=768, eps=1e-5)
    b.chain("tokens", "wte", "wpe", "drop")
    b.wire("drop", "blocks.x")
    b.chain("blocks.out", "ln_f")
    _lm_head(b, "wte", 768, 50257, True, "ln_f")
    return b.graph("GPT-2 small", {"gpt2_block": pre_ln_block(causal=True, act="gelu_tanh")})


def llama32_1b() -> Graph:
    scaling = {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
    return _decoder_lm("Llama 3.2 1B", 128256, 2048, 16, "llama_layer", llama_block(rope_scaling=scaling), True, 1e-5)


def qwen25_15b() -> Graph:
    return _decoder_lm("Qwen2.5-1.5B", 151936, 1536, 28, "qwen2_layer",
                       llama_block(D=1536, H=12, KV=2, F=8960, theta=1000000.0, eps=1e-6, qkv_bias=True), True, 1e-6)


def mixtral_8x7b() -> Graph:
    return _decoder_lm("Mixtral 8x7B", 32000, 4096, 32, "mixtral_layer", moe_block(), False, 1e-5)


def llama4_scout() -> Graph:
    return _decoder_lm("Llama 4 Scout (text)", 202048, 5120, 12, "llama4_group", llama4_group(), False, 1e-5,
                       label="12× group of 4 layers (48)")


def mamba_130m() -> Graph:
    return _decoder_lm("Mamba-130M", 50280, 768, 24, "mamba_block", mamba_block(), True, 1e-5, label="24× Mamba block")


def lstm_classifier() -> Graph:
    b = Body()
    b.add("tokens", "Input", shape=["B", "T"], dtype="i64", field="x")
    b.add("embed", "Embedding", num=20000, dim=128)
    b.add("lstm", "LSTM", **{"input": 128, "hidden": 256, "layers": 2, "bidirectional": True, "dropout": 0.2})
    b.add("drop", "Dropout", p=0.3)
    b.add("head", "Linear", **{"in": 512, "out": 2})
    b.add("loss", "CrossEntropy")
    b.chain("tokens", "embed", "lstm")
    b.wire("lstm.last", "drop.in")
    b.chain("drop", "head")
    b.wire("head", "loss.logits")
    return b.graph("BiLSTM text classifier")


def vae_mnist() -> Graph:
    b = Body()
    b.add("x", "Input", shape=["B", 784])
    b.add("enc", "Linear", **{"in": 784, "out": 400})
    b.add("enc_act", "ReLU")
    b.add("mu", "Linear", label="μ", **{"in": 400, "out": 20})
    b.add("logvar", "Linear", label="log σ²", **{"in": 400, "out": 20})
    b.add("z", "Reparameterize")
    b.add("dec", "Linear", **{"in": 20, "out": 400})
    b.add("dec_act", "ReLU")
    b.add("recon", "Linear", label="reconstruction logits", **{"in": 400, "out": 784})
    b.add("recon_loss", "BCEWithLogits", label="reconstruction")
    b.add("kl", "KLGaussian", label="KL to prior")
    b.chain("x", "enc", "enc_act", "mu")
    b.chain("enc_act", "logvar")
    b.wire("mu", "z.mu")
    b.wire("logvar", "z.logvar")
    b.chain("z", "dec", "dec_act", "recon")
    b.wire("recon", "recon_loss.pred")
    b.wire("x", "recon_loss.target")
    b.wire("mu", "kl.mu")
    b.wire("logvar", "kl.logvar")
    return b.graph("VAE · MNIST")


def mobilenet_v2(classes: int = 1000) -> Graph:
    b = Body()
    b.add("image", "Input", shape=["B", 3, 224, 224])
    b.add("stem", "Conv2d", in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
    b.add("stem_bn", "BatchNorm2d", features=32)
    b.add("stem_act", "ReLU6")
    # the first block has expansion 1: depthwise straight on the stem
    b.add("b0_dw", "Conv2d", label="depthwise 3×3", in_channels=32, out_channels=32, kernel_size=3, padding=1, groups=32, bias=False)
    b.add("b0_bn1", "BatchNorm2d", features=32)
    b.add("b0_act", "ReLU6")
    b.add("b0_project", "Conv2d", in_channels=32, out_channels=16, kernel_size=1, padding=0, bias=False)
    b.add("b0_bn2", "BatchNorm2d", features=16)
    b.chain("image", "stem", "stem_bn", "stem_act", "b0_dw", "b0_bn1", "b0_act", "b0_project", "b0_bn2")
    prev, cin = "b0_bn2", 16
    for i, (t, c, n, st) in enumerate([(6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)], 1):
        nid = f"stage{i}"
        b.add(nid, "Subgraph", label=f"stage {i} · {cin}→{c}" + (f" /{st}" if st > 1 else ""), graph="ir_proj", args={"Cin": cin, "Cout": c, "S": st, "T": t})
        b.wire(prev, f"{nid}.x")
        prev = f"{nid}.out"
        if n > 1:
            rid = f"stage{i}_res"
            b.add(rid, "Subgraph", label=f"stage {i} · {n - 1}× residual {c}", graph="ir", repeat=n - 1, args={"C": c, "T": t})
            b.wire(prev, f"{rid}.x")
            prev = f"{rid}.out"
        cin = c
    b.add("head", "Conv2d", in_channels=320, out_channels=1280, kernel_size=1, padding=0, bias=False)
    b.add("head_bn", "BatchNorm2d", features=1280)
    b.add("head_act", "ReLU6")
    b.add("pool", "AdaptiveAvgPool2d", output_size=1)
    b.add("flat", "Flatten")
    b.add("drop", "Dropout", p=0.2)
    b.add("fc", "Linear", **{"in": 1280, "out": classes})
    b.add("loss", "CrossEntropy")
    b.chain(prev, "head", "head_bn", "head_act", "pool", "flat", "drop", "fc")
    b.wire("fc", "loss.logits")
    return b.graph("MobileNetV2", {"ir": inverted_residual(), "ir_proj": inverted_residual_proj()})


def vqvae_mnist() -> Graph:
    b = Body()
    b.add("x", "Input", shape=["B", 1, 28, 28])
    b.add("enc1", "Conv2d", in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1)
    b.add("enc1_act", "ReLU")
    b.add("enc2", "Conv2d", in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
    b.add("enc2_act", "ReLU")
    b.add("to_latent", "Conv2d", label="to code space", in_channels=64, out_channels=64, kernel_size=1, padding=0)
    b.add("channels_last", "Permute", dims=[0, 2, 3, 1])
    b.add("vq", "VectorQuantizer", label="codebook · 512 codes", codes=512, dim=64, beta=0.25)
    b.add("channels_first", "Permute", dims=[0, 3, 1, 2])
    b.add("dec1", "ConvTranspose2d", in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
    b.add("dec1_act", "ReLU")
    b.add("dec2", "ConvTranspose2d", label="reconstruction logits", in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)
    b.add("recon", "BCEWithLogits", label="reconstruction")
    b.add("commit", "AuxLoss", label="codebook + commitment")
    b.chain("x", "enc1", "enc1_act", "enc2", "enc2_act", "to_latent", "channels_last", "vq")
    b.wire("vq.out", "channels_first.in")
    b.chain("channels_first", "dec1", "dec1_act", "dec2")
    b.wire("dec2", "recon.pred")
    b.wire("x", "recon.target")
    b.wire("vq.commit", "commit.in")
    return b.graph("VQ-VAE · MNIST")


def ddpm_mnist() -> Graph:
    b = Body()
    b.add("x", "Input", shape=["B", 784])
    b.add("to_signed", "Scale", label="× 2", factor=2.0)
    b.add("center", "AddConst", label="− 1 → [−1, 1]", value=-1.0)
    b.add("diffuse", "DiffusionForward", steps=1000)
    b.add("in_proj", "Linear", **{"in": 784, "out": 512})
    b.add("t_embed", "TimestepEmbedding", dim=512)
    b.add("cond", "Add", label="+ timestep")
    b.add("act1", "SiLU")
    b.add("mid", "Linear", **{"in": 512, "out": 512})
    b.add("act2", "SiLU")
    b.add("eps", "Linear", label="predicted noise", **{"in": 512, "out": 784})
    b.add("loss", "MSE", label="noise prediction")
    b.chain("x", "to_signed", "center", "diffuse")
    b.wire("diffuse.noisy", "in_proj.in")
    b.wire("diffuse.t", "t_embed.in")
    b.wire("in_proj", "cond.a")
    b.wire("t_embed", "cond.b")
    b.chain("cond", "act1", "mid", "act2", "eps")
    b.wire("eps", "loss.pred")
    b.wire("diffuse.noise", "loss.target")
    return b.graph("DDPM denoiser · MNIST")


def seq2seq_transformer() -> Graph:
    b = Body()
    b.add("src", "Input", label="source tokens", shape=["B", "S"], dtype="i64", field="x")
    b.add("tgt", "Input", label="target tokens (shifted)", shape=["B", "T"], dtype="i64", field="tgt")
    b.add("src_emb", "Embedding", num=37000, dim=512)
    b.add("tgt_emb", "Embedding", num=37000, dim=512)
    b.add("src_scale", "Scale", factor=math.sqrt(512))
    b.add("tgt_scale", "Scale", factor=math.sqrt(512))
    b.add("src_pos", "SinusoidalPositional", dim=512)
    b.add("tgt_pos", "SinusoidalPositional", dim=512)
    b.add("encoder", "Subgraph", label="6× encoder layer", graph="encoder_layer", repeat=6)
    b.add("decoder", "Subgraph", label="6× decoder layer", graph="decoder_layer", repeat=6)
    b.add("generator", "Linear", **{"in": 512, "out": 37000, "bias": False, "tie_to": "tgt_emb"})
    b.add("loss", "CrossEntropy", label_smoothing=0.1)
    b.chain("src", "src_emb", "src_scale", "src_pos")
    b.chain("tgt", "tgt_emb", "tgt_scale", "tgt_pos")
    b.wire("src_pos", "encoder.x")
    b.wire("tgt_pos", "decoder.x")
    b.wire("encoder.out", "decoder.memory")
    b.chain("decoder.out", "generator")
    b.wire("generator", "loss.logits")
    return b.graph("Transformer (Vaswani base)", {
        "encoder_layer": post_ln_block(D=512, H=8, F=2048, eps=1e-5, act="relu"),
        "decoder_layer": decoder_block(),
    })


# ================================================================= catalogue


@dataclass
class Template:
    id: str
    name: str
    kind: str  # "model" | "block"
    category: str
    doc: str
    build: Any
    reference: int | None = None  # published parameter count, checked by tests
    source: str = ""
    tags: list[str] = field(default_factory=list)


MODELS = [
    Template("mlp_mnist", "MLP · MNIST", "model", "vision", "Two-layer perceptron for 28×28 digits.", mlp_mnist),
    Template("lenet5", "LeNet-5", "model", "vision", "LeCun et al. 1998: conv-pool-conv-pool and three dense layers.", lenet5, 61_706, "LeCun et al. 1998"),
    Template("resnet18", "ResNet-18", "model", "vision", "He et al. 2015: 4 stages of basic residual blocks, 224×224, 1000 classes.", resnet18, 11_689_512, "torchvision resnet18"),
    Template("unet_small", "U-Net (small)", "model", "vision", "Encoder-decoder with skip connections for segmentation masks.", unet_small),
    Template("mobilenet_v2", "MobileNetV2", "model", "vision", "Sandler et al. 2018: inverted residuals with depthwise convs and ReLU6, 224×224.", mobilenet_v2, 3_504_872, "torchvision mobilenet_v2"),
    Template("vit_b16", "ViT-B/16", "model", "vision", "Dosovitskiy et al. 2020: 16×16 patches, [CLS] token, 12 pre-LN blocks.", vit_b16, 86_567_656, "torchvision vit_b_16"),
    Template("bert_base", "BERT-base", "model", "language", "Devlin et al. 2018: 12 post-LN encoder layers, token types, pooler.", bert_base, 109_482_240, "bert-base-uncased"),
    Template("minilm_l12", "MiniLM-L12 multilingual", "model", "language", "paraphrase-multilingual-MiniLM-L12-v2: 384-dim sentence encoder, mean pooling.", minilm_l12_multilingual, 117_653_760, "sentence-transformers"),
    Template("mpnet_base", "MPNet-base (all-mpnet-base-v2)", "model", "language", "Sentence encoder with a shared relative position bias, mean pooling, L2-normalised.", mpnet_base, 109_486_464, "sentence-transformers"),
    Template("gpt2_small", "GPT-2 small", "model", "language", "Radford et al. 2019: 12 pre-LN causal blocks, learned positions, tied head.", gpt2_small, 124_439_808, "openai-community/gpt2"),
    Template("llama32_1b", "Llama 3.2 1B", "model", "language", "16 layers, GQA 32/8, RoPE θ=5e5, SwiGLU, tied embeddings.", llama32_1b, 1_235_814_400, "meta-llama/Llama-3.2-1B"),
    Template("qwen25_15b", "Qwen2.5-1.5B", "model", "language", "28 layers, GQA 12/2 with q/k/v bias, RoPE θ=1e6, SwiGLU, tied embeddings.", qwen25_15b, 1_543_714_304, "Qwen/Qwen2.5-1.5B"),
    Template("mixtral_8x7b", "Mixtral 8x7B", "model", "language", "32 layers, 8 SwiGLU experts, top-2 softmax routing.", mixtral_8x7b, 46_702_792_704, "mistralai/Mixtral-8x7B-v0.1"),
    Template("llama4_scout", "Llama 4 Scout (text)", "model", "language", "48 layers in iRoPE groups of 4, 16 experts top-1 plus a shared expert.", llama4_scout, None, "computed; the release reports ~109B with the vision encoder"),
    Template("mamba_130m", "Mamba-130M", "model", "sequence", "Gu & Dao 2023: 24 selective state-space blocks, no attention.", mamba_130m, 129_135_360, "state-spaces/mamba-130m"),
    Template("lstm_classifier", "BiLSTM classifier", "model", "sequence", "Embedding → 2-layer bidirectional LSTM → final state → classes.", lstm_classifier),
    Template("seq2seq", "Transformer (Vaswani base)", "model", "sequence", "Vaswani et al. 2017: 6+6 layers, d=512, cross-attention, sinusoidal positions.", seq2seq_transformer),
    Template("vqvae_mnist", "VQ-VAE · MNIST", "model", "generative", "van den Oord et al. 2017: conv encoder, 512-code codebook, straight-through, conv decoder.", vqvae_mnist),
    Template("ddpm_mnist", "DDPM denoiser · MNIST", "model", "generative", "Ho et al. 2020: noise images at a random step, predict the noise from (x_t, t).", ddpm_mnist),
    Template("vae_mnist", "VAE · MNIST", "model", "generative", "Kingma & Welling 2013: Gaussian latent, reparameterisation, BCE + KL.", vae_mnist),
]

BLOCKS = [
    Template("pre_ln_block", "Transformer block (pre-LN)", "block", "attention", "GPT-2 / ViT layer.", pre_ln_block),
    Template("post_ln_block", "Transformer block (post-LN)", "block", "attention", "BERT layer.", post_ln_block),
    Template("llama_block", "Llama / Qwen layer", "block", "attention", "RMSNorm, GQA with RoPE, SwiGLU.", llama_block),
    Template("moe_block", "Mixtral MoE layer", "block", "attention", "Llama layer with a top-k routed expert MLP.", moe_block),
    Template("llama4_group", "Llama 4 layer group", "block", "attention", "3 RoPE/chunked + 1 NoPE layer, sigmoid MoE with shared expert.", llama4_group),
    Template("decoder_block", "Transformer decoder layer", "block", "attention", "Self-attention, cross-attention to a memory, MLP.", decoder_block),
    Template("resnet_basic", "ResNet basic block", "block", "convolution", "Two 3×3 conv-BN, identity shortcut.", resnet_basic),
    Template("resnet_basic_down", "ResNet downsample block", "block", "convolution", "Stride-2 basic block with a 1×1 shortcut.", resnet_basic_down),
    Template("double_conv", "U-Net double conv", "block", "convolution", "(conv → BN → ReLU) × 2.", double_conv),
    Template("inverted_residual", "MobileNetV2 inverted residual", "block", "convolution", "1×1 expand, 3×3 depthwise, 1×1 project, shortcut.", inverted_residual),
    Template("inverted_residual_proj", "MobileNetV2 inverted residual (stride / channels)", "block", "convolution", "Same without the shortcut; changes channels or stride.", inverted_residual_proj),
    Template("mamba_block", "Mamba block", "block", "sequence", "x + Mamba(RMSNorm(x)).", mamba_block),
]


def export() -> list[dict[str, Any]]:
    """JSON for the canvas library (scripts/gen_fixtures.py writes it to web/src/ir/templates.json)."""
    out = []
    for t in MODELS + BLOCKS:
        payload = t.build()
        out.append({
            "id": t.id, "name": t.name, "kind": t.kind, "category": t.category, "doc": t.doc,
            "reference": t.reference, "source": t.source,
            ("graph" if t.kind == "model" else "block"): copy.deepcopy(payload),
        })
    return out
