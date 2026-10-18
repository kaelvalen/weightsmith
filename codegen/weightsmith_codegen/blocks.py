"""Source of the reusable building blocks emitted into model.py.

Only the blocks a graph uses are emitted, so every model.py is self-contained:
it runs without Weightsmith installed.
"""

LORA_LINEAR = '''
class LoRALinear(nn.Module):
    """y = base(x) + (alpha / r) · B A x.  B starts at zero, so training starts from the base model."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0, freeze: bool = True, init: str = "zeros_b"):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.init = init
        self.reset_adapter()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.enabled = True  # False: behave exactly like the base layer
        if freeze:
            self.base.requires_grad_(False)

    @torch.no_grad()
    def reset_adapter(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if self.init == "gaussian":
            nn.init.normal_(self.lora_B, std=1e-3)
        else:
            self.lora_B.zero_()

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale

    def merged(self) -> nn.Linear:
        """A plain Linear with the low-rank update folded into its weight."""
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            out.weight.copy_(self.base.weight + (self.lora_B @ self.lora_A) * self.scale)
            if self.base.bias is not None:
                out.bias.copy_(self.base.bias)
        return out
'''

BOTTLENECK_ADAPTER = '''
class BottleneckAdapter(nn.Module):
    """y = base(x);  y + up(gelu(down(y))).  up starts at zero."""

    def __init__(self, base: nn.Module, features: int, bottleneck: int, freeze: bool = True):
        super().__init__()
        self.base = base
        self.down = nn.Linear(features, bottleneck)
        self.up = nn.Linear(bottleneck, features)
        self.reset_adapter()
        self.enabled = True
        if freeze:
            self.base.requires_grad_(False)

    @torch.no_grad()
    def reset_adapter(self):
        self.down.reset_parameters()
        self.up.weight.zero_()
        self.up.bias.zero_()

    def forward(self, *args, **kwargs):
        y = self.base(*args, **kwargs)
        return y + self.up(F.gelu(self.down(y))) if self.enabled else y
'''

ATTENTION = '''
class Attention(nn.Module):
    """Multi-head self-attention over B×T×D.

    Grouped-query when kv_heads < heads; RoPE (rotate-half) or none; per-head QK-norm
    ("rms" learned, "l2" plain); chunked local attention; optional additive bias heads×T×T.
    """

    def __init__(self, dim: int, heads: int, kv_heads: int | None = None, causal: bool = False, dropout: float = 0.0,
                 bias: bool = True, o_bias: bool = True, rope: bool = False, rope_theta: float = 10000.0,
                 rope_scaling: dict | None = None, qk_norm: str = "none", chunk: int = 0, nope_temp: bool = False):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads or heads
        self.head_dim = dim // heads
        self.causal, self.dropout, self.rope, self.qk_norm = causal, dropout, rope, qk_norm
        self.chunk, self.nope_temp = chunk, nope_temp
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(dim, dim, bias=o_bias)
        if qk_norm == "rms":
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        if rope:
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
            self.register_buffer("inv_freq", self._scaled(inv_freq, rope_scaling or {}), persistent=False)

    @staticmethod
    def _scaled(inv_freq, cfg: dict):
        """RoPE frequency scaling: "linear" divides all frequencies, "llama3" only the low ones (smoothly)."""
        kind = cfg.get("type") or cfg.get("rope_type")
        if kind == "linear":
            return inv_freq / cfg["factor"]
        if kind != "llama3":
            return inv_freq
        factor, low, high = cfg["factor"], cfg.get("low_freq_factor", 1.0), cfg.get("high_freq_factor", 4.0)
        old = cfg.get("original_max_position_embeddings", 8192)
        wavelen = 2 * math.pi / inv_freq
        scaled = torch.where(wavelen > old / low, inv_freq / factor, inv_freq)
        smooth = (old / wavelen - low) / (high - low)
        smoothed = (1 - smooth) * scaled / factor + smooth * scaled
        medium = (wavelen >= old / high) & (wavelen <= old / low)
        return torch.where(medium, smoothed, scaled)

    @staticmethod
    def _rotate(x, cos, sin):
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin

    def forward(self, x, bias=None, mask=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm == "rms":
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope:
            angles = torch.outer(torch.arange(T, device=x.device, dtype=torch.float32), self.inv_freq)
            angles = torch.cat([angles, angles], dim=-1)
            cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
            q, k = self._rotate(q, cos, sin), self._rotate(k, cos, sin)
        if self.qk_norm == "l2":
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6)
        if self.nope_temp and not self.rope:  # Llama 4: sharpen attention far into long contexts
            pos = torch.arange(T, device=x.device, dtype=torch.float32)
            q = q * (torch.log(torch.floor((pos + 1) / 8192) + 1) * 0.1 + 1).view(1, 1, T, 1).to(q.dtype)
        if self.kv_heads != self.heads:
            k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
            v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        attn_mask = None
        if bias is not None or mask is not None or (self.chunk and T > self.chunk):
            i = torch.arange(T, device=x.device)
            allowed = torch.ones(T, T, dtype=torch.bool, device=x.device)
            if self.causal:
                allowed &= i[:, None] >= i[None, :]
            if self.chunk:
                allowed &= (i[:, None] // self.chunk) == (i[None, :] // self.chunk)
            attn_mask = torch.zeros(T, T, dtype=q.dtype, device=x.device).masked_fill(~allowed, float("-inf"))
            if bias is not None:
                attn_mask = attn_mask + bias.to(q.dtype)
            if mask is not None:  # padding keys are never attended to
                attn_mask = attn_mask + torch.zeros(B, 1, 1, T, dtype=q.dtype, device=x.device).masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=self.causal and attn_mask is None,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, D))
'''

CROSS_ATTENTION = '''
class CrossAttention(nn.Module):
    """Queries from x (B×T×D), keys and values from context (B×S×Dc)."""

    def __init__(self, dim: int, heads: int, context_dim: int | None = None, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        cdim = context_dim or dim
        self.heads, self.dropout = heads, dropout
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(cdim, dim, bias=bias)
        self.v_proj = nn.Linear(cdim, dim, bias=bias)
        self.o_proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x, context):
        B, T, D = x.shape
        S = context.shape[1]
        q = self.q_proj(x).view(B, T, self.heads, -1).transpose(1, 2)
        k = self.k_proj(context).view(B, S, self.heads, -1).transpose(1, 2)
        v = self.v_proj(context).view(B, S, self.heads, -1).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, D))
'''

MOE = '''
class MoE(nn.Module):
    """SwiGLU experts behind a learned router.

    softmax: weights = softmax over the top-k logits, applied to the experts' outputs (Mixtral).
    sigmoid: weight = sigmoid(logit), applied to the expert's input (Llama 4).
    shared_hidden > 0 adds an always-on shared expert.
    """

    def __init__(self, dim: int, hidden: int, num_experts: int, top_k: int = 2, router: str = "softmax", shared_hidden: int = 0):
        super().__init__()
        self.num_experts, self.top_k, self.mode = num_experts, top_k, router
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = Experts(num_experts, dim, hidden)
        self.shared_expert = GatedMLP(dim, shared_hidden, dim) if shared_hidden else None

    def forward(self, x):
        shape = x.shape
        h = x.reshape(-1, shape[-1])
        logits = self.router(h)
        top_val, top_idx = logits.topk(self.top_k, dim=-1)
        if self.mode == "softmax":
            w = torch.softmax(top_val.float(), dim=-1).to(h.dtype)
        else:
            w = torch.sigmoid(top_val.float()).to(h.dtype)
        gates = torch.zeros_like(logits).scatter(1, top_idx, w).t()  # E × N
        out = self.shared_expert(h) if self.shared_expert is not None else torch.zeros_like(h)
        if h.is_meta:  # shape-only run: every expert on every token
            if self.mode == "softmax":
                y = (self.experts.all(h.expand(self.num_experts, -1, -1)) * gates.unsqueeze(-1)).sum(0)
            else:
                y = self.experts.all(h.unsqueeze(0) * gates.unsqueeze(-1)).sum(0)
            return (out + y).view(shape)
        for e in range(self.num_experts):  # each expert runs only on its own tokens
            idx = torch.nonzero(gates[e] > 0, as_tuple=True)[0]
            if not idx.numel():
                continue
            g = gates[e, idx, None]
            y = self.experts.one(e, h[idx] * g) if self.mode == "sigmoid" else self.experts.one(e, h[idx]) * g
            out = out.index_add(0, idx, y)
        return out.view(shape)


class Experts(nn.Module):
    """All experts' SwiGLU weights stacked: gate_up_proj E×D×2H, down_proj E×H×D."""

    def __init__(self, num_experts: int, dim: int, hidden: int):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, dim, 2 * hidden) * dim**-0.5)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, dim) * hidden**-0.5)

    def one(self, e: int, x):
        gate, up = (x @ self.gate_up_proj[e]).chunk(2, dim=-1)
        return (F.silu(gate) * up) @ self.down_proj[e]

    def all(self, x):  # x: E × N × D
        gate, up = torch.bmm(x, self.gate_up_proj).chunk(2, dim=-1)
        return torch.bmm(F.silu(gate) * up, self.down_proj)
'''

MAMBA = '''
class Mamba(nn.Module):
    """Selective state-space mixer (Mamba-1), with a plain sequential scan."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: int | None = None):
        super().__init__()
        e = expand * dim
        self.dt_rank = dt_rank or math.ceil(dim / 16)
        self.d_state = d_state
        self.in_proj = nn.Linear(dim, 2 * e, bias=False)
        self.conv1d = nn.Conv1d(e, e, d_conv, groups=e, padding=d_conv - 1)
        self.x_proj = nn.Linear(e, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, e)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(e, 1))
        self.D = nn.Parameter(torch.ones(e))
        self.out_proj = nn.Linear(e, dim, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv1d(u.transpose(1, 2))[..., :T].transpose(1, 2))
        dt, Bm, Cm = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # B × T × E
        A = -torch.exp(self.A_log.float())  # E × N
        y = self._scan(dt.float(), A, Bm.float(), Cm.float(), u.float()).to(x.dtype) + u * self.D
        return self.out_proj(y * F.silu(z))

    @staticmethod
    def _scan(dt, A, Bm, Cm, u, chunk: int = 64):
        """h_t = exp(dt_t·A)·h_{t-1} + dt_t·B_t·u_t,  y_t = C_t·h_t.

        The exponentials and products are computed for a chunk of steps at once; only the
        recurrence itself runs step by step (a fused CUDA kernel would remove that loop too).
        """
        B, T, E = u.shape
        h = torch.zeros(B, E, A.shape[1], device=u.device, dtype=u.dtype)
        ys = []
        for c0 in range(0, T, chunk):
            sl = slice(c0, min(T, c0 + chunk))
            dA = torch.exp(dt[:, sl, :, None] * A)  # B × L × E × N
            dBu = dt[:, sl, :, None] * Bm[:, sl, None, :] * u[:, sl, :, None]
            hs = []
            for t in range(dA.shape[1]):
                h = dA[:, t] * h + dBu[:, t]
                hs.append(h)
            ys.append((torch.stack(hs, 1) * Cm[:, sl, None, :]).sum(-1))
        return torch.cat(ys, dim=1)
'''

RECURRENT = '''
class Recurrent(nn.Module):
    """LSTM / GRU / RNN, batch-first. Returns (every step, final step of the last layer)."""

    def __init__(self, kind: str, input: int, hidden: int, layers: int = 1, bidirectional: bool = False,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        cls = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}[kind]
        self.dirs = 2 if bidirectional else 1
        self.rnn = cls(input, hidden, layers, bias=bias, batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        out, h = self.rnn(x)
        h = h[0] if isinstance(h, tuple) else h  # LSTM returns (h, c)
        last = h[-self.dirs:].transpose(0, 1).reshape(x.shape[0], -1)
        return out, last
'''

SINUSOIDAL = '''
class SinusoidalPositional(nn.Module):
    """x + fixed sine/cosine position encodings (Vaswani et al.)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device, dtype=torch.float32)[:, None]
        div = torch.exp(torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32) * (-math.log(10000.0) / self.dim))
        pe = torch.zeros(T, self.dim, device=x.device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return x + pe.to(x.dtype)
'''

CLASS_TOKEN = '''
class ClassToken(nn.Module):
    """Prepends a learned [CLS] token."""

    def __init__(self, dim: int):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return torch.cat([self.token.expand(x.shape[0], -1, -1), x], dim=1)
'''

PATCH_EMBED = '''
class PatchEmbed(nn.Module):
    """B×C×H×W → B×(H/p·W/p)×D with one strided convolution."""

    def __init__(self, in_channels: int, dim: int, patch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)
'''

RELATIVE_BIAS = '''
class RelativePositionBias(nn.Module):
    """Bucketed relative position bias (T5 / MPNet): one table, heads×T×T, shared by all layers."""

    def __init__(self, heads: int, buckets: int = 32, max_distance: int = 128, bidirectional: bool = True):
        super().__init__()
        self.buckets, self.max_distance, self.bidirectional = buckets, max_distance, bidirectional
        self.table = nn.Embedding(buckets, heads)

    def _bucket(self, rel):
        n, out = self.buckets, torch.zeros_like(rel)
        if self.bidirectional:
            n //= 2
            out += (rel > 0).long() * n
            rel = rel.abs()
        else:
            rel = -torch.clamp(rel, max=0)
        exact = n // 2
        large = exact + (torch.log(rel.float().clamp(min=1) / exact) / math.log(self.max_distance / exact) * (n - exact)).long()
        return out + torch.where(rel < exact, rel, large.clamp(max=n - 1))

    def forward(self, x):
        T = x.shape[1]
        i = torch.arange(T, device=x.device)
        return self.table(self._bucket(i[None, :] - i[:, None])).permute(2, 0, 1)
'''

DROP_PATH = '''
class DropPath(nn.Module):
    """Stochastic depth: zero the whole branch for a random subset of samples."""

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or not self.p:
            return x
        keep = torch.rand(x.shape[0], *([1] * (x.ndim - 1)), device=x.device) >= self.p
        return x * keep / (1 - self.p)
'''

MLP = '''
class MLP(nn.Module):
    """fc2(act(fc1(x)))."""

    def __init__(self, dim: int, hidden: int, out: int, act: str = "gelu", dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = {"gelu": nn.GELU(), "gelu_tanh": nn.GELU(approximate="tanh"), "relu": nn.ReLU(), "silu": nn.SiLU(), "tanh": nn.Tanh()}[act]
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, out)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))
'''

GATED_MLP = '''
class GatedMLP(nn.Module):
    """down(act(gate(x)) · up(x)) — SwiGLU with act=silu."""

    def __init__(self, dim: int, hidden: int, out: int, act: str = "silu", bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=bias)
        self.up_proj = nn.Linear(dim, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, out, bias=bias)
        self.act = {"silu": F.silu, "gelu": F.gelu}[act]

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
'''

POSITIONAL_EMBEDDING = '''
class PositionalEmbedding(nn.Module):
    """x + a learned vector per position."""

    def __init__(self, max_len: int, dim: int, offset: int = 0):
        super().__init__()
        self.offset = offset
        self.weight = nn.Parameter(torch.zeros(max_len, dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return x + self.weight[self.offset : self.offset + x.shape[1]]
'''

SWISH = '''
class Swish(nn.Module):
    """x · sigmoid(β·x), β learned per channel (starts at 1 = SiLU)."""

    def __init__(self, dim: int):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
'''

LEARNED_TENSOR = '''
class LearnedTensor(nn.Module):
    """A free learned tensor (queries, prompts, latents)."""

    def __init__(self, shape: tuple, init: str = "normal", std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(shape))
        if init == "normal":
            nn.init.normal_(self.weight, std=std)
        elif init == "ones":
            nn.init.ones_(self.weight)

    def forward(self):
        return self.weight
'''

LAYER_SCALE = '''
class LayerScale(nn.Module):
    """x · γ, γ per channel (last dim), initialised small so a new branch starts near identity."""

    def __init__(self, dim: int, init: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward(self, x):
        return x * self.gamma
'''

SQUEEZE_EXCITE = '''
class SqueezeExcite(nn.Module):
    """Channel gate: average over the spatial dims, bottleneck MLP, sigmoid, rescale channels."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = x.mean(dim=tuple(range(2, x.ndim)))
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s.view(*s.shape, *([1] * (x.ndim - 2)))
'''

ALIBI = '''
class ALiBi(nn.Module):
    """Attention with linear biases (Press et al.): −slope_h · |i − j| per head, → heads×T×T."""

    def __init__(self, heads: int, causal: bool = True):
        super().__init__()
        self.causal = causal
        n = 2 ** math.floor(math.log2(heads))
        slopes = [2 ** (-8 * (i + 1) / n) for i in range(n)]
        if n < heads:  # non-power-of-two: interleave the next power's slopes
            extra = [2 ** (-4 * (2 * i + 1) / n) for i in range(heads - n)]
            slopes += extra
        self.register_buffer("slopes", torch.tensor(slopes).view(heads, 1, 1), persistent=False)

    def forward(self, x):
        T = x.shape[1]
        i = torch.arange(T, device=x.device)
        dist = (i[None, :] - i[:, None]).abs().float()
        bias = -self.slopes * dist
        if self.causal:
            bias = bias.masked_fill(i[None, :] > i[:, None], float("-inf"))
        return bias
'''

SLIDING_WINDOW = '''
class SlidingWindowBias(nn.Module):
    """0 inside the window, −inf outside: each query sees at most `window` keys around it."""

    def __init__(self, heads: int, window: int, causal: bool = True):
        super().__init__()
        self.heads, self.window, self.causal = heads, window, causal

    def forward(self, x):
        T = x.shape[1]
        i = torch.arange(T, device=x.device)
        d = i[:, None] - i[None, :]
        allowed = (d < self.window) & ((d >= 0) if self.causal else (d > -self.window))
        bias = torch.zeros(T, T, device=x.device).masked_fill(~allowed, float("-inf"))
        return bias.expand(self.heads, T, T)
'''

LINEAR_ATTENTION = '''
class LinearAttention(nn.Module):
    """Kernelised attention (Katharopoulos et al.): φ(q)·(φ(k)ᵀv) with φ = elu + 1, linear in T."""

    def __init__(self, dim: int, heads: int = 8, causal: bool = False):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        h = lambda t: t.view(B, T, self.heads, D // self.heads).transpose(1, 2)  # noqa: E731
        q, k, v = F.elu(h(self.q_proj(x))) + 1, F.elu(h(self.k_proj(x))) + 1, h(self.v_proj(x))
        if self.causal:
            kv = torch.einsum("bhtd,bhte->bhtde", k, v).cumsum(2)
            num = torch.einsum("bhtd,bhtde->bhte", q, kv)
            den = torch.einsum("bhtd,bhtd->bht", q, k.cumsum(2)).unsqueeze(-1)
        else:
            num = q @ (k.transpose(-2, -1) @ v)
            den = (q @ k.sum(2, keepdim=True).transpose(-2, -1))
        y = num / (den + 1e-6)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, D))
'''

TIMESTEP_EMBEDDING = '''
class TimestepEmbedding(nn.Module):
    """Diffusion timestep → sinusoidal features → MLP (DDPM / DiT)."""

    def __init__(self, dim: int, hidden: int | None = None, max_period: float = 10000.0):
        super().__init__()
        hidden = hidden or 4 * dim
        self.dim, self.max_period = dim, max_period
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        return self.fc2(F.silu(self.fc1(emb.to(self.fc1.weight.dtype))))
'''

VECTOR_QUANTIZER = '''
class VectorQuantizer(nn.Module):
    """VQ-VAE codebook (van den Oord et al.): nearest code, straight-through gradient.

    Returns (quantised, code indices, codebook + β·commitment loss).
    """

    def __init__(self, codes: int, dim: int, beta: float = 0.25):
        super().__init__()
        self.beta = beta
        self.codebook = nn.Embedding(codes, dim)
        nn.init.uniform_(self.codebook.weight, -1 / codes, 1 / codes)

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        w = self.codebook.weight
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ w.T + w.pow(2).sum(1)[None]
        idx = dist.argmin(1)
        q = self.codebook(idx).view_as(x)
        loss = F.mse_loss(q, x.detach()) + self.beta * F.mse_loss(x, q.detach())
        return x + (q - x).detach(), idx.view(x.shape[:-1]), loss
'''

DIFFUSION_FORWARD = '''
class DiffusionForward(nn.Module):
    """DDPM forward (noising) process with a fixed schedule; returns (x_t, t, ε)."""

    def __init__(self, steps: int = 1000, schedule: str = "linear", beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        if schedule == "cosine":  # Nichol & Dhariwal 2021
            s = torch.arange(steps + 1, dtype=torch.float64) / steps
            f = torch.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
        else:
            betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
        self.steps = steps
        self.register_buffer("alphas_cumprod", torch.cumprod(1 - betas, 0).float(), persistent=False)

    def forward(self, x):
        t = torch.randint(0, self.steps, (x.shape[0],), device=x.device)
        noise = torch.randn_like(x)
        a = self.alphas_cumprod[t].view(-1, *([1] * (x.ndim - 1))).to(x.dtype)
        return a.sqrt() * x + (1 - a).sqrt() * noise, t, noise
'''

GRAPH_CONV = '''
class GraphConv(nn.Module):
    """Kipf & Welling GCN layer: D^-½ (A + I) D^-½ X W."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        a = adj + torch.eye(adj.shape[-1], device=adj.device, dtype=adj.dtype)
        d = a.sum(-1).clamp(min=1e-6).rsqrt()
        a = d[..., :, None] * a * d[..., None, :]
        return a.to(x.dtype) @ self.lin(x)
'''

BLOCKS = {
    "CrossAttention": CROSS_ATTENTION,
    "MoE": MOE,
    "Mamba": MAMBA,
    "Recurrent": RECURRENT,
    "SinusoidalPositional": SINUSOIDAL,
    "ClassToken": CLASS_TOKEN,
    "PatchEmbed": PATCH_EMBED,
    "RelativePositionBias": RELATIVE_BIAS,
    "DropPath": DROP_PATH,
    "LoRALinear": LORA_LINEAR,
    "BottleneckAdapter": BOTTLENECK_ADAPTER,
    "Attention": ATTENTION,
    "MLP": MLP,
    "GatedMLP": GATED_MLP,
    "PositionalEmbedding": POSITIONAL_EMBEDDING,
    "Swish": SWISH,
    "LearnedTensor": LEARNED_TENSOR,
    "LayerScale": LAYER_SCALE,
    "SqueezeExcite": SQUEEZE_EXCITE,
    "ALiBi": ALIBI,
    "SlidingWindowBias": SLIDING_WINDOW,
    "LinearAttention": LINEAR_ATTENTION,
    "TimestepEmbedding": TIMESTEP_EMBEDDING,
    "VectorQuantizer": VECTOR_QUANTIZER,
    "GraphConv": GRAPH_CONV,
    "DiffusionForward": DIFFUSION_FORWARD,
}
