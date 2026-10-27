"""Episode memory for System 2 (QwenGR00T_DualSystem dual_system.memory).

A small recurrent transformer over the VLM's last hidden states: M_t = MemoryTransformer(M_{t-1}, H_t) with
Q = the memory tokens, K / V = [memory tokens; projected VLM tokens]. Training unrolls it over a few past VLM frames
(dual_system.memory.update_offsets, e.g. every 0.5 s over the last 2 s) from the learned initial memory; at
deployment it is updated on every new VLM frame from the episode start (memory_init / memory_update).

Readout (-> System 2's cross-attention context): the memory tokens and one progress token. Progress = episode
progress t / (T - 1) classified into num_bins equal bins from the mean-pooled memory; the softmax over the bins goes
through an MLP into the progress token.

attention: "standard" (nn.MultiheadAttention) or "gated" (GatedAttention: a sigmoid output gate on every head
channel, computed from the memory tokens, as in the full-attention layers of Qwen3-Next / Qwen3.5).

token_dim: an MLP (token_resize) first maps every VLM token from vlm_dim to token_dim, and the rest of the memory works on
token_dim tokens, so VLMs of different hidden sizes (Qwen3.5-2B 2048, -4B 2560) share one memory architecture; the
readout width ctx_dim is then usually token_dim as well."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_ATTENTIONS = ("standard", "gated")


class GatedAttention(nn.Module):
    """Multi-head attention with a query-dependent elementwise sigmoid gate on the attention output, before the output
    projection (Qiu et al. 2025, "Gated Attention for Large Language Models"):
        out = W_o [ SDPA(W_q x_q, W_k x_kv, W_v x_kv) * sigmoid(W_g x_q) ]
    The gate (one value per head channel) lets each memory token scale down / switch off what it reads this step."""

    def __init__(self, d: int, num_heads: int):
        super().__init__()
        if d % num_heads:
            raise ValueError(f"GatedAttention: d_model {d} not divisible by num_heads {num_heads}")
        self.num_heads, self.head_dim = int(num_heads), d // int(num_heads)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.gate_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """q [B, N, d], kv [B, L, d], key_padding_mask [B, L] bool (True = ignore, as nn.MultiheadAttention)."""
        b, n, d = q.shape
        heads = lambda t: t.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        mask = None if key_padding_mask is None else ~key_padding_mask.bool()[:, None, None, :]
        o = F.scaled_dot_product_attention(heads(self.q_proj(q)), heads(self.k_proj(kv)), heads(self.v_proj(kv)), attn_mask=mask)
        return self.out_proj(o.transpose(1, 2).reshape(b, n, d) * torch.sigmoid(self.gate_proj(q)))


class MemoryBlock(nn.Module):
    """Pre-norm block: memory tokens attend to [memory; VLM tokens], then an MLP."""

    def __init__(self, d: int, num_heads: int, mlp_ratio: float, attention: str = "standard"):
        super().__init__()
        self.gated = attention == "gated"
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = GatedAttention(d, num_heads) if self.gated else nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(approximate="tanh"), nn.Linear(h, d))

    def forward(self, m: torch.Tensor, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """m [B, N, d], x [B, L, d], x_mask [B, L] bool (True = valid token)."""
        kv = self.norm_kv(torch.cat([m, x], dim=1))
        ignore = torch.cat([torch.zeros(m.shape[:2], dtype=torch.bool, device=m.device), ~x_mask.bool()], dim=1)
        if self.gated:
            m = m + self.attn(self.norm_q(m), kv, key_padding_mask=ignore)
        else:
            m = m + self.attn(self.norm_q(m), kv, kv, key_padding_mask=ignore, need_weights=False)[0]
        return m + self.mlp(self.norm_mlp(m))


class EpisodeMemory(nn.Module):
    def __init__(self, vlm_dim: int, ctx_dim: int, num_tokens: int = 16, d_model: int = 768, depth: int = 2,
                 num_heads: int = 12, mlp_ratio: float = 4.0, progress_bins: int = 50, attention: str = "standard",
                 token_dim: Optional[int] = None):
        super().__init__()
        if attention not in _ATTENTIONS:
            raise ValueError(f"EpisodeMemory: attention must be one of {_ATTENTIONS}, got {attention!r}")
        self.attention = attention
        self.num_tokens, self.d_model, self.num_bins = int(num_tokens), int(d_model), int(progress_bins)
        self.init_tokens = nn.Parameter(torch.randn(1, self.num_tokens, self.d_model) * 0.02)
        self.token_resize = None
        if token_dim:
            self.token_resize = nn.Sequential(nn.LayerNorm(vlm_dim), nn.Linear(vlm_dim, int(token_dim)),
                                              nn.GELU(approximate="tanh"), nn.Linear(int(token_dim), int(token_dim)))
            vlm_dim = int(token_dim)
        self.in_proj = nn.Sequential(nn.LayerNorm(vlm_dim), nn.Linear(vlm_dim, self.d_model))
        self.blocks = nn.ModuleList([MemoryBlock(self.d_model, int(num_heads), float(mlp_ratio), attention)
                                     for _ in range(int(depth))])
        self.out_norm = nn.LayerNorm(self.d_model)
        self.progress_head = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.GELU(approximate="tanh"),
                                           nn.Linear(self.d_model, self.num_bins))
        self.progress_embed = nn.Sequential(nn.Linear(self.num_bins, ctx_dim), nn.GELU(approximate="tanh"),
                                            nn.Linear(ctx_dim, ctx_dim))
        self.mem_proj = nn.Linear(self.d_model, ctx_dim)
        self.source_embed = nn.ParameterDict({k: nn.Parameter(torch.randn(1, 1, ctx_dim) * 0.02) for k in ("memory", "progress")})
        self.register_buffer("bin_centers", (torch.arange(self.num_bins, dtype=torch.float32) + 0.5) / self.num_bins,
                             persistent=False)

    @property
    def num_ctx_tokens(self) -> int:
        return self.num_tokens + 1

    def init(self, b: int) -> torch.Tensor:
        """Initial memory [B, N, d] (fp32)."""
        return self.init_tokens.float().expand(b, -1, -1).clone()

    def update(self, m: torch.Tensor, vlm_tokens: torch.Tensor, vlm_mask: torch.Tensor,
               valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One step M_t = f(M_{t-1}, H_t). valid [B] bool: samples whose frame lies outside the episode keep M_{t-1}."""
        if self.token_resize is not None:
            vlm_tokens = self.token_resize(vlm_tokens.to(self.token_resize[1].weight.dtype))
        x = self.in_proj(vlm_tokens.to(self.in_proj[1].weight.dtype))
        new = m
        for blk in self.blocks:
            new = blk(new, x, vlm_mask)
        new = new.to(m.dtype)
        if valid is not None:
            new = torch.where(valid.to(new.device).bool()[:, None, None], new, m)
        return new

    def readout(self, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (System 2 context tokens [B, N + 1, ctx_dim] = memory tokens + progress token, progress logits [B, bins])."""
        h = self.out_norm(m)
        logits = self.progress_head(h.mean(1))
        probs = logits.float().softmax(-1).to(self.progress_embed[0].weight.dtype)
        prog = self.progress_embed(probs)[:, None] + self.source_embed["progress"]
        mem = self.mem_proj(h) + self.source_embed["memory"]
        return torch.cat([mem, prog.to(mem.dtype)], dim=1), logits

    def progress_target(self, p: torch.Tensor) -> torch.Tensor:
        """progress in [0, 1] -> bin index floor(p * bins), the last bin closed."""
        return (p.float() * self.num_bins).long().clamp(0, self.num_bins - 1)

    def expected_progress(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits.float().softmax(-1) * self.bin_centers).sum(-1)

    def progress_loss(self, logits: torch.Tensor, p: torch.Tensor,
                      valid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """valid [B] bool: only these samples count (the loss stays connected to the graph when none does)."""
        tgt = self.progress_target(p.to(logits.device))
        if valid is not None:
            w = valid.to(logits.device).float()
            n = w.sum().clamp_min(1)
            ce = (F.cross_entropy(logits.float(), tgt, reduction="none") * w).sum() / n
            with torch.no_grad():
                acc = ((logits.argmax(-1) == tgt).float() * w).sum() / n
                mae = ((self.expected_progress(logits) - p.to(logits.device).float()).abs() * w).sum() / n
            return ce, {"memory/progress_ce": float(ce.detach()), "memory/progress_acc": float(acc),
                        "memory/progress_mae": float(mae)}
        ce = F.cross_entropy(logits.float(), tgt)
        with torch.no_grad():
            acc = (logits.argmax(-1) == tgt).float().mean()
            mae = (self.expected_progress(logits) - p.to(logits.device).float()).abs().mean()
        return ce, {"memory/progress_ce": float(ce.detach()), "memory/progress_acc": float(acc),
                    "memory/progress_mae": float(mae)}
