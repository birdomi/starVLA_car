"""Transformer building blocks of the Dual-System VLA (System 2 keyframe DiT and System 1 short-horizon DiT).

DiTBlock = full bidirectional self-attention over the primary sequence (optionally with 1-D temporal RoPE on q / k),
then cross-attention to the condition tokens, then an MLP. The diffusion-timestep vector c modulates self-attention and
MLP through adaLN-Zero (shift / scale / gate from c; gates start at 0); the cross-attention is a plain pre-LayerNorm
residual (PixArt-style) so the conditions are used from step 0. The zero-initialised FinalLayer makes the untrained
network output 0.

Shapes: x [B, N, d], c [B, d], ctx [B, L, d], ctx_mask [B, L] bool (True = attend), pos [B, N] float (seconds).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sincos_1d(pos: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of real-valued positions: pos [...] -> [..., dim] (float32)."""
    assert dim % 2 == 0, dim
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=pos.device, dtype=torch.float32) / half)
    args = pos.float()[..., None] * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)


def sincos_2d(h: int, w: int, dim: int) -> torch.Tensor:
    """Fixed 2D sin-cos embedding of a row-major patch grid, [h * w, dim]."""
    assert dim % 4 == 0, dim
    gy, gx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.cat([sincos_1d(gy.flatten(), dim // 2), sincos_1d(gx.flatten(), dim // 2)], dim=1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class TimestepEmbedder(nn.Module):
    """Diffusion step t in [0, T) -> [B, d]: sinusoidal features of (t / T) * 1000, then a 2-layer MLP (DiT)."""

    def __init__(self, dim: int, num_train_timesteps: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim, self.num_train_timesteps = freq_dim, num_train_timesteps
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = sincos_1d(t.float() * (1000.0 / self.num_train_timesteps), self.freq_dim)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class ContinuousRoPE(nn.Module):
    """1-D rotary position embedding over REAL-VALUED timestamps (seconds).

    angle_j = (t * time_scale) * base^(-2j / head_dim); time_scale = positions per second (e.g. the dense action
    rate, so adjacent dense actions are 1 position apart). q . k after rotation depends on t_q - t_k only.
    """

    def __init__(self, head_dim: int, time_scale: float = 20.0, base: float = 1000.0):
        super().__init__()
        assert head_dim % 2 == 0, head_dim
        self.time_scale = float(time_scale)
        inv = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.register_buffer("inv_freq", inv, persistent=False)

    def cos_sin(self, t: torch.Tensor):
        """t [B, N] seconds -> cos, sin [B, 1, N, head_dim] (broadcast over heads)."""
        ang = (t.float() * self.time_scale)[..., None] * self.inv_freq  # [B, N, hd/2]
        ang = torch.cat([ang, ang], dim=-1)[:, None]
        return ang.cos(), ang.sin()

    @staticmethod
    def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x [B, h, N, hd] -> rotated x (rotate-half convention), computed in float32."""
        x1, x2 = x.float().chunk(2, dim=-1)
        rot = torch.cat([-x2, x1], dim=-1)
        return (x.float() * cos + rot * sin).to(x.dtype)


class Attention(nn.Module):
    """Multi-head attention with RMS-normalized q / k. Self-attention when ctx is None (full bidirectional, no
    causal mask). rope = (cos, sin) rotates q and k (self-attention); cross-attention rotates q by rope and k by
    ctx_rope (the condition tokens' own timestamps, so q . k depends on t_q - t_k); ctx_mask [B, L] bool, True = attend."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True):
        super().__init__()
        assert dim % num_heads == 0, (dim, num_heads)
        self.h, self.hd = num_heads, dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.hd, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.hd, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, x, ctx=None, ctx_mask=None, rope=None, ctx_rope=None):
        b, n, d = x.shape
        src = x if ctx is None else ctx
        q = self.q_norm(self.q(x).view(b, n, self.h, self.hd)).transpose(1, 2)
        k, v = self.kv(src).view(b, src.shape[1], 2, self.h, self.hd).unbind(2)
        k, v = self.k_norm(k).transpose(1, 2), v.transpose(1, 2)
        if ctx is not None and (rope is None) != (ctx_rope is None):
            raise ValueError("cross-attention RoPE needs both the query (rope) and the condition (ctx_rope) timestamps")
        if ctx is None and ctx_rope is not None:
            raise ValueError("ctx_rope is for cross-attention only")
        if rope is not None:
            cos, sin = rope
            kcos, ksin = rope if ctx is None else ctx_rope
            q, k = ContinuousRoPE.apply(q, cos, sin), ContinuousRoPE.apply(k, kcos, ksin)
        mask = None if ctx_mask is None else ctx_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class DiTBlock(nn.Module):
    """x += g1 * SelfAttn(mod(LN x)) [full, RoPE];  x += CrossAttn(LN x, ctx) [RoPE iff ctx timestamps are given];
    x += g2 * MLP(mod(LN x))."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qk_norm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads, qk_norm)
        self.cross_norm = nn.LayerNorm(dim, eps=1e-6)
        self.cross = Attention(dim, num_heads, qk_norm)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, mlp_ratio)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, ctx, ctx_mask, rope_cos=None, rope_sin=None, ctx_cos=None, ctx_sin=None):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        rope = None if rope_cos is None else (rope_cos, rope_sin)
        x = x + g1[:, None] * self.attn(modulate(self.norm1(x), s1, sc1), rope=rope)
        if ctx_cos is None:
            x = x + self.cross(self.cross_norm(x), ctx, ctx_mask)
        else:
            x = x + self.cross(self.cross_norm(x), ctx, ctx_mask, rope=rope, ctx_rope=(ctx_cos, ctx_sin))
        return x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, sc2))


class FinalLayer(nn.Module):
    """adaLN-modulated LayerNorm + zero-initialised output projection."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, out_dim)
        for m in (self.ada[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, c):
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return self.out(modulate(self.norm(x), shift, scale))


class DiTStack(nn.Module):
    """depth x DiTBlock with optional activation checkpointing; one shared RoPE table per forward."""

    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0, qk_norm: bool = True,
                 rope: Optional[ContinuousRoPE] = None, grad_ckpt: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, mlp_ratio, qk_norm) for _ in range(depth)])
        self.rope, self.grad_ckpt = rope, grad_ckpt

    def forward(self, x, c, ctx, ctx_mask, pos: Optional[torch.Tensor] = None, ctx_pos: Optional[torch.Tensor] = None):
        """pos [B, N] timestamps in seconds (required iff the stack has RoPE); ctx_pos [B, L] timestamps of the condition
        tokens (optional: given, the cross-attention is RoPE'd too, so it sees t_query - t_ctx)."""
        cos = sin = kcos = ksin = None
        if self.rope is not None:
            if pos is None:
                raise ValueError("this DiT uses temporal RoPE: pass per-token timestamps `pos` [B, N]")
            cos, sin = self.rope.cos_sin(pos)
            if ctx_pos is not None:
                kcos, ksin = self.rope.cos_sin(ctx_pos)
        elif ctx_pos is not None:
            raise ValueError("ctx_pos needs a DiT with temporal RoPE")
        for blk in self.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, c, ctx, ctx_mask, cos, sin, kcos, ksin, use_reentrant=False)
            else:
                x = blk(x, c, ctx, ctx_mask, cos, sin, kcos, ksin)
        return x
