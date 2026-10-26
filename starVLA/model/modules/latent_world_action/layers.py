"""Transformer layers of the QwenLatentWorldAction framework.

AdaLNCrossBlock  (System 1.5 Flow DiT block)
    x = x + g1 * SelfAttn(adaLN_1(x; c))                      self-attention over the noisy latent-change tokens
    x = x + CrossAttn_s(LN_s(x), ctx_s, mask_s)  for s in streams   one independent cross-attention per condition
                                                                    stream ("zt": C-RADIO Z_t, "s2": Qwen tokens S)
    x = x + g2 * MLP(adaLN_2(x; c))
  c = embedding of the flow timestep tau; shift / scale / gate come from c (adaLN-Zero: gates start at 0, so every
  block starts as the identity). Parameter names match tri_system.blocks.AdaLNBlock, so a QwenGR00T_TriSystem
  System 1.5 checkpoint converts 1:1 (starVLA/training/latent_world_action_tools.py convert-trisystem).

FinalAdaNorm     final normalization of the DiT backbone (LayerNorm + adaLN shift/scale from c). Its output is the
                 hidden state H handed to System 1; the velocity head is a separate Linear on top of it.

QueryDecoderBlock (System 1 action decoder block, pre-LN)
    q = q + SelfAttn(LN(q));  q = q + CrossAttn(LN(q), memory, mask);  q = q + MLP(LN(q))
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from starVLA.model.modules.tri_system.blocks import Mlp, modulate, sincos_2d

__all__ = ["Attention", "Mlp", "AdaLNCrossBlock", "FinalAdaNorm", "QueryDecoderBlock", "GradScale", "sincos_2d"]


class Attention(nn.Module):
    """Multi-head attention (SDPA). Self-attention when ctx is None; ctx may have its own width ctx_dim.

    ctx_mask [B, L] bool, True = attend. Every query row must have at least one True key.
    """

    def __init__(self, dim: int, num_heads: int, ctx_dim: Optional[int] = None, qk_norm: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, (dim, num_heads)
        self.h, self.hd = num_heads, dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(ctx_dim or dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.hd, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.hd, eps=1e-6) if qk_norm else nn.Identity()
        self.attn_drop = float(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None,
                ctx_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, d = x.shape
        ctx = x if ctx is None else ctx
        q = self.q_norm(self.q(x).view(b, n, self.h, self.hd)).transpose(1, 2)
        k, v = self.kv(ctx).view(b, ctx.shape[1], 2, self.h, self.hd).unbind(2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        mask = None if ctx_mask is None else ctx_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask,
                                             dropout_p=self.attn_drop if self.training else 0.0)
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(b, n, d)))


class AdaLNCrossBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, cross_streams: Sequence[str] = (),
                 ctx_dim: Optional[int] = None, qk_norm: bool = True, dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads, None, qk_norm, attn_dropout, dropout)
        self.cross_norm = nn.ModuleDict({s: nn.LayerNorm(dim, eps=1e-6) for s in cross_streams})
        self.cross_attn = nn.ModuleDict(
            {s: Attention(dim, num_heads, ctx_dim, qk_norm, attn_dropout, dropout) for s in cross_streams})
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, mlp_ratio)
        self.mlp_drop = nn.Dropout(dropout)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                contexts: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]) -> torch.Tensor:
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1[:, None] * self.attn(modulate(self.norm1(x), s1, sc1))
        for name, attn in self.cross_attn.items():
            ctx, mask = contexts[name]
            x = x + attn(self.cross_norm[name](x), ctx, mask)
        return x + g2[:, None] * self.mlp_drop(self.mlp(modulate(self.norm2(x), s2, sc2)))


class FinalAdaNorm(nn.Module):
    """LayerNorm (no affine) modulated by shift / scale from c. Zero-initialised modulation = plain LayerNorm."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return modulate(self.norm(x), shift, scale)


class QueryDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qk_norm: bool = True,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = Attention(dim, num_heads, None, qk_norm, attn_dropout, dropout)
        self.cross_norm = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = Attention(dim, num_heads, None, qk_norm, attn_dropout, dropout)
        self.mlp_norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, mlp_ratio)
        self.mlp_drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, memory: torch.Tensor, memory_mask: Optional[torch.Tensor]) -> torch.Tensor:
        q = q + self.self_attn(self.self_norm(q))
        q = q + self.cross_attn(self.cross_norm(q), memory, memory_mask)
        return q + self.mlp_drop(self.mlp(self.mlp_norm(q)))


def run_blocks(blocks: nn.ModuleList, x: torch.Tensor, *args, grad_ckpt: bool = False, training: bool = False):
    """Apply blocks in order, with activation checkpointing when requested and gradients are being recorded."""
    for blk in blocks:
        if grad_ckpt and training and torch.is_grad_enabled():
            x = checkpoint(blk, x, *args, use_reentrant=False)
        else:
            x = blk(x, *args)
    return x


class GradScale(torch.autograd.Function):
    """Identity forward; multiplies the incoming gradient by `scale` (weights the action loss that reaches
    System 1.5 in the joint stage)."""

    @staticmethod
    def forward(ctx, x, scale: float):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None
