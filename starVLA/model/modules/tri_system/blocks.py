"""Transformer building blocks shared by the System 1.5 Flow Transformer and the System 1 DiT.

AdaLNBlock = self-attention on the generated tokens, then one cross-attention per condition stream
(e.g. "zt", "s2"), then an MLP. The timestep vector c modulates self-attention and MLP via adaLN-Zero
(shift/scale/gate from c, gates start at 0 so every block starts as the identity); the cross-attentions
are plain pre-LayerNorm residuals (PixArt-style) so the conditions are used from step 0.
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _sincos_1d(pos: torch.Tensor, dim: int) -> torch.Tensor:
    omega = 1.0 / (10000 ** (torch.arange(dim // 2, dtype=torch.float64) / (dim / 2)))
    out = pos.double()[:, None] * omega[None]
    return torch.cat([out.sin(), out.cos()], dim=1).float()


def sincos_2d(h: int, w: int, dim: int) -> torch.Tensor:
    """Fixed 2D sin-cos position embedding, [h * w, dim] (row-major, matching the patch order)."""
    assert dim % 4 == 0, dim
    gy, gx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.cat([_sincos_1d(gy.flatten(), dim // 2), _sincos_1d(gx.flatten(), dim // 2)], dim=1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class Attention(nn.Module):
    """Multi-head attention; self-attention when ctx is None. ctx_mask [B, L] bool, True = attend."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True):
        super().__init__()
        assert dim % num_heads == 0, (dim, num_heads)
        self.h, self.hd = num_heads, dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = nn.RMSNorm(self.hd, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.hd, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None, ctx_mask: Optional[torch.Tensor] = None):
        b, n, d = x.shape
        ctx = x if ctx is None else ctx
        q = self.q_norm(self.q(x).view(b, n, self.h, self.hd)).transpose(1, 2)
        k, v = self.kv(ctx).view(b, ctx.shape[1], 2, self.h, self.hd).unbind(2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
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


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, cross_streams: Sequence[str] = (),
                 qk_norm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads, qk_norm)
        self.cross_norm = nn.ModuleDict({s: nn.LayerNorm(dim, eps=1e-6) for s in cross_streams})
        self.cross_attn = nn.ModuleDict({s: Attention(dim, num_heads, qk_norm) for s in cross_streams})
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, mlp_ratio)
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
        return x + g2[:, None] * self.mlp(modulate(self.norm2(x), s2, sc2))


class FinalLayer(nn.Module):
    """adaLN-modulated LayerNorm + zero-initialised output projection (the network starts at output 0)."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, out_dim)
        for m in (self.ada[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return self.out(modulate(self.norm(x), shift, scale))


class AdaLNTransformer(nn.Module):
    """Stack of AdaLNBlocks with optional activation checkpointing."""

    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0,
                 cross_streams: Sequence[str] = (), qk_norm: bool = True, grad_ckpt: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AdaLNBlock(dim, num_heads, mlp_ratio, cross_streams, qk_norm) for _ in range(depth)]
        )
        self.grad_ckpt = grad_ckpt

    def forward(self, x, c, contexts):
        for blk in self.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, c, contexts, use_reentrant=False)
            else:
                x = blk(x, c, contexts)
        return x
