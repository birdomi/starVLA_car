"""Trainable parts of the geometry-guided local residual policy (no diffusion / flow: one regression pass).

    GeometryTokenizer : MoGe-2 neck features [M, C, gh, gw] -> LN + Linear -> + learned 2D position -> [M, gh * gw, D]
    TrajectoryEncoder : the System 2 local chunk (K steps) per hand, in wrist-pose space, + current wrist / finger state
                        + plan age -> 2K query tokens [M, 2K, D] (token = (step k, hand h))
    ResidualDecoder   : n_layers x (self-attention over queries, cross-attention to geometry tokens, MLP), pre-LN
    ResidualHead      : LN + Linear(D, 12), ZERO-initialised (step 0 = System 2 unchanged) -> per (k, h):
                        [dp (3) | dw (3) | dfinger (6)] in units of the residual scales (pos m, rot rad, finger cmd)
"""

import math

import torch
import torch.nn as nn


def fourier_xyz(p: torch.Tensor, n_freq: int = 6, base: float = 1.0) -> torch.Tensor:
    """[..., 3] metres -> [..., 3 * 2 * n_freq] sin / cos at base * 2^k * pi."""
    f = base * math.pi * (2.0 ** torch.arange(n_freq, device=p.device, dtype=p.dtype))
    x = p.unsqueeze(-1) * f
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1).flatten(-2)


class GeometryTokenizer(nn.Module):
    def __init__(self, in_dim: int, d_model: int, grid):
        super().__init__()
        self.grid = tuple(int(v) for v in grid)
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.grid[0] * self.grid[1], d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if tuple(feat.shape[-2:]) != self.grid:
            raise ValueError(f"GeometryTokenizer: feature grid {tuple(feat.shape[-2:])} != {self.grid}")
        x = feat.flatten(2).transpose(1, 2)  # [M, N, C]
        return self.out_norm(self.proj(self.norm(x)) + self.pos)


class TrajectoryEncoder(nn.Module):
    """Per (step k, hand h) token from: plan wrist pos (3) + Fourier(pos) + rot6d (6), plan finger command (6),
    plan pos - current wrist pos (3), current wrist rot6d (6), current finger state (6), plan age (1)."""

    def __init__(self, d_model: int, horizon: int, n_freq: int = 6):
        super().__init__()
        self.n_freq = n_freq
        in_dim = 3 + 3 * 2 * n_freq + 6 + 6 + 3 + 6 + 6 + 1
        self.mlp = nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.step = nn.Parameter(torch.zeros(1, horizon, 1, d_model))
        self.hand = nn.Parameter(torch.zeros(1, 1, 2, d_model))
        nn.init.normal_(self.step, std=0.02)
        nn.init.normal_(self.hand, std=0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, plan_p, plan_r6, plan_f, cur_p, cur_r6, cur_f, age):
        """plan_*: [M, K, 2, *], cur_*: [M, 2, *], age [M] (seconds) -> [M, K * 2, D]."""
        K = plan_p.shape[1]
        cur_p, cur_r6, cur_f = (t.unsqueeze(1).expand(-1, K, -1, -1) for t in (cur_p, cur_r6, cur_f))
        a = age.view(-1, 1, 1, 1).expand(-1, K, 2, 1)
        x = torch.cat([plan_p, fourier_xyz(plan_p, self.n_freq), plan_r6, plan_f, plan_p - cur_p, cur_r6, cur_f, a], dim=-1)
        h = self.mlp(x) + self.step[:, :K] + self.hand
        return self.norm(h.flatten(1, 2))


class _Block(nn.Module):
    def __init__(self, d: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.n1, self.sa = nn.LayerNorm(d), nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n2, self.ca = nn.LayerNorm(d), nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp_ratio)), nn.GELU(), nn.Linear(int(d * mlp_ratio), d))

    def forward(self, q, kv):
        h = self.n1(q)
        q = q + self.sa(h, h, h, need_weights=False)[0]
        q = q + self.ca(self.n2(q), kv, kv, need_weights=False)[0]
        return q + self.mlp(self.n3(q))


class ResidualDecoder(nn.Module):
    def __init__(self, d_model: int, heads: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(d_model, heads, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, queries, geo_tokens):
        for b in self.blocks:
            queries = b(queries, geo_tokens)
        return self.norm(queries)


class ResidualHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int = 12):
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h):
        return self.proj(h)
