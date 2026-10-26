"""System 1 - ActionQueryDecoder: learnable action queries read the System 1.5 final hidden states and regress the
action chunk directly (one deterministic forward pass; no diffusion / flow matching).

    Q_action  [H_a, d] learnable  --expand-->  [B, H_a, d]
    (optional) state [B, 1, D_st] --MLP--> [B, 1, d]: prepended as an extra self-attention token ("token") or
               added to every query ("add")
    memory    H_f [B, N, D_h] --LN, Linear--> [B, N, d] (+ 2D sin-cos position of the C-RADIO patch grid)
    depth x QueryDecoderBlock: self-attention over the queries, cross-attention (Q = queries, K/V = projected H_f,
               invalid patch tokens masked), MLP
    LN -> regression head Linear(d, D_a)  ->  A_hat [B, H_a, D_a]   (normalized action space of the loader)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import QueryDecoderBlock, sincos_2d

STATE_CONDITIONING = ("none", "token", "add")
LOSSES = ("smooth_l1", "l1", "mse")


class ActionQueryDecoder(nn.Module):
    def __init__(self, action_dim: int, horizon: int, memory_dim: int, grid: Tuple[int, int], state_dim: int = 0,
                 d_model: int = 512, depth: int = 4, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1,
                 attn_dropout: float = 0.0, qk_norm: bool = True, add_grid_pos: bool = True,
                 state_conditioning: str = "token"):
        super().__init__()
        if state_conditioning not in STATE_CONDITIONING:
            raise ValueError(f"system1.state_conditioning must be one of {STATE_CONDITIONING}, got {state_conditioning!r}")
        self.action_dim, self.horizon, self.memory_dim = int(action_dim), int(horizon), int(memory_dim)
        self.state_dim = int(state_dim or 0)
        self.state_conditioning = state_conditioning if self.state_dim else "none"
        self.queries = nn.Parameter(torch.randn(1, self.horizon, d_model) * 0.02)
        self.memory_proj = nn.Sequential(nn.LayerNorm(memory_dim), nn.Linear(memory_dim, d_model))
        self.add_grid_pos = bool(add_grid_pos)
        self.register_buffer("grid_pos", sincos_2d(*grid, d_model)[None], persistent=False)  # [1, N, d]
        if self.state_conditioning != "none":
            self.state_proj = nn.Sequential(nn.Linear(self.state_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.n_prefix = 1 if self.state_conditioning == "token" else 0
        self.blocks = nn.ModuleList([QueryDecoderBlock(d_model, num_heads, mlp_ratio, qk_norm, dropout, attn_dropout)
                                     for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.head = nn.Linear(d_model, self.action_dim)

    @property
    def uses_state(self) -> bool:
        return self.state_conditioning != "none"

    def embed_memory(self, hidden: torch.Tensor) -> torch.Tensor:
        """H_f [B, N, D_h] -> K/V memory [B, N, d]."""
        mem = self.memory_proj(hidden.to(self.queries.dtype))
        return mem + self.grid_pos if self.add_grid_pos else mem

    def forward(self, hidden: torch.Tensor, hidden_mask: Optional[torch.Tensor] = None,
                state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """hidden [B, N, D_h] (System 1.5 final hidden states), hidden_mask [B, N] bool (True = attend),
        state [B, 1, D_st] -> normalized action chunk [B, H_a, D_a]."""
        b = hidden.shape[0]
        q = self.queries.expand(b, -1, -1)
        if self.uses_state:
            if state is None:
                raise ValueError("ActionQueryDecoder is state-conditioned but no state was given")
            s = self.state_proj(state.to(self.queries.dtype).reshape(b, 1, -1))
            q = torch.cat([s, q], dim=1) if self.state_conditioning == "token" else q + s
        mem = self.embed_memory(hidden)
        for blk in self.blocks:
            q = blk(q, mem, hidden_mask)
        return self.head(self.out_norm(q[:, self.n_prefix:]))


def regression_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "smooth_l1",
                    beta: float = 0.1) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Element-mean regression loss over [B, H_a, D_a] (fp32) + logs (l1 / rmse of the same prediction)."""
    pred, target = pred.float(), target.float()
    if kind == "smooth_l1":
        loss = F.smooth_l1_loss(pred, target, beta=beta)
    elif kind == "l1":
        loss = F.l1_loss(pred, target)
    elif kind == "mse":
        loss = F.mse_loss(pred, target)
    else:
        raise ValueError(f"system1.loss must be one of {LOSSES}, got {kind!r}")
    with torch.no_grad():
        err = pred - target
        logs = {"system1/l1": err.abs().mean().item(), "system1/rmse": err.pow(2).mean().sqrt().item()}
    return loss, logs
