"""System 1.5 - LatentFlowDiT: conditional flow-matching DiT over C-RADIOv4 latent changes, p(dZ | Z_t, S).

Data X_1 = dZ / s  (dZ = Z_f - Z_t in C-RADIO patch space, s = per-channel delta scale, see the framework).

    training   eps ~ N(0, I), tau ~ p(tau)
               X_tau = (1 - tau) eps + tau X_1,   u_tau = X_1 - eps
               L_FM  = || v_theta(X_tau, tau | Z_t, S) - u_tau ||^2      (mean over valid patch tokens)
    inference  X_0 ~ N(0, I), dX/dtau = v_theta from tau = 0 to 1 (LatentFlowSampler) -> X_1

Network, split into a backbone and a velocity head:

    H_tau = backbone(X_tau, tau | Z_t, S)
        X_tau [B, N, D_c] --x_proj--> [B, N, d] + 2D sin-cos position          noisy latent-change tokens
        Z_t   [B, N, D_c] --zt_proj--> [B, N, d_c] + position                    cross-attention stream "zt"
        S     [B, L, D_s] --LN, s2_proj--> [B, L, d_c] (+ learned null key)     cross-attention stream "s2" (masked)
        tau   [B] --TimestepEmbedder--> c [B, d]   (adaln_cond "tau+s2": c += MLP(masked mean of the S stream))
        depth x AdaLNCrossBlock(self-attn, cross-attn zt, cross-attn s2, MLP; adaLN-Zero from c)
        FinalAdaNorm(c)                                      -> H_tau [B, N, d]   (= what System 1 reads at tau = 1)
    v_tau = velocity_head(H_tau)  [B, N, D_c]  (Linear, zero-initialised)

The conditions are projected once per (Z_t, S) by encode_conditions() and reused by every ODE step and by the
final tau = 1 hidden-state pass.

One-step endpoint projection (joint training): with the velocity of the FM forward pass, the clean endpoint of the
straight path is estimated in one differentiable step,  dZ~ = X_tau + (1 - tau) v_theta(X_tau, tau)  (exact for the
linear path if v_theta = X_1 - eps).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from starVLA.model.modules.tri_system.flow_matching import TimestepEmbedder, make_noisy, masked_mse, sample_tau

from .layers import AdaLNCrossBlock, FinalAdaNorm, run_blocks, sincos_2d

ADALN_CONDS = ("tau", "tau+s2")


def one_step_endpoint(x_tau: torch.Tensor, tau: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """dZ~ = X_tau + (1 - tau) v  [B, N, D_c] (fp32), differentiable in x_tau and v; tau [B]."""
    return x_tau.float() + (1.0 - tau.float()).view(-1, 1, 1) * v.float()


@dataclass
class FlowConditions:
    """Projected conditions of one batch: cross-attention contexts + the pooled S vector (or None)."""

    contexts: Dict[str, Tuple[torch.Tensor, torch.Tensor]]  # "zt": ([B, N, d_c], [B, N]), "s2": ([B, 1 + L, d_c], [B, 1 + L])
    pooled: Optional[torch.Tensor]  # [B, d] added to the tau embedding (adaln_cond "tau+s2"), else None

    @property
    def batch_size(self) -> int:
        return self.contexts["zt"][0].shape[0]

    def repeat(self, r: int) -> "FlowConditions":
        if r == 1:
            return self
        rep = lambda t: t.repeat(r, *([1] * (t.dim() - 1)))
        return FlowConditions({k: (rep(x), rep(m)) for k, (x, m) in self.contexts.items()},
                              None if self.pooled is None else rep(self.pooled))


class LatentFlowDiT(nn.Module):
    def __init__(self, feat_dim: int, s2_dim: int, grid: Tuple[int, int], d_model: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0, cond_dim: Optional[int] = None, qk_norm: bool = True,
                 dropout: float = 0.0, attn_dropout: float = 0.0, adaln_cond: str = "tau", grad_ckpt: bool = False):
        super().__init__()
        if adaln_cond not in ADALN_CONDS:
            raise ValueError(f"system15.adaln_cond must be one of {ADALN_CONDS}, got {adaln_cond!r}")
        cond_dim = int(cond_dim or d_model)
        self.grid, self.num_tokens = tuple(grid), int(grid[0] * grid[1])
        self.feat_dim, self.s2_dim, self.d_model, self.cond_dim = feat_dim, s2_dim, d_model, cond_dim
        self.adaln_cond, self.grad_ckpt = adaln_cond, grad_ckpt
        self.x_proj = nn.Linear(feat_dim, d_model)  # noisy latent-change tokens (input projection)
        self.zt_proj = nn.Linear(feat_dim, cond_dim)  # current C-RADIO tokens (own projection)
        self.s2_proj = nn.Sequential(nn.LayerNorm(s2_dim), nn.Linear(s2_dim, cond_dim))  # Qwen tokens (own projection)
        # always-visible extra key of the S stream: attention stays defined even if every S token were masked
        self.s2_null = nn.Parameter(torch.randn(1, 1, cond_dim) * 0.02)
        self.t_embed = TimestepEmbedder(d_model)
        if adaln_cond == "tau+s2":
            self.s2_pool = nn.Sequential(nn.LayerNorm(cond_dim), nn.Linear(cond_dim, d_model))
        self.register_buffer("pos", sincos_2d(*self.grid, d_model)[None], persistent=False)  # [1, N, d]
        self.register_buffer("cond_pos", sincos_2d(*self.grid, cond_dim)[None], persistent=False)  # [1, N, d_c]
        self.blocks = nn.ModuleList([
            AdaLNCrossBlock(d_model, num_heads, mlp_ratio, ("zt", "s2"), cond_dim, qk_norm, dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.final_norm = FinalAdaNorm(d_model)
        self.velocity_head = nn.Linear(d_model, feat_dim)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)

    # ------------------------------------------------------------------ conditions
    def encode_conditions(self, z_t: torch.Tensor, z_valid: torch.Tensor, s2_tokens: torch.Tensor,
                          s2_mask: torch.Tensor) -> FlowConditions:
        """z_t [B, N, D_c], z_valid [B, N] bool, s2_tokens [B, L, D_s], s2_mask [B, L] bool (False = padding)."""
        if z_t.shape[1] != self.num_tokens:
            raise ValueError(f"Z_t has {z_t.shape[1]} tokens, the DiT grid {self.grid} expects {self.num_tokens}")
        dtype = self.x_proj.weight.dtype
        b = z_t.shape[0]
        zt = self.zt_proj(z_t.to(dtype)) + self.cond_pos
        s2 = torch.cat([self.s2_null.expand(b, -1, -1), self.s2_proj(s2_tokens.to(dtype))], dim=1)
        s2_mask = torch.cat([s2_mask.new_ones(b, 1), s2_mask.bool()], dim=1)
        pooled = None
        if self.adaln_cond == "tau+s2":
            m = s2_mask[..., None].to(s2.dtype)
            pooled = self.s2_pool((s2 * m).sum(1) / m.sum(1))
        return FlowConditions({"zt": (zt, z_valid.bool()), "s2": (s2, s2_mask)}, pooled)

    def _c(self, tau: torch.Tensor, cond: FlowConditions) -> torch.Tensor:
        c = self.t_embed(tau)
        return c if cond.pooled is None else c + cond.pooled

    # ------------------------------------------------------------------ network
    def backbone(self, x_tau: torch.Tensor, tau: torch.Tensor, cond: FlowConditions) -> torch.Tensor:
        """H_tau = blocks + final normalization, [B, N, d] (the hidden state right before the velocity head)."""
        c = self._c(tau, cond)
        h = self.x_proj(x_tau.to(self.x_proj.weight.dtype)) + self.pos
        h = run_blocks(self.blocks, h, c, cond.contexts, grad_ckpt=self.grad_ckpt, training=self.training)
        return self.final_norm(h, c)

    def forward(self, x_tau: torch.Tensor, tau: torch.Tensor, cond: FlowConditions, return_hidden_states: bool = False):
        """v_theta(X_tau, tau | Z_t, S) [B, N, D_c]; with return_hidden_states also H_tau [B, N, d]."""
        h = self.backbone(x_tau, tau, cond)
        v = self.velocity_head(h)
        return (v, h) if return_hidden_states else v

    # ------------------------------------------------------------------ training
    def flow_matching_loss(self, x1: torch.Tensor, cond: FlowConditions, token_mask: torch.Tensor,
                           tau_schedule: Optional[Dict] = None, num_noise_samples: int = 1,
                           return_intermediates: bool = False):
        """x1 [B, N, D_c] = scaled GT change, token_mask [B, N] (False = excluded from the loss).

        Returns (L_FM, logs), plus {"x_tau", "tau", "v"} of the [B * num_noise_samples] draws (graph kept; the first
        B rows are the first draw) with return_intermediates. num_noise_samples > 1 draws several (eps, tau)."""
        r = int(num_noise_samples)
        rep = (lambda t: t.repeat(r, *([1] * (t.dim() - 1)))) if r > 1 else (lambda t: t)
        x1, token_mask, cond = rep(x1), rep(token_mask), cond.repeat(r)
        tau = sample_tau(x1.shape[0], x1.device, **dict(tau_schedule or {"schedule": "uniform"}))
        x_tau, u, _ = make_noisy(x1.float(), tau)
        v = self(x_tau, tau, cond)
        loss = masked_mse(v, u, token_mask)
        valid = token_mask.sum().clamp_min(1)
        logs = {"system15/fm_loss": loss.item(),
                "system15/x1_rms": (x1.float().pow(2).mean(-1) * token_mask).sum().div(valid).sqrt().item()}
        if return_intermediates:
            return loss, logs, {"x_tau": x_tau, "tau": tau, "v": v}
        return loss, logs
