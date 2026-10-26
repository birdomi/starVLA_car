"""System 1.5 — conditional flow-matching Latent Change Predictor, p(dZ | Z_t, S_t), in C-RADIOv4 token space.

Training (per sample):
    eps ~ N(0, I), tau ~ p(tau)
    X_tau = (1 - tau) eps + tau X_1          X_1 = dZ / dz_rms,  dZ = z_n(future) - z_n(current)
    loss  = || v_theta(X_tau, tau | Z_t, S_t) - (X_1 - eps) ||^2   (over valid patch tokens)
Inference: X_0 ~ N(0, I), ODE dX/dtau = v_theta from tau = 0 to 1, dZ_hat = X_1 * dz_rms.

Flow Transformer, shapes:
    X_tau  [B, N, D_c] --x_proj--> [B, N, d]  (+ 2D sin-cos position)   <- self-attention tokens
    Z_t    [B, N, D_c] --zt_proj-> [B, N, d]  (+ same positions)        <- cross-attention stream "zt"
    S_t    [B, L_s, D_s] --s2_proj-> [B, L_s, d] (+ learned null token) <- cross-attention stream "s2"
    tau    [B] --TimestepEmbedder--> [B, d] --(+ pooled S_t)--> adaLN-Zero modulation of every block
    out    [B, N, d] --FinalLayer--> v [B, N, D_c]
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .blocks import AdaLNTransformer, FinalLayer, sincos_2d
from .flow_matching import TimestepEmbedder, make_noisy, masked_mse, ode_solve, sample_tau


class ChangeFlowTransformer(nn.Module):
    def __init__(self, feat_dim: int, s2_dim: int, grid: Tuple[int, int], d_model: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0, qk_norm: bool = True, zt_add_to_input: bool = False,
                 grad_ckpt: bool = False):
        super().__init__()
        self.grid = tuple(grid)
        self.x_proj = nn.Linear(feat_dim, d_model)  # noisy change tokens
        self.zt_proj = nn.Linear(feat_dim, d_model)  # current C-RADIO tokens
        self.s2_proj = nn.Sequential(nn.LayerNorm(s2_dim), nn.Linear(s2_dim, d_model))  # Qwen tokens
        # Always-visible extra key for the S_t stream: keeps attention defined when S_t is dropped (CFG).
        self.s2_null = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Optional (ablation): also add the aligned Z_t projection to the input tokens.
        self.zt_add_to_input = zt_add_to_input
        self.t_embed = TimestepEmbedder(d_model)
        self.s2_pool = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.register_buffer("pos", sincos_2d(*self.grid, d_model)[None], persistent=False)  # [1, N, d]
        self.transformer = AdaLNTransformer(d_model, depth, num_heads, mlp_ratio, ("zt", "s2"), qk_norm, grad_ckpt)
        self.final = FinalLayer(d_model, feat_dim)

    def forward(self, x_tau, tau, z_t, z_valid, s2_tokens, s2_mask):
        """x_tau, z_t [B, N, D_c]; tau [B]; z_valid [B, N]; s2_tokens [B, L_s, D_s]; s2_mask [B, L_s] -> [B, N, D_c]."""
        b = x_tau.shape[0]
        zt = self.zt_proj(z_t) + self.pos
        h = self.x_proj(x_tau) + self.pos
        if self.zt_add_to_input:
            h = h + zt
        s2 = torch.cat([self.s2_null.expand(b, -1, -1), self.s2_proj(s2_tokens.to(self.s2_null.dtype))], dim=1)
        s2_mask = torch.cat([s2_mask.new_ones(b, 1), s2_mask], dim=1)
        m = s2_mask[..., None].to(s2.dtype)
        pooled = (s2 * m).sum(1) / m.sum(1)
        c = self.t_embed(tau) + self.s2_pool(pooled)
        h = self.transformer(h, c, {"zt": (zt, z_valid), "s2": (s2, s2_mask)})
        return self.final(h, c)


class LatentChangePredictor(nn.Module):
    """System 1.5. `target`: "delta" (dZ, the design) or "future" (ablation: Z_f itself; dZ = Z_f - Z_t after)."""

    def __init__(self, feat_dim: int, s2_dim: int, grid: Tuple[int, int], cfg: Dict):
        super().__init__()
        self.target = str(cfg.get("target", "delta"))
        assert self.target in ("delta", "future"), self.target
        self.net = ChangeFlowTransformer(
            feat_dim, s2_dim, grid, d_model=int(cfg.get("d_model", 768)), depth=int(cfg.get("depth", 12)),
            num_heads=int(cfg.get("num_heads", 12)), mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
            qk_norm=bool(cfg.get("qk_norm", True)), zt_add_to_input=bool(cfg.get("zt_add_to_input", False)),
            grad_ckpt=bool(cfg.get("grad_ckpt", False)),
        )
        self.tau_schedule = dict(cfg.get("tau_schedule", {"schedule": "logit_normal", "mean": 0.0, "std": 1.0}))
        self.num_noise_samples = int(cfg.get("num_noise_samples", 1))
        # > 0: drop S_t for that fraction of samples (unconditional branch); only useful with cfg_scale > 1
        self.s2_drop_prob = float(cfg.get("s2_drop_prob", 0.0))
        self.sample_steps = int(cfg.get("sample_steps", 4))
        self.sample_method = str(cfg.get("sample_method", "euler"))
        self.cfg_scale = float(cfg.get("cfg_scale", 1.0))

    # X_1 for target "delta" is dZ / dz_rms; for "future" it is z_n(future) = z_t + dZ (dz_rms [D_c]).
    def _x1(self, z_t, dz_scaled, dz_rms):
        return dz_scaled if self.target == "delta" else z_t + dz_scaled * dz_rms

    def _to_delta(self, x1, z_t, dz_rms):
        return x1 if self.target == "delta" else (x1 - z_t) / dz_rms

    def loss(self, z_t, dz_scaled, z_valid, s2_tokens, s2_mask, dz_rms) -> Tuple[torch.Tensor, Dict[str, float]]:
        """z_t, dz_scaled [B, N, D_c] (dz_scaled = dZ / dz_rms); returns the velocity MSE."""
        r = self.num_noise_samples
        rep = lambda t: t.repeat(r, *([1] * (t.dim() - 1))) if r > 1 else t
        z_t, dz_scaled, z_valid, s2_tokens, s2_mask = map(rep, (z_t, dz_scaled, z_valid, s2_tokens, s2_mask))
        b = z_t.shape[0]
        tau = sample_tau(b, z_t.device, **self.tau_schedule)
        x1 = self._x1(z_t, dz_scaled, dz_rms)
        x_tau, u, _ = make_noisy(x1, tau)
        if self.training and self.s2_drop_prob > 0:
            keep = torch.rand(b, device=z_t.device) >= self.s2_drop_prob
            s2_mask = s2_mask & keep[:, None]
        v = self.net(x_tau, tau, z_t, z_valid, s2_tokens, s2_mask)
        loss = masked_mse(v, u, z_valid)
        return loss, {"system15/fm_loss": loss.item(), "system15/x1_rms": x1[z_valid].float().pow(2).mean().sqrt().item()}

    def sample(self, z_t, z_valid, s2_tokens, s2_mask, dz_rms, steps: Optional[int] = None,
               method: Optional[str] = None, cfg_scale: Optional[float] = None,
               noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ODE sampling of dZ_hat / dz_rms, [B, N, D_c]. Runs under the caller's grad mode."""
        steps = self.sample_steps if steps is None else int(steps)
        method = self.sample_method if method is None else method
        cfg_scale = self.cfg_scale if cfg_scale is None else float(cfg_scale)
        x0 = torch.randn_like(z_t) if noise is None else noise
        uncond_mask = torch.zeros_like(s2_mask)

        def velocity(x, tau):
            v = self.net(x, tau, z_t, z_valid, s2_tokens, s2_mask)
            if cfg_scale != 1.0:
                v_u = self.net(x, tau, z_t, z_valid, s2_tokens, uncond_mask)
                v = v_u + cfg_scale * (v - v_u)
            return v

        x1 = ode_solve(velocity, x0, steps=steps, method=method)
        return self._to_delta(x1, z_t, dz_rms).masked_fill(~z_valid[..., None], 0.0)
