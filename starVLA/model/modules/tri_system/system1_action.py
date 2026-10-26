"""System 1 — Action Expert: GR00T-style flow-matching action DiT conditioned on the predicted change dZ_hat.

Action target A = [a_t, ..., a_{t+H-1}] in R^{H x D_a} (normalized by the loader). Training (per sample, R noise
draws as GR00T's repeated_diffusion_steps):
    eps ~ N(0, I), tau_a ~ p(tau_a)
    A_tau = (1 - tau_a) eps + tau_a A,   loss = || v_phi(A_tau, tau_a | S_t, Z_t, dZ, state) - (A - eps) ||^2
Inference: A_0 ~ N(0, I), ODE dA/dtau = v_phi from 0 to 1 (euler | midpoint | heun | rk4).

ActionDiT, shapes:
    state [B, 1, D_st] --MLP--> [B, 1, d];  A_tau [B, H, D_a] --Linear--> [B, H, d]
    sequence [B, 1 + H, d] (+ learned position)                      <- self-attention tokens
    dZ    [B, N, D_c]   --Linear--> [B, N, d] (+ 2D sin-cos)          <- cross-attention stream "dz" (default only)
    (opt) S_t [B, L_s, D_s] --LN+Linear--> "s2";  Z_t --Linear--> "zt";  Z_t + dZ --Linear--> "zf"
    cond_mode "separate": one cross-attention block per stream (default); "joint": the projected streams are
    concatenated (+ type embedding) into one cross-attention memory.
    tau_a [B] --TimestepEmbedder--> adaLN-Zero of every block;  out: action tokens --FinalLayer--> v [B, H, D_a]
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .blocks import AdaLNTransformer, FinalLayer, sincos_2d
from .flow_matching import TimestepEmbedder, make_noisy, masked_mse, ode_solve, sample_tau

_SPATIAL = ("zt", "dz", "zf")


class ActionDiT(nn.Module):
    def __init__(self, action_dim: int, state_dim: int, horizon: int, feat_dim: int, s2_dim: int,
                 grid: Tuple[int, int], d_model: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, streams: Sequence[str] = ("dz",),
                 cond_mode: str = "separate", qk_norm: bool = True, grad_ckpt: bool = False):
        super().__init__()
        assert cond_mode in ("separate", "joint"), cond_mode
        assert set(streams) <= {"s2", *_SPATIAL} and streams, streams
        self.horizon, self.streams, self.cond_mode = horizon, tuple(streams), cond_mode
        self.action_in = nn.Linear(action_dim, d_model)
        self.state_in = nn.Sequential(nn.Linear(state_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)) if state_dim else None
        self.n_prefix = 1 if state_dim else 0  # the state token precedes the action tokens
        self.seq_pos = nn.Parameter(torch.randn(1, self.n_prefix + horizon, d_model) * 0.02)
        # learnable projectors aligning every condition to d_model
        self.proj = nn.ModuleDict()
        for s in self.streams:
            self.proj[s] = nn.Sequential(nn.LayerNorm(s2_dim), nn.Linear(s2_dim, d_model)) if s == "s2" else nn.Linear(feat_dim, d_model)
        self.register_buffer("grid_pos", sincos_2d(*grid, d_model)[None], persistent=False)  # [1, N, d]
        # learned stand-in for the change stream(s) (condition dropout / "no dZ" ablation at test time)
        self.null = nn.ParameterDict({s: nn.Parameter(torch.randn(1, 1, d_model) * 0.02) for s in self.streams if s in ("dz", "zf")})
        if cond_mode == "joint":
            self.type_embed = nn.ParameterDict({s: nn.Parameter(torch.randn(1, 1, d_model) * 0.02) for s in self.streams})
        self.t_embed = TimestepEmbedder(d_model)
        cross = self.streams if cond_mode == "separate" else ("cond",)
        self.transformer = AdaLNTransformer(d_model, depth, num_heads, mlp_ratio, cross, qk_norm, grad_ckpt)
        self.final = FinalLayer(d_model, action_dim)

    def embed_conditions(self, conds: Dict[str, Tuple[torch.Tensor, torch.Tensor]], drop_delta=None):
        """conds[s] = (tokens, mask): s2 [B, L_s, D_s], zt / dz / zf [B, N, D_c] -> cross-attention contexts."""
        ctx = {}
        for s in self.streams:
            x, m = conds[s]
            e = self.proj[s](x.to(self.action_in.weight.dtype))
            if s in _SPATIAL:
                e = e + self.grid_pos
            if s in self.null and drop_delta is not None:
                e = torch.where(drop_delta[:, None, None], self.null[s].to(e.dtype).expand_as(e), e)
            ctx[s] = (e, m)
        if self.cond_mode == "joint":
            ctx = {"cond": (torch.cat([ctx[s][0] + self.type_embed[s] for s in self.streams], 1),
                            torch.cat([ctx[s][1] for s in self.streams], 1))}
        return ctx

    def forward(self, a_tau, tau, state, ctx):
        """a_tau [B, H, D_a], tau [B], state [B, 1, D_st] | None, ctx from embed_conditions -> v [B, H, D_a]."""
        b = a_tau.shape[0]
        parts = []
        if self.state_in is not None:
            parts.append(self.state_in(state.to(self.action_in.weight.dtype)))
        h = torch.cat(parts + [self.action_in(a_tau.to(self.action_in.weight.dtype))], dim=1) + self.seq_pos
        c = self.t_embed(tau)
        h = self.transformer(h, c, ctx)
        return self.final(h[:, self.n_prefix:], c)


class ActionExpert(nn.Module):
    """System 1 wrapper: flow-matching loss and ODE sampling of the action chunk."""

    def __init__(self, action_dim: int, state_dim: int, horizon: int, feat_dim: int, s2_dim: int,
                 grid: Tuple[int, int], cfg: Dict):
        super().__init__()
        self.net = ActionDiT(
            action_dim, state_dim, horizon, feat_dim, s2_dim, grid, d_model=int(cfg.get("d_model", 768)),
            depth=int(cfg.get("depth", 12)), num_heads=int(cfg.get("num_heads", 12)),
            mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
            streams=list(cfg.get("cond_streams", ["dz"])), cond_mode=str(cfg.get("cond_mode", "separate")),
            qk_norm=bool(cfg.get("qk_norm", True)), grad_ckpt=bool(cfg.get("grad_ckpt", False)),
        )
        self.horizon, self.action_dim = horizon, action_dim
        self.streams = self.net.streams
        # GR00T default: tau_a = s * (1 - Beta(1.5, 1)), i.e. more weight on the noisy end
        self.tau_schedule = dict(cfg.get("tau_schedule", {"schedule": "beta", "beta_a": 1.5, "beta_b": 1.0, "beta_s": 0.999}))
        self.num_noise_samples = int(cfg.get("num_noise_samples", 4))
        self.delta_drop_prob = float(cfg.get("delta_drop_prob", 0.0))
        self.sample_steps = int(cfg.get("sample_steps", 4))
        self.sample_method = str(cfg.get("sample_method", "euler"))

    def loss(self, actions, state, conds) -> Tuple[torch.Tensor, Dict[str, float]]:
        """actions [B, >=H, D_a] (last H used), state [B, 1, D_st] | None, conds (see ActionDiT.embed_conditions)."""
        target = actions[:, -self.horizon:].float()
        b = target.shape[0]
        drop = None
        if self.training and self.delta_drop_prob > 0:
            drop = torch.rand(b, device=target.device) < self.delta_drop_prob
        ctx = self.net.embed_conditions(conds, drop)  # projected once, shared by the R noise draws
        r = self.num_noise_samples
        rep = lambda t: None if t is None else t.repeat(r, *([1] * (t.dim() - 1)))
        ctx = {k: (rep(e), rep(m)) for k, (e, m) in ctx.items()}
        target, state = rep(target), rep(state)
        tau = sample_tau(target.shape[0], target.device, **self.tau_schedule)
        a_tau, u, _ = make_noisy(target, tau)
        v = self.net(a_tau, tau, state, ctx)
        loss = masked_mse(v, u)
        return loss, {"system1/fm_loss": loss.item()}

    def sample(self, state, conds, steps: Optional[int] = None, method: Optional[str] = None, noise=None,
               drop_delta=None) -> torch.Tensor:
        """-> normalized action chunk [B, H, D_a]. drop_delta [B] bool replaces the change streams by null tokens."""
        steps = self.sample_steps if steps is None else int(steps)
        method = self.sample_method if method is None else method
        ctx = self.net.embed_conditions(conds, drop_delta)
        b = next(iter(ctx.values()))[0].shape[0]
        x0 = torch.randn(b, self.horizon, self.action_dim, device=self.net.seq_pos.device) if noise is None else noise
        return ode_solve(lambda x, t: self.net(x, t, state, ctx), x0, steps=steps, method=method)
