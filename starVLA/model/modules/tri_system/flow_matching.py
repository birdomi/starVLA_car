"""Conditional flow-matching utilities shared by System 1.5 and System 1.

Convention (same direction as the repo's GR00T action head): tau in [0, 1], Gaussian noise at
tau = 0, data at tau = 1,

    X_tau = (1 - tau) * eps + tau * X_1,        u_tau = X_1 - eps   (velocity target),

and inference integrates dX/dtau = v_theta(X, tau) from tau = 0 (noise) to tau = 1.
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn


def sample_tau(n: int, device, schedule: str = "logit_normal", mean: float = 0.0, std: float = 1.0,
               beta_a: float = 1.5, beta_b: float = 1.0, beta_s: float = 0.999) -> torch.Tensor:
    """Flow timesteps [n] in [0, 1].

    uniform      : U(0, 1)
    logit_normal : sigmoid(N(mean, std)) (SD3); mean < 0 puts more weight on the noisy end
    beta         : beta_s * (1 - Beta(a, b)) (GR00T action head default, noisy end)
    """
    if schedule == "uniform":
        return torch.rand(n, device=device)
    if schedule == "logit_normal":
        return torch.sigmoid(torch.randn(n, device=device) * std + mean)
    if schedule == "beta":
        d = torch.distributions.Beta(torch.tensor(float(beta_a)), torch.tensor(float(beta_b)))
        return (beta_s * (1.0 - d.sample((n,)))).to(device)
    raise ValueError(f"unknown tau schedule {schedule!r}")


def _bcast(tau: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return tau.view(-1, *([1] * (like.dim() - 1))).to(like.dtype)


def make_noisy(x1: torch.Tensor, tau: torch.Tensor, eps: Optional[torch.Tensor] = None):
    """Returns (X_tau, u_tau = X_1 - eps, eps)."""
    eps = torch.randn_like(x1) if eps is None else eps
    t = _bcast(tau, x1)
    return (1.0 - t) * eps + t * x1, x1 - eps, eps


def velocity_from_x(x_hat: torch.Tensor, x_tau: torch.Tensor, tau: torch.Tensor, min_gap: float = 0.05) -> torch.Tensor:
    """v = (x_hat - X_tau) / (1 - tau) for an x-predicting network (JiT-style), gap clamped near tau = 1."""
    return (x_hat - x_tau) / (1.0 - _bcast(tau, x_tau)).clamp_min(min_gap)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """MSE over [B, N, C] tensors; token_mask [B, N] (True = counted) averages over valid tokens only."""
    err = (pred.float() - target.float()).pow(2).mean(-1)  # [B, N]
    if token_mask is None:
        return err.mean()
    m = token_mask.to(err.dtype)
    return (err * m).sum() / m.sum().clamp_min(1.0)


def ode_solve(velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x0: torch.Tensor, steps: int = 10,
              method: str = "euler", t0: float = 0.0, t1: float = 1.0) -> torch.Tensor:
    """Integrate dx/dtau = velocity_fn(x, tau[B]) from t0 to t1 with a fixed-step solver.

    euler (1 NFE/step), midpoint / heun (2), rk4 (4). Gradients flow if the caller enables them
    (used by System 1's `pred_grad` conditioning); wrap in torch.no_grad() otherwise.
    """
    ts = torch.linspace(t0, t1, steps + 1, device=x0.device, dtype=torch.float32)
    b = x0.shape[0]
    x = x0
    for i in range(steps):
        t, dt = ts[i], ts[i + 1] - ts[i]
        tb = t.expand(b)
        if method == "euler":
            x = x + dt * velocity_fn(x, tb)
        elif method == "midpoint":
            k1 = velocity_fn(x, tb)
            x = x + dt * velocity_fn(x + 0.5 * dt * k1, (t + 0.5 * dt).expand(b))
        elif method == "heun":
            k1 = velocity_fn(x, tb)
            k2 = velocity_fn(x + dt * k1, ts[i + 1].expand(b))
            x = x + 0.5 * dt * (k1 + k2)
        elif method == "rk4":
            th = (t + 0.5 * dt).expand(b)
            k1 = velocity_fn(x, tb)
            k2 = velocity_fn(x + 0.5 * dt * k1, th)
            k3 = velocity_fn(x + 0.5 * dt * k2, th)
            k4 = velocity_fn(x + dt * k3, ts[i + 1].expand(b))
            x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown ODE method {method!r}")
    return x


class TimestepEmbedder(nn.Module):
    """tau in [0, 1] -> [B, dim]: sinusoidal features of tau * scale, then a 2-layer MLP (DiT)."""

    def __init__(self, dim: int, freq_dim: int = 256, scale: float = 1000.0):
        super().__init__()
        self.freq_dim, self.scale = freq_dim, scale
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=tau.device, dtype=torch.float32) / half)
        args = tau.float()[:, None] * self.scale * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))
