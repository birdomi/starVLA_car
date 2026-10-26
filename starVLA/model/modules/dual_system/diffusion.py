"""Generative schedulers of the Dual-System VLA, independent of any network. `build_scheduler(cfg)` picks one by
cfg["type"]; both systems default to GR00T-style flow matching.

flow_matching (GR00T N1.5 action head, starVLA/model/modules/action_model/GR00T_ActionHeader.py):
    tau = (s - u) / s,  u ~ Beta(alpha=1.5, beta=1.0) clipped to s = 0.999  (more weight on the noisy end, tau -> 0)
    x_tau = (1 - tau) eps + tau x_1,  network predicts the velocity v = x_1 - eps,  loss = MSE(v_hat, v)
    timestep input = floor(tau * num_timestep_buckets) (1000 buckets)
    inference: x ~ N(0, I), Euler: for i in 0..N-1: x += (1 / N) v_hat(x, floor(i / N * 1000))   (N = 4 by default)

x0 (DDPM forward / DDIM reverse, x0-prediction; kept as an option):
    q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I), t in {0..T-1}; network predicts x0_hat
    eps_hat = (x_t - sqrt(abar_t) x0_hat) / sqrt(1 - abar_t)
    DDIM(eta): x_{t'} = sqrt(abar_t') x0_hat + sqrt(1 - abar_t' - sigma^2) eps_hat + sigma z; the last step returns x0_hat.

regression (no diffusion; e.g. system1.diffusion.type regression):
    the network gets an all-zero action input at timestep 0 and predicts the actions themselves, loss = MSE(x_hat, x);
    inference = that single forward pass (deterministic, sample_steps / noise ignored).

Common interface: .prediction ("velocity" | "x0"), .num_train_timesteps (timestep-embedding range),
.training_inputs(x1) -> (x_t, t [B] long, aux {"noise", "target", ...}), .sample(fn(x, t) -> net output, shape, ...).
"""

import math
from typing import Callable, Optional

import torch


def make_betas(num_train_timesteps: int, schedule: str = "squaredcos_cap_v2", beta_start: float = 1e-4,
               beta_end: float = 0.02) -> torch.Tensor:
    t = num_train_timesteps
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, t, dtype=torch.float64)
    if schedule == "scaled_linear":
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, t, dtype=torch.float64) ** 2
    if schedule == "squaredcos_cap_v2":  # Nichol & Dhariwal cosine schedule
        f = lambda s: math.cos((s / t + 0.008) / 1.008 * math.pi / 2) ** 2
        return torch.tensor([min(1 - f(i + 1) / f(i), 0.999) for i in range(t)], dtype=torch.float64)
    raise ValueError(f"unknown beta schedule {schedule!r} (linear | scaled_linear | squaredcos_cap_v2)")


class X0DiffusionScheduler(torch.nn.Module):
    """cfg keys: num_train_timesteps (100), beta_schedule (squaredcos_cap_v2), beta_start, beta_end,
    sample_steps (DDIM steps at inference), eta (0 = deterministic DDIM), clip_x0 (None | float: clamp x0_hat)."""

    prediction = "x0"

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__()
        cfg = dict(cfg or {})
        self.num_train_timesteps = int(cfg.get("num_train_timesteps", 100))
        betas = make_betas(self.num_train_timesteps, str(cfg.get("beta_schedule", "squaredcos_cap_v2")),
                           float(cfg.get("beta_start", 1e-4)), float(cfg.get("beta_end", 0.02)))
        abar = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", abar.float(), persistent=False)
        self.sample_steps = int(cfg.get("sample_steps", 10))
        self.eta = float(cfg.get("eta", 0.0))
        clip = cfg.get("clip_x0", None)
        self.clip_x0 = None if clip in (None, "None", "none") else float(clip)

    # ------------------------------------------------------------------ training
    def sample_timesteps(self, batch: int, device) -> torch.Tensor:
        return torch.randint(0, self.num_train_timesteps, (batch,), device=device)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """-> (x_t, noise). x0 [B, ...], t [B] long."""
        noise = torch.randn_like(x0) if noise is None else noise
        a = self.alphas_cumprod.to(x0.device)[t].view(-1, *([1] * (x0.dim() - 1)))
        return a.sqrt() * x0 + (1 - a).sqrt() * noise, noise

    def training_inputs(self, x1: torch.Tensor, noise: Optional[torch.Tensor] = None):
        t = self.sample_timesteps(x1.shape[0], x1.device)
        x_t, noise = self.add_noise(x1, t, noise)
        return x_t, t, {"noise": noise, "target": x1}

    # ------------------------------------------------------------------ inference
    def inference_timesteps(self, steps: Optional[int] = None):
        """Descending, unique integer timesteps from T-1 to 0."""
        n = min(int(steps or self.sample_steps), self.num_train_timesteps)
        ts = torch.linspace(self.num_train_timesteps - 1, 0, n).round().long().tolist()
        return list(dict.fromkeys(ts))

    @torch.no_grad()
    def sample(self, denoise: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], shape, device, dtype=torch.float32,
               steps: Optional[int] = None, eta: Optional[float] = None, noise: Optional[torch.Tensor] = None,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """denoise(x_t [B, ...], t [B] long) -> x0_hat. Returns x0 [B, ...]. Grad stays off (the caller may wrap
        `sample_with_grad` instead when the gradient is needed)."""
        return self._sample(denoise, shape, device, dtype, steps, eta, noise, generator)

    def sample_with_grad(self, denoise, shape, device, dtype=torch.float32, steps=None, eta=None, noise=None,
                         generator=None):
        return self._sample(denoise, shape, device, dtype, steps, eta, noise, generator)

    def _sample(self, denoise, shape, device, dtype, steps, eta, noise, generator):
        eta = self.eta if eta is None else float(eta)
        abar = self.alphas_cumprod.to(device)
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator) if noise is None else noise.to(device, dtype)
        ts = self.inference_timesteps(steps)
        b = shape[0]
        x0 = x
        for i, t in enumerate(ts):
            tt = torch.full((b,), t, device=device, dtype=torch.long)
            x0 = denoise(x, tt).to(dtype)
            if self.clip_x0 is not None:
                x0 = x0.clamp(-self.clip_x0, self.clip_x0)
            if i == len(ts) - 1:
                break
            a_t, a_p = abar[t], abar[ts[i + 1]]
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()
            sigma = eta * ((1 - a_p) / (1 - a_t)).sqrt() * (1 - a_t / a_p).sqrt()
            x = a_p.sqrt() * x0 + (1 - a_p - sigma ** 2).clamp_min(0).sqrt() * eps
            if eta > 0:
                x = x + sigma * torch.randn(shape, device=device, dtype=dtype, generator=generator)
        return x0


class FlowMatchingScheduler(torch.nn.Module):
    """GR00T-style flow matching. cfg keys: beta_alpha (1.5), beta_beta (1.0), noise_s (0.999),
    num_timestep_buckets (1000), sample_steps (Euler steps at inference, 4)."""

    prediction = "velocity"

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__()
        cfg = dict(cfg or {})
        self.alpha, self.beta = float(cfg.get("beta_alpha", 1.5)), float(cfg.get("beta_beta", 1.0))
        self.noise_s = float(cfg.get("noise_s", 0.999))
        self.num_train_timesteps = int(cfg.get("num_timestep_buckets", 1000))
        self.sample_steps = int(cfg.get("sample_steps", 4))

    def sample_tau(self, batch: int, device) -> torch.Tensor:
        """tau in [0, 1] (0 = noise, 1 = data), [B] float32."""
        dist = torch.distributions.Beta(torch.tensor(self.alpha, device=device), torch.tensor(self.beta, device=device))
        u = dist.sample((batch,)).clamp(max=self.noise_s)
        return (self.noise_s - u) / self.noise_s

    def discretize(self, tau: torch.Tensor) -> torch.Tensor:
        return (tau * self.num_train_timesteps).long().clamp(0, self.num_train_timesteps - 1)

    def training_inputs(self, x1: torch.Tensor, noise: Optional[torch.Tensor] = None, tau: Optional[torch.Tensor] = None):
        """-> (x_tau, t [B] long, {"noise", "tau", "target" = x1 - noise})."""
        noise = torch.randn_like(x1) if noise is None else noise
        tau = self.sample_tau(x1.shape[0], x1.device) if tau is None else tau
        tt = tau.view(-1, *([1] * (x1.dim() - 1))).to(x1.dtype)
        x_tau = (1 - tt) * noise + tt * x1
        return x_tau, self.discretize(tau), {"noise": noise, "tau": tau, "target": x1 - noise}

    @torch.no_grad()
    def sample(self, velocity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], shape, device, dtype=torch.float32,
               steps: Optional[int] = None, eta=None, noise: Optional[torch.Tensor] = None,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """velocity(x_tau [B, ...], t [B] long) -> v_hat. Euler from tau = 0 (noise) to 1 -> x_1 (no gradient)."""
        n = int(steps or self.sample_steps)
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator) if noise is None else noise.to(device, dtype)
        dt = 1.0 / n
        for i in range(n):
            t = torch.full((shape[0],), int(i / n * self.num_train_timesteps), device=device, dtype=torch.long)
            x = x + dt * velocity(x, t).to(dtype)
        return x


class RegressionScheduler(torch.nn.Module):
    """Direct regression with the diffusion interface: input x = 0, t = 0, target = x1 (prediction "x0"). No cfg keys."""

    prediction = "x0"
    num_train_timesteps = 1
    sample_steps = 1

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__()

    def training_inputs(self, x1: torch.Tensor, noise: Optional[torch.Tensor] = None):
        zeros = torch.zeros_like(x1)
        return zeros, torch.zeros(x1.shape[0], dtype=torch.long, device=x1.device), {"noise": zeros, "target": x1}

    @torch.no_grad()
    def sample(self, denoise: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], shape, device, dtype=torch.float32,
               steps: Optional[int] = None, eta=None, noise: Optional[torch.Tensor] = None,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """denoise(x = 0, t = 0) -> the predicted actions (one forward pass; steps / noise / generator unused)."""
        x = torch.zeros(shape, device=device, dtype=dtype)
        return denoise(x, torch.zeros((shape[0],), device=device, dtype=torch.long)).to(dtype)


def build_scheduler(cfg: Optional[dict] = None):
    cfg = dict(cfg or {})
    kind = str(cfg.get("type", "flow_matching"))
    if kind == "flow_matching":
        return FlowMatchingScheduler(cfg)
    if kind == "x0":
        return X0DiffusionScheduler(cfg)
    if kind == "regression":
        return RegressionScheduler(cfg)
    raise ValueError(f"diffusion.type must be flow_matching | x0 | regression, got {kind!r}")
