"""System 2 — slow, long-horizon keyframe action predictor (GR00T-style flow matching over 61 keyframes).

Condition (built once per System 2 observation):
    vlm_tokens    [B, L_v, D_v] Qwen last hidden states (vision + text), padding masked --vlm_projector-->       [B, L_v, d]
    radio_prefix  [B, P, D_c]   frozen C-RADIOv4 summary tokens (SigLIP2-g / DINOv3 CLS = its 2304-d summary, P = 2)
                                --radio_prefix_projector_s2 (+ per-slot embedding)-->                            [B, P, d]
    radio_tokens  [B, N, D_c]   frozen C-RADIOv4 patches, artifact tokens masked --radio_projector_s2 (+2D pos)--> [B, N, d]
    state         [B, 1, D_st]  optional --state_projector (MLP)-->                                              [B, 1, d]
    s2_context = concat([vlm, radio_prefix, radio, (state)], dim=1)   (TOKEN dimension)            [B, L_v + P + N (+1), d]
    (each source also gets a learned source embedding so the shared cross-attention can tell them apart)
    vlm_age_embed (framework dual_system.stale_input vlm): the VLM tokens may come from an older frame than C-RADIO /
    state; MLP(sin-cos(age_s * rope_time_scale)) is added to every VLM token so the DiT knows how stale they are.
    V camera views (framework dual_system.num_views): radio_prefix [B, V * P, D_c] / radio_tokens [B, V * N, D_c], the
    views concatenated on the token dim (view-major); one prefix slot embedding per (view, slot), and the patch tokens
    get the 2-D grid position + a learned per-view embedding (RadioPositions).

Keyframe DiT (primary sequence = the K noisy keyframe action tokens):
    x_t [B, K, D_a] --action_in--> [B, K, d] (+ sin-cos of the keyframe time)  -> depth x DiTBlock:
        full bidirectional self-attention with temporal RoPE over t_i, cross-attention to s2_context, MLP,
        adaLN-Zero(flow / diffusion step t) -> FinalLayer -> v_hat (flow matching) or x0_hat [B, K, D_a]

Training (default diffusion.type flow_matching, GR00T style; x_1 = GT keyframes = the dense GT sampled at t_i):
    tau = (0.999 - Beta(1.5, 1)) / 0.999, x_tau = (1 - tau) eps + tau x_1, v_hat = DiT(x_tau, floor(1000 tau))
    L_S2 = MSE(interp(v_hat) at the dense GT timestamps, dense GT - interp(eps))   over valid dense steps (only loss)
    (diffusion.type x0: x0_hat = DiT(q_sample(x_1, t), t), L_S2 = MSE(interp(x0_hat), dense GT))
Inference: Euler (4 steps) / DDIM -> clean keyframes [B, K, D_a].
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import VisualFeatures
from .diffusion import build_scheduler
from .interpolation import DenseTrajectoryLoss, KeyframeTimeline, TrajectoryInterpolator
from .layers import ContinuousRoPE, DiTStack, FinalLayer, TimestepEmbedder, sincos_1d, sincos_2d


def token_projector(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim))


def state_projector(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))


class PrefixProjector(nn.Module):
    """C-RADIO prefix tokens [B, P, D] -> [B, P, d]: LayerNorm + Linear + a learned embedding per slot (every slot has
    its own role: one CLS per teacher)."""

    def __init__(self, in_dim: int, num_prefix: int, d_model: int):
        super().__init__()
        self.proj = token_projector(in_dim, d_model)
        self.slot_embed = nn.Parameter(torch.randn(1, num_prefix, d_model) * 0.02)
        self.num_prefix = num_prefix

    def forward(self, prefix: Optional[torch.Tensor]) -> torch.Tensor:
        if prefix is None or prefix.shape[1] != self.num_prefix:
            raise ValueError(f"expected {self.num_prefix} C-RADIO prefix tokens, got "
                             f"{None if prefix is None else tuple(prefix.shape)}")
        return self.proj(prefix.to(self.proj[1].weight.dtype)) + self.slot_embed


class RadioPositions(nn.Module):
    """Position of every C-RADIO patch token, [1, V * N, d] for V camera views concatenated on the token dim (view-major):
    the fixed 2-D sin-cos grid position + a learned embedding per view (V > 1 only, so single-view state dicts are
    unchanged)."""

    def __init__(self, grid: Tuple[int, int], d_model: int, num_views: int = 1):
        super().__init__()
        self.num_views = int(num_views)
        self.register_buffer("grid_pos", sincos_2d(*grid, d_model)[None], persistent=False)  # [1, N, d]
        self.view_embed = nn.Parameter(torch.randn(1, self.num_views, 1, d_model) * 0.02) if self.num_views > 1 else None

    def forward(self) -> torch.Tensor:
        if self.view_embed is None:
            return self.grid_pos
        return (self.grid_pos[:, None] + self.view_embed).flatten(1, 2)


class System2Conditioner(nn.Module):
    def __init__(self, vlm_dim: int, radio_dim: int, radio_grid: Tuple[int, int], state_dim: int, d_model: int,
                 num_prefix: int = 0, prefix_dim: Optional[int] = None, num_views: int = 1, vlm_age_embed: bool = False,
                 age_time_scale: float = 20.0):
        super().__init__()
        self.vlm_projector = token_projector(vlm_dim, d_model)
        # age (s) of the VLM tokens relative to the C-RADIO / state frame -> added to every VLM token; zero-init output
        self.vlm_age_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)) if vlm_age_embed else None
        if self.vlm_age_mlp is not None:
            nn.init.zeros_(self.vlm_age_mlp[2].weight)
            nn.init.zeros_(self.vlm_age_mlp[2].bias)
        self.age_time_scale = float(age_time_scale)
        # one prefix slot embedding per (view, prefix token): radio.prefix is [B, V * P, D] (view-major)
        self.radio_prefix_projector_s2 = (PrefixProjector(prefix_dim or radio_dim, num_prefix * num_views, d_model)
                                          if num_prefix else None)
        self.radio_projector_s2 = token_projector(radio_dim, d_model)
        self.state_projector = state_projector(state_dim, d_model) if state_dim else None
        names = ["vlm"] + (["radio_prefix"] if num_prefix else []) + ["radio"] + (["state"] if state_dim else [])
        self.source_embed = nn.ParameterDict({k: nn.Parameter(torch.randn(1, 1, d_model) * 0.02) for k in names})
        self.radio_pos = RadioPositions(radio_grid, d_model, num_views)  # [1, V * N, d]

    @property
    def num_prefix(self) -> int:
        return 0 if self.radio_prefix_projector_s2 is None else self.radio_prefix_projector_s2.num_prefix

    def forward(self, vlm_tokens, vlm_mask, radio: VisualFeatures, state: Optional[torch.Tensor] = None,
                vlm_age_s: Optional[torch.Tensor] = None):
        """vlm_age_s [B] (or None = 0): age of the VLM tokens' frame, used with vlm_age_embed only.
        -> (s2_context [B, L_v + P + N (+1), d], mask [B, L_v + P + N (+1)] bool)."""
        dt = self.vlm_projector[1].weight.dtype
        vlm = self.vlm_projector(vlm_tokens.to(dt)) + self.source_embed["vlm"]
        if self.vlm_age_mlp is not None:
            b = vlm.shape[0]
            age = torch.zeros(b, device=vlm.device) if vlm_age_s is None else \
                torch.as_tensor(vlm_age_s, dtype=torch.float32, device=vlm.device).reshape(-1).expand(b)
            vlm = vlm + self.vlm_age_mlp(sincos_1d(age * self.age_time_scale, vlm.shape[-1]).to(dt))[:, None]
        parts, masks = [vlm], [vlm_mask.bool()]
        if self.radio_prefix_projector_s2 is not None:
            pre = self.radio_prefix_projector_s2(radio.prefix) + self.source_embed["radio_prefix"]
            parts.append(pre)
            masks.append(torch.ones(pre.shape[:2], dtype=torch.bool, device=pre.device))
        parts.append(self.radio_projector_s2(radio.tokens.to(dt)) + self.radio_pos() + self.source_embed["radio"])
        masks.append(radio.valid.bool())
        if self.state_projector is not None:
            if state is None:
                raise ValueError("System 2 is state-conditioned (system2.use_state): pass state [B, 1, D_st]")
            parts.append(self.state_projector(state.to(dt)) + self.source_embed["state"])
            masks.append(torch.ones(state.shape[:2], dtype=torch.bool, device=state.device))
        return torch.cat(parts, dim=1), torch.cat(masks, dim=1)


class KeyframeDiT(nn.Module):
    def __init__(self, action_dim: int, timeline: KeyframeTimeline, num_train_timesteps: int, d_model: int = 768,
                 depth: int = 12, num_heads: int = 12, mlp_ratio: float = 4.0, qk_norm: bool = True,
                 rope_time_scale: float = 20.0, rope_base: float = 1000.0, abs_time_embed: bool = True,
                 grad_ckpt: bool = False):
        super().__init__()
        self.action_in = nn.Linear(action_dim, d_model)
        self.register_buffer("key_times", timeline.times.clone(), persistent=False)  # [K]
        self.abs_time_embed = abs_time_embed
        self.time_scale = float(rope_time_scale)
        self.t_embed = TimestepEmbedder(d_model, num_train_timesteps)
        rope = ContinuousRoPE(d_model // num_heads, rope_time_scale, rope_base)
        self.stack = DiTStack(d_model, depth, num_heads, mlp_ratio, qk_norm, rope=rope, grad_ckpt=grad_ckpt)
        self.final = FinalLayer(d_model, action_dim)

    def forward(self, x_t, t, ctx, ctx_mask):
        """x_t [B, K, D_a], t [B] long, ctx [B, L, d], ctx_mask [B, L] -> v_hat / x0_hat [B, K, D_a]."""
        b = x_t.shape[0]
        h = self.action_in(x_t.to(self.action_in.weight.dtype))
        pos = self.key_times.to(h.device)[None].expand(b, -1)  # System 2 reference time (s)
        if self.abs_time_embed:
            h = h + sincos_1d(pos * self.time_scale, h.shape[-1]).to(h.dtype)
        c = self.t_embed(t)
        h = self.stack(h, c, ctx.to(h.dtype), ctx_mask, pos=pos)
        return self.final(h, c)


class System2KeyframePlanner(nn.Module):
    """System 2 heads: conditioner (projectors) + keyframe DiT + flow-matching (or x0) scheduler + interpolation loss.

    cfg keys: d_model, depth, num_heads, mlp_ratio, qk_norm, rope_time_scale, rope_base, abs_time_embed, grad_ckpt,
    use_state, use_radio_prefix, vlm_age_embed, num_noise_samples (noise draws per sample per step), diffusion {...},
    interpolation {groups, default_mode, strict}.
    """

    def __init__(self, vlm_dim: int, radio_dim: int, radio_grid: Tuple[int, int], state_dim: int, action_dim: int,
                 timeline: KeyframeTimeline, cfg: Dict, num_prefix: int = 0, prefix_dim: Optional[int] = None,
                 num_views: int = 1):
        super().__init__()
        d = int(cfg.get("d_model", 768))
        self.timeline, self.action_dim = timeline, action_dim
        self.scheduler = build_scheduler(cfg.get("diffusion", {}))
        self.conditioner = System2Conditioner(vlm_dim, radio_dim, radio_grid, state_dim if cfg.get("use_state", True) else 0, d,
                                              num_prefix=num_prefix if cfg.get("use_radio_prefix", True) else 0,
                                              prefix_dim=prefix_dim, num_views=num_views,
                                              vlm_age_embed=bool(cfg.get("vlm_age_embed", False)),
                                              age_time_scale=float(cfg.get("rope_time_scale", 20.0)))
        self.dit = KeyframeDiT(
            action_dim, timeline, self.scheduler.num_train_timesteps, d_model=d, depth=int(cfg.get("depth", 12)),
            num_heads=int(cfg.get("num_heads", 12)), mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
            qk_norm=bool(cfg.get("qk_norm", True)), rope_time_scale=float(cfg.get("rope_time_scale", 20.0)),
            rope_base=float(cfg.get("rope_base", 1000.0)), abs_time_embed=bool(cfg.get("abs_time_embed", True)),
            grad_ckpt=bool(cfg.get("grad_ckpt", False)),
        )
        ic = dict(cfg.get("interpolation", {}) or {})
        self.interpolator = TrajectoryInterpolator(action_dim, ic.get("groups", None), ic.get("default_mode", "linear"),
                                                   strict=bool(ic.get("strict", True)))
        self.dense_loss = DenseTrajectoryLoss(self.interpolator)
        self.num_noise_samples = int(cfg.get("num_noise_samples", 1))
        self.register_buffer("key_times", timeline.times.clone(), persistent=False)

    @property
    def uses_state(self) -> bool:
        return self.conditioner.state_projector is not None

    def extract_keyframes(self, dense: torch.Tensor, dense_times: torch.Tensor) -> torch.Tensor:
        """Clean GT keyframes: the dense GT trajectory [B, Q, D_a] (timestamps [B, Q]) sampled at t_i -> [B, K, D_a]."""
        q = self.key_times.to(dense.device)[None].expand(dense.shape[0], -1)
        kf, in_range = self.interpolator(dense.float(), dense_times, q)
        if not bool(in_range.all()):
            raise ValueError("the dense action window does not cover the keyframe span "
                             f"[{self.timeline.t_min:+.2f}, {self.timeline.t_max:+.2f}] s")
        return kf

    def loss(self, ctx, ctx_mask, dense_gt, dense_times, dense_valid) -> Tuple[torch.Tensor, Dict[str, float]]:
        """ctx [B, L, d] / ctx_mask [B, L] from the conditioner; dense_gt [B, Q, D_a]; dense_times, dense_valid [B, Q]."""
        x1 = self.extract_keyframes(dense_gt, dense_times)
        r = self.num_noise_samples
        rep = (lambda v: v.repeat(r, *([1] * (v.dim() - 1)))) if r > 1 else (lambda v: v)
        x1, ctx, ctx_mask, dense_gt, dense_times, dense_valid = map(rep, (x1, ctx, ctx_mask, dense_gt, dense_times, dense_valid))
        x_t, t, aux = self.scheduler.training_inputs(x1)
        out = self.dit(x_t, t, ctx, ctx_mask)
        if self.scheduler.prediction == "velocity":
            loss, info = self.dense_loss.velocity(out, aux["noise"], self.key_times, dense_gt, dense_times, dense_valid)
        else:
            loss, info = self.dense_loss(out, self.key_times, dense_gt, dense_times, dense_valid)
        key = "system2/dense_fm_loss" if self.scheduler.prediction == "velocity" else "system2/dense_mse"
        return loss, {key: loss.item(), "system2/valid_frac": info["valid_frac"]}

    def sample(self, ctx, ctx_mask, steps: Optional[int] = None, noise: Optional[torch.Tensor] = None,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """-> clean keyframes [B, K, D_a] (no gradient)."""
        b = ctx.shape[0]
        return self.scheduler.sample(lambda x, t: self.dit(x, t, ctx, ctx_mask), (b, self.timeline.num, self.action_dim),
                                     ctx.device, steps=steps, noise=noise, generator=generator)
