"""System 1 — fast, short-horizon (1 s) action diffusion conditioned on System 2's keyframe trajectory.

Interface to System 2 = its predicted keyframe ACTIONS (never System 2's DiT hidden states):
    TrajectoryTokenizer: keyframes [B, K=61, D_a] --MLP--> [B, K, d1] + token-type embedding (past / current / future,
    by the keyframe time in System 2's reference) -> trajectory_tokens [B, K, d1]

Short-horizon DiT, primary sequence (full bidirectional self-attention over all of it):
    X = concat([noisy_action_tokens [B, N_a, d1], past [B, 20, d1], current [B, 1, d1], future [B, 40, d1]], dim=1)
    N_total = N_a + 61. Every token has a timestamp RELATIVE TO THE CURRENT SYSTEM 1 EXECUTION TIME:
        action token k:      k / action_rate                         (k = 0 .. N_a - 1, the next 1 s)
        trajectory token i:  t_i - delay,  delay = t_S1 - t_S2obs >= 0 (age of the System 2 plan)
    -> temporal RoPE on q / k (+ optional sin-cos absolute time embedding), so a stale plan is placed correctly.
    cross-attention: C-RADIOv4 of the CURRENT frame: summary tokens --radio_prefix_projector_s1--> [B, P, d1] and
                     patch tokens --radio_projector_s1 (+2D pos)--> [B, N, d1], concatenated on the token dim
                     + the robot state AT THE SYSTEM 1 TIME --state_projector (MLP) + type embedding--> [B, 1, d1]
                     (system1.use_state; kept in the cross-attention memory so X stays N_a + 61 tokens)
                     V camera views: [B, V * P + V * N (+1), d1], views on the token dim (as System 2's conditioner)
    adaLN-Zero(flow step t); FinalLayer on the N_a action positions -> v_hat [B, N_a, D_a]
    (layout above = the default; system1.plan_input condition + state_input sequence instead gives
     X = [noisy_action_tokens [B, N_a, d1], state token [B, 1, d1] at 0 s], cross-attention memory =
     [trajectory tokens [B, 61, d1], C-RADIO prefix + patches], and the cross-attention is RoPE'd as well: q at the action
     times, k at t_i - delay for the trajectory tokens and 0 s (now) for the C-RADIO tokens)
Training (default, GR00T flow matching): tau = (0.999 - Beta(1.5, 1)) / 0.999, x_tau = (1 - tau) eps + tau a,
    L_S1 = MSE(v_hat, a - eps) over valid steps, a = a[t_S1 .. t_S1 + 1 s); inference: Euler, 4 steps.
    (diffusion.type x0: L_S1 = MSE(x0_hat, a), DDIM; diffusion.type regression: all-zero action input at t = 0,
     L_S1 = MSE(a_hat, a), one forward pass at inference.)

plan_input action (diffusion.type regression only): System 2's plan enters ONLY as the action-token input. The framework
    cuts the plan (absolute normalised actions) to the first plan_chunk_s seconds of System 1's window and smooths it in
    time (smooth_time: Gaussian, plan_smooth_sigma_s, edge-replicated inside the chunk, so nothing beyond the chunk is
    read) -> plan_chunk [B, n_in, D_a]. X = [N_a action tokens], token k < n_in = embed(plan_chunk[k]), token k >= n_in =
    embed(0) + a learned no-plan embedding; no trajectory tokens (no tokenizer); memory = [C-RADIO, state] as usual.

ActionMap option (system1.action_map {groups, names, plan_grid}; default off = the layout above):
    the action chunk is a grid, rows = time, columns = G joint groups (as GR00T_GridDiTHeader's ActionMap DiT):
    every (step, group) cell is one token (own Linear per group + learned group embedding), N_a x G action tokens
    row-major; plan_grid (default true) splits the 61 keyframes the same way into 61 x G trajectory tokens (own MLP
    per group + group embedding + past / current / future type). All cells of a row share its timestamp (temporal
    RoPE, no group axis); FinalLayer has one zero-initialised output projection per group.
    X = [N_a x G action cells, 61 x G (or 61) trajectory tokens].
    noise_std [G] (optional, flow matching only): the flow's source noise per column is N(0, noise_std_g^2) instead of
    N(0, 1), in training (x_tau = (1 - tau) sigma eps + tau a, target a - sigma eps) and as the Euler start at inference
    (= a higher signal-to-noise ratio for the columns with sigma < 1).
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .backbone import VisualFeatures
from .diffusion import RegressionScheduler, build_scheduler
from .interpolation import KeyframeTimeline
from .layers import ContinuousRoPE, DiTStack, FinalLayer, TimestepEmbedder, modulate, sincos_1d
from .system2 import PrefixProjector, RadioPositions, state_projector, token_projector

# where System 1 reads its inputs: the plan (trajectory tokens) and the state token, each in the self-attention
# sequence or in the cross-attention memory ("condition")
_PLAN_INPUTS = ("sequence", "condition", "action")
_STATE_INPUTS = ("condition", "sequence")


def smooth_time(x: torch.Tensor, sigma_steps: float) -> torch.Tensor:
    """x [B, N, D] -> Gaussian-smoothed along N (sigma in steps, kernel +-3 sigma, edges replicated; sigma <= 0: x)."""
    if sigma_steps <= 0:
        return x
    r = int(math.ceil(3 * sigma_steps))
    w = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=torch.float32) / sigma_steps) ** 2)
    w = (w / w.sum()).to(x.dtype)
    xp = torch.cat([x[:, :1].expand(-1, r, -1), x, x[:, -1:].expand(-1, r, -1)], dim=1)
    return (xp.unfold(1, 2 * r + 1, 1) * w).sum(-1)  # [B, N, D, 2r + 1] . w


def parse_action_groups(specs: Sequence, action_dim: int) -> List[List[int]]:
    """system1.action_map.groups: one index spec per ActionMap column ("0:2", "2:9,22:29" half-open, or [0, 1, ...])
    -> dim lists. The columns must partition 0 .. action_dim - 1."""
    groups = []
    for spec in specs:
        dims = []
        if isinstance(spec, str):
            for part in spec.replace(" ", "").split(","):
                if ":" in part:
                    lo, hi = part.split(":")
                    dims.extend(range(int(lo), int(hi)))
                elif part:
                    dims.append(int(part))
        else:
            dims = [int(d) for d in spec]
        groups.append(dims)
    if not groups or any(not g for g in groups) or sorted(d for g in groups for d in g) != list(range(action_dim)):
        raise ValueError(f"system1.action_map.groups must partition the {action_dim} action dims into non-empty columns, "
                         f"got {groups}")
    return groups


class _GroupIndex(nn.Module):
    """Index buffers of the ActionMap columns: split [..., D] into per-group slices / merge them back."""

    def __init__(self, groups: List[List[int]]):
        super().__init__()
        self.num = len(groups)
        self.dims = [len(g) for g in groups]
        for i, g in enumerate(groups):
            self.register_buffer(f"idx{i}", torch.tensor(g, dtype=torch.long), persistent=False)
        perm = torch.tensor([d for g in groups for d in g], dtype=torch.long)
        self.register_buffer("inv_perm", torch.argsort(perm), persistent=False)

    def split(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [x.index_select(-1, getattr(self, f"idx{i}")) for i in range(self.num)]

    def merge(self, parts: List[torch.Tensor]) -> torch.Tensor:
        """per-group [..., d_g] (group order) -> [..., D] (loader order)."""
        return torch.cat(parts, dim=-1).index_select(-1, self.inv_perm)


class PlanNoise:
    """Structured noise added to System 2's sampled plan while training System 1 (system1.traj_noise), so System 1
    sees plans as wrong as System 2's held-out plans instead of its near-perfect plans on the training episodes.

        n[b, k, d] = s_b * time_std[k] * dim_std[d] * z[b, k, c(d)],   s_b ~ U(0, scale_max) w.p. prob, else 0
        z = sqrt(white_frac) * white + sqrt(1 - white_frac) * AR(1)(rho)  (unit variance per keyframe)
    c(d) maps every dim of a shared_dims group to the group's first dim (one shared error, e.g. all fingers of a hand
    = a grasp-timing error). scale_max = sqrt(3) gives E[s^2] = 1, i.e. on average the full time_std x dim_std
    profile. cfg: scale_max (0 = off), prob, white_frac, rho, std (dim_std fallback), time_std [K] | None (= 1),
    dim_std [D] | None (= std), shared_dims [[d, ...], ...]. No parameters (not part of the state dict)."""

    def __init__(self, cfg: Optional[dict], num_keyframes: int, action_dim: int):
        cfg = dict(cfg or {})
        self.scale_max = float(cfg.get("scale_max", 0.0))
        self.prob = float(cfg.get("prob", 1.0))
        self.white_frac = float(cfg.get("white_frac", 0.35))
        self.rho = float(cfg.get("rho", 0.91))
        if not (0.0 <= self.white_frac <= 1.0 and 0.0 <= self.rho < 1.0):
            raise ValueError(f"traj_noise: need 0 <= white_frac <= 1 and 0 <= rho < 1, got {self.white_frac}, {self.rho}")
        ts, ds = cfg.get("time_std", None), cfg.get("dim_std", None)
        self.time_std = torch.ones(num_keyframes) if ts is None else torch.tensor([float(x) for x in ts])
        self.dim_std = (torch.full((action_dim,), float(cfg.get("std", 0.1))) if ds is None
                        else torch.tensor([float(x) for x in ds]))
        if self.time_std.numel() != num_keyframes or self.dim_std.numel() != action_dim:
            raise ValueError(f"traj_noise: time_std needs {num_keyframes} entries and dim_std {action_dim}, got "
                             f"{self.time_std.numel()} / {self.dim_std.numel()}")
        chan = list(range(action_dim))
        for group in cfg.get("shared_dims", None) or []:
            group = [int(d) for d in group]
            for d in group:
                chan[d] = group[0]
        self.chan = torch.tensor(chan)

    @property
    def enabled(self) -> bool:
        return self.scale_max > 0 and self.prob > 0

    def sample(self, batch: int, device, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """-> noise [B, K, D] float32."""
        k, d = self.time_std.numel(), self.dim_std.numel()
        rn = lambda *s: torch.randn(*s, device=device, generator=generator)
        slow = [rn(batch, d)]
        for _ in range(k - 1):
            slow.append(self.rho * slow[-1] + (1 - self.rho ** 2) ** 0.5 * rn(batch, d))
        z = self.white_frac ** 0.5 * rn(batch, k, d) + (1 - self.white_frac) ** 0.5 * torch.stack(slow, 1)
        z = z[..., self.chan.to(device)]
        s = torch.rand(batch, device=device, generator=generator) * self.scale_max
        s = s * (torch.rand(batch, device=device, generator=generator) < self.prob)
        prof = self.time_std.to(device)[:, None] * self.dim_std.to(device)[None]
        return s[:, None, None] * prof[None] * z


class TrajectoryTokenizer(nn.Module):
    """One trajectory token per System 2 keyframe: MLP(action) + type embedding (0 past, 1 current, 2 future)."""

    def __init__(self, action_dim: int, d_model: int, timeline: KeyframeTimeline):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(action_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.type_embed = nn.Embedding(3, d_model)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        self.register_buffer("types", timeline.types.clone(), persistent=False)  # [K]

    def forward(self, keyframes: torch.Tensor) -> torch.Tensor:
        """keyframes [B, K, D_a] -> trajectory_tokens [B, K, d]."""
        if keyframes.shape[1] != self.types.numel():
            raise ValueError(f"expected {self.types.numel()} keyframes, got {keyframes.shape[1]}")
        x = self.proj(keyframes.to(self.proj[0].weight.dtype))
        return x + self.type_embed(self.types)[None].to(x.dtype)


class GridTrajectoryTokenizer(nn.Module):
    """ActionMap plan tokens: one token per (keyframe, joint group) cell, K x G row-major:
    MLP_g(keyframe[group g dims]) + group embedding + type embedding (0 past, 1 current, 2 future)."""

    def __init__(self, groups: List[List[int]], d_model: int, timeline: KeyframeTimeline):
        super().__init__()
        self.index = _GroupIndex(groups)
        self.group_proj = nn.ModuleList(nn.Sequential(nn.Linear(len(g), d_model), nn.SiLU(), nn.Linear(d_model, d_model))
                                        for g in groups)
        self.group_embed = nn.Parameter(torch.randn(len(groups), d_model) * 0.02)
        self.type_embed = nn.Embedding(3, d_model)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        self.register_buffer("types", timeline.types.clone(), persistent=False)  # [K]

    def forward(self, keyframes: torch.Tensor) -> torch.Tensor:
        """keyframes [B, K, D_a] -> trajectory_tokens [B, K * G, d] (row-major: keyframe, group)."""
        if keyframes.shape[1] != self.types.numel():
            raise ValueError(f"expected {self.types.numel()} keyframes, got {keyframes.shape[1]}")
        dt = self.group_proj[0][0].weight.dtype
        x = torch.stack([p(k.to(dt)) for p, k in zip(self.group_proj, self.index.split(keyframes))], dim=2)  # [B, K, G, d]
        x = x + self.group_embed.to(x.dtype) + self.type_embed(self.types)[None, :, None].to(x.dtype)
        return x.flatten(1, 2)


class GroupFinalLayer(nn.Module):
    """FinalLayer for ActionMap cells: shared adaLN-modulated LayerNorm, one zero-initialised projection per group."""

    def __init__(self, dim: int, index: _GroupIndex):
        super().__init__()
        self.index = index
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.outs = nn.ModuleList(nn.Linear(dim, d) for d in index.dims)
        for m in (self.ada[1], *self.outs):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, c):
        """x [B, N_a * G, d] row-major cells, c [B, d] -> [B, N_a, D_a] (loader order)."""
        shift, scale = self.ada(c).chunk(2, dim=-1)
        h = modulate(self.norm(x), shift, scale)
        h = h.reshape(h.shape[0], -1, self.index.num, h.shape[-1])  # [B, N_a, G, d]
        return self.index.merge([out(h[:, :, g]) for g, out in enumerate(self.outs)])


class ShortHorizonDiT(nn.Module):
    """plan_input sequence: X = [action tokens, trajectory tokens] (+ state tokens), full self-attention.
    plan_input condition: X = [action tokens] (+ state tokens); the trajectory tokens lead the cross-attention memory
    [trajectory tokens, ctx], whose RoPE puts them at their plan time (the other condition tokens at 0 s = now).
    plan_input action: X = [action tokens] (+ state tokens), the plan is a_t itself (no trajectory tokens); the action
    tokens from plan_steps on (no plan input) get noplan_embed."""

    def __init__(self, action_dim: int, horizon: int, num_train_timesteps: int, d_model: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0, qk_norm: bool = True, rope_time_scale: float = 20.0,
                 rope_base: float = 1000.0, abs_time_embed: bool = True, grad_ckpt: bool = False,
                 groups: Optional[List[List[int]]] = None, plan_grid: bool = False, plan_input: str = "sequence",
                 plan_steps: Optional[int] = None):
        super().__init__()
        if plan_input not in _PLAN_INPUTS:
            raise ValueError(f"system1.plan_input must be one of {_PLAN_INPUTS}, got {plan_input!r}")
        self.plan_in_sequence = plan_input == "sequence"
        self.plan_as_input = plan_input == "action"
        self.horizon = horizon
        self.index = _GroupIndex(groups) if groups else None  # ActionMap columns (None: one token per step)
        self.cells = 1 if self.index is None else self.index.num  # action tokens per step
        self.plan_cells = self.cells if plan_grid else 1  # trajectory tokens per keyframe
        if self.index is None:
            self.action_in = nn.Linear(action_dim, d_model)
        else:
            self.group_in = nn.ModuleList(nn.Linear(d, d_model) for d in self.index.dims)
            self.group_embed = nn.Parameter(torch.randn(self.index.num, d_model) * 0.02)
        self.action_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)  # tells action tokens from trajectory tokens
        self.plan_steps = int(plan_steps if plan_steps is not None else horizon) if self.plan_as_input else None
        self.noplan_embed = (nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
                             if self.plan_as_input and self.plan_steps < horizon else None)
        self.abs_time_embed, self.time_scale = abs_time_embed, float(rope_time_scale)
        self.t_embed = TimestepEmbedder(d_model, num_train_timesteps)
        rope = ContinuousRoPE(d_model // num_heads, rope_time_scale, rope_base)
        self.stack = DiTStack(d_model, depth, num_heads, mlp_ratio, qk_norm, rope=rope, grad_ckpt=grad_ckpt)
        if self.index is None:
            self.final = FinalLayer(d_model, action_dim)
        else:
            self.group_final = GroupFinalLayer(d_model, self.index)

    def embed_actions(self, a_t: torch.Tensor) -> torch.Tensor:
        """a_t [B, N_a, D_a] -> action tokens [B, N_a * cells, d] (ActionMap: row-major (step, group) cells)."""
        if self.index is None:
            return self.action_in(a_t.to(self.action_in.weight.dtype)) + self.action_type
        dt = self.group_in[0].weight.dtype
        h = torch.stack([lin(a.to(dt)) for lin, a in zip(self.group_in, self.index.split(a_t))], dim=2)  # [B, N_a, G, d]
        return (h + self.group_embed.to(h.dtype)).flatten(1, 2) + self.action_type

    def forward(self, a_t, t, traj_tokens, action_times, traj_times, ctx, ctx_mask, state_tokens=None):
        """a_t [B, N_a, D_a], t [B] long, traj_tokens [B, K * plan_cells, d], action_times [B, N_a], traj_times [B, K]
        (seconds, relative to the current System 1 time), ctx [B, L, d], ctx_mask [B, L], state_tokens [B, S, d]
        (optional, joins the self-attention sequence at 0 s) -> v_hat / x0_hat [B, N_a, D_a].
        ActionMap: every cell of a step / keyframe gets that row's timestamp. plan_input action: traj_tokens /
        traj_times are None (a_t carries the plan)."""
        h_a = self.embed_actions(a_t)
        if self.cells > 1:
            action_times = action_times.repeat_interleave(self.cells, dim=1)
        if self.plan_as_input:
            if self.noplan_embed is not None:
                n_in = self.plan_steps * self.cells
                h_a = torch.cat([h_a[:, :n_in], h_a[:, n_in:] + self.noplan_embed.to(h_a.dtype)], dim=1)
        elif self.plan_cells > 1:
            traj_times = traj_times.repeat_interleave(self.plan_cells, dim=1)
        if not self.plan_as_input and traj_tokens.shape[1] != traj_times.shape[1]:
            raise ValueError(f"{traj_tokens.shape[1]} trajectory tokens for {traj_times.shape[1]} timestamps "
                             f"(plan cells per keyframe: {self.plan_cells})")
        seq, pos = [h_a], [action_times]
        if self.plan_in_sequence:
            seq.append(traj_tokens.to(h_a.dtype))
            pos.append(traj_times)
        if state_tokens is not None:
            seq.append(state_tokens.to(h_a.dtype))
            pos.append(action_times.new_zeros(state_tokens.shape[:2]))
        h = torch.cat(seq, dim=1)  # [B, N_a * cells (+ K * plan_cells) (+ S), d]
        pos = torch.cat(pos, dim=1).float()
        if self.abs_time_embed:
            h = h + sincos_1d(pos * self.time_scale, h.shape[-1]).to(h.dtype)
        ctx, ctx_pos = ctx.to(h.dtype), None
        if not self.plan_in_sequence and not self.plan_as_input:  # memory = [trajectory tokens (at their plan time), ctx (at 0 s)]
            traj = traj_tokens.to(h.dtype)
            if self.abs_time_embed:
                traj = traj + sincos_1d(traj_times.float() * self.time_scale, h.shape[-1]).to(h.dtype)
            ctx_pos = torch.cat([traj_times.float(), traj_times.new_zeros(ctx.shape[:2]).float()], dim=1)
            ctx = torch.cat([traj, ctx], dim=1)
            ctx_mask = torch.cat([ctx_mask.new_ones(traj.shape[:2]), ctx_mask], dim=1)
        c = self.t_embed(t)
        h = self.stack(h, c, ctx, ctx_mask, pos=pos, ctx_pos=ctx_pos)
        n = self.horizon * self.cells
        return self.final(h[:, :n], c) if self.index is None else self.group_final(h[:, :n], c)


class System1ActionPolicy(nn.Module):
    """System 1: trajectory tokenizer + visual adapter + short-horizon DiT + flow-matching (or x0) scheduler.

    cfg keys: d_model, depth, num_heads, mlp_ratio, qk_norm, rope_time_scale, rope_base, abs_time_embed, grad_ckpt,
    use_state, use_radio_prefix, num_noise_samples, diffusion {...}, action_map {groups, names, plan_grid, noise_std} (optional),
    plan_input (sequence | condition | action), state_input (condition | sequence), traj_token_dropout, state_drop_dims,
    plan_chunk_s / plan_smooth_sigma_s (plan_input action: the plan chunk's length and its Gaussian smoothing, seconds).
    Default layout (plan_input sequence, state_input condition): X = [actions, plan], memory = [C-RADIO, state].
    plan_input condition + state_input sequence: X = [actions, state], memory = [plan, C-RADIO] (plan RoPE'd in the
    cross-attention at t_i - delay).
    traj_token_dropout p > 0 (training only): every trajectory token is independently replaced, with probability p, by
    a learned mask token + its past / current / future type embedding; its plan time (RoPE / sin-cos) is kept.
    state_drop_dims [i, ...]: these state dims are zeroed for System 1 (= not an input; System 2 still sees them).
    """

    def __init__(self, radio_dim: int, radio_grid: Tuple[int, int], state_dim: int, action_dim: int, horizon: int,
                 action_rate: float, timeline: KeyframeTimeline, cfg: Dict, num_prefix: int = 0,
                 prefix_dim: Optional[int] = None, num_views: int = 1):
        super().__init__()
        d = self.d_model = int(cfg.get("d_model", 768))
        self.plan_input = str(cfg.get("plan_input", "sequence"))
        self.state_input = str(cfg.get("state_input", "condition"))
        if self.plan_input not in _PLAN_INPUTS or self.state_input not in _STATE_INPUTS:
            raise ValueError(f"system1.plan_input must be one of {_PLAN_INPUTS} and system1.state_input one of "
                             f"{_STATE_INPUTS}, got {self.plan_input!r} / {self.state_input!r}")
        self.timeline, self.action_dim, self.horizon, self.action_rate = timeline, action_dim, horizon, float(action_rate)
        self.scheduler = build_scheduler(cfg.get("diffusion", {}))
        # plan_input action: the smoothed plan chunk is the action-token input (needs the regression "scheduler")
        self.plan_as_input = self.plan_input == "action"
        self.plan_chunk_steps = self.plan_smooth_sigma_s = None
        if self.plan_as_input:
            if not isinstance(self.scheduler, RegressionScheduler):
                raise ValueError("system1.plan_input action needs system1.diffusion.type regression")
            self.plan_chunk_steps = int(round(float(cfg.get("plan_chunk_s", 1.0)) * self.action_rate))
            self.plan_smooth_sigma_s = float(cfg.get("plan_smooth_sigma_s", 0.0))
            if not 1 <= self.plan_chunk_steps <= horizon or self.plan_smooth_sigma_s < 0:
                raise ValueError(f"system1.plan_chunk_s must give 1..{horizon} steps and plan_smooth_sigma_s >= 0, got "
                                 f"{self.plan_chunk_steps} steps / {self.plan_smooth_sigma_s}")
        am = cfg.get("action_map", None)
        self.groups = parse_action_groups(am.get("groups"), action_dim) if am else None
        self.plan_grid = bool(am.get("plan_grid", True)) if am else False
        names = am.get("names", None) if am else None
        self.group_names = ([str(n) for n in names] if names else [f"g{i}" for i in range(len(self.groups))]) if am else []
        if am and len(self.group_names) != len(self.groups):
            raise ValueError(f"system1.action_map.names needs {len(self.groups)} entries, got {self.group_names}")
        self.tokenizer = (None if self.plan_as_input else
                          GridTrajectoryTokenizer(self.groups, d, timeline) if self.plan_grid
                          else TrajectoryTokenizer(action_dim, d, timeline))
        use_prefix = bool(cfg.get("use_radio_prefix", True)) and num_prefix > 0
        self.radio_prefix_projector_s1 = (PrefixProjector(prefix_dim or radio_dim, num_prefix * num_views, d)
                                          if use_prefix else None)  # radio.prefix [B, V * P, D] (view-major)
        self.radio_projector_s1 = token_projector(radio_dim, d)
        self.radio_pos = RadioPositions(radio_grid, d, num_views)  # [1, V * N, d]
        self.state_projector = state_projector(state_dim, d) if (state_dim and cfg.get("use_state", True)) else None
        self.state_type = nn.Parameter(torch.randn(1, 1, d) * 0.02) if self.state_projector is not None else None
        self.dit = ShortHorizonDiT(
            action_dim, horizon, self.scheduler.num_train_timesteps, d_model=d, depth=int(cfg.get("depth", 12)),
            num_heads=int(cfg.get("num_heads", 12)), mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
            qk_norm=bool(cfg.get("qk_norm", True)), rope_time_scale=float(cfg.get("rope_time_scale", 20.0)),
            rope_base=float(cfg.get("rope_base", 1000.0)), abs_time_embed=bool(cfg.get("abs_time_embed", True)),
            grad_ckpt=bool(cfg.get("grad_ckpt", False)), groups=self.groups, plan_grid=self.plan_grid,
            plan_input=self.plan_input, plan_steps=self.plan_chunk_steps,
        )
        self.num_noise_samples = int(cfg.get("num_noise_samples", 1))
        self.register_buffer("key_times", timeline.times.clone(), persistent=False)
        self.register_buffer("action_times", torch.arange(horizon, dtype=torch.float32) / self.action_rate, persistent=False)
        # keyframe token dropout (created last, so the other modules' init does not depend on it)
        self.traj_token_dropout = float(cfg.get("traj_token_dropout", 0.0))
        if not 0.0 <= self.traj_token_dropout < 1.0:
            raise ValueError(f"system1.traj_token_dropout must be in [0, 1), got {self.traj_token_dropout}")
        if self.traj_token_dropout > 0 and self.plan_as_input:
            raise ValueError("system1.traj_token_dropout needs trajectory tokens (plan_input sequence | condition)")
        if self.traj_token_dropout > 0 and self.plan_grid:
            raise ValueError("system1.traj_token_dropout needs one token per keyframe (action_map.plan_grid: false)")
        self.traj_mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02) if self.traj_token_dropout > 0 else None
        # state dims System 1 does not see (system1.state_drop_dims): zeroed before the state projector, the same as
        # removing them (their projector columns get neither input nor gradient) with unchanged checkpoint shapes
        self.state_drop_dims = [int(i) for i in (cfg.get("state_drop_dims", None) or [])]
        if self.state_drop_dims and (self.state_projector is None or not all(0 <= i < state_dim for i in self.state_drop_dims)):
            raise ValueError(f"system1.state_drop_dims {self.state_drop_dims} needs use_state and indices < state_dim {state_dim}")
        keep = None
        if self.state_drop_dims:
            keep = torch.ones(state_dim)
            keep[self.state_drop_dims] = 0.0
        self.register_buffer("state_keep", keep, persistent=False)  # None = System 1 sees the whole state
        # action_map.noise_std: per-column std of the flow's source noise (None = N(0, 1) everywhere, the default path)
        ns = am.get("noise_std", None) if am else None
        std = None
        if ns is not None:
            ns = [float(v) for v in ns]
            if len(ns) != len(self.groups) or min(ns) <= 0:
                raise ValueError(f"system1.action_map.noise_std needs {len(self.groups)} positive entries, got {ns}")
            if self.scheduler.prediction != "velocity":
                raise ValueError("system1.action_map.noise_std needs diffusion.type flow_matching")
            std = torch.ones(action_dim)
            for g, v in zip(self.groups, ns):
                std[g] = v
        self.register_buffer("noise_std", std, persistent=False)  # [D_a] or None

    @property
    def uses_state(self) -> bool:
        return self.state_projector is not None

    @property
    def num_prefix(self) -> int:
        return 0 if self.radio_prefix_projector_s1 is None else self.radio_prefix_projector_s1.num_prefix

    def describe_sequence(self) -> str:
        """Self-attention sequence layout, e.g. '[B, 30 x 3 + 61 x 3, 768]' (ActionMap) or '[B, 30 + 61, 768]'."""
        k = self.timeline.num
        g = len(self.groups) if self.groups is not None else 1
        acts = f"{self.horizon}" if self.groups is None else f"{self.horizon} x {g}"
        plan = f"{k} x {g}" if self.plan_grid else f"{k}"
        if self.plan_as_input:
            acts = f"{acts} ({self.plan_chunk_steps} with the smoothed plan chunk)"
        parts = [acts] + ([plan] if self.plan_input == "sequence" else [])
        if self.uses_state and self.state_input == "sequence":
            parts.append("1 state")
        cols = "" if self.groups is None else f"  ActionMap columns {self.group_names}"
        return f"[B, {' + '.join(parts)}, {self.d_model}]{cols}"

    def describe_context(self, num_patches: int) -> str:
        """Cross-attention memory layout, e.g. '[B, 61 plan + 2 + 256, 768]'."""
        g = len(self.groups) if self.groups is not None else 1
        parts = []
        if self.plan_input == "condition":
            parts.append(f"{self.timeline.num}{f' x {g}' if self.plan_grid else ''} plan")
        parts += [str(self.num_prefix), str(num_patches)] if self.num_prefix else [str(num_patches)]
        if self.uses_state and self.state_input == "condition":
            parts.append("1")
        return f"[B, {' + '.join(parts)}, {self.d_model}]"

    def state_token(self, state: Optional[torch.Tensor]) -> torch.Tensor:
        """state [B, 1, D_st] -> the state token [B, 1, d] (state_projector + type embedding)."""
        if state is None:
            raise ValueError("System 1 is state-conditioned (system1.use_state): pass state [B, 1, D_st]")
        if self.state_keep is not None:  # system1.state_drop_dims
            state = state * self.state_keep.to(state.dtype)
        return self.state_projector(state.to(self.radio_projector_s1[1].weight.dtype)) + self.state_type

    def sequence_state(self, state: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """The state token for the self-attention sequence (state_input sequence), else None."""
        return self.state_token(state) if (self.uses_state and self.state_input == "sequence") else None

    def context(self, radio: VisualFeatures, state: Optional[torch.Tensor] = None):
        """Current-frame visual condition [prefix, patches] (+ the state token when state_input is condition)
        -> (ctx [B, P + N (+1), d], mask)."""
        dt = self.radio_projector_s1[1].weight.dtype
        parts, masks = [], []
        if self.radio_prefix_projector_s1 is not None:
            pre = self.radio_prefix_projector_s1(radio.prefix)
            parts.append(pre)
            masks.append(torch.ones(pre.shape[:2], dtype=torch.bool, device=pre.device))
        parts.append(self.radio_projector_s1(radio.tokens.to(dt)) + self.radio_pos())
        masks.append(radio.valid.bool())
        if self.state_projector is not None and self.state_input == "condition":
            parts.append(self.state_token(state))
            masks.append(torch.ones(state.shape[:2], dtype=torch.bool, device=state.device))
        return torch.cat(parts, dim=1), torch.cat(masks, dim=1)

    def timestamps(self, delay: torch.Tensor):
        """delay [B] seconds (plan age) -> (action_times [B, N_a], traj_times [B, K]) relative to System 1 now."""
        b = delay.shape[0]
        a = self.action_times.to(delay.device)[None].expand(b, -1)
        k = self.key_times.to(delay.device)[None] - delay.float()[:, None]
        return a, k

    def drop_trajectory_tokens(self, traj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """traj [B, K, d] -> (traj with each token replaced by mask token + its type embedding w.p. traj_token_dropout,
        drop [B, K] bool). Training only; the token keeps its plan time (timestamps are applied in the DiT)."""
        drop = torch.rand(traj.shape[:2], device=traj.device) < self.traj_token_dropout
        masked = self.traj_mask_token.to(traj.dtype) + self.tokenizer.type_embed(self.tokenizer.types)[None].to(traj.dtype)
        return torch.where(drop[..., None], masked, traj), drop

    def plan_input_actions(self, plan_chunk: Optional[torch.Tensor], b: int, device) -> torch.Tensor:
        """plan_input action: plan_chunk [B, n_in, D_a] -> the action-token input [B, N_a, D_a] (zeros after n_in)."""
        if plan_chunk is None or plan_chunk.shape[1:] != (self.plan_chunk_steps, self.action_dim):
            raise ValueError(f"system1.plan_input action needs plan_chunk [B, {self.plan_chunk_steps}, {self.action_dim}], "
                             f"got {None if plan_chunk is None else tuple(plan_chunk.shape)}")
        pad = plan_chunk.new_zeros(b, self.horizon - self.plan_chunk_steps, self.action_dim)
        return torch.cat([plan_chunk.float(), pad.float()], dim=1).to(device)

    def loss(self, keyframes, delay, radio: VisualFeatures, state, actions, valid,
             plan_chunk: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """keyframes [B, K, D_a] (System 2 plan), delay [B] s, actions [B, N_a, D_a] GT from t_S1, valid [B, N_a];
        plan_chunk [B, n_in, D_a] (plan_input action only: the smoothed plan chunk, absolute normalised)."""
        traj = None if self.plan_as_input else self.tokenizer(keyframes)
        ctx, ctx_mask = self.context(radio, state)
        st_tok = self.sequence_state(state)
        a_times, k_times = self.timestamps(delay)
        x_in = self.plan_input_actions(plan_chunk, actions.shape[0], actions.device) if self.plan_as_input else None
        r = self.num_noise_samples
        rep = (lambda v: v.repeat(r, *([1] * (v.dim() - 1)))) if r > 1 else (lambda v: v)
        rep_opt = lambda v: None if v is None else rep(v)
        traj, k_times, st_tok, x_in = map(rep_opt, (traj, k_times, st_tok, x_in))
        ctx, ctx_mask, a_times, actions, valid = map(rep, (ctx, ctx_mask, a_times, actions, valid))
        drop = None
        if self.traj_mask_token is not None and self.training:  # per noise draw, after the repeat
            traj, drop = self.drop_trajectory_tokens(traj)
        noise = None if self.noise_std is None else torch.randn_like(actions.float()) * self.noise_std
        x_t, t, aux = self.scheduler.training_inputs(actions.float(), noise=noise)
        if x_in is not None:  # plan_input action: the plan chunk replaces the regression's zero input
            x_t = x_in
            k_times = None
        out = self.dit(x_t, t, traj, a_times, k_times, ctx, ctx_mask, st_tok).float()
        m = valid[..., None].float()
        loss = ((out - aux["target"]) ** 2 * m).sum() / (m.sum() * out.shape[-1]).clamp_min(1)
        key = "system1/fm_loss" if self.scheduler.prediction == "velocity" else "system1/mse"
        logs = {key: loss.item(), "system1/valid_frac": float(valid.float().mean())}
        if drop is not None:
            logs["system1/traj_drop_frac"] = float(drop.float().mean())
        if self.groups is not None:  # per ActionMap column (same normalisation as the loss)
            with torch.no_grad():
                se = (out - aux["target"]) ** 2 * m
                for name, g in zip(self.group_names, self.groups):
                    logs[f"{key}_{name}"] = float(se[..., g].sum() / (m.sum() * len(g)).clamp_min(1))
        return loss, logs

    def sample(self, keyframes, delay, radio: VisualFeatures, state=None, steps: Optional[int] = None,
               noise: Optional[torch.Tensor] = None, generator: Optional[torch.Generator] = None,
               plan_chunk: Optional[torch.Tensor] = None) -> torch.Tensor:
        """-> normalized actions for the next horizon / action_rate seconds, [B, N_a, D_a]."""
        b = keyframes.shape[0]
        ctx, ctx_mask = self.context(radio, state)
        st_tok = self.sequence_state(state)
        a_times, k_times = self.timestamps(delay)
        if self.plan_as_input:  # one forward pass on the plan chunk (no trajectory tokens)
            x_in = self.plan_input_actions(plan_chunk, b, keyframes.device)
            with torch.no_grad():
                return self.dit(x_in, torch.zeros(b, dtype=torch.long, device=keyframes.device), None, a_times, None,
                                ctx, ctx_mask, st_tok).float()
        traj = self.tokenizer(keyframes)
        if noise is None and self.noise_std is not None:  # action_map.noise_std: the Euler start N(0, sigma_g^2)
            noise = torch.randn((b, self.horizon, self.action_dim), device=keyframes.device, generator=generator) * self.noise_std
        return self.scheduler.sample(lambda x, t: self.dit(x, t, traj, a_times, k_times, ctx, ctx_mask, st_tok),
                                     (b, self.horizon, self.action_dim), keyframes.device, steps=steps, noise=noise,
                                     generator=generator)
