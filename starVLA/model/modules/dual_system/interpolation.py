"""Keyframe timeline, differentiable keyframe -> dense interpolation and the dense-trajectory loss of System 2.

Timeline (System 2 reference time = the timestamp of the observation System 2 was run on):
    t_i = t_min + i * dt, i = 0..K-1, both endpoints included. Default [-2.0, 4.0] s at 0.1 s -> K = 61:
    20 past keyframes (-2.0 .. -0.1), 1 current (0.0), 40 future (+0.1 .. +4.0).

Interpolation of keyframe values y [B, K, D] (knots at t_i) to query times q [B, Q] (seconds, same reference):
    i = the knot interval containing q, w = (q - t_i) / (t_{i+1} - t_i) in [0, 1]
    linear   y_i + w (y_{i+1} - y_i)                           (continuous joints / positions, the default)
    step     y_i (zero-order hold; y_{K-1} at q = t_{K-1})     (discrete commands, e.g. a binary gripper)
    nearest  y_i if w < 0.5 else y_{i+1}                       (discrete, centred)
    angle    y_i + w wrap(y_{i+1} - y_i)                       (periodic, shortest arc; `period` in model units)
    slerp    spherical interpolation of 4-D unit quaternions   (rotations; q and -q identified)
Gradients flow into the keyframe values for every mode (step / nearest pass the gradient to the selected knot).
Queries outside [t_0, t_{K-1}] are NEVER extrapolated: they are returned masked (in_range False), and with
strict=True a query that the data marks valid but lies outside the keyframe span raises an error.
Actions are interpolated in the loader's normalized space. Linear interpolation commutes with the per-dimension affine
min_max / mean_std normalization, so this equals interpolating raw actions; for `angle` give the period in normalized
units (e.g. 2.0 for a min_max-normalized angle whose raw range is one full turn).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

MODES = ("linear", "step", "nearest", "angle", "slerp")


class KeyframeTimeline:
    """Fixed keyframe timestamps and their token types (0 past, 1 current, 2 future)."""

    PAST, CURRENT, FUTURE = 0, 1, 2

    def __init__(self, t_min: float = -2.0, t_max: float = 4.0, dt: float = 0.1):
        if dt <= 0 or t_max <= t_min:
            raise ValueError(f"bad keyframe timeline t_min={t_min} t_max={t_max} dt={dt}")
        n = int(round((t_max - t_min) / dt)) + 1
        if abs(t_min + (n - 1) * dt - t_max) > 1e-6 * max(1.0, abs(t_max)):
            raise ValueError(f"(t_max - t_min) / dt must be an integer: t_min={t_min} t_max={t_max} dt={dt}")
        self.t_min, self.t_max, self.dt, self.num = float(t_min), float(t_max), float(dt), n
        t = torch.arange(n, dtype=torch.float64) * dt + t_min
        t[t.abs() < 1e-9] = 0.0
        self.times = t.float()  # [K]
        tol = 1e-6
        self.types = torch.where(t < -tol, self.PAST, torch.where(t > tol, self.FUTURE, self.CURRENT)).long()
        if int((self.types == self.CURRENT).sum()) != 1:
            raise ValueError(f"the keyframe timeline must contain t = 0 exactly once (t_min={t_min}, dt={dt})")

    def counts(self) -> Dict[str, int]:
        return {k: int((self.types == v).sum()) for k, v in (("past", 0), ("current", 1), ("future", 2))}

    def __repr__(self):
        c = self.counts()
        return (f"KeyframeTimeline([{self.t_min:+.2f}, {self.t_max:+.2f}] s, dt {self.dt:.3f} s, K={self.num}: "
                f"{c['past']} past / {c['current']} current / {c['future']} future)")


def _wrap(d: torch.Tensor, period: float) -> torch.Tensor:
    return torch.remainder(d + 0.5 * period, period) - 0.5 * period


class TrajectoryInterpolator(nn.Module):
    """groups: [{"dims": [start, end), "mode": <MODES>, "period": float (angle only)}, ...]; dims not covered by a
    group use `default_mode` (must be linear / step / nearest). slerp groups must be 4 dims wide."""

    def __init__(self, action_dim: int, groups: Optional[Sequence[dict]] = None, default_mode: str = "linear",
                 strict: bool = True, eps: float = 1e-5):
        super().__init__()
        if default_mode not in ("linear", "step", "nearest"):
            raise ValueError(f"default_mode must be linear | step | nearest, got {default_mode!r}")
        self.action_dim, self.strict, self.eps = int(action_dim), bool(strict), float(eps)
        owner = [None] * self.action_dim
        self.groups: List[Tuple[int, int, str, Optional[float]]] = []
        for g in groups or []:
            lo, hi = (int(v) for v in g["dims"])
            mode = str(g.get("mode", "linear"))
            if mode not in MODES:
                raise ValueError(f"interpolation mode {mode!r} not in {MODES}")
            if not 0 <= lo < hi <= self.action_dim:
                raise ValueError(f"interpolation group dims [{lo}, {hi}) outside the action dim {self.action_dim}")
            if mode == "slerp" and hi - lo != 4:
                raise ValueError(f"slerp group [{lo}, {hi}) must be a 4-D quaternion")
            period = g.get("period", None)
            if mode == "angle" and (period is None or float(period) <= 0):
                raise ValueError(f"angle group [{lo}, {hi}) needs a positive `period` (in normalized action units)")
            for d in range(lo, hi):
                if owner[d] is not None:
                    raise ValueError(f"action dim {d} is in two interpolation groups")
                owner[d] = mode
            self.groups.append((lo, hi, mode, None if period is None else float(period)))
        d = 0
        while d < self.action_dim:  # contiguous runs of uncovered dims use the default mode
            if owner[d] is not None:
                d += 1
                continue
            lo = d
            while d < self.action_dim and owner[d] is None:
                d += 1
            self.groups.append((lo, d, default_mode, None))
        self.groups.sort()

    def describe(self) -> str:
        return ", ".join(f"[{lo}:{hi}] {m}" + (f"(period {p})" if p else "") for lo, hi, m, p in self.groups)

    def forward(self, values: torch.Tensor, knot_times: torch.Tensor, query_times: torch.Tensor, linear_only: bool = False):
        """values [B, K, D], knot_times [K] or [B, K] ascending, query_times [B, Q]
        -> (out [B, Q, D], in_range [B, Q] bool). Out-of-range queries are 0 and masked.
        linear_only: angle / slerp groups use plain linear weights (interpolating a VELOCITY or the noise of a flow,
        which are not angles / unit quaternions); step / nearest keep their selection rule (also linear operators)."""
        b, k, d = values.shape
        if d != self.action_dim:
            raise ValueError(f"interpolator built for action dim {self.action_dim}, got {d}")
        if k < 2:
            raise ValueError("need at least 2 keyframes")
        knots = knot_times.to(query_times.device, torch.float32)
        knots = knots.expand(b, k) if knots.dim() == 1 else knots
        q = query_times.float()
        in_range = (q >= knots[:, :1] - self.eps) & (q <= knots[:, -1:] + self.eps)
        idx = (torch.searchsorted(knots.contiguous(), q.contiguous(), right=True) - 1).clamp(0, k - 2)  # [B, Q]
        t0, t1 = knots.gather(1, idx), knots.gather(1, idx + 1)
        w = ((q - t0) / (t1 - t0)).clamp(0.0, 1.0)  # [B, Q]
        gidx = lambda i: i[..., None].expand(-1, -1, d)
        y0, y1 = values.gather(1, gidx(idx)), values.gather(1, gidx(idx + 1))  # [B, Q, D]
        w_ = w[..., None].to(values.dtype)
        parts = []
        for lo, hi, mode, period in self.groups:
            a, c = y0[..., lo:hi], y1[..., lo:hi]
            if mode == "linear" or (linear_only and mode in ("angle", "slerp")):
                parts.append(a + w_ * (c - a))
            elif mode == "step":
                parts.append(torch.where(w_ >= 1.0 - self.eps, c, a))
            elif mode == "nearest":
                parts.append(torch.where(w_ < 0.5, a, c))
            elif mode == "angle":
                parts.append(a + w_ * _wrap(c - a, period))
            else:  # slerp
                parts.append(self._slerp(a, c, w_))
        out = torch.cat(parts, dim=-1)
        return out.masked_fill(~in_range[..., None], 0.0), in_range

    @staticmethod
    def _slerp(q0: torch.Tensor, q1: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dot = (q0 * q1).sum(-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)  # shortest path
        dot = dot.abs().clamp(max=1.0 - 1e-6)
        theta = torch.acos(dot)
        s = torch.sin(theta)
        small = s < 1e-4
        s_safe = torch.where(small, torch.ones_like(s), s)
        out = torch.where(small, (1 - w) * q0 + w * q1,
                          (torch.sin((1 - w) * theta) * q0 + torch.sin(w * theta) * q1) / s_safe)
        return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)


class DenseTrajectoryLoss(nn.Module):
    """System 2's only objective, computed on the dense GT trajectory (never on the 61 keyframes directly).

    x0 prediction   : L = MSE(interp(x0_hat keyframes)(dense t), dense GT)                      (`forward`)
    flow matching   : L = MSE(interp(v_hat)(dense t), dense GT - interp(eps)(dense t))          (`velocity`)
                      i.e. the GR00T velocity target v = x_1 - eps taken on the interpolated dense trajectory; with
                      linear interpolation the Euler-integrated keyframes converge to the least-squares keyframe fit of
                      the dense trajectory.
    The error is computed per interpolation group: plain difference, except angle groups (wrapped difference) and
    slerp groups (x0 only: the predicted quaternion's sign is matched to the GT, q ~ -q).
    """

    def __init__(self, interpolator: TrajectoryInterpolator):
        super().__init__()
        self.interp = interpolator

    def error(self, pred: torch.Tensor, gt: torch.Tensor, velocity: bool = False) -> torch.Tensor:
        parts = []
        for lo, hi, mode, period in self.interp.groups:
            p, g = pred[..., lo:hi], gt[..., lo:hi]
            if mode == "angle":
                parts.append(_wrap(p - g, period))
            elif mode == "slerp" and not velocity:
                sign = torch.where((p * g).sum(-1, keepdim=True) < 0, -1.0, 1.0).to(p.dtype)
                parts.append(sign * p - g)
            else:
                parts.append(p - g)
        return torch.cat(parts, dim=-1)

    def forward(self, keyframes: torch.Tensor, knot_times: torch.Tensor, dense_gt: torch.Tensor,
                dense_times: torch.Tensor, dense_valid: torch.Tensor):
        """keyframes [B, K, D] (predicted CLEAN keyframes), dense_gt [B, Q, D], dense_times / dense_valid [B, Q]."""
        pred, in_range = self.interp(keyframes.float(), knot_times, dense_times)
        outside = dense_valid & ~in_range
        if self.interp.strict and bool(outside.any()):
            t = dense_times[outside]
            raise ValueError(f"{int(outside.sum())} valid dense targets lie outside the keyframe span "
                             f"[{float(knot_times.min()):+.3f}, {float(knot_times.max()):+.3f}] s "
                             f"(e.g. t = {float(t[0]):+.4f} s); extrapolation is disabled — fix the data window")
        return self._masked(self.error(pred, dense_gt.float()), dense_valid, in_range)

    def velocity(self, v_hat: torch.Tensor, noise: torch.Tensor, knot_times: torch.Tensor, dense_gt: torch.Tensor,
                 dense_times: torch.Tensor, dense_valid: torch.Tensor):
        """v_hat, noise [B, K, D] (predicted keyframe velocity and the keyframe noise of x_tau)."""
        v_d, in_range = self.interp(v_hat.float(), knot_times, dense_times, linear_only=True)
        e_d, _ = self.interp(noise.float(), knot_times, dense_times, linear_only=True)
        # v_d - (dense - e_d), with angle groups wrapped
        return self._masked(self.error(v_d + e_d, dense_gt.float(), velocity=True), dense_valid, in_range, dense_times, knot_times)

    def _masked(self, err, dense_valid, in_range, dense_times=None, knot_times=None):
        outside = dense_valid & ~in_range
        if dense_times is not None and self.interp.strict and bool(outside.any()):
            raise ValueError(f"{int(outside.sum())} valid dense targets lie outside the keyframe span "
                             f"[{float(knot_times.min()):+.3f}, {float(knot_times.max()):+.3f}] s; extrapolation is disabled")
        valid = dense_valid & in_range
        loss = (err * valid[..., None]).pow(2).sum() / (valid.sum() * err.shape[-1]).clamp_min(1)
        return loss, {"valid_frac": float(valid.float().mean()), "outside_frac": float(outside.float().mean())}
