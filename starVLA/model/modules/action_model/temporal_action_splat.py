"""Differentiable action-value splatting onto a future-time grid.

Each point stores [action[D], time_offset, radius_offset, importance].
Compact triangular footprints accumulate action values, not probability
densities. Softmax importance resolves overlapping contributions.

The point set may be adaptive (3DGS-style densify / prune): the module owns a
fixed capacity of ``num_points`` slots, of which ``initial_points`` start
active on a regular grid. Callers that manage their own point population pass
explicit ``anchors`` / ``active`` tensors to ``forward``; slots that are
inactive contribute nothing, and a time slot that no active point supports
falls back to its nearest active point so the output is always defined.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def inverse_softplus(y):
    y = torch.as_tensor(y, dtype=torch.float64)
    return (y + torch.log(-torch.expm1(-y))).float()


class TemporalActionSplat(nn.Module):
    """Map [B,...,K,D+3] points to [B,...,H,D] actions.

    Static grid (initial_points == num_points): point i is anchored at
    (i+0.5)/K with at most max_shift/K displacement, and a minimum radius
    strictly above (0.5+max_shift)/K guarantees positive coverage everywhere.

    Adaptive (initial_points < num_points): anchors live in the caller,
    radii may shrink down to the capacity-K_max spacing, and coverage is
    guaranteed by the nearest-active-point fallback instead of geometry.
    """

    def __init__(self, action_dim, action_horizon, num_points=32, temperature=1.0,
                 max_shift=0.25, initial_radius=1.5, initial_points=None):
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.num_points = int(num_points)  # capacity
        self.initial_points = int(initial_points if initial_points is not None else num_points)
        self.temperature = float(temperature)
        self.max_shift = float(max_shift)
        if min(self.action_dim, self.action_horizon, self.initial_points) < 1 or self.initial_points > self.num_points:
            raise ValueError("action_dim, action_horizon, initial_points must be positive and initial_points <= num_points")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.max_shift) or not 0 <= self.max_shift < 0.5:
            raise ValueError("max_shift must be in [0, 0.5) anchor spacings")
        # Margin protects coverage at saturated offsets and minimum radii.
        minimum = 0.5 + self.max_shift + 0.05
        if not math.isfinite(initial_radius) or initial_radius <= minimum:
            raise ValueError("initial_radius must exceed 0.55 + max_shift anchor spacings")
        # Geometry in absolute time units. Spacing of the initial grid sets the
        # shift bound and the initial radius; the capacity spacing sets the
        # radius floor so split children can keep shrinking.
        self.grid_spacing = 1.0 / self.initial_points
        self.min_radius = minimum / self.num_points
        self.max_shift_abs = self.max_shift * self.grid_spacing
        self.initial_radius_abs = initial_radius * self.grid_spacing
        self.register_buffer("radius_bias", inverse_softplus(self.initial_radius_abs - self.min_radius))
        anchors = torch.zeros(self.num_points)
        anchors[: self.initial_points] = (torch.arange(self.initial_points, dtype=torch.float32) + 0.5) * self.grid_spacing
        self.register_buffer("anchors", anchors)
        active = torch.zeros(self.num_points, dtype=torch.bool)
        active[: self.initial_points] = True
        self.register_buffer("active", active)
        self.register_buffer("times", torch.linspace(0, 1, self.action_horizon))

    @property
    def parameter_dim(self):
        return self.action_dim + 3

    def radius_raw_for(self, radius):
        """Raw radius parameter that decodes to ``radius`` (absolute time units)."""
        radius = torch.as_tensor(radius, dtype=torch.float32)
        return inverse_softplus((radius - self.min_radius).clamp_min(1e-6)) - self.radius_bias.to(radius.device)

    def decode_points(self, points, anchors=None):
        if points.ndim < 2 or points.shape[-2:] != (self.num_points, self.parameter_dim):
            raise ValueError(f"Expected [...,{self.num_points},{self.parameter_dim}] points")
        # Float64 also supports finite-difference checks. AMP inputs use FP32.
        dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
        p = points.to(dtype)
        anchors = self.anchors if anchors is None else anchors
        values = p[..., :self.action_dim]
        centers = anchors.to(dtype) + self.max_shift_abs * p[..., self.action_dim].tanh()
        radii = self.min_radius + F.softplus(p[..., self.action_dim + 1] + self.radius_bias.to(dtype))
        importance = p[..., self.action_dim + 2] / self.temperature
        return values, centers, radii, importance

    def forward(self, points, anchors=None, active=None, return_weights=False):
        """points [..., K, D+3]; anchors [..., K] or None; active [..., K] bool or None."""
        with torch.autocast(device_type=points.device.type, enabled=False):
            values, centers, radii, importance = self.decode_points(points, anchors)
            active = self.active if active is None else active
            distance = self.times.to(values.dtype)[:, None] - centers.unsqueeze(-2)  # [..., H, K]
            footprint = (1 - distance.abs() / radii.unsqueeze(-2)).clamp_min(0)
            support = (footprint > 0) & active.unsqueeze(-2)
            # Normalize among supporting points only. A huge importance outside
            # a slot's support must not underflow every valid contribution.
            log_footprint = footprint.clamp_min(torch.finfo(values.dtype).tiny).log()
            logits = (log_footprint + importance.unsqueeze(-2)).masked_fill(~support, -torch.inf)
            # Coverage fallback: a slot with no supporting active point takes
            # its nearest active point (in radius-normalized distance) alone.
            uncovered = ~support.any(dim=-1, keepdim=True)  # [..., H, 1]
            if bool(uncovered.any()):
                normalized = (distance.abs() / radii.unsqueeze(-2)).masked_fill(~active.unsqueeze(-2), torch.inf)
                nearest = F.one_hot(normalized.argmin(dim=-1), self.num_points).bool()
                logits = torch.where(uncovered, torch.zeros_like(logits).masked_fill(~nearest, -torch.inf), logits)
            weights = logits.softmax(dim=-1)
            out = torch.einsum("...hk,...kd->...hd", weights, values)
            return (out, weights) if return_weights else out
