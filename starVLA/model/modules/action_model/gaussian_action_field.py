"""An explicit Gaussian primitive field in normalized action + relative time.

Each primitive factors as p(t|k) p(a|t,k), with a local linear action mean.
Positive diagonal action covariance and positive time variance imply a valid
joint Gaussian, including action-time correlation through the local slope.
Splatting here means additive density evaluation, not graphics alpha blending.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value):
    return value + math.log(-math.expm1(-value))


class GaussianActionField(nn.Module):
    """Raw parameters: time center/width, log mass, action center/slope/std.

    Shape [B,K,3+3D]. The module evaluates a supplied bank; ownership of the
    trainable shared bank and observation deformation belongs to the head.
    """

    def __init__(self, action_dim, action_horizon, num_gaussians=32,
                 min_time_width=0.01, initial_time_width=0.15,
                 min_action_std=0.03, initial_action_std=0.3):
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.num_gaussians = int(num_gaussians)
        if min(self.action_dim, self.action_horizon, self.num_gaussians) < 1:
            raise ValueError("action_dim, action_horizon and num_gaussians must be positive")
        for lower, initial in ((min_time_width, initial_time_width), (min_action_std, initial_action_std)):
            if not 0 < lower < initial or not math.isfinite(initial):
                raise ValueError("Require finite initial scale > minimum scale > 0")
        self.min_time_width = float(min_time_width)
        self.min_action_std = float(min_action_std)
        centers = (torch.arange(self.num_gaussians, dtype=torch.float32) + 0.5) / self.num_gaussians
        self.register_buffer("center_logits", torch.logit(centers))
        self.register_buffer("time_width_offset", torch.tensor(_inverse_softplus(initial_time_width - min_time_width)))
        self.register_buffer("action_std_offset", torch.tensor(_inverse_softplus(initial_action_std - min_action_std)))
        self.register_buffer("times", torch.linspace(0, 1, self.action_horizon))

    @property
    def parameter_dim(self):
        return 3 + 3 * self.action_dim

    def components(self, parameters, times=None):
        """Return log p(k|t), conditional means [B,H,K,D], stds [B,K,D]."""
        if parameters.ndim != 3 or parameters.shape[1:] != (self.num_gaussians, self.parameter_dim):
            raise ValueError(f"Expected [B, {self.num_gaussians}, {self.parameter_dim}] parameters")
        with torch.autocast(device_type=parameters.device.type, enabled=False):
            p = parameters.float()
            t = self.times if times is None else torch.as_tensor(times, device=p.device, dtype=torch.float32)
            if t.ndim != 1 or t.numel() < 1:
                raise ValueError("times must be a nonempty 1D sequence")
            t = t.to(device=p.device, dtype=torch.float32)
            center = torch.sigmoid(p[..., 0] + self.center_logits.float())
            width = self.min_time_width + F.softplus(p[..., 1] + self.time_width_offset.float())
            dt = t[None, :, None] - center[:, None, :]
            log_weight = p[:, None, :, 2] - width[:, None, :].log() - 0.5 * (dt / width[:, None, :]).square()
            log_weight = log_weight.log_softmax(dim=-1)
            u, slope, raw_std = p[..., 3:].split(self.action_dim, dim=-1)
            std = self.min_action_std + F.softplus(raw_std + self.action_std_offset.float())
            mean = u[:, None, :, :] + dt[..., None] * slope[:, None, :, :]
            return log_weight, mean, std

    def log_prob(self, parameters, actions, action_mask=None):
        """Conditional log density [B,H]; invalid dimensions are marginalized.

        Mask True=valid, shape [B,H] or [B,H,D]. Fully masked steps return zero.
        """
        log_weight, mean, std = self.components(parameters)
        if actions.shape != (parameters.shape[0], self.action_horizon, self.action_dim):
            raise ValueError("actions must have shape [B, action_horizon, action_dim]")
        valid = self.valid_mask(actions, action_mask)
        with torch.autocast(device_type=parameters.device.type, enabled=False):
            # Replace invalid targets BEFORE arithmetic; masked NaNs must not
            # poison gradients into learned means or covariance.
            target = torch.where(valid, actions.float(), 0.0)
            z = (target[:, :, None, :] - mean) / std[:, None, :, :]
            terms = -0.5 * z.square() - std[:, None, :, :].log() - 0.5 * math.log(2 * math.pi)
            terms = torch.where(valid[:, :, None, :], terms, 0.0)
            result = torch.logsumexp(log_weight + terms.sum(-1), dim=-1)
            return torch.where(valid.any(-1), result, 0.0)

    @staticmethod
    def valid_mask(actions, action_mask):
        if action_mask is None:
            return torch.ones_like(actions, dtype=torch.bool)
        mask = torch.as_tensor(action_mask, device=actions.device, dtype=torch.bool)
        if mask.shape == actions.shape[:2]:
            return mask.unsqueeze(-1).expand_as(actions)
        if mask.shape != actions.shape:
            raise ValueError("action_mask must have shape [B,H] or [B,H,D]")
        return mask

    @torch.no_grad()
    def select_actions(self, parameters, num_candidates=8, continuity_weight=1.0):
        """Pick a dense, continuous path through component-center candidates.

        Candidate pruning ranks weighted component peaks. Each candidate is
        then scored by the ENTIRE mixture. Dynamic programming maximizes
        log-density/D minus continuity_weight * mean squared action changes.
        This is an approximate mode path, not the global continuous MAP, and
        never averages distinct action modes into an unsupported action.
        """
        if num_candidates < 1 or not math.isfinite(continuity_weight) or continuity_weight < 0:
            raise ValueError("Require num_candidates > 0 and finite continuity_weight >= 0")
        log_weight, mean, std = self.components(parameters)
        with torch.autocast(device_type=parameters.device.type, enabled=False):
            count = min(int(num_candidates), self.num_gaussians)
            peak_score = log_weight - std.log().sum(-1)[:, None, :]
            indices = peak_score.topk(count, dim=-1).indices
            candidates = mean.gather(2, indices[..., None].expand(-1, -1, -1, self.action_dim))
            # Evaluate one candidate at a time to bound B*H*C*K*D memory.
            scores = []
            for i in range(count):
                z = (candidates[:, :, i, None, :] - mean) / std[:, None, :, :]
                log_density = (-0.5 * z.square() - std[:, None, :, :].log()).sum(-1)
                scores.append(torch.logsumexp(log_weight + log_density, -1) / self.action_dim)
            scores = torch.stack(scores, -1)
            best_score = scores[:, 0]
            parents = []
            for h in range(1, self.action_horizon):
                jumps = (candidates[:, h - 1, :, None, :] - candidates[:, h, None, :, :]).square().mean(-1)
                path_score, parent = (best_score[:, :, None] - continuity_weight * jumps).max(dim=1)
                best_score = path_score + scores[:, h]
                parents.append(parent)
            choice = best_score.argmax(-1)
            batch = torch.arange(parameters.shape[0], device=parameters.device)
            path = [candidates[batch, -1, choice]]
            for h in range(self.action_horizon - 2, -1, -1):
                choice = parents[h][batch, choice]
                path.append(candidates[batch, h, choice])
            return torch.stack(path[::-1], dim=1)
