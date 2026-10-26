"""Observation-conditioned action points, splatted directly into one action chunk.

One shared bank of K action points; the trunk predicts per-sample offsets to
every point; a triangular softmax splat turns the points into [B,H,D] actions.
No candidate branches, WTA, score head or chunk selection.

Optionally adaptive (3DGS-style densification): points that keep receiving
large gradients are split (large footprint) or cloned (small footprint), and
points whose splat weight stays negligible are pruned. The population lives in
a fixed-capacity bank with an ``active`` mask so tensor shapes (and therefore
DDP / DeepSpeed / optimizer state) never change during training.
"""

import math

import torch
import torch.distributed as dist
from torch import nn

from starVLA.model.modules.action_model.GR00T_ActionHeader import DiTConfig, MLP
from starVLA.model.modules.action_model.GR00T_RegressorHeader import RegressionDiT
from starVLA.model.modules.action_model.temporal_action_splat import TemporalActionSplat


class ActionSplatHead(nn.Module):
    """Shared K-token trunk, one point bank, one splatted chunk."""

    def __init__(self, full_config):
        super().__init__()
        self.full_config = full_config
        self.config = c = full_config.framework.action_model
        if c.get("multi_windows") or int(c.get("num_past_steps", 0)):
            raise ValueError("ActionSplatHead currently supports single future-only windows")
        if int(c.get("moe_expert_layers", 0)):
            raise ValueError("ActionSplatHead uses a single decoder; set moe_expert_layers=0")
        self.action_dim = int(c.action_dim)
        self.action_horizon = int(c.action_horizon)
        initial_points = int(c.get("num_splat_points", 32))
        capacity = int(c.get("num_splat_points_max", initial_points) or initial_points)
        self.splatter = TemporalActionSplat(
            self.action_dim, self.action_horizon,
            num_points=capacity, initial_points=initial_points,
            temperature=c.get("splat_temperature", 1.0),
            max_shift=c.get("splat_max_shift", 0.25),
            initial_radius=c.get("splat_initial_radius", 1.5),
        )
        model_config = {**DiTConfig[c.action_model_type], **dict(c.diffusion_model_cfg or {})}
        for key in ("norm_type", "num_embeds_ada_norm", "compute_dtype"):
            model_config.pop(key, None)
        self.model = RegressionDiT(**model_config)
        dim = self.model.inner_dim
        self.state_encoder = MLP(c.state_dim, c.hidden_size, dim) if c.state_dim else None
        count, params = self.splatter.num_points, self.splatter.parameter_dim
        self.point_bank = nn.Parameter(torch.zeros(count, params))
        with torch.no_grad():
            self.point_bank[:, :self.action_dim].normal_(std=float(c.get("splat_init_value_std", 0.25)))
        self.point_encoder = MLP(params, c.hidden_size, dim)
        self.point_queries = nn.Embedding(count, dim)
        nn.init.normal_(self.point_queries.weight, std=0.02)
        n_future = int(c.get("num_target_vision_tokens", 0) or 0)
        self.future_tokens = nn.Embedding(n_future, dim) if n_future else None
        self.point_decoder = MLP(self.model.config.output_dim, c.hidden_size, params)
        nn.init.normal_(self.point_decoder.layer2.weight, std=0.001)
        nn.init.zeros_(self.point_decoder.layer2.bias)
        self.loss_type = c.get("loss_type", "l1")
        if self.loss_type not in ("l1", "mse", "l1+l2"):
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        self.l1_weight = float(c.get("l1_weight", 1.0))
        self.l2_weight = float(c.get("l2_weight", 1.0))
        self.delta_weight = float(c.get("splat_delta_loss_weight", 0.1))
        if any(not math.isfinite(w) or w < 0 for w in (self.l1_weight, self.l2_weight, self.delta_weight)):
            raise ValueError("Loss weights must be finite and nonnegative")

        # ---- adaptive population --------------------------------------
        # anchors / active are checkpointed buffers; statistics are not.
        self.register_buffer("point_anchor", self.splatter.anchors.clone())
        self.register_buffer("point_active", self.splatter.active.clone())
        self.register_buffer("_grad_accum", torch.zeros(count), persistent=False)
        self.register_buffer("_usage_accum", torch.zeros(count), persistent=False)
        self.register_buffer("_stat_count", torch.zeros(()), persistent=False)
        self.adaptive = capacity > initial_points or bool(c.get("splat_adaptive", False))
        self.densify_from = int(c.get("splat_densify_from", 500))
        self.densify_until = int(c.get("splat_densify_until", 30000))
        self.densify_interval = int(c.get("splat_densify_interval", 500))
        self.densify_quantile = float(c.get("splat_densify_quantile", 0.75))  # points above this grad quantile densify
        self.densify_grad_floor = float(c.get("splat_densify_grad_floor", 0.0))  # absolute floor on accumulated grad
        self.max_new_per_interval = int(c.get("splat_max_new_per_interval", initial_points))
        self.prune_usage_ratio = float(c.get("splat_prune_usage_ratio", 0.1))  # prune below ratio * (1 / active)
        self.min_points = int(c.get("splat_min_points", min(4, initial_points)))
        self.split_radius_ratio = float(c.get("splat_split_radius_ratio", 1.0))  # split if radius > ratio / active
        self.split_shrink = float(c.get("splat_split_shrink", 1.6))
        self.importance_reset_interval = int(c.get("splat_importance_reset_interval", 0))
        if self.densify_interval < 1 or not 0 <= self.densify_quantile < 1 or self.min_points < 1:
            raise ValueError("splat_densify_interval >= 1, splat_densify_quantile in [0,1), splat_min_points >= 1")
        self.last_wta_stats = {}
        self._refresh_population_stats()

    # ------------------------------------------------------------------ core
    def _predict_points(self, vl_embs, state=None, encoder_attention_mask=None):
        batch = vl_embs.shape[0]
        parts = []
        if state is not None and self.state_encoder is not None:
            parts.append(self.state_encoder(state))
        if self.future_tokens is not None:
            parts.append(self.future_tokens.weight.unsqueeze(0).expand(batch, -1, -1))
        queries = self.point_queries.weight + self.point_encoder(self.point_bank)
        parts.append(queries.unsqueeze(0).expand(batch, -1, -1))
        features = self.model(
            hidden_states=torch.cat(parts, dim=1).to(vl_embs.dtype),
            encoder_hidden_states=vl_embs, encoder_attention_mask=encoder_attention_mask,
        )[:, -self.splatter.num_points:]
        return self.point_bank.unsqueeze(0) + self.point_decoder(features).float()

    def _splat(self, points, return_weights=False):
        return self.splatter(points, anchors=self.point_anchor, active=self.point_active, return_weights=return_weights)

    @staticmethod
    def _valid_mask(actions, action_mask):
        if action_mask is None:
            return torch.ones_like(actions, dtype=torch.bool)
        mask = torch.as_tensor(action_mask, device=actions.device, dtype=torch.bool)
        if mask.shape == actions.shape[:2]:
            return mask.unsqueeze(-1).expand_as(actions)
        if mask.shape != actions.shape:
            raise ValueError("action_mask must have shape [B,H] or [B,H,D]")
        return mask

    def forward(self, vl_embs, actions, state=None, encoder_attention_mask=None, action_mask=None):
        points = self._predict_points(vl_embs, state, encoder_attention_mask)
        predictions, weights = self._splat(points, return_weights=True)
        if self.adaptive and self.training:
            self._accumulate_statistics(points, weights)
        expected = (predictions.shape[0], self.action_horizon, self.action_dim)
        if actions.shape != expected:
            raise ValueError(f"Expected actions {expected}, got {tuple(actions.shape)}")
        valid = self._valid_mask(actions, action_mask)
        target = torch.where(valid, actions.float(), 0.0)
        error = torch.where(valid, predictions - target, 0.0)
        count = valid.sum((1, 2))
        denominator = count.clamp_min(1)
        l1 = error.abs().sum((1, 2)) / denominator
        l2 = error.square().sum((1, 2)) / denominator
        errors = l1 if self.loss_type == "l1" else l2
        if self.loss_type == "l1+l2":
            errors = self.l1_weight * l1 + self.l2_weight * l2
        if self.action_horizon > 1 and self.delta_weight:
            pairs = valid[:, 1:] & valid[:, :-1]
            delta = torch.where(pairs, error[:, 1:] - error[:, :-1], 0.0)
            errors = errors + self.delta_weight * delta.abs().sum((1, 2)) / pairs.sum((1, 2)).clamp_min(1)
        sample_valid = (count > 0).to(errors.dtype)
        return (errors * sample_valid).sum() / sample_valid.sum().clamp_min(1)

    @torch.no_grad()
    def predict_action(self, vl_embs, state=None, encoder_attention_mask=None):
        return self._splat(self._predict_points(vl_embs, state, encoder_attention_mask))

    # -------------------------------------------------------- adaptive K
    def _accumulate_statistics(self, points, weights):
        """3DGS analogue of the accumulated view-space positional gradient.

        ``points`` [B,K,P] are the per-sample decoded point parameters; the
        hook records the batch-mean L2 norm of their gradient per slot.
        ``weights`` [B,H,K] are the splat weights; their mean over batch and
        time is the slot's usage (uniform coverage would give 1/K_active).
        """
        with torch.no_grad():
            self._usage_accum += weights.detach().float().mean((0, 1))
            self._stat_count += 1
        if points.requires_grad:
            def _hook(g):
                with torch.no_grad():
                    self._grad_accum += g.detach().float().norm(dim=-1).mean(0)
            points.register_hook(_hook)

    def _reset_statistics(self):
        self._grad_accum.zero_()
        self._usage_accum.zero_()
        self._stat_count.zero_()

    def _refresh_population_stats(self, extra=None):
        _, _, radii, _ = self.splatter.decode_points(self.point_bank.detach(), anchors=self.point_anchor)
        active = self.point_active
        stats = {"splat/active": float(active.sum().item()),
                 "splat/radius_mean": float(radii[active].mean().item()) if bool(active.any()) else 0.0}
        if extra:
            stats.update(extra)
        self.last_wta_stats = stats

    @staticmethod
    def _sync_(tensor):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    @staticmethod
    def _reset_optimizer_rows(optimizer, param, rows):
        """Zero Adam moments for the rows of ``param`` that were rewritten."""
        opt = optimizer
        for _ in range(4):  # unwrap accelerate / DeepSpeed wrappers
            if opt is None or hasattr(opt, "state"):
                break
            opt = getattr(opt, "optimizer", None)
        if opt is None or not hasattr(opt, "state") or param not in opt.state:
            return
        for value in opt.state[param].values():
            if torch.is_tensor(value) and value.shape == param.shape:
                value[rows] = 0

    @torch.no_grad()
    def adaptive_step(self, step, optimizer=None):
        """Call once per optimizer step on every rank. Returns population stats.

        Statistics are all-reduced so every rank makes the same decision.
        """
        if not self.adaptive:
            return self.last_wta_stats
        in_window = self.densify_from <= step <= self.densify_until
        if self.importance_reset_interval and in_window and step % self.importance_reset_interval == 0:
            self.point_bank[:, self.action_dim + 2] = 0
        if not in_window or step % self.densify_interval != 0:
            return self.last_wta_stats
        count = self._sync_(self._stat_count.clone())
        if count.item() < 1:
            return self.last_wta_stats
        grad = (self._sync_(self._grad_accum.clone()) / count).cpu()
        usage = (self._sync_(self._usage_accum.clone()) / count).cpu()
        # (world-size factor cancels: grad is compared only against its own quantile, usage against its own share)
        if dist.is_available() and dist.is_initialized():
            usage /= dist.get_world_size()
        bank = self.point_bank
        _, _, radii, _ = self.splatter.decode_points(bank, anchors=self.point_anchor)
        radii, anchor = radii.cpu(), self.point_anchor.cpu()
        active = self.point_active.cpu().clone()
        generator = torch.Generator(device="cpu").manual_seed(int(step))
        n_split = n_clone = n_prune = 0
        touched = []

        # ---- prune: usage far below the uniform share --------------------
        n_active = int(active.sum())
        if n_active > self.min_points:
            low = active & (usage < self.prune_usage_ratio / n_active)
            removable = n_active - self.min_points
            if int(low.sum()) > removable:  # drop only the least-used ones
                keep = torch.zeros_like(low)
                keep[torch.argsort(usage.masked_fill(~low, torch.inf))[:removable]] = True
                low &= keep
            active &= ~low
            n_prune = int(low.sum())
            n_active = int(active.sum())

        # ---- densify: top-quantile accumulated gradient ------------------
        free = (~active).nonzero().flatten().tolist()
        if free and n_active > 0:
            threshold = max(torch.quantile(grad[active], self.densify_quantile).item(), self.densify_grad_floor)
            candidates = ((grad > threshold) & active).nonzero().flatten().tolist()
            candidates.sort(key=lambda k: -grad[k].item())
            for k in candidates[: min(len(free), self.max_new_per_interval)]:
                j = free.pop(0)
                r = radii[k].item()
                a = anchor[k].item()
                bank[j] = bank[k]
                if r > self.split_radius_ratio / n_active:
                    # split: two children half a radius apart, both shrunk
                    child_r = max(r / self.split_shrink, self.splatter.min_radius * 1.01)
                    anchor[k] = min(max(a - r / 2, 0.0), 1.0)
                    anchor[j] = min(max(a + r / 2, 0.0), 1.0)
                    raw = self.splatter.radius_raw_for(child_r).to(bank.device)
                    bank[k, self.action_dim + 1] = raw
                    bank[j, self.action_dim + 1] = raw
                    n_split += 1
                else:
                    # clone: same parameters, jittered position; gradients separate them
                    jitter = (torch.rand((), generator=generator).item() - 0.5) * r
                    anchor[j] = min(max(a + jitter, 0.0), 1.0)
                    n_clone += 1
                bank[k, self.action_dim] = 0  # time offsets reset; anchors carry the position
                bank[j, self.action_dim] = 0
                noise = 0.01 * torch.randn(self.point_queries.weight.shape[1], generator=generator)
                self.point_queries.weight[j] = self.point_queries.weight[k] + noise.to(self.point_queries.weight)
                active[j] = True
                touched += [k, j]
                n_active += 1

        self.point_anchor.copy_(anchor.to(self.point_anchor.device))
        self.point_active.copy_(active.to(self.point_active.device))
        if optimizer is not None and touched:
            rows = torch.tensor(sorted(set(touched)))
            self._reset_optimizer_rows(optimizer, self.point_bank, rows)
            self._reset_optimizer_rows(optimizer, self.point_queries.weight, rows)
        self._reset_statistics()
        self._refresh_population_stats({"splat/n_split": n_split, "splat/n_clone": n_clone, "splat/n_prune": n_prune})
        return self.last_wta_stats

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
