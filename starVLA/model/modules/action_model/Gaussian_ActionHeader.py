"""Observation-deformed Gaussian action bank for QwenGR00T_Regressor."""

import math

import torch
from torch import nn

from starVLA.model.modules.action_model.GR00T_ActionHeader import DiTConfig, MLP
from starVLA.model.modules.action_model.GR00T_RegressorHeader import RegressionDiT
from starVLA.model.modules.action_model.gaussian_action_field import GaussianActionField


class GaussianActionHead(nn.Module):
    """Shared action-time primitives, deformed and gated by observation.

    Train by demonstration likelihood. Execute a dense, continuous mode path.
    No success reward is inferred from demonstration-only training.
    """

    def __init__(self, full_config):
        super().__init__()
        self.full_config = full_config
        self.config = config = full_config.framework.action_model
        if int(config.get("num_hypotheses", 1)) != 1 or config.get("multi_windows"):
            raise ValueError("GaussianActionHead requires num_hypotheses=1 and no multi_windows")
        if int(config.get("num_past_steps", 0)):
            raise ValueError("GaussianActionHead currently supports future-only chunks")
        self.action_dim = int(config.action_dim)
        self.action_horizon = int(config.action_horizon)
        self.field = GaussianActionField(
            self.action_dim, self.action_horizon,
            num_gaussians=config.get("num_gaussians", 32),
            min_time_width=config.get("gaussian_min_time_width", 0.01),
            initial_time_width=config.get("gaussian_initial_time_width", 0.15),
            min_action_std=config.get("gaussian_min_action_std", 0.03),
            initial_action_std=config.get("gaussian_initial_action_std", 0.3),
        )
        model_config = {**DiTConfig[config.action_model_type], **dict(config.diffusion_model_cfg or {})}
        for key in ("norm_type", "num_embeds_ada_norm", "compute_dtype"):
            model_config.pop(key, None)
        self.model = RegressionDiT(**model_config)
        dim = self.model.inner_dim
        self.state_encoder = MLP(config.state_dim, config.hidden_size, dim) if config.state_dim else None
        self.gaussian_queries = nn.Embedding(self.field.num_gaussians, dim)
        nn.init.normal_(self.gaussian_queries.weight, std=0.02)
        n_future = int(config.get("num_target_vision_tokens", 0) or 0)
        self.future_tokens = nn.Embedding(n_future, dim) if n_future else None
        self.gaussian_bank = nn.Parameter(torch.zeros(self.field.num_gaussians, self.field.parameter_dim))
        with torch.no_grad():
            self.gaussian_bank[:, 3:3 + self.action_dim].normal_(std=0.25)
        self.primitive_encoder = MLP(self.field.parameter_dim, config.hidden_size, dim)
        self.gaussian_decoder = MLP(self.model.config.output_dim, config.hidden_size, self.field.parameter_dim)
        # Small residuals preserve time coverage while allowing gradients into
        # queries/trunk on the first step (unlike an all-zero output matrix).
        nn.init.normal_(self.gaussian_decoder.layer2.weight, std=0.001)
        nn.init.zeros_(self.gaussian_decoder.layer2.bias)
        self.num_candidates = int(config.get("gaussian_num_candidates", 8))
        self.continuity_weight = float(config.get("gaussian_continuity_weight", 1.0))
        if self.num_candidates < 1 or not math.isfinite(self.continuity_weight) or self.continuity_weight < 0:
            raise ValueError("Require gaussian_num_candidates > 0 and finite gaussian_continuity_weight >= 0")

    def predict_parameters(self, vl_embs, state=None, encoder_attention_mask=None):
        batch = vl_embs.shape[0]
        parts = []
        if state is not None and self.state_encoder is not None:
            parts.append(self.state_encoder(state))
        if self.future_tokens is not None:
            parts.append(self.future_tokens.weight.unsqueeze(0).expand(batch, -1, -1))
        queries = self.gaussian_queries.weight + self.primitive_encoder(self.gaussian_bank)
        parts.append(queries.unsqueeze(0).expand(batch, -1, -1))
        features = self.model(
            hidden_states=torch.cat(parts, dim=1).to(dtype=vl_embs.dtype),
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
        )[:, -self.field.num_gaussians:]
        # Shared canonical primitives + observation-conditioned deformation.
        return self.gaussian_bank.unsqueeze(0) + self.gaussian_decoder(features).float()

    def forward(self, vl_embs, actions, state=None, encoder_attention_mask=None, action_mask=None):
        """Conditional demonstration NLL per valid action dimension.

        action_mask may be [B,H] or [B,H,D]. The standard framework currently
        supplies no padding mask, so all returned dataset slots are supervised.
        """
        parameters = self.predict_parameters(vl_embs, state, encoder_attention_mask)
        log_prob = self.field.log_prob(parameters, actions, action_mask)
        count = self.field.valid_mask(actions, action_mask).sum().clamp_min(1)
        return -log_prob.sum() / count

    @torch.no_grad()
    def predict_action(self, vl_embs, state=None, encoder_attention_mask=None):
        return self.field.select_actions(
            self.predict_parameters(vl_embs, state, encoder_attention_mask),
            num_candidates=self.num_candidates, continuity_weight=self.continuity_weight,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
