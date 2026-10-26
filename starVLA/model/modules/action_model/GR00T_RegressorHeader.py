# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
#
# Regression variant of GR00T_ActionHeader:
#   - no noised-action input (learnable action query tokens instead)
#   - no timestep embedding / AdaLN conditioning (plain LayerNorm)
#   - single forward pass regression instead of iterative flow-matching denoising


import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import BasicTransformerBlock
from starVLA.model.modules.action_model.GR00T_ActionHeader import MLP, DiTConfig


class RegressionDiT(ModelMixin, ConfigMixin):
    """Cross-attention DiT without timestep conditioning.

    Same block layout as GR00T's DiT (optionally interleaving self-attention
    layers), but every norm is a plain LayerNorm and the output projection is
    unconditioned, so no `timestep` argument exists anywhere.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        final_dropout: bool = True,
        positional_embeddings: str = "sinusoidal",
        interleave_self_attention: bool = False,
        cross_attention_dim: int = None,
        **kwargs,
    ):
        super().__init__()
        self._interleave_self_attention = bool(interleave_self_attention)
        self.inner_dim = num_attention_heads * attention_head_dim
        self.gradient_checkpointing = False

        all_blocks = []
        for idx in range(num_layers):
            use_self_attn = idx % 2 == 1 and self._interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_type="layer_norm",  # no AdaLN: temb is never consumed
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        # Unconditioned output block (DiT uses AdaLN modulation here)
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=True, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, output_dim)
        print(
            "Total number of RegressionDiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D) or list of layer-wise tensors
        encoder_attention_mask=None,
        return_all_hidden_states: bool = False,
        return_pre_output: bool = False,
    ):
        hidden_states = hidden_states.contiguous()
        is_layerwise_encoder = isinstance(encoder_hidden_states, (list, tuple))
        encoder_hidden_states = (
            [state.contiguous() for state in encoder_hidden_states]
            if is_layerwise_encoder
            else encoder_hidden_states.contiguous()
        )

        all_hidden_states = [hidden_states]

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self._interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                )
            else:
                if is_layerwise_encoder:
                    block_encoder_hidden_states = encoder_hidden_states[idx]
                else:
                    block_encoder_hidden_states = encoder_hidden_states
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=block_encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )
            all_hidden_states.append(hidden_states)

        if return_pre_output:
            if return_all_hidden_states:
                return hidden_states, all_hidden_states
            return hidden_states

        output = self.proj_out(self.norm_out(hidden_states))
        if return_all_hidden_states:
            return output, all_hidden_states
        return output


class RegressionActionHead(nn.Module):
    """GR00T-style action head that regresses the action chunk directly.

    Differences from FlowmatchingActionHead:
      - the DiT input for the action slots is a set of learnable query tokens
        (one per action step), not an encoding of the noised trajectory
      - no timestep sampling / embedding anywhere
      - forward() computes an L1 (default) or MSE regression loss;
        predict_action() is a single forward pass

    Single-hypothesis regression only: the winner-takes-all multiple-hypothesis
    mode (num_hypotheses / wta_relax / score head / per-hypothesis MoE experts)
    was removed. `moe_expert_layers` survives because the multi-window "experts"
    mode still uses it as the per-window stack depth.
    """

    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config

        # DiT architecture selection: same DiT-B / DiT-L shapes as GR00T.
        # diffusion_model_cfg.cross_attention_dim must be set by the framework
        # to the VLM hidden size before get_action_model() is called.
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]

        diffusion_model_cfg = dict(config.diffusion_model_cfg or {})
        # Flow-matching-only knobs are meaningless here; drop them if a YAML
        # written for the flow-matching head is reused as-is.
        for key in ("norm_type", "num_embeds_ada_norm", "compute_dtype"):
            diffusion_model_cfg.pop(key, None)
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = RegressionDiT(**diffusion_model_cfg)

        self.action_horizon = int(config.action_horizon)
        self.action_dim = config.action_dim
        self.hidden_size = config.hidden_size

        # "l1" (default) or "mse"
        self.loss_type = getattr(config, "loss_type", "l1")
        assert self.loss_type in ("l1", "mse", "l1+l2"), f"Unsupported loss_type: {self.loss_type}"
        # "l1+l2": l1_weight * L1 + l2_weight * MSE
        self.l1_weight = float(getattr(config, "l1_weight", 1.0))
        self.l2_weight = float(getattr(config, "l2_weight", 1.0))
        # For bidirectional windows: the first num_past_steps slots are PAST
        # actions, the rest FUTURE. Only used to log past/future errors separately.
        self.num_past_steps = int(getattr(config, "num_past_steps", 0) or 0)
        # Multi-window mode: ONE model predicts several bidirectional windows at
        # once, e.g. [8, 16, 32, 64] -> 120 query slots. Window W owns W slots
        # (W/2 past + W/2 future) and its target is the centre slice of the
        # dataset's longest window (action_horizon == max(multi_windows)).
        self.multi_windows = [int(w) for w in (getattr(config, "multi_windows", None) or [])]
        # multi_window_mode:
        #   "experts" (default): shared trunk over max-window slots; one extra
        #       transformer stack per window (moe_expert_layers blocks, cross-attn
        #       to vl_embs) takes the trunk's centre W tokens and the shared
        #       decoder emits that window's trajectory. Output = concat of windows.
        #   "nested": no extra stacks; window W = centre W slots of the trunk output.
        #   "separate": each window owns its own trunk slots (sum of windows tokens).
        self.multi_window_mode = str(getattr(config, "multi_window_mode", "experts"))
        if self.multi_windows:
            assert all(w % 2 == 0 for w in self.multi_windows), "windows must be even (half past / half future)"
            assert max(self.multi_windows) == self.action_horizon, "action_horizon must equal the longest window"
            assert self.multi_window_mode in ("experts", "nested", "separate")
            c = self.action_horizon // 2
            self.window_slices = []  # (pred_lo, pred_hi, tgt_lo, tgt_hi)
            self.window_trunk_slices = []  # trunk-slot slice feeding each window (experts mode)
            if self.multi_window_mode == "nested":
                for w in self.multi_windows:
                    self.window_slices.append((c - w // 2, c + w // 2, c - w // 2, c + w // 2))
                self.num_slots = self.action_horizon
            elif self.multi_window_mode == "experts":
                start = 0
                for w in self.multi_windows:
                    self.window_slices.append((start, start + w, c - w // 2, c + w // 2))
                    self.window_trunk_slices.append((c - w // 2, c + w // 2))
                    start += w
                self.num_slots = self.action_horizon  # trunk slots
            else:  # separate
                start = 0
                for w in self.multi_windows:
                    self.window_slices.append((start, start + w, c - w // 2, c + w // 2))
                    start += w
                self.num_slots = start
        else:
            self.window_slices = None
            self.num_slots = self.action_horizon

        # Per-window expert stack depth (multi_window_mode == "experts"):
        # L transformer blocks per window on top of the shared DiT trunk.
        self.moe_expert_layers = int(getattr(config, "moe_expert_layers", 1))

        self.state_encoder = (
            MLP(
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.state_dim
            else None
        )

        # Learnable per-step query tokens replace the noised-action encoding.
        # They are position-specific, so no extra positional embedding is needed
        # on top (the DiT blocks also carry sinusoidal positions internally).
        self.action_queries = nn.Embedding(self.num_slots, self.input_embedding_dim)
        nn.init.normal_(self.action_queries.weight, mean=0.0, std=0.02)

        # Window experts (multi_window_mode == "experts"): one transformer stack
        # per window on top of the shared trunk, same block as the MoE experts.
        self.use_window_experts = bool(self.multi_windows) and self.multi_window_mode == "experts"
        if self.use_window_experts:
            assert self.moe_expert_layers > 0, "set moe_expert_layers >= 1 for window experts"
            mcfg = self.model.config
            self.window_experts = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            BasicTransformerBlock(
                                self.model.inner_dim,
                                mcfg.num_attention_heads,
                                mcfg.attention_head_dim,
                                dropout=mcfg.dropout,
                                activation_fn=mcfg.activation_fn,
                                attention_bias=mcfg.attention_bias,
                                upcast_attention=mcfg.upcast_attention,
                                norm_type="layer_norm",
                                norm_elementwise_affine=mcfg.norm_elementwise_affine,
                                norm_eps=mcfg.norm_eps,
                                positional_embeddings=mcfg.positional_embeddings,
                                num_positional_embeddings=mcfg.max_num_positional_embeddings,
                                final_dropout=mcfg.final_dropout,
                                cross_attention_dim=mcfg.cross_attention_dim,
                            )
                            for _ in range(self.moe_expert_layers)
                        ]
                    )
                    for _ in self.multi_windows
                ]
            )
        else:
            self.window_experts = None

        # Action decoder: per-slot features -> action_dim.
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # Learnable "planning" tokens prepended before the action queries,
        # kept from GR00T so the DiT has dedicated scratch slots.
        # num_target_vision_tokens = 0 disables them (they carry no loss here:
        # GR00T used them for FLARE future-prediction, which is off).
        self.num_future_tokens = int(getattr(config, "num_target_vision_tokens", 0) or 0)
        if self.num_future_tokens > 0:
            self.future_tokens = nn.Embedding(self.num_future_tokens, self.input_embedding_dim)
            nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        else:
            self.future_tokens = None

        self.config = config

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _build_sa_embs(self, batch_size: int, state: torch.Tensor = None) -> torch.Tensor:
        """Assemble [state | future_tokens | action_queries] along the sequence dim."""
        parts = []
        if state is not None and self.state_encoder is not None:
            parts.append(self.state_encoder(state))
        if self.future_tokens is not None:
            parts.append(self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1))
        parts.append(self.action_queries.weight.unsqueeze(0).expand(batch_size, -1, -1))
        return torch.cat(parts, dim=1)

    def _run(self, vl_embs: torch.Tensor, state: torch.Tensor, encoder_attention_mask):
        """One DiT pass over the action slots.

        pred_actions: (B, num_slots, action_dim)
        action_feats: (B, num_slots, D)
        """
        B = vl_embs.shape[0]
        sa_embs = self._build_sa_embs(B, state).to(dtype=vl_embs.dtype)

        if self.use_window_experts:
            trunk = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                return_pre_output=True,
            )[:, -self.num_slots :]  # (B, 64, inner): the trunk's action slots
            feats, preds = [], []
            for expert, (lo, hi) in zip(self.window_experts, self.window_trunk_slices):
                h = trunk[:, lo:hi]  # centre W tokens
                for blk in expert:
                    h = blk(h, encoder_hidden_states=vl_embs, encoder_attention_mask=encoder_attention_mask)
                h = self.model.proj_out(self.model.norm_out(h))
                feats.append(h)
                preds.append(self.action_decoder(h))
            return torch.cat(preds, dim=1), torch.cat(feats, dim=1)  # (B, sum(W), A), (B, sum(W), D)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
        )
        action_feats = model_output[:, -self.num_slots :]  # (B, S, D)
        return self.action_decoder(action_feats), action_feats  # (B, S, A), (B, S, D)

    def expand_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """(B, action_horizon, A) dataset window -> (B, num_slots, A) slot targets."""
        if self.window_slices is None or self.multi_window_mode == "nested":
            return actions  # nested: slots == dataset window
        return torch.cat([actions[:, tlo:thi] for (_, _, tlo, thi) in self.window_slices], dim=1)

    def split_windows(self, pred: torch.Tensor) -> dict:
        """(B, num_slots, A) -> {window: (B, window, A)} (past half first)."""
        if self.window_slices is None:
            return {self.action_horizon: pred}
        return {w: pred[:, lo:hi] for w, (lo, hi, _, _) in zip(self.multi_windows, self.window_slices)}

    def _regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        if self.loss_type == "mse":
            return F.mse_loss(pred, target)
        return self.l1_weight * F.l1_loss(pred, target) + self.l2_weight * F.mse_loss(pred, target)

    def forward(
        self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None
    ):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, action_horizon, action_dim)
        """
        pred, _ = self._run(vl_embs, state, encoder_attention_mask)  # (B, S, A)

        if self.window_slices is not None:
            # per-window loss (each window weighted equally), targets sliced
            # from the centre of the dataset window
            losses, stats = [], {}
            for w, (lo, hi, tlo, thi) in zip(self.multi_windows, self.window_slices):
                pw, tw = pred[:, lo:hi], actions[:, tlo:thi]
                losses.append(self._regression_loss(pw, tw))
                with torch.no_grad():
                    d = (pw - tw).abs()
                    stats[f"loss/w{w}_l1"] = float(d.mean())
                    stats[f"loss/w{w}_past_l1"] = float(d[:, : w // 2].mean())
                    stats[f"loss/w{w}_future_l1"] = float(d[:, w // 2 :].mean())
            self.last_wta_stats = stats
            return torch.stack(losses).mean()

        loss = self._regression_loss(pred, actions)
        with torch.no_grad():
            stats = {}
            if 0 < self.num_past_steps < pred.shape[1]:
                d = (pred - actions).abs()
                stats["loss/past_l1"] = float(d[:, : self.num_past_steps].mean())
                stats["loss/future_l1"] = float(d[:, self.num_past_steps :].mean())
            self.last_wta_stats = stats
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        pred, _ = self._run(vl_embs, state, encoder_attention_mask)
        return pred

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """
    Factory: build a regression, Gaussian-field, or direct action-splat head.

    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        nn.Module: Head selected by action_model.head_type (default regression).
    """
    head_type = config.framework.action_model.get("head_type", "regression")
    if head_type == "action_splat":
        from starVLA.model.modules.action_model.ActionSplatHead import ActionSplatHead

        return ActionSplatHead(full_config=config)
    if head_type == "gaussian":
        from starVLA.model.modules.action_model.Gaussian_ActionHeader import GaussianActionHead

        return GaussianActionHead(full_config=config)
    if head_type != "regression":
        raise ValueError(f"Unknown regressor head_type: {head_type}")
    return RegressionActionHead(full_config=config)
