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

    Multiple-hypothesis / winner-takes-all mode (num_hypotheses = K > 1):
      - the DiT still processes only `horizon` action tokens; the K chunks
        branch off the shared per-step features inside the action decoder
        (output layer widened to K*action_dim), so extra compute is a few
        linear ops — negligible vs. the DiT pass. Only the hypothesis
        closest to GT (per sample) receives the regression loss, so the
        decoder branches specialise into distinct modes
      - wta_relax (relaxed WTA, Rupprecht et al.) gives non-winners a small
        loss weight to prevent dead hypotheses
      - a score head regresses each hypothesis' L1 error from its (detached)
        DiT features; at inference the lowest-predicted-error chunk is chosen,
        keeping one-shot inference while removing mode averaging
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
        assert self.loss_type in ("l1", "mse"), f"Unsupported loss_type: {self.loss_type}"

        # ------------------------------------------------------------------
        # Winner-takes-all multiple hypotheses
        #   num_hypotheses: K chunks per sample (1 = plain regression).
        #   wta_relax: ε weight on non-winner hypotheses (0.05 is a common
        #     choice; 0.0 = hard WTA, risks dead hypotheses).
        #   score_loss_coeff: weight of the score-head regression loss.
        # ------------------------------------------------------------------
        self.num_hypotheses = int(getattr(config, "num_hypotheses", 8))
        self.wta_relax = float(getattr(config, "wta_relax", 0.05))
        self.score_loss_coeff = float(getattr(config, "score_loss_coeff", 1.0))

        # MoE hypothesis branch:
        #   moe_expert_layers = 0: hypotheses branch only in the decoder's
        #     last linear layer (cheapest).
        #   moe_expert_layers = L > 0 (default 1): the shared DiT trunk is
        #     followed by K expert stacks of L transformer blocks each
        #     (cross-attending to vl_embs), giving every hypothesis
        #     attention-level capacity; actions are decoded from each
        #     expert's features by ONE shared decoder.
        #     Extra cost ≈ K*L / num_layers of one DiT pass.
        #   WTA acts as the GT-based router in training; the score head is
        #   the router at inference — no gating network needed.
        self.moe_expert_layers = int(getattr(config, "moe_expert_layers", 1))
        self.use_moe = self.num_hypotheses > 1 and self.moe_expert_layers > 0

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
        # The DiT sequence length is independent of K: hypotheses branch off
        # the shared per-step features in the decoder below.
        self.action_queries = nn.Embedding(self.action_horizon, self.input_embedding_dim)
        nn.init.normal_(self.action_queries.weight, mean=0.0, std=0.02)

        # K expert transformer stacks on top of the shared DiT trunk
        # (only in MoE mode). Each expert sees the trunk's pre-output hidden
        # states and cross-attends to vl_embs like a regular DiT block; the
        # trunk's norm_out/proj_out are reused per expert.
        if self.use_moe:
            mcfg = self.model.config
            self.expert_blocks = nn.ModuleList(
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
                    for _ in range(self.num_hypotheses)
                ]
            )
        else:
            self.expert_blocks = None

        # Action decoder:
        #   MoE mode: shared decoder applied to each expert's features (-> A).
        #   Linear-branch mode: shared hidden, K-way output (-> K*A).
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim if self.use_moe else self.num_hypotheses * self.action_dim,
        )

        # Score head: regresses every hypothesis' mean L1 error from mean-
        # pooled action features (per-expert features in MoE mode, shared
        # features otherwise). Inputs are detached so it acts purely as a
        # selector and never shapes the backbone.
        self.score_head = (
            MLP(
                input_dim=self.model.config.output_dim,
                hidden_dim=self.hidden_size,
                output_dim=1 if self.use_moe else self.num_hypotheses,
            )
            if self.num_hypotheses > 1
            else None
        )

        # Learnable "planning" tokens prepended before the action queries,
        # kept from GR00T so the DiT has dedicated scratch slots.
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.config = config

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _build_sa_embs(self, batch_size: int, state: torch.Tensor = None) -> torch.Tensor:
        """Assemble [state | future_tokens | action_queries] along the sequence dim."""
        action_features = self.action_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        state_features = self.state_encoder(state) if state is not None else None
        return (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

    def _run_hypotheses(self, vl_embs: torch.Tensor, state: torch.Tensor, encoder_attention_mask):
        """One shared-trunk DiT pass over `horizon` action tokens.

        pred_actions: (B, K, horizon, action_dim)
        action_feats: (B, K, horizon, D) in MoE mode (per-expert features),
                      (B, horizon, D) otherwise (shared across hypotheses)
        """
        B = vl_embs.shape[0]
        sa_embs = self._build_sa_embs(B, state).to(dtype=vl_embs.dtype)

        if self.use_moe:
            trunk = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                return_pre_output=True,
            )  # (B, T, inner_dim)
            expert_feats = []
            for expert in self.expert_blocks:
                h = trunk
                for blk in expert:
                    h = blk(
                        h,
                        encoder_hidden_states=vl_embs,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                h = self.model.proj_out(self.model.norm_out(h))  # shared output proj
                expert_feats.append(h[:, -self.action_horizon :])
            action_feats = torch.stack(expert_feats, dim=1)  # (B, K, H, D)
            pred_actions = self.action_decoder(action_feats)  # (B, K, H, A)
            return pred_actions, action_feats

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
        )
        action_feats = model_output[:, -self.action_horizon :]  # (B, H, D)
        pred = self.action_decoder(action_feats)  # (B, H, K*A)
        pred_actions = pred.reshape(B, self.action_horizon, self.num_hypotheses, self.action_dim).permute(0, 2, 1, 3)
        return pred_actions, action_feats

    def _score_hypotheses(self, action_feats: torch.Tensor) -> torch.Tensor:
        """Predict each hypothesis' error from action features -> (B, K)."""
        if self.use_moe:
            return self.score_head(action_feats.mean(dim=2)).squeeze(-1)  # per-expert pooled
        return self.score_head(action_feats.mean(dim=1))  # shared pooled, K-way output

    def forward(
        self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None
    ):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, action_horizon, action_dim)
        """
        pred_actions, action_feats = self._run_hypotheses(vl_embs, state, encoder_attention_mask)

        if self.num_hypotheses == 1:
            pred = pred_actions[:, 0]
            if self.loss_type == "l1":
                return F.l1_loss(pred, actions)
            return F.mse_loss(pred, actions)

        # Per-hypothesis error vs GT: (B, K)
        diff = pred_actions - actions.unsqueeze(1)
        errors = diff.abs().mean(dim=(2, 3)) if self.loss_type == "l1" else diff.pow(2).mean(dim=(2, 3))

        # Winner-takes-all: backprop only through the closest hypothesis,
        # plus ε-weighted loss on the others (relaxed WTA).
        winner = errors.argmin(dim=1)  # (B,)
        winner_loss = errors.gather(1, winner[:, None]).mean()
        loss = winner_loss
        if self.wta_relax > 0 and self.num_hypotheses > 1:
            others_loss = (errors.sum(dim=1) - errors.gather(1, winner[:, None]).squeeze(1)).mean() / (
                self.num_hypotheses - 1
            )
            loss = winner_loss + self.wta_relax * others_loss

        # Score head learns to predict each hypothesis' L1 error (selector).
        # Detached on both sides: gradients touch only the score head.
        score_target = pred_actions.detach().sub(actions.unsqueeze(1)).abs().mean(dim=(2, 3))  # (B, K)
        pred_scores = self._score_hypotheses(action_feats.detach())  # (B, K)
        score_loss = F.mse_loss(pred_scores, score_target)

        # lightweight per-batch stats for trainer logging (no grad)
        self.last_wta_stats = {
            "wta/unique_winners": int(winner.unique().numel()),
            "wta/score_loss": float(score_loss.detach()),
        }

        return loss + self.score_loss_coeff * score_loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        return_all_hypotheses: bool = False,
    ) -> torch.Tensor:
        pred_actions, action_feats = self._run_hypotheses(vl_embs, state, encoder_attention_mask)

        if self.num_hypotheses == 1:
            return pred_actions[:, 0]

        # Pick the hypothesis with the lowest predicted error.
        pred_scores = self._score_hypotheses(action_feats)  # (B, K)
        best = pred_scores.argmin(dim=1)  # (B,)
        chosen = pred_actions[torch.arange(pred_actions.shape[0], device=pred_actions.device), best]

        if return_all_hypotheses:
            return chosen, pred_actions, pred_scores
        return chosen

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """
    Factory: build RegressionActionHead from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        RegressionActionHead: Initialized regression action head.
    """
    return RegressionActionHead(full_config=config)
