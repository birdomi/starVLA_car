# Copyright 2025 starVLA community. All rights reserved.
"""Flow-matching action head with a state-history context and a progress readout.

Subclasses ``GR00T_ActionHeader.FlowmatchingActionHead`` so the DiT, the
encoders and the flow-matching objective stay bit-for-bit identical; only two
things are added.

1. **State history inside ``sa_embs``.** The base head already runs
   ``state_encoder`` over the state tensor and concatenates the result in front
   of the action tokens, so a state modality with
   ``state_indices = [-14, ..., 0]`` becomes 15 tokens with no structural
   change. What is missing there is *order*: the base head adds a position
   embedding to the action tokens only, which leaves the history
   permutation-invariant — the model could not tell the oldest frame from the
   newest. ``history_position_embedding`` fixes that.

2. **``future_tokens`` predict normalized progress t/T.** Those query tokens
   are already allocated in the base head and their outputs are discarded
   (``pred[:, -action_horizon:]`` keeps only the action tail, and the DiT runs
   with ``return_all_hidden_states=False`` — "not using flare now"). Here they
   are pooled and regressed onto how far into the episode the current frame is,
   so the policy reports its own progress at deploy time with no extra input
   and no external planner.

Why t/T and not the absolute frame index: on car-left-turn-signal the same task
runs 20.6-36.2 s, so the index mostly encodes how fast the operator happened to
move. A ridge regression from a 15-frame state window predicts t/T with MAE
0.145 against 0.246 for predicting the mean (41% better) but absolute t only to
3.89 s. Progress is the part that is actually determined by the observation.

Do not expect the progress loss to improve action accuracy on a single-task
dataset: only 1.8% of frame pairs are state-similar-but-action-different, and
those differ by 0.76 deg, which is below the model's own error. The head earns
its keep for failure detection, resume, and hand-off between tasks.
"""
import torch
import torch.nn.functional as F
from torch import nn

from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
)


class FlowmatchingActionHeadProgress(FlowmatchingActionHead):
    def __init__(self, full_config):
        super().__init__(full_config)
        config = full_config.framework.action_model

        # Order for the state-history tokens. Sized generously; only the first
        # n_hist rows are ever read, so a shorter history costs nothing.
        self.max_state_history = int(getattr(config, "max_state_history", 32))
        self.history_position_embedding = nn.Embedding(self.max_state_history, self.input_embedding_dim)
        # ZERO init, not the usual normal(0, 0.02). When fine-tuning from a
        # checkpoint trained with a single state token, a random embedding adds
        # an offset the pretrained DiT has never seen, and the action path does
        # not recover from it: a 20k fine-tune at lr 2e-06 left the val rollout
        # at arm 2.28 deg / coverage 144% against 1.33 deg / 100.3% for the
        # checkpoint it started from, essentially flat from 5k to 20k. With
        # zeros, index 0 reproduces the pretrained input exactly and the
        # remaining rows grow only as far as the history actually needs them.
        nn.init.zeros_(self.history_position_embedding.weight)

        self.num_future_tokens = int(config.num_target_vision_tokens)
        out_dim = self.model.config.output_dim
        # Sigmoid at the end because t/T lives in [0, 1].
        self.progress_head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.progress_loss_weight = float(getattr(config, "progress_loss_weight", 0.1))

    # ------------------------------------------------------------------ utils
    def _encode_state(self, state):
        """(B, n_hist, state_dim) -> (B, n_hist, D), ordered oldest-first.

        The dataloader stacks the state modality in ``state_indices`` order, so
        with [-14, ..., 0] row 0 is the oldest frame and row -1 is the current
        one. n_hist == 1 reproduces the base head up to a constant shift.
        """
        if state is None:
            return None
        feats = self.state_encoder(state)
        n_hist = feats.shape[1]
        if n_hist > self.max_state_history:
            raise ValueError(
                f"state history of {n_hist} frames exceeds max_state_history="
                f"{self.max_state_history}; raise it under framework.action_model."
            )
        pos = torch.arange(n_hist, device=feats.device, dtype=torch.long)
        return feats + self.history_position_embedding(pos).unsqueeze(0)

    def _build_sa_embs(self, state_features, action_features, batch_size):
        future = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        if state_features is None:
            return torch.cat((future, action_features), dim=1), 0
        return torch.cat((state_features, future, action_features), dim=1), state_features.shape[1]

    def _progress_from(self, model_output, n_state):
        """Pool the future-token outputs and regress t/T from them."""
        q = model_output[:, n_state : n_state + self.num_future_tokens]
        return torch.sigmoid(self.progress_head(q.mean(dim=1))).squeeze(-1)

    # ---------------------------------------------------------------- forward
    def forward(self, vl_embs, actions, state=None, encoder_attention_mask=None, progress=None):
        device = vl_embs.device

        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)
        state_features = self._encode_state(state)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        sa_embs, n_state = self._build_sa_embs(state_features, action_features, vl_embs.shape[0])
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]
        action_loss = ((pred_actions - velocity) ** 2).mean()

        zero = action_loss.new_zeros(())
        if progress is None:
            return {"action_loss": action_loss, "progress_loss": zero,
                    "progress_mae": zero, "loss": action_loss}

        # The batch is repeated `repeated_diffusion_steps` times upstream, and
        # the progress label is repeated the same way, so these line up 1:1.
        progress_pred = self._progress_from(model_output, n_state)
        progress = progress.to(progress_pred.dtype)
        progress_loss = F.smooth_l1_loss(progress_pred, progress, beta=0.1)
        return {
            "action_loss": action_loss,
            "progress_loss": progress_loss,
            "progress_mae": (progress_pred - progress).abs().mean().detach(),
            "loss": action_loss + self.progress_loss_weight * progress_loss,
        }

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_action(self, vl_embs, state=None, encoder_attention_mask=None, return_progress=False):
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        state_features = self._encode_state(state)
        progress_pred = None

        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            sa_embs, n_state = self._build_sa_embs(state_features, action_features, batch_size)
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)
            actions = actions + dt * pred[:, -self.action_horizon :]
            if step == num_steps - 1:
                # Read progress off the final, least-noisy pass.
                progress_pred = self._progress_from(model_output, n_state)

        if return_progress:
            return actions, progress_pred
        return actions


def get_action_model(config=None):
    return FlowmatchingActionHeadProgress(full_config=config)
