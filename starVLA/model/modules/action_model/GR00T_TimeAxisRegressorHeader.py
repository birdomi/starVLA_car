"""Time-axis GR00T regression head (standalone; touches neither the plain nor the group-query head).

The action chunk is the same (T x A) grid as in the group-query head, but the grid is read along the
OTHER axis: one query token per ACTION DIMENSION, and each token regresses that dimension's whole
trajectory over the chunk.

  * queries : A learned query tokens (29 for GR1: 7 left arm + 7 right arm + 6 left hand + 6 right hand
              + 3 waist). The DiT sequence length is the action dimensionality, NOT the horizon.
  * decoder : token d is decoded to T values -> dimension d's time series. decoder_mode picks how much
              is shared: "per_dim" (one MLP per dimension, default), "per_group" (one MLP per joint
              group, shared by its dimensions) or "shared" (a single MLP for all dimensions).
  * RoPE    : each dimension token carries a (joint group, index inside that group) coordinate and the
              DiT SELF-attention layers rotate q/k with axial 2-D RoPE on those coordinates. There is no
              time coordinate here: time lives INSIDE a token's output vector, not across tokens.
  * state   : split into the same joint groups, one encoder each, appended to the cross-attention
              KEYS/VALUES by default (state_as_kv), so the dimension tokens keep the sequence to
              themselves.

Trade-off vs the group-query head: attention now reasons across joints (29 tokens, every joint sees
every other joint) while smoothness along time has to come out of one MLP per dimension, instead of
attention over time steps.

Plain L1/MSE regression: no WTA hypotheses, no MoE experts, no multi-window slots, no past history.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.GR00T_ActionHeader import MLP, DiTConfig
from starVLA.model.modules.action_model.GR00T_RegressorHeader import RegressionDiT


# ----------------------------------------------------------------------------- axial 2-D RoPE
def _rope_cos_sin(coords: torch.Tensor, dim: int, theta: float):
    half = dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=coords.device, dtype=torch.float32) / half))
    ang = coords.float()[:, None] * inv_freq[None, :]
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos(), ang.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class AxialRoPE:
    """cos/sin tables for the (joint group, index in group) token layout."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for axial RoPE"
        self.head_dim = head_dim
        self.theta = theta
        self.cos = None
        self.sin = None

    def update(self, a_coord: torch.Tensor, b_coord: torch.Tensor):
        d = self.head_dim // 2
        ca, sa = _rope_cos_sin(a_coord, d, self.theta)
        cb, sb = _rope_cos_sin(b_coord, d, self.theta)
        self.cos = torch.cat([ca, cb], dim=-1)[None, None]
        self.sin = torch.cat([sa, sb], dim=-1)[None, None]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cos.to(x.dtype), self.sin.to(x.dtype)
        d = self.head_dim // 2
        xa, xb = x[..., :d], x[..., d:]
        xa = xa * cos[..., :d] + _rotate_half(xa) * sin[..., :d]
        xb = xb * cos[..., d:] + _rotate_half(xb) * sin[..., d:]
        return torch.cat([xa, xb], dim=-1)


class RoPEAttnProcessor:
    """diffusers Attention processor: SDPA with axial RoPE on q/k for SELF-attention calls only."""

    def __init__(self, rope: AxialRoPE):
        self.rope = rope

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        B, N, _ = hidden_states.shape
        is_self = encoder_hidden_states is None
        kv_in = hidden_states if is_self else encoder_hidden_states
        q = attn.to_q(hidden_states)
        k = attn.to_k(kv_in)
        v = attn.to_v(kv_in)
        h = attn.heads
        q = q.view(B, N, h, -1).transpose(1, 2)
        k = k.view(B, kv_in.shape[1], h, -1).transpose(1, 2)
        v = v.view(B, kv_in.shape[1], h, -1).transpose(1, 2)
        if is_self and self.rope.cos is not None and self.rope.cos.shape[2] == N:
            q = self.rope.apply(q)
            k = self.rope.apply(k)
        mask = None
        if attention_mask is not None:
            mask = attention_mask
            if mask.dim() == 2:
                mask = mask.bool()[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, N, -1).to(q.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


# ----------------------------------------------------------------------------- head
class TimeAxisRegressionActionHead(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config
        self.config = config

        dit_cfg = DiTConfig[config.action_model_type]
        self.input_embedding_dim = dit_cfg["input_embedding_dim"]
        diffusion_model_cfg = dict(config.diffusion_model_cfg or {})
        for key in ("norm_type", "num_embeds_ada_norm", "compute_dtype"):
            diffusion_model_cfg.pop(key, None)
        diffusion_model_cfg = {**dit_cfg, **diffusion_model_cfg}
        self.model = RegressionDiT(**diffusion_model_cfg)

        self.action_dim = int(config.action_dim)
        self.state_dim = int(config.state_dim or 0)
        self.action_horizon = int(config.action_horizon)
        self.hidden_size = int(config.hidden_size)
        self.loss_type = str(getattr(config, "loss_type", "l1"))
        assert self.loss_type in ("l1", "mse"), f"Unsupported loss_type: {self.loss_type}"

        # ---- joint groups (used for RoPE coordinates, decoders and state) ---
        self.action_groups: List[int] = [int(g) for g in (getattr(config, "action_groups", None) or [])]
        assert self.action_groups, "action_groups must list the per-group action dims, e.g. [7, 7, 6, 6, 3]"
        assert sum(self.action_groups) == self.action_dim, (
            f"action_groups {self.action_groups} sum to {sum(self.action_groups)} != action_dim {self.action_dim}"
        )
        self.num_groups = len(self.action_groups)
        bounds = [0]
        for g in self.action_groups:
            bounds.append(bounds[-1] + g)
        self.group_slices = [(bounds[i], bounds[i + 1]) for i in range(self.num_groups)]
        # per-dimension coordinates: which group, and which joint inside that group
        self.register_buffer(
            "dim_group", torch.tensor([g for g, n in enumerate(self.action_groups) for _ in range(n)]), persistent=False
        )
        self.register_buffer(
            "dim_index", torch.tensor([j for n in self.action_groups for j in range(n)]), persistent=False
        )

        # ---- one query token per ACTION DIMENSION ---------------------------
        self.action_queries = nn.Embedding(self.action_dim, self.input_embedding_dim)
        nn.init.normal_(self.action_queries.weight, mean=0.0, std=0.02)

        # ---- decoders: token d -> that dimension's T-step trajectory ---------
        self.decoder_mode = str(getattr(config, "decoder_mode", "per_dim"))
        assert self.decoder_mode in ("per_dim", "per_group", "shared")
        out_dim = self.model.config.output_dim
        if self.decoder_mode == "per_dim":
            self.decoders = nn.ModuleList(
                [MLP(input_dim=out_dim, hidden_dim=self.hidden_size, output_dim=self.action_horizon) for _ in range(self.action_dim)]
            )
        elif self.decoder_mode == "per_group":
            self.decoders = nn.ModuleList(
                [MLP(input_dim=out_dim, hidden_dim=self.hidden_size, output_dim=self.action_horizon) for _ in range(self.num_groups)]
            )
        else:
            self.decoders = nn.ModuleList([MLP(input_dim=out_dim, hidden_dim=self.hidden_size, output_dim=self.action_horizon)])

        # ---- state -> per-group KV tokens -----------------------------------
        self.state_as_kv = bool(getattr(config, "state_as_kv", True))
        state_groups = [int(g) for g in (getattr(config, "state_groups", None) or [])]
        if not state_groups and self.state_dim:
            if self.state_dim == self.action_dim:
                state_groups = list(self.action_groups)
            elif self.state_dim == 2 * self.action_dim:  # loader sin/cos: [sin | cos] per key
                state_groups = [2 * g for g in self.action_groups]
            else:
                raise AssertionError(
                    f"state_dim {self.state_dim} is neither action_dim nor 2*action_dim; set state_groups explicitly"
                )
        self.state_groups = state_groups
        if self.state_dim:
            sb = [0]
            for g in self.state_groups:
                sb.append(sb[-1] + g)
            self.state_group_slices = [(sb[i], sb[i + 1]) for i in range(len(self.state_groups))]
            enc_out = int(diffusion_model_cfg["cross_attention_dim"]) if self.state_as_kv else self.input_embedding_dim
            self.state_encoders = nn.ModuleList(
                [MLP(input_dim=g, hidden_dim=self.hidden_size, output_dim=enc_out) for g in self.state_groups]
            )
        else:
            self.state_group_slices = []
            self.state_encoders = None

        # ---- axial RoPE over (joint group, index in group) -------------------
        self.use_rope = bool(getattr(config, "use_rope", True))
        self.rope = None
        if self.use_rope:
            self.rope = AxialRoPE(int(self.model.config.attention_head_dim), float(getattr(config, "rope_theta", 10000.0)))
            proc = RoPEAttnProcessor(self.rope)
            for module in self.model.modules():
                if module.__class__.__name__ == "Attention":
                    module.set_processor(proc)

        self.num_past_steps = 0
        self.last_wta_stats = {}

    # ------------------------------------------------------------------ utils
    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def expand_targets(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def _state_tokens(self, state: Optional[torch.Tensor], dtype) -> Optional[torch.Tensor]:
        if state is None or self.state_encoders is None:
            return None
        if state.dim() == 3:
            state = state[:, -1]
        elif state.dim() == 4:
            state = state[:, 0, -1]
        state = state.to(dtype)
        return torch.stack([enc(state[:, lo:hi]) for enc, (lo, hi) in zip(self.state_encoders, self.state_group_slices)], dim=1)

    def _decode(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, A, D) dimension tokens -> (B, T, A) trajectories."""
        if self.decoder_mode == "per_dim":
            traj = torch.stack([dec(feats[:, d]) for d, dec in enumerate(self.decoders)], dim=1)  # (B, A, T)
        elif self.decoder_mode == "per_group":
            traj = torch.cat([self.decoders[g](feats[:, lo:hi]) for g, (lo, hi) in enumerate(self.group_slices)], dim=1)
        else:
            traj = self.decoders[0](feats)  # (B, A, T)
        return traj.transpose(1, 2)  # (B, T, A)

    def _run(self, vl_embs: torch.Tensor, state: Optional[torch.Tensor], encoder_attention_mask):
        B = vl_embs.shape[0]
        if self.rope is not None:
            self.rope.update(self.dim_group.to(vl_embs.device), self.dim_index.to(vl_embs.device))
        queries = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1).to(dtype=vl_embs.dtype)

        s_tok = self._state_tokens(state, vl_embs.dtype)
        if s_tok is not None and self.state_as_kv:
            vl_embs = torch.cat([vl_embs, s_tok], dim=1)
            if encoder_attention_mask is not None:
                pad = torch.ones(B, s_tok.shape[1], dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device)
                encoder_attention_mask = torch.cat([encoder_attention_mask, pad], dim=1)
            hidden = queries
        elif s_tok is not None:
            hidden = torch.cat([s_tok, queries], dim=1)
        else:
            hidden = queries

        out = self.model(
            hidden_states=hidden,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
        )
        return self._decode(out[:, -self.action_dim :])

    def _regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target) if self.loss_type == "l1" else F.mse_loss(pred, target)

    # ---------------------------------------------------------------- forward
    def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None):
        pred = self._run(vl_embs, state, encoder_attention_mask)
        loss = self._regression_loss(pred, actions)
        with torch.no_grad():
            d = (pred - actions).abs()
            self.last_wta_stats = {f"loss/g{i}_l1": float(d[..., lo:hi].mean()) for i, (lo, hi) in enumerate(self.group_slices)}
        return loss

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None, **kwargs):
        return self._run(vl_embs, state, encoder_attention_mask)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_time_axis_action_model(config=None):
    return TimeAxisRegressionActionHead(config)
