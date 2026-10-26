"""Group-query GR00T regression head (standalone; does not touch the plain/WTA regressor).

Same idea as the patch regressor's image2image grid, but with NO past action/state grid as input:
the action chunk (T x A) is still treated as a 2-D grid (rows = time, columns = joint groups), yet
only the OUTPUT side is factorised.

  * action queries : one learned query per (timestep, joint group) cell -> T * G tokens
                     (time-major, index = t * G + g) instead of ONE query per timestep.
                     Each cell is decoded by its own MLP to that group's slice of the action vector.
  * axial RoPE     : every query token carries a (time, group) coordinate; the DiT SELF-attention
                     layers rotate q/k with 2-D RoPE on those coordinates (use_rope: false disables it,
                     leaving the learned per-cell queries as the only positional signal).
  * state as KV    : the proprioceptive state is split into the same joint groups, each group encoded
                     by its own MLP, and the resulting G tokens are CONCATENATED to the VLM tokens on
                     the cross-attention KEY/VALUE side (state_as_kv: true, the default). The queries
                     therefore attend to "scene tokens + my own joints" in one place, and the state
                     never occupies self-attention slots. state_as_kv: false falls back to the plain
                     head's behaviour (one state token prepended to the query sequence).

State layout: the GR00T loader's sin/cos transform emits [sin(d) | cos(d)] PER KEY, so with
state_dim == 2 * action_dim each group owns 2 * group_dim contiguous values; `state_groups` can be
given explicitly when a dataset does not follow that rule.

Plain L1/MSE regression only: no WTA hypotheses, no MoE experts, no multi-window slots.
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
    """coords: (N,) -> cos, sin of shape (N, dim) for a `dim`-wide rotary slice (dim even)."""
    half = dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=coords.device, dtype=torch.float32) / half))
    ang = coords.float()[:, None] * inv_freq[None, :]  # (N, half)
    ang = torch.cat([ang, ang], dim=-1)  # (N, dim)
    return ang.cos(), ang.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class AxialRoPE:
    """cos/sin tables for the (time, group) token layout; applied to q/k inside self-attention."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for axial (time, group) RoPE"
        self.head_dim = head_dim
        self.theta = theta
        self.cos = None  # (1, 1, N, head_dim)
        self.sin = None

    def update(self, t_coord: torch.Tensor, g_coord: torch.Tensor):
        d = self.head_dim // 2
        ct, st = _rope_cos_sin(t_coord, d, self.theta)
        cg, sg = _rope_cos_sin(g_coord, d, self.theta)
        self.cos = torch.cat([ct, cg], dim=-1)[None, None]
        self.sin = torch.cat([st, sg], dim=-1)[None, None]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, N, head_dim) -> first half of the dims rotated by time, second half by group."""
        cos, sin = self.cos.to(x.dtype), self.sin.to(x.dtype)
        d = self.head_dim // 2
        xt, xg = x[..., :d], x[..., d:]
        xt = xt * cos[..., :d] + _rotate_half(xt) * sin[..., :d]
        xg = xg * cos[..., d:] + _rotate_half(xg) * sin[..., d:]
        return torch.cat([xt, xg], dim=-1)


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
        q = q.view(B, N, h, -1).transpose(1, 2)  # (B, h, N, d)
        k = k.view(B, kv_in.shape[1], h, -1).transpose(1, 2)
        v = v.view(B, kv_in.shape[1], h, -1).transpose(1, 2)
        if is_self and self.rope.cos is not None:
            q = self.rope.apply(q)
            k = self.rope.apply(k)
        mask = None
        if attention_mask is not None:
            mask = attention_mask
            if mask.dim() == 2:  # (B, S) bool/0-1 -> (B, 1, 1, S)
                mask = mask.bool()[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, N, -1).to(q.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


# ----------------------------------------------------------------------------- head
class GroupQueryRegressionActionHead(nn.Module):
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

        # ---- joint groups -------------------------------------------------
        self.query_groups: List[int] = [int(g) for g in (getattr(config, "query_groups", None) or [])]
        assert self.query_groups, "query_groups must list the per-group action dims, e.g. [7, 7, 6, 6, 3]"
        assert sum(self.query_groups) == self.action_dim, (
            f"query_groups {self.query_groups} sum to {sum(self.query_groups)} != action_dim {self.action_dim}"
        )
        self.num_groups = len(self.query_groups)
        bounds = [0]
        for g in self.query_groups:
            bounds.append(bounds[-1] + g)
        self.group_slices = [(bounds[i], bounds[i + 1]) for i in range(self.num_groups)]
        self.num_query_tokens = self.action_horizon * self.num_groups

        # ---- action queries (one per (t, g) cell) --------------------------
        self.action_queries = nn.Embedding(self.num_query_tokens, self.input_embedding_dim)
        nn.init.normal_(self.action_queries.weight, mean=0.0, std=0.02)
        self.group_decoders = nn.ModuleList(
            [MLP(input_dim=self.model.config.output_dim, hidden_dim=self.hidden_size, output_dim=g) for g in self.query_groups]
        )

        # ---- state -> per-group KV tokens ---------------------------------
        self.state_as_kv = bool(getattr(config, "state_as_kv", True))
        state_groups = [int(g) for g in (getattr(config, "state_groups", None) or [])]
        if not state_groups and self.state_dim:
            if self.state_dim == self.action_dim:
                state_groups = list(self.query_groups)
            elif self.state_dim == 2 * self.action_dim:  # loader sin/cos, [sin|cos] per key
                state_groups = [2 * g for g in self.query_groups]
            else:
                raise AssertionError(
                    f"state_dim {self.state_dim} is neither action_dim nor 2*action_dim; set state_groups explicitly"
                )
        self.state_groups = state_groups
        if self.state_dim:
            assert sum(self.state_groups) == self.state_dim, (
                f"state_groups {self.state_groups} sum to {sum(self.state_groups)} != state_dim {self.state_dim}"
            )
            sb = [0]
            for g in self.state_groups:
                sb.append(sb[-1] + g)
            self.state_group_slices = [(sb[i], sb[i + 1]) for i in range(len(self.state_groups))]
            # one encoder per group; KV tokens must match the CROSS-attention width, self tokens the DiT width
            out_dim = int(diffusion_model_cfg["cross_attention_dim"]) if self.state_as_kv else self.input_embedding_dim
            self.state_encoders = nn.ModuleList(
                [MLP(input_dim=g, hidden_dim=self.hidden_size, output_dim=out_dim) for g in self.state_groups]
            )
        else:
            self.state_group_slices = []
            self.state_encoders = None

        # ---- axial RoPE over the (time, group) coordinates -----------------
        self.use_rope = bool(getattr(config, "use_rope", True))
        self.rope = None
        if self.use_rope:
            head_dim = int(self.model.config.attention_head_dim)
            self.rope = AxialRoPE(head_dim, float(getattr(config, "rope_theta", 10000.0)))
            proc = RoPEAttnProcessor(self.rope)
            for module in self.model.modules():
                if module.__class__.__name__ == "Attention":
                    module.set_processor(proc)

        # kept for trainer/framework compatibility
        self.num_past_steps = 0
        self.last_wta_stats = {}

    # ------------------------------------------------------------------ utils
    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def expand_targets(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def _rope_update(self, device):
        if self.rope is None:
            return
        t = torch.arange(self.action_horizon, device=device).repeat_interleave(self.num_groups)  # time-major
        g = torch.arange(self.num_groups, device=device).repeat(self.action_horizon)
        self.rope.update(t, g)

    def _state_tokens(self, state: Optional[torch.Tensor], dtype) -> Optional[torch.Tensor]:
        """(B, *, state_dim) -> (B, G, D): one token per joint group, from the LAST state step."""
        if state is None or self.state_encoders is None:
            return None
        if state.dim() == 3:
            state = state[:, -1]
        elif state.dim() == 4:
            state = state[:, 0, -1]
        state = state.to(dtype)
        return torch.stack([enc(state[:, lo:hi]) for enc, (lo, hi) in zip(self.state_encoders, self.state_group_slices)], dim=1)

    def _run(self, vl_embs: torch.Tensor, state: Optional[torch.Tensor], encoder_attention_mask):
        B = vl_embs.shape[0]
        self._rope_update(vl_embs.device)
        queries = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1).to(dtype=vl_embs.dtype)

        s_tok = self._state_tokens(state, vl_embs.dtype)
        if s_tok is not None and self.state_as_kv:
            # joint-group state tokens ride along on the cross-attention KEY/VALUE side
            vl_embs = torch.cat([vl_embs, s_tok], dim=1)
            if encoder_attention_mask is not None:
                pad = torch.ones(B, s_tok.shape[1], dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device)
                encoder_attention_mask = torch.cat([encoder_attention_mask, pad], dim=1)
            hidden = queries
        elif s_tok is not None:
            hidden = torch.cat([s_tok, queries], dim=1)  # plain-head style: state on the self-attention side
        else:
            hidden = queries

        out = self.model(
            hidden_states=hidden,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
        )
        feats = out[:, -self.num_query_tokens :].reshape(B, self.action_horizon, self.num_groups, -1)
        return torch.cat([dec(feats[:, :, g]) for g, dec in enumerate(self.group_decoders)], dim=-1)  # (B, T, A)

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


def get_group_query_action_model(config=None):
    return GroupQueryRegressionActionHead(config)
