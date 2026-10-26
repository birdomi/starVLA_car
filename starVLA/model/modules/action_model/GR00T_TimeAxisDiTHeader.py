"""Time-axis flow-matching DiT head (standalone; touches neither GR00T_ActionHeader nor the grid DiT).

The DiT counterpart of GR00T_TimeAxisRegressorHeader: the noised action chunk (T x A) is read along the
ACTION-DIMENSION axis. One token per action dimension (29 for GR1), and each token carries that
dimension's whole T-step noised trajectory.

  * input projection  : the chunk is transposed to (A, T); dimension d's T values go through their OWN
                        encoder (Linear(T -> D) + flow-time embedding + 2 layers). encoder_mode picks the
                        sharing: "per_dim" (default), "per_group" (one encoder per joint group) or "shared".
  * axial RoPE        : each dimension token carries a (joint group, index inside the group) coordinate;
                        the DiT SELF-attention layers rotate q/k with 2-D RoPE on those coordinates.
                        There is no time coordinate: time lives INSIDE a token, not across tokens.
  * output projection : token d is decoded to T values -> dimension d's velocity trajectory
                        (decoder_mode per_dim | per_group | shared), then transposed back to (T, A).
  * state             : split into joint groups, one encoder each, appended to the cross-attention
                        KEYS/VALUES by default (state_as_kv).

Training and sampling are unchanged GR00T flow matching (Beta time sampling, velocity target, Euler).
The DiT sequence is A tokens (29) instead of the grid DiT's T*G (80) or the plain DiT's T (16).
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
from torch.distributions import Beta
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.GR00T_ActionHeader import MLP, DiTConfig
from starVLA.model.modules.action_model.flow_matching_head.action_encoder import SinusoidalPositionalEncoding, swish
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


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
    """cos/sin tables for the (time, group) token layout; applied to q/k inside self-attention."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for axial (time, group) RoPE"
        self.head_dim = head_dim
        self.theta = theta
        self.cos = None
        self.sin = None

    def update(self, t_coord: torch.Tensor, g_coord: torch.Tensor):
        d = self.head_dim // 2
        ct, st = _rope_cos_sin(t_coord, d, self.theta)
        cg, sg = _rope_cos_sin(g_coord, d, self.theta)
        self.cos = torch.cat([ct, cg], dim=-1)[None, None]
        self.sin = torch.cat([st, sg], dim=-1)[None, None]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cos.to(x.dtype), self.sin.to(x.dtype)
        d = self.head_dim // 2
        xt, xg = x[..., :d], x[..., d:]
        xt = xt * cos[..., :d] + _rotate_half(xt) * sin[..., :d]
        xg = xg * cos[..., d:] + _rotate_half(xg) * sin[..., d:]
        return torch.cat([xt, xg], dim=-1)


class RoPEAttnProcessor:
    """diffusers Attention processor: SDPA with axial RoPE on q/k for SELF-attention calls only.

    The grid tokens sit at the END of the sequence (state/future tokens may precede them), so the RoPE
    table is padded with zero coordinates for those leading tokens.
    """

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
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, N, -1).to(q.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


# ----------------------------------------------------------------------------- per-dimension action encoder
class DimActionEncoder(nn.Module):
    """GR00T's ActionEncoder, but one projection stack PER ACTION DIMENSION over the TIME axis.

    actions: (B, T, action_dim) -> (B, action_dim, D): token d = dimension d's T-step trajectory.
    mode: "per_dim" (A stacks), "per_group" (one stack per joint group, shared by its dims), "shared" (1 stack).
    """

    def __init__(self, action_groups: List[int], horizon: int, hidden_size: int, mode: str = "per_dim"):
        super().__init__()
        assert mode in ("per_dim", "per_group", "shared")
        self.mode = mode
        self.action_dim = sum(action_groups)
        self.horizon = horizon
        self.dim_group = [g for g, n in enumerate(action_groups) for _ in range(n)]
        n_stacks = {"per_dim": self.action_dim, "per_group": len(action_groups), "shared": 1}[mode]
        self.layer1 = nn.ModuleList([nn.Linear(horizon, hidden_size) for _ in range(n_stacks)])
        self.layer2 = nn.ModuleList([nn.Linear(2 * hidden_size, hidden_size) for _ in range(n_stacks)])
        self.layer3 = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_stacks)])
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def _stack(self, d: int) -> int:
        return d if self.mode == "per_dim" else (self.dim_group[d] if self.mode == "per_group" else 0)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        B, T, A = actions.shape
        assert T == self.horizon and A == self.action_dim, (actions.shape, self.horizon, self.action_dim)
        assert timesteps.dim() == 1 and timesteps.shape[0] == B, "timesteps must be (B,)"
        traj = actions.transpose(1, 2)  # (B, A, T)
        tau = self.pos_encoding(timesteps.unsqueeze(1).expand(-1, A))  # (B, A, D)
        toks = []
        for d in range(A):
            k = self._stack(d)
            a = self.layer1[k](traj[:, d])  # (B, D)
            x = swish(self.layer2[k](torch.cat([a, tau[:, d].to(a.dtype)], dim=-1)))
            toks.append(self.layer3[k](x))
        return torch.stack(toks, dim=1)  # (B, A, D)


# ----------------------------------------------------------------------------- head
class TimeAxisFlowmatchingActionHead(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config
        self.config = config

        dit_cfg = DiTConfig[config.action_model_type]
        self.input_embedding_dim = dit_cfg["input_embedding_dim"]
        diffusion_model_cfg = {**dit_cfg, **dict(config.diffusion_model_cfg or {})}
        self.model = DiT(**diffusion_model_cfg)

        self.action_dim = int(config.action_dim)
        self.state_dim = int(config.state_dim or 0)
        self.action_horizon = int(config.action_horizon)
        self.hidden_size = int(config.hidden_size)

        # ---- joint groups -------------------------------------------------
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
        self.num_grid_tokens = self.action_dim  # one token per action dimension
        self.register_buffer("dim_group", torch.tensor([g for g, n in enumerate(self.action_groups) for _ in range(n)]), persistent=False)
        self.register_buffer("dim_index", torch.tensor([j for n in self.action_groups for j in range(n)]), persistent=False)

        # ---- per-dimension input / output projections over the TIME axis ----
        self.encoder_mode = str(getattr(config, "encoder_mode", "per_dim"))
        self.decoder_mode = str(getattr(config, "decoder_mode", "per_dim"))
        assert self.decoder_mode in ("per_dim", "per_group", "shared")
        self.action_encoder = DimActionEncoder(self.action_groups, self.action_horizon, self.input_embedding_dim, self.encoder_mode)
        n_dec = {"per_dim": self.action_dim, "per_group": self.num_groups, "shared": 1}[self.decoder_mode]
        self.decoders = nn.ModuleList(
            [MLP(input_dim=self.model.config.output_dim, hidden_dim=self.hidden_size, output_dim=self.action_horizon) for _ in range(n_dec)]
        )

        # ---- state -> per-group tokens ------------------------------------
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
            out_dim = int(diffusion_model_cfg["cross_attention_dim"]) if self.state_as_kv else self.input_embedding_dim
            self.state_encoders = nn.ModuleList(
                [MLP(input_dim=g, hidden_dim=self.hidden_size, output_dim=out_dim) for g in self.state_groups]
            )
        else:
            self.state_group_slices = []
            self.state_encoders = None

        # ---- learnable "planning" tokens (GR00T's future tokens) -----------
        self.num_future_tokens = int(getattr(config, "num_target_vision_tokens", 0) or 0)
        if self.num_future_tokens > 0:
            self.future_tokens = nn.Embedding(self.num_future_tokens, self.input_embedding_dim)
            nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        else:
            self.future_tokens = None

        # ---- learned per-cell positional embedding (on top of RoPE) --------
        self.add_pos_embed = bool(getattr(config, "add_pos_embed", True))
        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(self.num_grid_tokens, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.position_embedding = None

        # ---- axial RoPE ----------------------------------------------------
        self.use_rope = bool(getattr(config, "use_rope", True))
        self.rope = None
        if self.use_rope:
            self.rope = AxialRoPE(int(self.model.config.attention_head_dim), float(getattr(config, "rope_theta", 10000.0)))
            proc = RoPEAttnProcessor(self.rope)
            for module in self.model.modules():
                if module.__class__.__name__ == "Attention":
                    module.set_processor(proc)

        # ---- flow matching -------------------------------------------------
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = int(config.num_timestep_buckets)
        self.num_inference_timesteps = int(config.num_inference_timesteps)

    # ------------------------------------------------------------------ utils
    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.noise_s)
        return (self.config.noise_s - sample) / self.config.noise_s

    def _rope_update(self, device, n_lead: int):
        """RoPE coordinates for [n_lead leading tokens | A dimension tokens]: (joint group, index in group)."""
        if self.rope is None:
            return
        g = self.dim_group.to(device)
        j = self.dim_index.to(device)
        if n_lead:
            pad = torch.zeros(n_lead, device=device, dtype=g.dtype)
            g, j = torch.cat([pad, g]), torch.cat([pad, j])
        self.rope.update(g, j)

    def _state_tokens(self, state: Optional[torch.Tensor], dtype) -> Optional[torch.Tensor]:
        if state is None or self.state_encoders is None:
            return None
        if state.dim() == 3:
            state = state[:, -1]
        elif state.dim() == 4:
            state = state[:, 0, -1]
        state = state.to(dtype)
        return torch.stack([enc(state[:, lo:hi]) for enc, (lo, hi) in zip(self.state_encoders, self.state_group_slices)], dim=1)

    def _build_inputs(self, vl_embs, action_features, s_tok, encoder_attention_mask):
        """Returns (hidden_states, encoder_hidden_states, encoder_attention_mask, n_lead)."""
        B = vl_embs.shape[0]
        if self.add_pos_embed:
            pos = self.position_embedding(torch.arange(self.num_grid_tokens, device=action_features.device))
            action_features = action_features + pos.unsqueeze(0).to(action_features.dtype)
        parts = []
        if s_tok is not None and self.state_as_kv:
            vl_embs = torch.cat([vl_embs, s_tok], dim=1)
            if encoder_attention_mask is not None:
                pad = torch.ones(B, s_tok.shape[1], dtype=encoder_attention_mask.dtype, device=encoder_attention_mask.device)
                encoder_attention_mask = torch.cat([encoder_attention_mask, pad], dim=1)
        elif s_tok is not None:
            parts.append(s_tok)
        if self.future_tokens is not None:
            parts.append(self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1).to(action_features.dtype))
        n_lead = sum(p.shape[1] for p in parts)
        hidden = torch.cat(parts + [action_features], dim=1) if parts else action_features
        return hidden, vl_embs, encoder_attention_mask, n_lead

    def _decode(self, model_output: torch.Tensor) -> torch.Tensor:
        """(B, ..., D) trunk output -> (B, T, action_dim) velocity: token d -> dimension d's T values."""
        feats = model_output[:, -self.num_grid_tokens :]  # (B, A, D)
        if self.decoder_mode == "per_dim":
            traj = torch.stack([dec(feats[:, d]) for d, dec in enumerate(self.decoders)], dim=1)  # (B, A, T)
        elif self.decoder_mode == "per_group":
            traj = torch.cat([self.decoders[g](feats[:, lo:hi]) for g, (lo, hi) in enumerate(self.group_slices)], dim=1)
        else:
            traj = self.decoders[0](feats)
        return traj.transpose(1, 2)  # (B, T, A)

    # ---------------------------------------------------------------- forward
    def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None):
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

        action_features = self.action_encoder(noisy_trajectory, t_discretized)
        s_tok = self._state_tokens(state, vl_embs.dtype)
        hidden, enc, enc_mask, n_lead = self._build_inputs(vl_embs, action_features, s_tok, encoder_attention_mask)
        self._rope_update(vl_embs.device, n_lead)

        model_output = self.model(
            hidden_states=hidden,
            encoder_hidden_states=enc,
            encoder_attention_mask=enc_mask,
            timestep=t_discretized,
        )
        pred = self._decode(model_output)
        loss = ((pred - velocity) ** 2).mean()
        with torch.no_grad():
            d = (pred - velocity).abs()
            self.last_wta_stats = {f"loss/g{i}_l1": float(d[..., lo:hi].mean()) for i, (lo, hi) in enumerate(self.group_slices)}
        return loss

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None, **kwargs):
        B, device = vl_embs.shape[0], vl_embs.device
        actions = torch.randn((B, self.action_horizon, self.action_dim), dtype=vl_embs.dtype, device=device)
        s_tok = self._state_tokens(state, vl_embs.dtype)
        dt = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            t_disc = int((step / float(self.num_inference_timesteps)) * self.num_timestep_buckets)
            timesteps = torch.full((B,), t_disc, device=device)
            action_features = self.action_encoder(actions, timesteps)
            hidden, enc, enc_mask, n_lead = self._build_inputs(vl_embs, action_features, s_tok, encoder_attention_mask)
            self._rope_update(device, n_lead)
            model_output = self.model(
                hidden_states=hidden,
                encoder_hidden_states=enc,
                encoder_attention_mask=enc_mask,
                timestep=timesteps,
            )
            actions = actions + dt * self._decode(model_output)
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_time_axis_dit_action_model(config=None):
    return TimeAxisFlowmatchingActionHead(config)
