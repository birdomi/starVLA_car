"""Patch-based GR00T regression head.

The (T x D) action chunk is treated as a 2-D grid: rows = time, columns = joints. Columns are split into
semantic joint groups (e.g. GR1: left_arm 7, right_arm 7, left_hand 6, right_hand 6, waist 3) and rows into
time patches of `patch_t` steps, giving (T / patch_t) x n_groups patches. One query token per FUTURE patch
is decoded to that patch's block of values (unpatchify); PAST patches (the executed action history) are
patch-embedded and prepended as context tokens. Every token carries a (time, group) coordinate and the DiT
self-attention layers use axial 2-D RoPE on those coordinates (cross-attention to the VLM tokens is unchanged).

Action representation (matches the loader's bidirectional `action_mode: delta`, anchored at the current step t):
  - future slots  : one-step displacements  d[0] = a[t] - s[t], d[i] = a[t+i] - a[t+i-1]   -> regression target
  - past slots    : the loader gives d[i] = a[i] - a[i+1] (pointing away from t); the head converts them to
                    displacements from the START of the past window, a[i] - a[t-P], via a cumulative sum
                    (`past_repr: from_start`), or feeds the per-step deltas as-is (`past_repr: per_step`).
Both are in the loader's normalized delta space, so the server only has to normalize raw past actions.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.GR00T_ActionHeader import MLP, DiTConfig
from starVLA.model.modules.action_model.GR00T_RegressorHeader import RegressionDiT


# ----------------------------------------------------------------------------- axial 2-D RoPE
def _rope_cos_sin(coords: torch.Tensor, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
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
    """Holds cos/sin tables for the current token layout; applied to q/k inside self-attention."""

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
        self.cos = torch.cat([ct, cg], dim=-1)[None, None]  # (1,1,N,head_dim)
        self.sin = torch.cat([st, sg], dim=-1)[None, None]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, N, head_dim) -> rotated (first half of dims by time, second half by group)."""
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
        q = q.view(B, N, h, -1).transpose(1, 2)  # (B,h,N,d)
        k = k.view(B, kv_in.shape[1], h, -1).transpose(1, 2)
        v = v.view(B, kv_in.shape[1], h, -1).transpose(1, 2)
        if is_self and self.rope.cos is not None:
            q = self.rope.apply(q)
            k = self.rope.apply(k)
        mask = None
        if attention_mask is not None:
            mask = attention_mask
            if mask.dim() == 2:  # (B, S) bool/0-1 -> (B,1,1,S)
                mask = mask.bool()[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, N, -1).to(q.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


# ----------------------------------------------------------------------------- head
class PatchRegressionActionHead(nn.Module):
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
        self.action_horizon = int(config.action_horizon)  # future steps
        self.num_past_steps = int(getattr(config, "num_past_steps", 0) or 0)
        self.patch_t = int(getattr(config, "patch_t", 4))
        groups = list(getattr(config, "patch_groups", None) or [self.action_dim])
        assert sum(groups) == self.action_dim, f"patch_groups {groups} must sum to action_dim {self.action_dim}"
        assert self.action_horizon % self.patch_t == 0 and self.num_past_steps % self.patch_t == 0
        self.groups = [int(g) for g in groups]
        self.group_offsets = [sum(self.groups[:i]) for i in range(len(self.groups))]
        self.n_groups = len(self.groups)
        self.n_tpatch_future = self.action_horizon // self.patch_t
        self.n_tpatch_past = self.num_past_steps // self.patch_t
        self.n_future_tokens = self.n_tpatch_future * self.n_groups
        self.n_past_tokens = self.n_tpatch_past * self.n_groups
        self.hidden_size = int(config.hidden_size)
        self.loss_type = getattr(config, "loss_type", "l1")
        self.past_repr = str(getattr(config, "past_repr", "from_start"))
        assert self.past_repr in ("from_start", "per_step", "raw")  # raw: normalized past fed as-is (same space as the output)
        # token_mode:
        #   "context"     : past patches are context tokens; separate learnable FUTURE query tokens are decoded.
        #   "image2image" : the (P x D) past window is the input image and the (H x D) future window the output
        #                   image: each past patch token (time patch pt, group g) is itself decoded into the future
        #                   patch at the same grid position (pt, g). No future query tokens; needs P == H.
        self.token_mode = str(getattr(config, "token_mode", "context"))
        assert self.token_mode in ("context", "image2image")
        if self.token_mode == "image2image":
            assert self.num_past_steps == self.action_horizon, "image2image needs num_past_steps == action_horizon (same grid in and out)"
        # loader action_mode the targets were built with: "delta" (one-step displacements, past slots point away
        # from t) or "rel" (every slot is a[i] - s[t]). Only affects how the past window is re-anchored.
        self.target_mode = str(getattr(config, "target_mode", "delta"))
        assert self.target_mode in ("delta", "rel", "abs")  # rel/abs: past re-anchored by subtracting its first slot
        self.use_rope = bool(getattr(config, "use_rope", True))
        # past_source: "action" (executed-action history) | "state" (OBSERVED state history s[t-P+1..t]; loader state
        #   window delta_indices range(-P+1, 1); client sends the last P raw states). Observed in training AND eval -> no
        #   exposure bias. The current-state token uses s[t].
        self.past_source = str(getattr(config, "past_source", "action"))
        assert self.past_source in ("action", "state")
        if self.past_source == "state":
            assert self.num_past_steps > 0 and int(config.state_dim) == self.action_dim, "state grid needs state_dim == action_dim"
        # output_residual (image2image only): pred = past grid + f(tokens); decoder output layers zero-initialised.
        self.output_residual = bool(getattr(config, "output_residual", False))
        if self.output_residual:
            assert self.token_mode == "image2image", "output_residual needs token_mode image2image (aligned grids)"

        D = self.input_embedding_dim
        self.state_encoder = MLP(input_dim=config.state_dim, hidden_dim=self.hidden_size, output_dim=D) if config.state_dim else None
        # shared query token for future patches (position comes from RoPE) + a learnable per-group bias
        self.future_query = nn.Parameter(torch.zeros(1, 1, D)) if self.token_mode == "context" else None
        self.group_embed = nn.Embedding(self.n_groups + 1, D)  # +1: state token
        if self.future_query is not None:
            nn.init.normal_(self.future_query, std=0.02)
        nn.init.normal_(self.group_embed.weight, std=0.02)
        # past patch embed: per-group Linear(patch_t * g -> D)
        self.past_embed = nn.ModuleList([nn.Linear(self.patch_t * g, D) for g in self.groups]) if self.num_past_steps else None
        self.time_embed_past = nn.Embedding(max(self.n_tpatch_past, 1), D) if self.num_past_steps else None
        # decoder: per-group MLP(out_dim -> hidden -> patch_t * g)
        out_dim = self.model.config.output_dim
        self.decoders = nn.ModuleList([MLP(input_dim=out_dim, hidden_dim=self.hidden_size, output_dim=self.patch_t * g) for g in self.groups])
        if self.output_residual:
            for dec in self.decoders:
                last = [m for m in dec.modules() if isinstance(m, nn.Linear)][-1]
                nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

        # token coordinates (time = patch start step relative to t, group index; state token = (0, n_groups))
        t_coords, g_coords = [], []
        if self.state_encoder is not None:
            t_coords.append(0.0); g_coords.append(float(self.n_groups))
        if self.token_mode == "image2image":
            # one token per grid cell (pt, g); its coordinate is the cell's row within the window
            for pt in range(self.n_tpatch_past):
                for g in range(self.n_groups):
                    t_coords.append(float(pt * self.patch_t)); g_coords.append(float(g))
        else:
            for pt in range(self.n_tpatch_past):
                for g in range(self.n_groups):
                    t_coords.append(float(-self.num_past_steps + pt * self.patch_t)); g_coords.append(float(g))
            for pt in range(self.n_tpatch_future):
                for g in range(self.n_groups):
                    t_coords.append(float(pt * self.patch_t)); g_coords.append(float(g))
        self.register_buffer("t_coords", torch.tensor(t_coords), persistent=False)
        self.register_buffer("g_coords", torch.tensor(g_coords), persistent=False)

        self.rope = None
        if self.use_rope:
            mcfg = self.model.config
            self.rope = AxialRoPE(mcfg.attention_head_dim, float(getattr(config, "rope_theta", 10000.0)))
            for blk in self.model.transformer_blocks:
                blk.attn1.set_processor(RoPEAttnProcessor(self.rope))
        self.last_wta_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------ patch helpers
    def patchify(self, x: torch.Tensor, n_tpatch: int) -> List[torch.Tensor]:
        """(B, n_tpatch*patch_t, D) -> list over time-patch-major tokens of per-group flattened blocks.
        Returns list of length n_tpatch*n_groups, element (B, patch_t*g)."""
        B = x.shape[0]
        toks = []
        for pt in range(n_tpatch):
            xs = x[:, pt * self.patch_t : (pt + 1) * self.patch_t]  # (B, patch_t, D)
            for g, off in zip(self.groups, self.group_offsets):
                toks.append(xs[:, :, off : off + g].reshape(B, self.patch_t * g))
        return toks

    def unpatchify(self, blocks: List[torch.Tensor], n_tpatch: int) -> torch.Tensor:
        """inverse of patchify: list of (B, patch_t*g) -> (B, n_tpatch*patch_t, D)."""
        B = blocks[0].shape[0]
        out = blocks[0].new_zeros(B, n_tpatch * self.patch_t, self.action_dim)
        i = 0
        for pt in range(n_tpatch):
            for g, off in zip(self.groups, self.group_offsets):
                out[:, pt * self.patch_t : (pt + 1) * self.patch_t, off : off + g] = blocks[i].reshape(B, self.patch_t, g)
                i += 1
        return out

    def past_to_from_start(self, past: torch.Tensor) -> torch.Tensor:
        """loader past deltas d[i] = a[i] - a[i+1]  ->  a[i] - a[0] = -cumsum(d)[i-1], rel[0] = 0."""
        c = torch.cumsum(past, dim=1)
        return torch.cat([torch.zeros_like(past[:, :1]), -c[:, :-1]], dim=1)

    # ------------------------------------------------------------------ tokens
    def _past_from_state(self, state: torch.Tensor) -> torch.Tensor:
        """state window (B, T, D) -> past grid (B, P, D): last P observed states (repeat s[t] if T < P)."""
        P = self.num_past_steps
        if state.shape[1] >= P:
            return state[:, -P:]
        return torch.cat([state[:, :1].expand(-1, P - state.shape[1], -1), state], dim=1)

    def _build_tokens(self, B: int, state: Optional[torch.Tensor], past: Optional[torch.Tensor], dtype, device):
        parts = []
        if state is not None and state.ndim == 4:  # (B, 1, T, D) from the server wrapper
            state = state[:, 0]
        if self.past_source == "state":
            if state is None:
                state = torch.zeros(B, self.num_past_steps, int(self.config.state_dim), device=device, dtype=dtype)
            past = self._past_from_state(state.to(dtype))
            self._last_past_grid = past
            state = state[:, -1:]  # current state token
        if self.state_encoder is not None:
            if state is None:
                state = torch.zeros(B, 1, int(self.config.state_dim), device=device, dtype=dtype)
            s = self.state_encoder(state.to(dtype)[:, -1:])  # (B,1,D)
            parts.append(s + self.group_embed.weight[self.n_groups][None, None].to(dtype))
        if self.num_past_steps:
            if past is None:
                past = torch.zeros(B, self.num_past_steps, self.action_dim, device=device, dtype=dtype)
            past = past.to(dtype)
            if self.past_source == "action":
                self._last_past_grid = past
            if self.past_repr == "from_start":
                past = self.past_to_from_start(past) if self.target_mode == "delta" else past - past[:, :1]
            # "per_step" / "raw": feed the loader-normalized past window unchanged
            blocks = self.patchify(past, self.n_tpatch_past)
            toks = []
            i = 0
            for pt in range(self.n_tpatch_past):
                for g in range(self.n_groups):
                    e = self.past_embed[g](blocks[i]) + self.group_embed.weight[g] + self.time_embed_past.weight[pt]
                    toks.append(e)
                    i += 1
            parts.append(torch.stack(toks, dim=1).to(dtype))  # (B, n_past_tokens, D)
        if self.token_mode == "image2image":
            return torch.cat(parts, dim=1)  # past patch tokens double as the output queries
        fq = []
        for pt in range(self.n_tpatch_future):
            for g in range(self.n_groups):
                fq.append(self.future_query[0, 0] + self.group_embed.weight[g])
        fq = torch.stack(fq, dim=0)[None].expand(B, -1, -1).to(dtype)  # (B, n_future_tokens, D)
        parts.append(fq)
        return torch.cat(parts, dim=1)

    def _run(self, vl_embs, state, past, encoder_attention_mask):
        B = vl_embs.shape[0]
        tokens = self._build_tokens(B, state, past, vl_embs.dtype, vl_embs.device)
        if self.rope is not None:
            self.rope.update(self.t_coords.to(vl_embs.device), self.g_coords.to(vl_embs.device))
        out = self.model(hidden_states=tokens, encoder_hidden_states=vl_embs, encoder_attention_mask=encoder_attention_mask)
        # image2image: the last n_past_tokens ARE the (pt, g) grid cells; context: the trailing future queries
        fut = out[:, -self.n_future_tokens :]  # (B, n_future_tokens, out_dim)
        blocks = []
        i = 0
        for pt in range(self.n_tpatch_future):
            for g in range(self.n_groups):
                blocks.append(self.decoders[g](fut[:, i]))
                i += 1
        pred = self.unpatchify(blocks, self.n_tpatch_future)  # (B, action_horizon, action_dim)
        if self.output_residual:
            pred = pred + self._last_past_grid.to(pred.dtype)  # same normalized space as the targets
        return pred

    # ------------------------------------------------------------------ API
    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def split_window(self, actions: torch.Tensor):
        """dataset window (B, past+future, D) -> (past (B,P,D) or None, future (B,H,D))."""
        if self.num_past_steps and actions.shape[1] >= self.num_past_steps + self.action_horizon:
            return actions[:, -self.num_past_steps - self.action_horizon : -self.action_horizon], actions[:, -self.action_horizon :]
        return None, actions[:, -self.action_horizon :]

    def forward(self, vl_embs, actions, state=None, encoder_attention_mask=None):
        past, target = self.split_window(actions)
        pred = self._run(vl_embs, state, past, encoder_attention_mask)
        diff = pred - target
        if self.loss_type == "l1":
            loss = diff.abs().mean()
        elif self.loss_type == "mse":
            loss = diff.pow(2).mean()
        else:
            loss = diff.abs().mean() + diff.pow(2).mean()
        with torch.no_grad():
            d = diff.abs()
            stats = {f"loss/g{gi}_l1": float(d[:, :, off : off + g].mean()) for gi, (g, off) in enumerate(zip(self.groups, self.group_offsets))}
            stats["loss/future_l1"] = float(d.mean())
            self.last_wta_stats = stats
        return loss

    @torch.no_grad()
    def predict_action(self, vl_embs, state=None, encoder_attention_mask=None, past_actions=None, **kwargs):
        return self._run(vl_embs, state, past_actions, encoder_attention_mask)

    def expand_targets(self, actions: torch.Tensor) -> torch.Tensor:
        return self.split_window(actions)[1]

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
