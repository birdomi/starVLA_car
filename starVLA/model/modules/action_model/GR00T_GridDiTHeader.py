"""Action-grid flow-matching DiT head (standalone; does not touch GR00T_ActionHeader).

The GR00T flow-matching head embeds the whole 29-D noised action vector of a timestep into ONE token,
so left arm, right arm, left hand, right hand and waist share a single projection and a single token.
Here the action chunk is treated as a 2-D grid instead — rows = time, columns = joint groups:

  * input projection  : every (time patch, joint group) cell gets its OWN encoder
                        (Linear(time_patch * group_dim -> D) + flow-time embedding + 2 more per-group
                        layers), giving (T / time_patch) * G action tokens instead of T. Token order is
                        row-major (index = row * G + g), matching GR00T_GroupQueryRegressorHeader.
  * keyframes         : keyframe_stride / keyframe_indices switch the head from a Dense ActionMap
                        (every timestep is a token) to a Keyframe ActionMap: only temporal anchors are
                        predicted (loss included), and predict_action fills the gaps by linear
                        interpolation, optionally ramping from a caller-provided start_anchor.
  * share_groups      : groups listed together reuse ONE encoder / decoder / state encoder, e.g.
                        [[0, 1]] to tie a left and a right column of equal width.
  * time_patch_size   : how many consecutive timesteps one row covers. 1 = one token per timestep
                        (the original action grid), 2 / 4 = coarser rows, T = a single row whose tokens
                        each carry a group's whole trajectory (GR00T_TimeAxisDiTHeader's layout).
  * axial RoPE        : each token carries its (time, group) coordinate and the DiT SELF-attention
                        layers rotate q/k with 2-D RoPE on those coordinates. Cross-attention to the
                        VLM tokens is untouched. use_rope: false disables it.
  * output projection : one decoder MLP per joint group; cell (row, g) predicts group g's velocity for all
                        `time_patch` timesteps of that row, and the slices are concatenated back into
                        (B, T, action_dim).
  * state             : split into the same joint groups, one encoder each, and by default appended to
                        the cross-attention KEYS/VALUES (state_as_kv) rather than to the token sequence.

Training and sampling are unchanged GR00T flow matching: x_t = (1-t) * noise + t * a, target velocity
a - noise, Beta(alpha, beta) time sampling, Euler integration at inference.

Cost note: the DiT sequence grows from T (+state/future tokens) to (T / time_patch) * G, and
predict_action runs it num_inference_timesteps times, so the head's inference cost grows roughly with
G / time_patch.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
from torch.distributions import Beta
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.GR00T_ActionHeader import MLP, DiTConfig
from starVLA.model.modules.action_model.flow_matching_head.action_encoder import SinusoidalPositionalEncoding, swish
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT, TimestepEncoder


# ----------------------------------------------------------------------------- index specs
def _parse_index_spec(spec, dim: int) -> List[int]:
    """One group's dims. Accepts "0:7,14:20" (half-open, python-style), [0, 1, 2] or [[0, 7], [14, 20]]."""
    out: List[int] = []
    if isinstance(spec, str):
        for part in spec.replace(" ", "").split(","):
            if not part:
                continue
            if ":" in part:
                lo, hi = part.split(":")
                out.extend(range(int(lo or 0), int(hi or dim)))
            else:
                out.append(int(part))
        return out
    for item in spec:
        if isinstance(item, (list, tuple)):
            lo, hi = item
            out.extend(range(int(lo), int(hi)))
        else:
            out.append(int(item))
    return out


def _shared_module_list(owner: List[int], build) -> nn.ModuleList:
    """One module per group, except groups that share: they hold the SAME module object.

    `owner[i]` is the group whose module group i uses (owner[i] == i for an owner).
    Owners always come first (owner[i] <= i), so the referenced module already exists.
    nn.Module.parameters() de-duplicates shared tensors, so a shared module is counted once.
    """
    mods: List[nn.Module] = []
    for i, o in enumerate(owner):
        mods.append(mods[o] if o != i else build(i))
    return nn.ModuleList(mods)


def _resolve_share_groups(share_spec, group_dims: List[int], what: str) -> List[int]:
    """[[0, 1]] -> owner list. Shared groups must have identical dims and may not overlap."""
    owner = list(range(len(group_dims)))
    seen = set()
    for grp in share_spec or []:
        idxs = sorted({int(i) for i in grp})
        assert len(idxs) > 1, f"share_groups entry {grp} needs at least two groups"
        for i in idxs:
            assert 0 <= i < len(group_dims), f"share_groups index {i} out of range for {len(group_dims)} groups"
            assert i not in seen, f"group {i} appears in more than one share_groups entry"
            seen.add(i)
        head = idxs[0]
        for i in idxs:
            assert group_dims[i] == group_dims[head], (
                f"share_groups {idxs} needs equal {what} dims, got "
                f"{[group_dims[j] for j in idxs]} -- groups of different width cannot share a projection"
            )
            owner[i] = head
    return owner


def _perm_buffers(index_lists: List[List[int]], dim: int, what: str):
    """Flatten per-group index lists into a gather permutation + its inverse."""
    perm = [i for g in index_lists for i in g]
    assert sorted(perm) == list(range(dim)), (
        f"{what} must partition range({dim}) exactly once; got {sorted(perm)}"
    )
    fwd = torch.as_tensor(perm, dtype=torch.long)
    inv = torch.empty(dim, dtype=torch.long)
    inv[fwd] = torch.arange(dim, dtype=torch.long)
    return fwd, inv


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
    """cos/sin tables for the grid token layout; applied to q/k inside self-attention.

    axes = "time_group" (default): head_dim split in half, first half rotated by the TIME
    coordinate, second half by the GROUP coordinate (2-axis axial RoPE).
    axes = "time": standard 1-axis RoPE over the FULL head_dim using only the time coordinate;
    the group axis carries no positional signal at all.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0, axes: str = "time_group"):
        assert axes in ("time_group", "time"), axes
        if axes == "time_group":
            assert head_dim % 4 == 0, "head_dim must be divisible by 4 for axial (time, group) RoPE"
        else:
            assert head_dim % 2 == 0, "head_dim must be even for 1-axis time RoPE"
        self.axes = axes
        self.head_dim = head_dim
        self.theta = theta
        self.cos = None
        self.sin = None

    def update(self, t_coord: torch.Tensor, g_coord: torch.Tensor = None):
        if self.axes == "time":
            c, sn = _rope_cos_sin(t_coord, self.head_dim, self.theta)
            self.cos, self.sin = c[None, None], sn[None, None]
            return
        d = self.head_dim // 2
        ct, st = _rope_cos_sin(t_coord, d, self.theta)
        cg, sg = _rope_cos_sin(g_coord, d, self.theta)
        self.cos = torch.cat([ct, cg], dim=-1)[None, None]
        self.sin = torch.cat([st, sg], dim=-1)[None, None]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cos.to(x.dtype), self.sin.to(x.dtype)
        if self.axes == "time":
            return x * cos + _rotate_half(x) * sin
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


# ----------------------------------------------------------------------------- per-group action encoder
class GroupActionEncoder(nn.Module):
    """GR00T's ActionEncoder, but with one projection stack PER (TIME PATCH, JOINT GROUP) CELL TYPE.

    A cell covers `time_patch` consecutive timesteps of one joint group, so the grid is
    (T / time_patch) rows x G columns:

      time_patch = 1  -> one token per (timestep, group)            [the original action grid]
      time_patch = T  -> one token per group, carrying its whole trajectory  [time-axis style]

    actions: (B, T, action_dim) -> (B, (T / time_patch) * G, D), row-major cells.
    """

    def __init__(self, group_dims: List[int], hidden_size: int, time_patch: int = 1, owner: Optional[List[int]] = None):
        super().__init__()
        self.group_dims = list(group_dims)
        self.time_patch = int(time_patch)
        self.owner = list(owner) if owner is not None else list(range(len(self.group_dims)))
        bounds = [0]
        for g in self.group_dims:
            bounds.append(bounds[-1] + g)
        self.slices = [(bounds[i], bounds[i + 1]) for i in range(len(self.group_dims))]
        self.layer1 = _shared_module_list(self.owner, lambda i: nn.Linear(self.time_patch * self.group_dims[i], hidden_size))
        self.layer2 = _shared_module_list(self.owner, lambda i: nn.Linear(2 * hidden_size, hidden_size))
        self.layer3 = _shared_module_list(self.owner, lambda i: nn.Linear(hidden_size, hidden_size))
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        B, T, _ = actions.shape
        assert timesteps.dim() == 1 and timesteps.shape[0] == B, "timesteps must be (B,)"
        p = self.time_patch
        assert T % p == 0, f"action_horizon {T} is not divisible by time_patch_size {p}"
        rows = T // p
        # the flow-matching time tau is per-sample, so one copy per ROW
        tau = self.pos_encoding(timesteps.unsqueeze(1).expand(-1, rows))  # (B, rows, D)
        cells = []
        for g, (lo, hi) in enumerate(self.slices):
            chunk = actions[..., lo:hi].reshape(B, rows, p * (hi - lo))  # (B, rows, p * dg)
            a = self.layer1[g](chunk)  # (B, rows, D)
            x = swish(self.layer2[g](torch.cat([a, tau.to(a.dtype)], dim=-1)))
            cells.append(self.layer3[g](x))
        # (B, rows, G, D) -> row-major (B, rows*G, D)
        return torch.stack(cells, dim=2).flatten(1, 2)


# ----------------------------------------------------------------------------- head
class GridFlowmatchingActionHead(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config
        self.config = config

        dit_cfg = DiTConfig[config.action_model_type]
        self.input_embedding_dim = dit_cfg["input_embedding_dim"]
        diffusion_model_cfg = {**dit_cfg, **dict(config.diffusion_model_cfg or {})}
        # split-mode group AdaLN owns its own (inner/2)-wide TimestepEncoder and ALWAYS overrides the
        # conditioning, so the trunk must not build its unused 768-wide encoder (dead params break DDP).
        _split_cond = bool(getattr(config, "group_adaln_cond", False)) and             str(getattr(config, "group_cond_mode", "proj") or "proj") == "split"
        if _split_cond:
            diffusion_model_cfg.setdefault("build_timestep_encoder", False)
        self.model = DiT(**diffusion_model_cfg)

        self.action_dim = int(config.action_dim)
        self.state_dim = int(config.state_dim or 0)
        self.action_horizon = int(config.action_horizon)
        self.hidden_size = int(config.hidden_size)

        # ---- joint groups -------------------------------------------------
        # Two ways to describe the grid's columns:
        #   action_groups        : contiguous per-group dims, e.g. [7, 7, 6, 6, 3]
        #   action_group_indices : explicit dim index sets, e.g. ["0:7,14:20", "7:14,20:26", "26:29"]
        #                          (left arm+hand, right arm+hand, waist). A group may be NON-CONTIGUOUS in
        #                          the loader's action layout, so the head gathers the dims into group order
        #                          on the way in and scatters the predicted velocity back on the way out.
        action_index_spec = list(getattr(config, "action_group_indices", None) or [])
        if action_index_spec:
            action_index_lists = [_parse_index_spec(spec, self.action_dim) for spec in action_index_spec]
            self.action_groups: List[int] = [len(g) for g in action_index_lists]
            fwd, inv = _perm_buffers(action_index_lists, self.action_dim, "action_group_indices")
            self.register_buffer("action_perm", fwd, persistent=False)
            self.register_buffer("action_inv_perm", inv, persistent=False)
        else:
            self.action_groups = [int(g) for g in (getattr(config, "action_groups", None) or [])]
            assert self.action_groups, (
                "action_groups (or action_group_indices) must describe the joint groups, e.g. [7, 7, 6, 6, 3]"
            )
            self.action_perm = None
            self.action_inv_perm = None
        assert sum(self.action_groups) == self.action_dim, (
            f"action_groups {self.action_groups} sum to {sum(self.action_groups)} != action_dim {self.action_dim}"
        )
        self.num_groups = len(self.action_groups)
        # ---- time patching: a row spans `time_patch_size` consecutive timesteps ----
        self.time_patch = int(getattr(config, "time_patch_size", 1) or 1)
        assert self.action_horizon % self.time_patch == 0, (
            f"action_horizon {self.action_horizon} must be divisible by time_patch_size {self.time_patch}"
        )
        # ---- keyframes: Dense ActionMap (default) models every timestep; Keyframe ActionMap models
        # only temporal anchors and fills the rest by linear interpolation at inference.
        #   keyframe_stride: s   -> anchors at 0-indexed steps [s-1, 2s-1, ..., H-1]  (H=16, s=4 -> 3,7,11,15)
        #   keyframe_indices: [] -> explicit 0-indexed steps (overrides the stride); must end at H-1.
        # Training loss touches ONLY the keyframe steps (flow matching runs in keyframe space).
        kf = [int(i) for i in (getattr(config, "keyframe_indices", None) or [])]
        stride = int(getattr(config, "keyframe_stride", 0) or 0)
        if not kf and stride > 0:
            assert self.action_horizon % stride == 0, (
                f"keyframe_stride {stride} must divide action_horizon {self.action_horizon}"
            )
            kf = list(range(stride - 1, self.action_horizon, stride))
        if kf:
            assert kf == sorted(set(kf)) and 0 <= kf[0] and kf[-1] == self.action_horizon - 1, (
                f"keyframe_indices {kf} must be ascending, unique, within [0, {self.action_horizon - 1}] "
                "and END at the last horizon step so interpolation covers the whole chunk"
            )
            assert self.time_patch == 1, "keyframes require time_patch_size = 1 (a row per anchor)"
            self.register_buffer("keyframe_idx", torch.as_tensor(kf, dtype=torch.long), persistent=False)
        else:
            self.keyframe_idx = None
        self.keyframe_indices = kf or None
        # Optional auxiliary loss (Keyframe ActionMap only): reconstruct the DENSE trajectory by
        # interpolating the CLEAN keyframe estimate (x_t + (1 - t) * v_hat, exact when v_hat is the
        # true velocity) and penalize its MSE against the full ground-truth trajectory. The
        # interpolation uses the same no-anchor convention as inference (ramp-in holds keyframe 0),
        # so the keyframes are trained to be good anchors for exactly the reconstruction used at test.
        self.dense_recon_weight = float(getattr(config, "dense_recon_weight", 0.0) or 0.0)
        assert self.dense_recon_weight == 0.0 or kf, (
            "dense_recon_weight needs keyframes (dense heads already supervise every step)"
        )
        # Weight of the flow-matching velocity MSE itself (default 1.0). fm_loss_weight: 0 +
        # dense_recon_weight > 0 trains PURELY from the interpolated dense reconstruction; at the
        # anchor steps that term equals the FM loss reweighted by (1 - t)^2, so the velocity field
        # still gets a training signal, just vanishing near t = 1.
        self.fm_loss_weight = float(getattr(config, "fm_loss_weight", 1.0))
        assert self.fm_loss_weight > 0 or self.dense_recon_weight > 0, (
            "fm_loss_weight and dense_recon_weight cannot both be zero"
        )
        # rows the DiT actually models; pred_horizon = rows * time_patch (== len(keyframes) when active)
        self.num_time_rows = (len(kf) if kf else self.action_horizon) // self.time_patch
        self.pred_horizon = self.num_time_rows * self.time_patch
        # RoPE time coordinate per row: the REAL timestep of the anchor, so non-uniform keyframes keep
        # their temporal spacing (dense: 0..rows-1, identical to the previous behaviour for time_patch 1).
        coords = torch.as_tensor(kf, dtype=torch.long) if kf else torch.arange(self.num_time_rows)
        self.register_buffer("time_coords", coords, persistent=False)
        names = [str(n) for n in (getattr(config, "group_names", None) or [])]
        self.group_names = names if len(names) == self.num_groups else [f"g{i}" for i in range(self.num_groups)]
        bounds = [0]
        for g in self.action_groups:
            bounds.append(bounds[-1] + g)
        self.group_slices = [(bounds[i], bounds[i + 1]) for i in range(self.num_groups)]
        self.num_grid_tokens = self.num_time_rows * self.num_groups

        # ---- weight sharing between groups ---------------------------------
        # share_groups: [[0, 1]] makes groups 0 and 1 (e.g. left and right) use ONE encoder,
        # decoder and state encoder instead of two. Shared groups must have equal dims.
        # NOTE: this shares raw joint projections; on a mirror-symmetric robot the two sides
        # often use opposite sign conventions, which a shared projection cannot undo by itself.
        self.share_groups = [list(g) for g in (getattr(config, "share_groups", None) or [])]
        self.group_owner = _resolve_share_groups(self.share_groups, self.action_groups, "action")

        # ---- per-cell input / output projections ---------------------------
        # each cell decodes its whole time patch at once: output_dim = time_patch * group_dim
        self.action_encoder = GroupActionEncoder(
            self.action_groups, self.input_embedding_dim, time_patch=self.time_patch, owner=self.group_owner
        )
        self.group_decoders = _shared_module_list(
            self.group_owner,
            lambda i: MLP(
                input_dim=self.model.config.output_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.time_patch * self.action_groups[i],
            ),
        )

        # ---- state -> per-group tokens ------------------------------------
        self.state_as_kv = bool(getattr(config, "state_as_kv", True))
        # state_single: 기본 DiT와 동일하게 58-D state 전체를 하나의 MLP로 한 토큰에 인코딩
        # (그룹 분할 없음). state_as_kv=false와 함께 쓰면 배치까지 기본 DiT와 같아진다
        # (시퀀스 선행 토큰; RoPE 좌표 0, group AdaLN은 no-group 인덱스).
        self.state_single = bool(getattr(config, "state_single", False))
        state_index_spec = [] if self.state_single else list(getattr(config, "state_group_indices", None) or [])
        state_groups = [] if self.state_single else [int(g) for g in (getattr(config, "state_groups", None) or [])]
        state_perm = None
        if self.state_single and self.state_dim:
            state_groups = [self.state_dim]          # 통짜 한 그룹
        if state_index_spec and self.state_dim:
            state_index_lists = [_parse_index_spec(spec, self.state_dim) for spec in state_index_spec]
            state_groups = [len(g) for g in state_index_lists]
            state_perm, _ = _perm_buffers(state_index_lists, self.state_dim, "state_group_indices")
        elif self.state_single:
            pass
        elif not state_groups and self.state_dim:
            if self.state_dim == self.action_dim:
                state_groups = list(self.action_groups)
                if self.action_perm is not None:  # same layout as the action vector -> same gather
                    state_perm = self.action_perm.clone()
            elif self.state_dim == 2 * self.action_dim and self.action_perm is None:  # sin/cos: [sin | cos] per key
                state_groups = [2 * g for g in self.action_groups]
            else:
                raise AssertionError(
                    f"state_dim {self.state_dim} cannot be split automatically for these action groups; "
                    "set state_groups (contiguous) or state_group_indices (explicit) in the yaml"
                )
        assert self.state_single or not self.state_dim or len(state_groups) == self.num_groups, (
            f"state groups {state_groups} must have one entry per action group ({self.num_groups})"
        )
        if state_perm is None:
            self.state_perm = None
        else:
            self.register_buffer("state_perm", state_perm, persistent=False)
        self.state_groups = state_groups
        if self.state_dim:
            sb = [0]
            for g in self.state_groups:
                sb.append(sb[-1] + g)
            self.state_group_slices = [(sb[i], sb[i + 1]) for i in range(len(self.state_groups))]
            out_dim = int(diffusion_model_cfg["cross_attention_dim"]) if self.state_as_kv else self.input_embedding_dim
            if self.state_single:
                # 기본 DiT처럼 통짜 state 하나 -> MLP 하나 -> 토큰 1개
                self.state_encoders = nn.ModuleList(
                    [MLP(input_dim=self.state_dim, hidden_dim=self.hidden_size, output_dim=out_dim)]
                )
            else:
                # the same sharing applies to the state encoders; their dims must line up too
                _resolve_share_groups(self.share_groups, self.state_groups, "state")
                self.state_encoders = _shared_module_list(
                    self.group_owner,
                    lambda i: MLP(input_dim=self.state_groups[i], hidden_dim=self.hidden_size, output_dim=out_dim),
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

        # ---- learned per-GROUP (body-part) embedding -----------------------
        # add_group_embed: one learnable vector per column (group), added to every grid token of that
        # column -- body identity only, no time information. Pair with add_pos_embed: false and
        # use_rope: true + rope_axes: "time" so the time axis is carried by temporal RoPE on q/k.
        self.add_group_embed = bool(getattr(config, "add_group_embed", False))
        if self.add_group_embed:
            self.group_embedding = nn.Embedding(self.num_groups, self.input_embedding_dim)
            nn.init.normal_(self.group_embedding.weight, mean=0.0, std=0.02)
        else:
            self.group_embedding = None

        # ---- axial RoPE ----------------------------------------------------
        self.use_rope = bool(getattr(config, "use_rope", True))
        # rope_axes: "time_group" (2축 axial, 기존) | "time" (시간축만 1-axis, 그룹 위치신호 없음)
        self.rope_axes = str(getattr(config, "rope_axes", "time_group") or "time_group")
        self.rope = None
        if self.use_rope:
            self.rope = AxialRoPE(
                int(self.model.config.attention_head_dim),
                float(getattr(config, "rope_theta", 10000.0)),
                axes=self.rope_axes,
            )
            proc = RoPEAttnProcessor(self.rope)
            for module in self.model.modules():
                if module.__class__.__name__ == "Attention":
                    module.set_processor(proc)

        # ---- group identity via AdaLN conditioning -------------------------
        # group_adaln_cond: the diffusion-time embedding is CONCATENATED with a learned per-group
        # embedding and projected back, giving a PER-TOKEN AdaLN conditioning (B, N, D). This is the
        # group-identity channel when add_pos_embed is off and RoPE carries only the time axis.
        # Leading tokens (state-as-sequence / future tokens) use a dedicated "no-group" index G.
        self.group_adaln_cond = bool(getattr(config, "group_adaln_cond", False))
        # group_cond_mode:
        #   "proj"  : concat(temb 768, group 64) -> Linear(832 -> 768). trunk의 timestep encoder 재사용.
        #   "split" : concat(temb 384, group 384) = 768 직결, projection 없음. 시간x그룹 상호작용은
        #             각 AdaLN의 자체 linear(768->1536)가 담당하므로 별도 혼합층이 불필요.
        self.group_cond_mode = str(getattr(config, "group_cond_mode", "proj") or "proj")
        assert self.group_cond_mode in ("proj", "split"), self.group_cond_mode
        if self.group_adaln_cond:
            inner = int(self.model.config.num_attention_heads) * int(self.model.config.attention_head_dim)
            if self.group_cond_mode == "split":
                half = inner // 2
                # 전용 (inner/2)-폭 TimestepEncoder. trunk 인코더는 build_timestep_encoder=False로
                # 아예 생성하지 않으므로 미사용 파라미터가 없다. concat 뒤 혼합층도 없다.
                self.group_time_encoder = TimestepEncoder(embedding_dim=half)
                self.group_cond_embed = nn.Embedding(self.num_groups + 1, inner - half)
                nn.init.normal_(self.group_cond_embed.weight, mean=0.0, std=0.02)
                self.group_cond_proj = None
            else:
                gdim = int(getattr(config, "group_cond_dim", 64) or 64)
                self.group_cond_embed = nn.Embedding(self.num_groups + 1, gdim)  # +1 = no-group
                nn.init.normal_(self.group_cond_embed.weight, mean=0.0, std=0.02)
                self.group_cond_proj = nn.Linear(inner + gdim, inner)

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
        """RoPE coordinates for [n_lead leading tokens | rows*G grid cells]; leading tokens get coord 0.

        The time coordinate is the ROW index (patch index), not the raw timestep.
        """
        if self.rope is None:
            return
        t = self.time_coords.to(device).repeat_interleave(self.num_groups)
        g = torch.arange(self.num_groups, device=device).repeat(self.num_time_rows)
        if n_lead:
            pad = torch.zeros(n_lead, device=device, dtype=t.dtype)
            t, g = torch.cat([pad, t]), torch.cat([pad, g])
        self.rope.update(t, g)

    def _to_group_order(self, x: torch.Tensor) -> torch.Tensor:
        """(..., action_dim) in loader order -> group order (identity unless action_group_indices is set)."""
        return x if self.action_perm is None else x.index_select(-1, self.action_perm)

    def _from_group_order(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.action_inv_perm is None else x.index_select(-1, self.action_inv_perm)

    def _state_tokens(self, state: Optional[torch.Tensor], dtype) -> Optional[torch.Tensor]:
        if state is None or self.state_encoders is None:
            return None
        if state.dim() == 3:
            state = state[:, -1]
        elif state.dim() == 4:
            state = state[:, 0, -1]
        state = state.to(dtype)
        if self.state_single:
            return self.state_encoders[0](state)[:, None, :]   # (B, 1, out) — 기본 DiT식 통짜 토큰
        if self.state_perm is not None:
            state = state.index_select(-1, self.state_perm)
        return torch.stack([enc(state[:, lo:hi]) for enc, (lo, hi) in zip(self.state_encoders, self.state_group_slices)], dim=1)

    def _build_inputs(self, vl_embs, action_features, s_tok, encoder_attention_mask):
        """Returns (hidden_states, encoder_hidden_states, encoder_attention_mask, n_lead)."""
        B = vl_embs.shape[0]
        if self.add_pos_embed:
            pos = self.position_embedding(torch.arange(self.num_grid_tokens, device=action_features.device))
            action_features = action_features + pos.unsqueeze(0).to(action_features.dtype)
        if self.add_group_embed:
            # grid tokens are laid out row-major (time row, group), as in _rope_update / _decode
            g = torch.arange(self.num_groups, device=action_features.device).repeat(self.num_time_rows)
            action_features = action_features + self.group_embedding(g).unsqueeze(0).to(action_features.dtype)
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

    def _token_cond(self, t_discretized: torch.Tensor, n_lead: int, device) -> Optional[torch.Tensor]:
        """timestep 임베딩에 그룹 임베딩을 concat -> (B, N, inner) per-token AdaLN 조건."""
        if not self.group_adaln_cond:
            return None
        g = torch.arange(self.num_groups, device=device).repeat(self.num_time_rows)
        if n_lead:
            g = torch.cat([torch.full((n_lead,), self.num_groups, device=device, dtype=g.dtype), g])
        ge = self.group_cond_embed(g)                                          # (N, gdim)
        if self.group_cond_mode == "split":
            temb = self.group_time_encoder(t_discretized)                      # (B, inner/2)
            B, N = temb.shape[0], g.shape[0]
            return torch.cat([temb[:, None, :].expand(B, N, -1), ge[None].expand(B, N, -1)], dim=-1)
        temb = self.model.timestep_encoder(t_discretized)                      # (B, inner)
        B, N = temb.shape[0], g.shape[0]
        cond = torch.cat([temb[:, None, :].expand(B, N, -1), ge[None].expand(B, N, -1)], dim=-1)
        return self.group_cond_proj(cond)                                      # (B, N, inner)

    def _decode(self, model_output: torch.Tensor) -> torch.Tensor:
        """(B, ..., D) trunk output -> (B, T, action_dim) velocity, in GROUP order."""
        B = model_output.shape[0]
        feats = model_output[:, -self.num_grid_tokens :].reshape(B, self.num_time_rows, self.num_groups, -1)
        outs = []
        for g, dec in enumerate(self.group_decoders):
            cell = dec(feats[:, :, g])  # (B, rows, time_patch * dg)
            outs.append(cell.reshape(B, self.pred_horizon, self.action_groups[g]))  # unpatch time
        return torch.cat(outs, dim=-1)

    # ---------------------------------------------------------------- forward
    def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None,
                progress_vector: Optional[torch.Tensor] = None):
        actions = self._to_group_order(actions)  # noise/velocity/pred all live in group order below
        full_actions = actions
        if self.keyframe_idx is not None:
            # Keyframe ActionMap: flow matching runs in keyframe space, so the loss below touches
            # ONLY the anchor steps; intermediate steps never enter the objective (unless
            # dense_recon_weight adds the interpolated-reconstruction term).
            actions = actions.index_select(1, self.keyframe_idx)
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
            temb_override=self._token_cond(t_discretized, n_lead, vl_embs.device),
            progress_vector=progress_vector,  # FFN pre-norm progress AdaLN (QwenGR00T_GridDiT_ProgressShift); None = off
        )
        pred = self._decode(model_output)
        loss = self.fm_loss_weight * ((pred - velocity) ** 2).mean()
        recon = None
        if self.keyframe_idx is not None and self.dense_recon_weight > 0:
            # clean keyframe estimate from the flow-matching prediction: x_t + (1 - t) * v_hat
            clean_kf = noisy_trajectory + (1 - t) * pred            # (B, K, D)
            dense = self._interpolate_keyframes(clean_kf)           # (B, H, D), same rule as inference
            recon = ((dense - full_actions) ** 2).mean()
            loss = loss + self.dense_recon_weight * recon
        with torch.no_grad():
            d = (pred - velocity).abs()
            self.last_wta_stats = {
                f"loss/{self.group_names[i]}_l1": float(d[..., lo:hi].mean())
                for i, (lo, hi) in enumerate(self.group_slices)
            }
            if recon is not None:
                self.last_wta_stats["loss/dense_recon"] = float(recon)
        return loss

    def _interpolate_keyframes(self, kf: torch.Tensor, start_anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, K, D) keyframe actions -> (B, H, D) by linear interpolation in normalized action space.

        Normalization (min_max / mean_std) is affine per dim, so interpolating normalized values equals
        interpolating raw values. `start_anchor` (B, D) or (D,) is the value BEFORE step 0 (e.g. the
        current joint position expressed in the same normalized action space; with rel actions this is
        the normalized zero displacement). Without an anchor, the ramp-in steps hold the first keyframe.
        """
        B, K, D = kf.shape
        out = kf.new_empty(B, self.action_horizon, D)
        prev_i = -1
        if start_anchor is None:
            prev_v = kf[:, 0]
        else:
            prev_v = start_anchor.to(kf.dtype).to(kf.device)
            if prev_v.dim() == 1:
                prev_v = prev_v[None].expand(B, D)
        for j, i in enumerate(self.keyframe_indices):
            cur = kf[:, j]
            span = i - prev_i
            steps = torch.arange(1, span + 1, device=kf.device, dtype=kf.dtype) / span  # (span,)
            out[:, prev_i + 1 : i + 1] = prev_v[:, None, :] + (cur - prev_v)[:, None, :] * steps[None, :, None]
            prev_i, prev_v = i, cur
        return out

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None, start_anchor: Optional[torch.Tensor] = None,
                       progress_vector: Optional[torch.Tensor] = None, **kwargs):
        B, device = vl_embs.shape[0], vl_embs.device
        actions = torch.randn((B, self.pred_horizon, self.action_dim), dtype=vl_embs.dtype, device=device)
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
                temb_override=self._token_cond(timesteps, n_lead, device),
                progress_vector=progress_vector,  # same vector at every flow step (observation-time quantity)
            )
            actions = actions + dt * self._decode(model_output)
        if self.keyframe_idx is not None:
            # anchor arrives in loader order; interpolation happens in group order
            if start_anchor is not None:
                start_anchor = self._to_group_order(torch.as_tensor(start_anchor))
            actions = self._interpolate_keyframes(actions, start_anchor)
        return self._from_group_order(actions)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_grid_dit_action_model(config=None):
    return GridFlowmatchingActionHead(config)
