"""Explicit tensor-shape bookkeeping for the three-system pipeline.

Named dimensions used throughout (bound on first use per forward, then enforced):

    B    batch                        H, W   frame size (pixels)
    T    frames per example (t + future offsets)
    N    C-RADIO patch tokens = (H / 16) * (W / 16)          D_c  C-RADIO feature dim (1152)
    L_s  System 2 tokens (vision + text, padded)             D_s  Qwen hidden dim (1024)
    H_a  action chunk length (action_horizon)                D_a  action dim, D_st state dim
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


class ShapeChecker:
    """`check("Z_t", z, "B N D_c")` binds new names and asserts already-bound ones; integer literals are exact."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.dims: Dict[str, int] = {}

    def reset(self, **known: int) -> "ShapeChecker":
        self.dims = dict(known)
        return self

    def __call__(self, name: str, t: torch.Tensor, spec: str) -> torch.Tensor:
        if not self.enabled:
            return t
        names = spec.split()
        if t.dim() != len(names):
            raise ValueError(f"{name}: expected [{spec}], got shape {tuple(t.shape)}")
        for dim_name, size in zip(names, t.shape):
            if dim_name.isdigit():
                ok = int(dim_name) == size
            else:
                ok = self.dims.setdefault(dim_name, size) == size
            if not ok:
                raise ValueError(f"{name}: dim {dim_name}={size} in {tuple(t.shape)} != bound {self.dims.get(dim_name, dim_name)} ([{spec}])")
        return t


@dataclass
class System2Tokens:
    """System 2 output: the Qwen last hidden state of EVERY token (vision, text, chat template)."""

    tokens: torch.Tensor  # [B, L_s, D_s]
    mask: torch.Tensor  # [B, L_s] bool, True = real token (left padding is False)


@dataclass
class PatchState:
    """C-RADIOv4 spatial patch tokens of one frame, normalized per channel (summary/register tokens dropped)."""

    z: torch.Tensor  # [B, N, D_c]
    valid: torch.Tensor  # [B, N] bool; False at C-RADIO's fixed high-norm artifact positions
    grid: Tuple[int, int]  # (H / 16, W / 16)


@dataclass
class TriSystemOutput:
    """predict_future(): every intermediate of image -> (S_t, Z_t) -> dZ_hat -> action chunk."""

    s2: System2Tokens
    z_t: PatchState
    delta_z: torch.Tensor  # [B, N, D_c] predicted change, normalized-feature units (Z_f_hat = z_t.z + delta_z)
    z_future: torch.Tensor  # [B, N, D_c]
    actions: Optional[torch.Tensor] = None  # [B, H_a, D_a] normalized action chunk
