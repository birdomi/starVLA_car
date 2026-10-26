"""CRADIOVisualEncoder: frozen C-RADIOv4 (SO400M or H, chosen by `radio.path`) -> spatial patch tokens Z [B, N, D_c].

Reuses the repo's C-RADIO loader (tri_system.radio_encoder.RadioPatchEncoder):
  * official preprocessing: RGB in [0, 1] -> the model's own input_conditioner (CLIP mean / std), its own
    position handling (CPE), any multiple of the 16-px patch;
  * every image (current and future) is resized to the same `input_size` with the same bilinear/antialias
    resampling, so N = (H / 16) * (W / 16) tokens in row-major (2D) order;
  * summary / CLS / register tokens are dropped (no global summary feature);
  * C-RADIO's fixed high-norm artifact tokens (top row, norm ~760) are flagged invalid (`valid` False) and zeroed.
Optional per-channel feature normalization (PatchFeatureStats, `normalize`), and an optional on-disk cache of the
raw frozen features keyed by (dataset, camera, episode, absolute frame) from the loader's `frame_keys`.
"""

import warnings
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from starVLA.model.modules.tri_system.radio_encoder import PatchFeatureStats, RadioPatchEncoder
from starVLA.model.modules.tri_system.shapes import PatchState

from .cache import TensorCache


class CRADIOVisualEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, normalize: bool = True, stats_path: Optional[str] = None,
                 cache_dir: Optional[str] = None, cache_meta: Optional[dict] = None):
        super().__init__()
        self.backbone = backbone
        self.stats = PatchFeatureStats(self.feat_dim, stats_path)
        self.normalize = bool(normalize)
        self.cache = TensorCache(cache_dir, cache_meta or {}) if cache_dir else None
        self._warned = False

    @classmethod
    def from_config(cls, rc, backbone: Optional[nn.Module] = None) -> "CRADIOVisualEncoder":
        """rc = framework.radio. `backbone` injects a stand-in encoder (unit tests)."""
        input_size = tuple(int(v) for v in rc.get("input_size", (256, 256)))
        if backbone is None:
            backbone = RadioPatchEncoder(
                rc.path, feature=rc.get("feature", "backbone"), input_size=input_size,
                outlier_norm=float(rc.get("outlier_norm", 100.0)), mask_outliers=bool(rc.get("mask_outliers", True)),
                dtype=rc.get("dtype", "bfloat16"), teacher_ckpt=rc.get("teacher_ckpt", None),
            )
        meta = {"path": str(rc.get("path", "")), "feature": str(rc.get("feature", "backbone")),
                "input_size": list(input_size), "dtype": str(rc.get("dtype", "bfloat16")), "kind": "radio_raw_features"}
        return cls(backbone, normalize=bool(rc.get("normalize_features", True)), stats_path=rc.get("stats_path", None),
                   cache_dir=rc.get("cache_dir", None), cache_meta=meta)

    # ------------------------------------------------------------------ properties
    @property
    def feat_dim(self) -> int:
        return int(self.backbone.feat_dim)

    @property
    def grid(self) -> Tuple[int, int]:
        return tuple(self.backbone.grid)

    @property
    def num_tokens(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def input_size(self) -> Tuple[int, int]:
        return tuple(self.backbone.input_size)

    @property
    def trainable(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())

    # ------------------------------------------------------------------ encoding
    def encode_raw(self, images: torch.Tensor, keys: Optional[Sequence[Optional[str]]] = None):
        """images [M, 3, H, W] in [0, 1] -> (raw tokens [M, N, D_c] float32, valid [M, N] bool).

        With a cache and keys, cached frames are loaded and only the missing ones are encoded (frozen encoder only)."""
        use_cache = self.cache is not None and keys is not None and all(k is not None for k in keys)
        if use_cache and self.trainable:
            if not self._warned:
                warnings.warn("radio.cache_dir is ignored while C-RADIO is trainable")
                self._warned = True
            use_cache = False
        if not use_cache:
            with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()):
                return self.backbone(images)
        got = [self.cache.get(k) for k in keys]
        miss = [i for i, g in enumerate(got) if g is None]
        if miss:
            with torch.no_grad():
                z_m, v_m = self.backbone(images[miss])
            for j, i in enumerate(miss):
                got[i] = {"z": z_m[j].to(torch.bfloat16), "valid": v_m[j]}
                self.cache.put(keys[i], got[i])
        z = torch.stack([g["z"] for g in got]).to(images.device, torch.float32)
        valid = torch.stack([g["valid"] for g in got]).to(images.device)
        return z, valid

    def latent(self, z_raw: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Raw tokens -> model latent (per-channel normalized if `normalize`), invalid tokens zeroed."""
        z = self.stats.normalize(z_raw, valid) if self.normalize else z_raw
        return z.masked_fill(~valid[..., None], 0.0)

    def forward(self, images: torch.Tensor, keys: Optional[List[Optional[str]]] = None) -> PatchState:
        """images [M, 3, H, W] in [0, 1] -> PatchState(z [M, N, D_c], valid [M, N], grid)."""
        z_raw, valid = self.encode_raw(images, keys)
        return PatchState(self.latent(z_raw, valid), valid, self.grid)
