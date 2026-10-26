"""Frozen MoGe-2 (official microsoft/MoGe code, vendored in third_party/) -> its shared GEOMETRY NECK features.

    from moge.model.v2 import MoGeModel; MoGeModel.from_pretrained(<model.pt or "Ruicheng/moge-2-vitb-normal">)

Only the neck output is used (user 10-22: "Geometry Neck의 출력 feature를 사용하자. ViT feature나 다른 거 빼고"): the
encoder and neck run exactly as the first half of MoGeModel.forward (v2.py: encoder at the base token grid, UV planes
concatenated at all 5 levels, self.neck); the point / normal / mask / scale heads are never run. MoGe-2 ViT-B neck
outputs (crop 172 x 256, num_tokens 1200 -> base grid 28 x 42):
    level 0 [768, 28, 42] (no res blocks: ~the projected ViT feature, NOT used by default)
    level 1 [256, 56, 84] | level 2 [128, 112, 168] | level 3 [64, 224, 336] | level 4 [32, 448, 672]
The chosen levels are area-pooled to the base grid (level l: 2^l x 2^l patches) and concatenated
-> [M, sum(C_l), gh, gw] (default levels 1-4: 480 channels).

The RoboCasa GR1 ego_view frames are letterboxed (content rows 42..213 of 256, measured on 5 tasks 10-22), so the frame
is cropped to `crop` (top, bottom, left, right) before MoGe. Always eval / no_grad (frozen).
"""

import os
import sys
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_VENDOR = [os.path.join(_REPO, "third_party", "MoGe"), os.path.join(_REPO, "third_party", "utils3d-moge")]


def _import_moge_v2():
    try:
        from moge.model.v2 import MoGeModel
    except ImportError:
        for p in _VENDOR:
            if p not in sys.path:
                sys.path.append(p)
        from moge.model.v2 import MoGeModel
    from moge.utils.geometry_torch import normalized_view_plane_uv
    return MoGeModel, normalized_view_plane_uv


class MoGe2NeckFeatures(nn.Module):
    def __init__(self, path: str, num_tokens: int = 1200, crop: Sequence[int] = (42, 214, 0, 256),
                 levels: Sequence[int] = (1, 2, 3, 4), dtype: str = "bfloat16"):
        super().__init__()
        MoGeModel, self._uv = _import_moge_v2()
        self.model = MoGeModel.from_pretrained(path)
        self.model.requires_grad_(False)
        self.model.eval()
        self.num_tokens = int(num_tokens)
        self.crop = tuple(int(v) for v in crop)
        self.levels = tuple(int(v) for v in levels)
        dims = list(self.model.neck.dim_res_blocks) if hasattr(self.model.neck, "dim_res_blocks") else None
        if dims is None:  # ConvStack keeps no attribute: read the checkpoint config
            dims = list(torch.load(path, map_location="cpu", weights_only=True)["model_config"]["neck"]["dim_res_blocks"])
        if not self.levels or any(not 0 <= l < len(dims) for l in self.levels):
            raise ValueError(f"moge neck levels must be in 0..{len(dims) - 1}, got {self.levels}")
        self.out_dim = int(sum(dims[l] for l in self.levels))
        h, w = self.crop[1] - self.crop[0], self.crop[3] - self.crop[2]
        ar = w / h
        self.grid = (round((self.num_tokens / ar) ** 0.5), round((self.num_tokens * ar) ** 0.5))
        self.dtype = getattr(torch, dtype)

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images [M, 3, H, W] in [0, 1] -> neck features [M, out_dim, gh, gw] float32."""
        t, b, l, r = self.crop
        x = images[..., t:b, l:r].float()
        m, _, img_h, img_w = x.shape
        base_h, base_w = self.grid
        ar = img_w / img_h
        with torch.autocast("cuda", dtype=self.dtype, enabled=x.is_cuda and self.dtype != torch.float32):
            # == MoGeModel.forward (v2.py) up to the shared neck ==
            features, _cls = self.model.encoder(x, base_h, base_w, return_class_token=True)
            features = [features, None, None, None, None]
            for level in range(5):
                uv = self._uv(width=base_w * 2 ** level, height=base_h * 2 ** level, aspect_ratio=ar,
                              dtype=x.dtype, device=x.device)
                uv = uv.permute(2, 0, 1).unsqueeze(0).expand(m, -1, -1, -1)
                features[level] = uv if features[level] is None else torch.concat([features[level], uv.to(features[level].dtype)], dim=1)
            neck = self.model.neck(features)
        out = [F.avg_pool2d(neck[lv].float(), 2 ** lv) if lv > 0 else neck[lv].float() for lv in self.levels]
        return torch.cat(out, dim=1)
