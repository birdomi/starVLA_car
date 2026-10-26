"""MoGe encoder (DINOv2 ViT/14 fine-tuned for monocular geometry) with the SharedRadioBackbone contract (a drop-in for
System 1's C-RADIO eye).

MoGe-2 / MoGe-3 (github microsoft/MoGe, moge/model/v2.py / v3.py; HF Ruicheng/moge-2-vitb-normal, Ruicheng/moge-3-vitl,
...: one model.pt = {model_config, model}) are a DINOv2 encoder + conv neck + point / normal / mask / scale heads (+ a
sparse 3D refiner in v3). Only the ENCODER is kept: the DINOv2 backbone (built with timm, weights loaded strict) and its
1x1 output projections, i.e. exactly the feature map MoGe's geometry neck consumes; neck, heads and refiner are not
built (no moge / utils3d dependency).

    images [M, 3, H, W] in [0, 1] -> bilinear (antialias) resize to (gh * 14, gw * 14) -> ImageNet mean / std
      -> DINOv2 layers intermediate_layers (MoGe-2 ViT-B: 5 / 11; MoGe-3 ViT-L: 5 / 11 / 17 / 23), each final-LayerNorm'ed
         -> 1x1 conv -> SUM
      -> VisualFeatures(tokens [M, gh * gw, dim_out] float32, valid [M, N] bool (all True), grid (gh, gw),
                        prefix [M, 1, D] = CLS token of the last intermediate layer (normed), as MoGe's scale head uses)
    grid 16 x 16 (224 x 224 inside): 256 tokens; MoGe-2 ViT-B x 768 (86.6M backbone + 1.2M projections),
    MoGe-3 ViT-L x 1024 (304M + 4.2M).

The position embedding (pretrained on a 37 x 37 grid) is resampled ONCE to the fixed grid the way DINOv2's
interpolate_pos_encoding does (bicubic, scale factor (g + 0.1) / 37, no antialias), so the encoder matches MoGe's own
code (CPU parity 10-22, helix_vla/moge_parity_check.py). MoGe itself runs at 1200-3600 tokens.

Same rules as SharedRadioBackbone: eval mode always (no dropout / drop-path in DINOv2 inference anyway);
trainable = True (backbone + projections) / False (frozen, forward under no_grad); compute in bf16 autocast (dtype).
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.dual_system.backbone import VisualFeatures

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 14
# MoGe encoder.backbone name -> timm model (plain-MLP DINOv2 without registers; ViT-g uses SwiGLU keys, not supported)
TIMM_NAMES = {"dinov2_vits14": "vit_small_patch14_dinov2", "dinov2_vitb14": "vit_base_patch14_dinov2",
              "dinov2_vitl14": "vit_large_patch14_dinov2"}


class MoGeEncoderBackbone(nn.Module):
    def __init__(self, path: str, input_size: Tuple[int, int] = (256, 256), grid: Tuple[int, int] = (16, 16),
                 dtype: str = "bfloat16", trainable=False):
        super().__init__()
        import timm

        if trainable not in (True, False, None):
            raise ValueError(f"moge trainable must be true / false, got {trainable!r}")
        ck = torch.load(path, map_location="cpu", weights_only=True)
        ecfg = ck["model_config"]["encoder"]
        if ecfg["backbone"] not in TIMM_NAMES:
            raise ValueError(f"moge encoder backbone {ecfg['backbone']!r} not supported (only {sorted(TIMM_NAMES)})")
        if isinstance(ecfg["intermediate_layers"], int):
            raise ValueError("moge encoder intermediate_layers must be a list of layer indices")
        self.layers = [int(v) for v in ecfg["intermediate_layers"]]
        sd = {k[len("encoder."):]: v for k, v in ck["model"].items() if k.startswith("encoder.")}
        del ck
        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        self.grid = tuple(int(v) for v in grid)
        self.input_size = tuple(int(v) for v in input_size)

        self.backbone = timm.create_model(TIMM_NAMES[ecfg["backbone"]], pretrained=False, num_classes=0, init_values=1.0,
                                          img_size=(self.grid[0] * PATCH, self.grid[1] * PATCH))
        pe = bb["pos_embed"].float()  # [1, 1 + M * M, D]
        m = int(math.sqrt(pe.shape[1] - 1))
        if m * m != pe.shape[1] - 1:
            raise ValueError(f"moge pos_embed {tuple(pe.shape)} is not a square grid")
        patch_pe = F.interpolate(pe[:, 1:].reshape(1, m, m, -1).permute(0, 3, 1, 2), mode="bicubic", antialias=False,
                                 scale_factor=((self.grid[0] + 0.1) / m, (self.grid[1] + 0.1) / m))
        if tuple(patch_pe.shape[-2:]) != self.grid:
            raise ValueError(f"moge pos_embed resample gave {tuple(patch_pe.shape[-2:])}, expected {self.grid}")
        bb = {k: v for k, v in bb.items() if k != "mask_token"}
        bb["pos_embed"] = torch.cat([pe[:, :1], patch_pe.permute(0, 2, 3, 1).flatten(1, 2)], dim=1)
        self.backbone.load_state_dict(bb, strict=True)
        width = int(self.backbone.embed_dim)
        self.output_projections = nn.ModuleList([nn.Conv2d(width, int(ecfg["dim_out"]), 1) for _ in self.layers])
        self.output_projections.load_state_dict(
            {k[len("output_projections."):]: v for k, v in sd.items() if k.startswith("output_projections.")}, strict=True)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.dtype = getattr(torch, dtype)

        self.requires_grad_(bool(trainable))
        self.eval()
        self.trainable = bool(trainable)
        self.trainable_desc = (f"MoGe {ecfg['backbone']} encoder (layers {self.layers}, grid {self.grid}) "
                               + ("trainable: all" if self.trainable else "frozen"))
        self.feat_dim = int(ecfg["dim_out"])
        self.prefix_dim = width
        self.prefix_idxs, self.prefix_names = [0], ["cls"]
        self.num_prefix = 1

    @property
    def num_tokens(self) -> int:
        return self.grid[0] * self.grid[1]

    def train(self, mode: bool = True):
        return super().train(False)  # eval mode in every training stage (also when trainable)

    def forward(self, images: torch.Tensor) -> VisualFeatures:
        if not self.trainable and any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("the frozen MoGe encoder had requires_grad re-enabled")
        if tuple(images.shape[-2:]) != self.input_size:
            raise ValueError(f"MoGe encoder: input {tuple(images.shape[-2:])}, expected {self.input_size}")
        gh, gw = self.grid
        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()), \
                torch.autocast("cuda", dtype=self.dtype, enabled=images.is_cuda and self.dtype != torch.float32):
            x = F.interpolate(images.float(), (gh * PATCH, gw * PATCH), mode="bilinear", align_corners=False, antialias=True)
            x = (x - self.mean) / self.std
            inter = self.backbone.forward_intermediates(x, indices=self.layers, norm=True, output_fmt="NCHW",
                                                        intermediates_only=True, return_prefix_tokens=True)
            z = sum(proj(f) for proj, (f, _) in zip(self.output_projections, inter))  # [M, dim_out, gh, gw]
        tok = z.float().flatten(2).transpose(1, 2)  # [M, gh * gw, dim_out]
        prefix = inter[-1][1][:, :1].float()  # [M, 1, D] CLS of the last intermediate layer
        valid = torch.ones(tok.shape[:2], dtype=torch.bool, device=tok.device)
        return VisualFeatures(tok, valid, self.grid, prefix)
