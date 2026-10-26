"""DINOv3 ConvNeXt patch backbone with the SharedRadioBackbone contract (a drop-in for System 1's C-RADIO eye).

DINOv3 ConvNeXt (facebook/dinov3-convnext-{tiny,small,base,large}-pretrain-lvd1689m, distilled from DINOv3 ViT-7B on
LVD-1689M) loaded with transformers' DINOv3ConvNextModel and CUT after stage `stage` (1..4): the later stages and the
final LayerNorm / pooling are dropped from the module (no unused parameters under DDP). The output is the raw feature
map of that stage, one token per position; there are no CLS / register tokens (num_prefix = 0) and no artifact tokens
(valid is all True).

    images [M, 3, H, W] in [0, 1] -> ImageNet mean / std -> stages 1..stage
      -> VisualFeatures(tokens [M, N, C] float32, valid [M, N] bool (all True), grid (gh, gw), prefix None)
    stride 4 / 8 / 16 / 32 at stage 1 / 2 / 3 / 4; ConvNeXt-Small widths 96 / 192 / 384 / 768, depths 3 / 3 / 27 / 3.
    ConvNeXt-Small stage 3, 256 x 256 input -> 16 x 16 = 256 tokens x 384 (stages 1-3: ~33M parameters).

Same rules as SharedRadioBackbone: eval mode always (ConvNeXt has LayerNorm only, so eval only disables drop-path);
trainable = True (all kept stages) / False (frozen, forward under no_grad); compute in bf16 autocast (dtype).
"""

from typing import Tuple

import torch
import torch.nn as nn

from starVLA.model.modules.dual_system.backbone import VisualFeatures

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoConvNextBackbone(nn.Module):
    def __init__(self, path: str, input_size: Tuple[int, int] = (256, 256), stage: int = 3, dtype: str = "bfloat16",
                 trainable=True):
        super().__init__()
        from transformers import DINOv3ConvNextModel

        full = DINOv3ConvNextModel.from_pretrained(path, torch_dtype=torch.float32)
        cfg = full.config
        n_stages = len(cfg.hidden_sizes)
        if not 1 <= int(stage) <= n_stages:
            raise ValueError(f"dinov3_convnext stage must be 1..{n_stages}, got {stage!r}")
        if trainable not in (True, False, None):
            raise ValueError(f"dinov3_convnext trainable must be true / false, got {trainable!r}")
        self.stage = int(stage)
        self.stages = nn.ModuleList(list(full.model.stages[: self.stage]))
        del full
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.dtype = getattr(torch, dtype)

        self.requires_grad_(bool(trainable))
        self.eval()
        self.trainable = bool(trainable)
        self.trainable_desc = (f"DINOv3 ConvNeXt stages 1-{self.stage} "
                               + ("trainable: all" if self.trainable else "frozen"))
        stride = 4 * 2 ** (self.stage - 1)
        self.input_size = tuple(int(v) for v in input_size)
        if self.input_size[0] % stride or self.input_size[1] % stride:
            raise ValueError(f"input_size {self.input_size} must be divisible by the stage-{self.stage} stride {stride}")
        self.grid = (self.input_size[0] // stride, self.input_size[1] // stride)
        self.feat_dim = int(cfg.hidden_sizes[self.stage - 1])
        self.prefix_dim = self.feat_dim
        self.prefix_idxs, self.prefix_names = [], []
        self.num_prefix = 0

    @property
    def num_tokens(self) -> int:
        return self.grid[0] * self.grid[1]

    def train(self, mode: bool = True):
        return super().train(False)  # eval mode in every training stage (also when trainable)

    def forward(self, images: torch.Tensor) -> VisualFeatures:
        if not self.trainable and any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("the frozen DINOv3 ConvNeXt backbone had requires_grad re-enabled")
        if tuple(images.shape[-2:]) != self.input_size:
            raise ValueError(f"DINOv3 ConvNeXt: input {tuple(images.shape[-2:])}, expected {self.input_size}")
        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()), \
                torch.autocast("cuda", dtype=self.dtype, enabled=images.is_cuda and self.dtype != torch.float32):
            x = (images.float() - self.mean) / self.std
            for st in self.stages:
                x = st(x)
        tok = x.float().flatten(2).transpose(1, 2)  # [M, gh * gw, C]
        valid = torch.ones(tok.shape[:2], dtype=torch.bool, device=tok.device)
        return VisualFeatures(tok, valid, self.grid, None)
