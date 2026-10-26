"""Shared C-RADIOv4 backbone of the Dual-System VLA (frozen by default; radio.trainable fine-tunes it).

One instance serves both systems (no duplicated weights). It wraps the C-RADIOv4 patch encoder of
starVLA/model/modules/tri_system/radio_encoder.py (HF-dir loader that works under transformers 5.5, top-row artifact
token masking) and adds the Dual-System contract:
  * every parameter has requires_grad = False and the module stays in eval mode, whatever the trainer does;
  * the forward always runs under torch.no_grad() (no activation memory, no gradient path into the backbone);
  * trainable = True (patch embedder + position embedding + all blocks) or n (the last n blocks) instead leaves those
    parameters trainable and runs the forward with gradient (when grad mode is on); still eval mode (C-RADIOv4 has no
    dropout / drop-path); the trainer must not list radio_encoder in trainer.freeze_modules;
  * output = all spatial patch tokens + the PREFIX tokens (raw features), plus a patch validity mask.
    prefix_tokens = "summary" (default): the CLS tokens of the teachers with use_summary (C-RADIOv4: SigLIP2-g and
    DINOv3-7B) = RADIO's 2304-d summary as 2 x 1152 tokens (bit-identical); "cls" = all 3 CLS (+ SAM3, which has no
    summary loss and is a near-constant sink); "all" = 3 CLS + 7 registers.
    Measured (RoboCasa / car frames): the SAM3 CLS and 6 of the 7 registers are nearly image-independent "sink" tokens
    (norm ~550-1150, cross-image cos ~1.0); SigLIP2 / DINOv3 CLS and register 3 vary with the image.
The per-system trainable adapters (radio_projector_s2 / radio_prefix_projector_s2 and the _s1 pair) live in the
systems, not here.

    images [M, 3, H, W] in [0, 1] -> VisualFeatures(tokens [M, N, D_c] float32, valid [M, N] bool, grid (gh, gw),
                                                    prefix [M, P, D_c] float32)
    N = (input_size / 16)^2, P = 2; C-RADIOv4-SO400M: D_c = 1152, 256 x 256 input -> 16 x 16 = 256 patch tokens.

C-RADIOv4 is released only as H, SO400M and 1D-H (no small / base v4 model), so SO400M is the smallest C-RADIOv4
(the path is configurable).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from starVLA.model.modules.tri_system.radio_encoder import RadioPatchEncoder


@dataclass
class VisualFeatures:
    tokens: torch.Tensor  # [M, N, D_c] raw C-RADIO patch features (invalid positions zeroed)
    valid: torch.Tensor  # [M, N] bool, False at C-RADIO's fixed high-norm artifact positions
    grid: Tuple[int, int]
    prefix: Optional[torch.Tensor] = None  # [M, P, D_c] raw C-RADIO CLS / register tokens

    def index(self, idx) -> "VisualFeatures":
        return VisualFeatures(self.tokens[idx], self.valid[idx], self.grid,
                              None if self.prefix is None else self.prefix[idx])


class SharedRadioBackbone(nn.Module):
    def __init__(self, path: str, input_size: Tuple[int, int] = (256, 256), feature: str = "backbone",
                 outlier_norm: float = 100.0, mask_outliers: bool = True, dtype: str = "bfloat16",
                 teacher_ckpt: Optional[str] = None, prefix_tokens: str = "summary", trainable=False):
        super().__init__()
        self.encoder = RadioPatchEncoder(path, feature=feature, input_size=tuple(input_size), outlier_norm=outlier_norm,
                                         mask_outliers=mask_outliers, dtype=dtype, teacher_ckpt=teacher_ckpt)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        vit = self.encoder.model.radio_model.model
        if trainable is True:
            train = [vit.patch_generator, *vit.blocks]
        elif trainable is False or trainable is None:
            train = []
        else:
            n = int(trainable)
            if not 0 < n <= len(vit.blocks) or feature != "backbone":
                raise ValueError(f"radio.trainable must be true / false / 1..{len(vit.blocks)} (last blocks, feature backbone), "
                                 f"got {trainable!r} (feature {feature!r})")
            train = list(vit.blocks[-n:])
        for mod in train:
            mod.requires_grad_(True)
        self.trainable = bool(train)
        self.trainable_desc = ("frozen" if not train else "trainable: all" if trainable is True
                               else f"trainable: last {len(train)} of {len(vit.blocks)} blocks")
        self.feat_dim = self.encoder.feat_dim
        self.grid = self.encoder.grid
        self.input_size = self.encoder.input_size
        self.prefix_idxs = self.encoder.prefix_indices(prefix_tokens)
        self.prefix_names = self.encoder.prefix_names(self.prefix_idxs)
        self.num_prefix = len(self.prefix_idxs)  # P
        self.prefix_dim = int(self.encoder.model.radio_model.model.embed_dim)  # backbone width (== feat_dim for backbone)

    @property
    def num_tokens(self) -> int:
        return self.grid[0] * self.grid[1]

    def train(self, mode: bool = True):
        return super().train(False)  # eval mode in every training stage (also when trainable)

    def forward(self, images: torch.Tensor) -> VisualFeatures:
        if not self.trainable and any(p.requires_grad for p in self.encoder.parameters()):
            raise RuntimeError("the shared C-RADIOv4 backbone must stay frozen (requires_grad was re-enabled)")
        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()):
            z, valid, prefix = self.encoder.forward_tokens(images, prefix_idxs=self.prefix_idxs)
        return VisualFeatures(z.masked_fill(~valid[..., None], 0.0), valid, self.grid, prefix)
