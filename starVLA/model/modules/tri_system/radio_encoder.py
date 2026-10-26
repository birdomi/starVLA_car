"""C-RADIOv4 visual-state encoder: image -> full spatial patch tokens Z [B, N, D_c].

The summary / CLS / register tokens are dropped; only the N = (H/16)(W/16) patch tokens are kept.

Measured on the car data (examples/realRobots/H1_2_car_real/analyze_radio_outputs.py):
  * The backbone has no final LayerNorm. 1-6 patch tokens per image, always in the top row at ~85-88% of
    the width (independent of content, even on a gray image), come out with norm ~760 (others ~36),
    99.7% of it in channel 640. They are 0.1% of tokens but 60% of the variance, so they are masked
    (`valid` False) and zeroed after normalization. The penultimate block has none of them.
  * Raw tokens share a large mean (random-pair cosine 0.55), hence the per-channel normalization
    (PatchFeatureStats, computed by starVLA/training/trisystem_prepare_stats.py).
"""

import importlib
import importlib.util
import os
import sys
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_TEACHER_HEADS = ("siglip2-g", "dino_v3_7b", "sam3")


def load_radio_hf_dir(path: str, pkg: str = "cradio_hf"):
    """Load a RADIO HF repo dir (remote code) without AutoModel.

    transformers 5.5 AutoModel(trust_remote_code) copies only part of the nested relative imports
    (utils.py is missing) and fails, so the dir is registered as a package and RADIOModel is used directly.
    """
    path = os.path.abspath(path)
    if pkg not in sys.modules:
        mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(pkg, loader=None, is_package=True))
        mod.__path__ = [path]
        sys.modules[pkg] = mod
    hf = importlib.import_module(pkg + ".hf_model")
    return hf.RADIOModel.from_pretrained(path)


def attach_teacher_heads(model, pth: str, pkg: str = "cradio_hf"):
    """Per-teacher adaptor heads (SigLIP2-g / DINOv3-7B / SAM3) from the training checkpoint (the HF
    safetensors ships the backbone only)."""
    import argparse

    torch.serialization.add_safe_globals([argparse.Namespace])
    ck = torch.load(pth, map_location="cpu", weights_only=True)
    args, sd = ck["args"], ck["state_dict"]
    generic = sys.modules[pkg + ".adaptor_generic"].GenericAdaptor
    for idx, t in enumerate(args.teachers):
        st = {}
        for src, dst in [(f"_heads.{t['name']}.", "summary."), (f"_feature_projections.{t['name']}.", "feature.")]:
            st.update({dst + k[len(src):]: v.float() for k, v in sd.items() if k.startswith(src)})
        ad = generic(args, t, st)
        ad.head_idx = idx
        model.radio_model.adaptors[t["name"]] = ad


class RadioPatchEncoder(nn.Module):
    """Frozen by default (trainer.freeze_modules: ...,radio_encoder). Fine-tuning is possible, but the
    framework always encodes the future frame (the dZ target) with no gradient (stop-grad target)."""

    def __init__(self, path: str, feature: str = "backbone", input_size: Tuple[int, int] = (368, 640),
                 outlier_norm: float = 100.0, mask_outliers: bool = True, dtype: str = "bfloat16",
                 teacher_ckpt: Optional[str] = None):
        super().__init__()
        self.model = load_radio_hf_dir(path)
        self.feature = feature
        if feature in _TEACHER_HEADS:
            if not teacher_ckpt:
                raise ValueError(f"radio.feature={feature!r} needs radio.teacher_ckpt (the *_half.pth.tar)")
            attach_teacher_heads(self.model, teacher_ckpt)
        elif feature not in ("backbone", "penultimate"):
            raise ValueError(f"radio.feature must be backbone | penultimate | one of {_TEACHER_HEADS}, got {feature!r}")
        self.patch_size = int(self.model.patch_size)
        self.input_size = tuple(int(v) for v in input_size)
        if any(v % self.patch_size for v in self.input_size):
            raise ValueError(f"radio.input_size {self.input_size} must be a multiple of {self.patch_size}")
        self.outlier_norm, self.mask_outliers = float(outlier_norm), bool(mask_outliers)
        self.dtype = getattr(torch, dtype)
        self.num_prefix = int(self.model.radio_model.model.patch_generator.num_skip)
        # prefix tokens of the backbone output: num_cls CLS tokens (one per teacher), then the registers.
        # summary_idxs = the CLS tokens of the teachers trained with a summary loss (C-RADIOv4: SigLIP2-g and DINOv3-7B,
        # RADIO's 2304-d summary; the SAM3 CLS has no summary loss)
        self.num_cls = int(self.model.radio_model.model.patch_generator.num_cls_tokens)
        self.num_registers = self.num_prefix - self.num_cls
        sidx = getattr(self.model.radio_model, "summary_idxs", None)
        self.summary_idxs = list(range(self.num_cls)) if sidx is None else [int(i) for i in sidx.tolist()]
        if feature in ("backbone", "penultimate"):
            self.feat_dim = int(self.model.radio_model.model.embed_dim)
        else:
            head = self.model.radio_model.adaptors[feature].feat_mlp  # MLP2 or AttnFDHead(blocks + MLP2)
            self.feat_dim = int([m for m in head.modules() if isinstance(m, nn.Linear)][-1].out_features)
        self.model.eval()

    def prefix_indices(self, mode: str = "all") -> List[int]:
        """Positions of the prefix tokens to return: "summary" (RADIO's summary CLS tokens), "cls" (every CLS token) or
        "all" (every CLS + register token)."""
        if mode == "summary":
            return list(self.summary_idxs)
        if mode == "cls":
            return list(range(self.num_cls))
        if mode == "all":
            return list(range(self.num_prefix))
        raise ValueError(f"radio prefix_tokens must be summary | cls | all, got {mode!r}")

    def prefix_names(self, idxs: Sequence[int]) -> List[str]:
        args = getattr(getattr(self.model, "config", None), "args", None)
        teachers = args.get("teachers", []) if isinstance(args, dict) else []
        cls = lambda i: f"cls:{teachers[i]['name']}" if i < len(teachers) else f"cls{i}"
        return [cls(i) if i < self.num_cls else f"reg{i - self.num_cls}" for i in idxs]

    @property
    def grid(self) -> Tuple[int, int]:
        return self.input_size[0] // self.patch_size, self.input_size[1] // self.patch_size

    def train(self, mode: bool = True):
        # stays in eval mode while frozen (no dropout / drop-path either way, but keeps intent explicit)
        trainable = any(p.requires_grad for p in self.parameters())
        return super().train(mode and trainable)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """images [B, 3, H, W] in [0, 1] -> (Z [B, N, D_c] float32 raw features, valid [B, N] bool)."""
        z, valid, _ = self.forward_tokens(images, prefix_idxs=None)
        return z, valid

    def forward_tokens(self, images: torch.Tensor, prefix_idxs: Optional[Sequence[int]] = None):
        """-> (Z [B, N, D_c], valid [B, N], prefix [B, P, D_c] float32 | None): prefix = the backbone's CLS / register
        tokens at prefix_idxs (see prefix_indices), always from the last block (also when `feature` selects another
        layer / a teacher head for Z)."""
        if tuple(images.shape[-2:]) != self.input_size:
            images = F.interpolate(images.float(), self.input_size, mode="bilinear", antialias=True, align_corners=False)
        rm = self.model.radio_model
        captured = {}
        hook = None
        if self.feature == "penultimate":
            hook = rm.model.blocks[-2].register_forward_hook(lambda m, i, o: captured.__setitem__("x", o))
        try:
            with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
                xc = rm.input_conditioner(images)
                y = rm.model.forward_features(xc)
                patches = y[:, self.num_prefix:]
                if self.feature == "backbone":
                    z = patches
                elif self.feature == "penultimate":
                    z = captured["x"][:, self.num_prefix:]
                else:
                    z = rm._extract_final(xc, y)[self.feature].features
        finally:
            if hook is not None:
                hook.remove()
        if self.mask_outliers:
            valid = patches.detach().float().norm(dim=-1) <= self.outlier_norm
        else:
            valid = torch.ones(patches.shape[:2], dtype=torch.bool, device=patches.device)
        prefix = None if prefix_idxs is None else y[:, list(prefix_idxs)].float()
        return z.float(), valid, prefix


class PatchFeatureStats(nn.Module):
    """Per-channel normalization of C-RADIO tokens and the scale of the change target.

        z_n  = (Z - z_mean) / z_std                     (normalized visual state)
        dZ   = z_n(future) - z_n(current)               (the change; mean cancels)
        X_1  = dZ / dz_rms                              (flow-matching data, unit RMS per channel)

    The buffers are persistent, so a checkpoint carries them; `path` (from the prepare-stats script) is only
    needed for the first training stage.
    """

    def __init__(self, dim: int, path: Optional[str] = None):
        super().__init__()
        self.register_buffer("z_mean", torch.zeros(dim))
        self.register_buffer("z_std", torch.ones(dim))
        self.register_buffer("dz_rms", torch.ones(dim))
        self.register_buffer("loaded", torch.zeros((), dtype=torch.bool))
        if path:
            if os.path.exists(path):
                st = torch.load(path, map_location="cpu", weights_only=True)
                for k in ("z_mean", "z_std", "dz_rms"):
                    if st[k].shape != (dim,):
                        raise ValueError(f"{path}: {k} has shape {tuple(st[k].shape)}, expected ({dim},)")
                    getattr(self, k).copy_(st[k].float())
                self.loaded.fill_(True)
            else:
                print(f"[PatchFeatureStats] WARNING: stats file {path} not found; identity normalization "
                      "(fine when loading a checkpoint, which carries the buffers)")

    def normalize(self, z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        z = (z - self.z_mean) / self.z_std
        return z.masked_fill(~valid[..., None], 0.0)

    def denormalize(self, z_n: torch.Tensor) -> torch.Tensor:
        return z_n * self.z_std + self.z_mean
