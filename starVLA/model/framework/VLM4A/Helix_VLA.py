"""
Helix_VLA: the basic System 2 / System 1 VLA (Helix-style): a slow VLM on a STALE frame + a fast visuomotor flow DiT on
the CURRENT frame, connected only by the VLM's latent tokens. One DiT (System 1); no plan, no AdaLN, no memory.

System 2 (slow, semantic): Qwen3.5-0.8B on frame t - s2_lag (helix.s2_lag env steps, default 12 = 0.6 s at 20 Hz = one
  executed 12-step chunk of the sim eval: at deployment System 2 works on the previous call's frame while System 1
  acts on the current one) + instruction -> the LM's last hidden states of all prompt tokens [B, L, H]
  -> LayerNorm with a learned affine (helix.s2_norm, as GR00T's vlln) = the System 2 latent tokens.
System 1 (fast, visuomotor): its own eye C-RADIOv4-SO400M on frame t (radio.trainable: fine-tuned or frozen) -> patch
  tokens [N, D_c] (artifact token masked) + summary prefix -> LN + Linear to H + learned position embeddings
  (RadioCondition of QwenGR00T_RadioAdaLN); state token at t; GR00T flow-matching DiT, input = the noisy action chunk
  a[t .. t + action_horizon - 1], cross-attention condition = [System 2 latent tokens ; System 1 C-RADIO tokens].
Controls: helix.s2_lag 0 = synchronous (both systems on frame t); helix.s1_vision false = System 2 only (no C-RADIO).
System 1 eye swap: radio.backbone "dinov3_convnext" (default "c_radio") = DINOv3 ConvNeXt cut after radio.stage (default
  3: ConvNeXt-Small 256 x 256 -> 16 x 16 tokens x 384, no prefix tokens), radio.path its HF dir; same module names
  (radio_encoder / radio_cond), so the lr / freeze keys of the yaml are unchanged (DinoConvNextBackbone).
  radio.backbone "moge" = MoGe-2 / MoGe-3 ENCODER only (DINOv2 ViT/14 fine-tuned for geometry, its summed 1x1-projected
  intermediate layers + the CLS token as 1 prefix), radio.path its model.pt, radio.grid tokens (default [16, 16]);
  MoGe-2 ViT-B 256 x 256 -> 16 x 16 tokens x 768 + 1 CLS (MoGeEncoderBackbone).

Frames: every example's "image" holds, per camera (camera-major), the frames of the video delta_indices oldest first:
  s2_lag > 0 -> [frame t - s2_lag, frame t] (yaml datasets.vla_data.delta_indices_override {video.ego_view: [-12, 0]};
  sim eval client ROBOCASA_VLM_PREV_FRAME=1 sends [previous call's frame, frame t], frame t twice at an episode start;
  the loader clamps t - s2_lag to the episode's first frame), s2_lag 0 -> [frame t].
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenGR00T_RadioAdaLN import RadioCondition, _example_images, _images_to_tensor
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.dual_system.backbone import SharedRadioBackbone
from starVLA.model.modules.dual_system.dinov3_convnext import DinoConvNextBackbone
from starVLA.model.modules.dual_system.moge_encoder import MoGeEncoderBackbone
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@dataclass
class HelixVLADefaultConfig(QwenGR00TDefaultConfig):
    name: str = "Helix_VLA"
    # System 1's eye (same keys as QwenGR00T_DualSystem framework.radio)
    radio: dict = field(default_factory=lambda: {
        "path": "playground/Pretrained_models/C-RADIOv4-SO400M",
        "feature": "backbone",
        "input_size": [256, 256],
        "outlier_norm": 100.0,
        "mask_outliers": True,
        "dtype": "bfloat16",
        "prefix_tokens": "summary",  # SigLIP2-g + DINOv3 CLS (RADIO's summary)
        "trainable": True,           # false = frozen; n = last n blocks (drop radio_encoder from trainer.freeze_modules)
    })
    helix: dict = field(default_factory=lambda: {
        "s2_lag": 12,        # env steps from System 2's frame to System 1's frame (== -video delta_indices[0]); 0 = sync
        "s2_norm": True,     # LayerNorm (learned affine) on the System 2 latent tokens
        "s1_vision": True,   # System 1's own C-RADIO tokens of frame t in the condition (false = System 2 only)
        "use_prefix": True,  # also the C-RADIO summary tokens (else patches only)
    })


@FRAMEWORK_REGISTRY.register("Helix_VLA")
class Helix_VLA(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(HelixVLADefaultConfig, config)
        fw = self.config.framework
        hc = fw.helix
        self.s2_lag = int(hc.get("s2_lag", 12))
        if self.s2_lag < 0:
            raise ValueError(f"helix.s2_lag must be >= 0, got {self.s2_lag}")
        self.num_views = int(fw.get("num_views", 1))
        self.frames_per_view = 2 if self.s2_lag > 0 else 1

        self.qwen_vl_interface = get_vlm_model(config=self.config)
        h = int(self.qwen_vl_interface.model.config.hidden_size)
        fw.action_model.diffusion_model_cfg.cross_attention_dim = h
        self.s2_norm = nn.LayerNorm(h) if bool(hc.get("s2_norm", True)) else None

        self.radio_encoder, self.radio_cond = None, None
        if bool(hc.get("s1_vision", True)):
            rc = fw.radio
            kind = str(rc.get("backbone", "c_radio"))
            if kind == "c_radio":
                self.radio_encoder = SharedRadioBackbone(
                    rc.path, input_size=tuple(rc.get("input_size", [256, 256])), feature=rc.get("feature", "backbone"),
                    outlier_norm=float(rc.get("outlier_norm", 100.0)), mask_outliers=bool(rc.get("mask_outliers", True)),
                    dtype=rc.get("dtype", "bfloat16"), teacher_ckpt=rc.get("teacher_ckpt", None),
                    prefix_tokens=str(rc.get("prefix_tokens", "summary")), trainable=rc.get("trainable", True),
                )
            elif kind == "dinov3_convnext":
                self.radio_encoder = DinoConvNextBackbone(
                    rc.path, input_size=tuple(rc.get("input_size", [256, 256])), stage=int(rc.get("stage", 3)),
                    dtype=rc.get("dtype", "bfloat16"), trainable=rc.get("trainable", True),
                )
            elif kind == "moge":
                self.radio_encoder = MoGeEncoderBackbone(
                    rc.path, input_size=tuple(rc.get("input_size", [256, 256])), grid=tuple(rc.get("grid", [16, 16])),
                    dtype=rc.get("dtype", "bfloat16"), trainable=rc.get("trainable", True),
                )
            else:
                raise ValueError(f"radio.backbone must be c_radio / dinov3_convnext / moge, got {kind!r}")
            if self.radio_encoder.trainable:
                tr = self.config.get("trainer", None)
                frozen = str(tr.get("freeze_modules", "") or "") if tr is not None and hasattr(tr, "get") else ""
                if "radio_encoder" in [p.strip() for p in frozen.split(",")]:
                    raise ValueError("radio.trainable is set but trainer.freeze_modules lists radio_encoder")
            self.radio_size = tuple(int(v) for v in self.radio_encoder.input_size)
            n_p = int(self.radio_encoder.num_prefix) if bool(hc.get("use_prefix", True)) else 0
            self.radio_cond = RadioCondition(int(self.radio_encoder.feat_dim), int(self.radio_encoder.prefix_dim), h,
                                             self.radio_encoder.num_tokens, n_p, self.num_views)

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(fw.action_model.action_horizon)
        logger.info(f"[Helix_VLA] System 2 Qwen (hidden {h}) on frame t - {self.s2_lag}, latent LayerNorm "
                    f"{self.s2_norm is not None}; System 1 flow DiT on frame t with "
                    + ("no own vision (System 2 only)" if self.radio_encoder is None else
                       f"{type(self.radio_encoder).__name__} ({self.radio_encoder.trainable_desc}, "
                       f"{self.radio_encoder.num_tokens} patches + "
                       f"{self.radio_cond.n_prefix} prefix)"))

    @property
    def _device(self):
        return next(self.action_model.parameters()).device

    # ------------------------------------------------------------------ frames -> System 2 / System 1 images
    def _split_frames(self, images_per_example: List[list]):
        """Per example: camera-major [frame t - s2_lag, frame t] per camera -> (System 2 images, System 1 images)."""
        s2, s1 = [], []
        v, k = self.num_views, self.frames_per_view
        for ims in images_per_example:
            if len(ims) != v * k:
                raise ValueError(f"Helix_VLA (s2_lag {self.s2_lag}): {len(ims)} images per example, expected {v} camera(s) "
                                 f"x {k} frame(s) ([frame t - lag, frame t] per camera; sim eval: ROBOCASA_VLM_PREV_FRAME=1)")
            s2.append(list(ims[0::k]))
            s1.append(list(ims[k - 1::k]))
        return s2, s1

    # ------------------------------------------------------------------ the two systems
    def _system2(self, images: List[list], instructions: List[str]):
        """Qwen on the stale frame -> latent tokens [B, L, H] + mask [B, L] bool."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
        with torch.autocast("cuda", dtype=getattr(self.qwen_vl_interface, "autocast_dtype", torch.bfloat16)):
            out = self.qwen_vl_interface(**qwen_inputs, output_attentions=False, output_hidden_states=True,
                                         return_dict=True)
            z = out.hidden_states[-1]
            if self.s2_norm is not None:
                z = self.s2_norm(z).to(z.dtype)
        mask = qwen_inputs.get("attention_mask", None)
        mask = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device) if mask is None else mask.to(z.device).bool()
        return z, mask

    def _condition(self, images_per_example: List[list], instructions: List[str]):
        """[System 2 latent tokens ; System 1 C-RADIO tokens of frame t] + mask, and timings."""
        s2_images, s1_images = self._split_frames(images_per_example)
        t0 = time.perf_counter()
        z, mask = self._system2(s2_images, instructions)
        t1 = time.perf_counter()
        if self.radio_encoder is not None:
            flat = [im for ims in s1_images for im in ims]
            f = self.radio_encoder(_images_to_tensor(flat, self.radio_size, self._device))
            r_tok, r_mask = self.radio_cond(f, len(images_per_example))
            z = torch.cat([z, r_tok.to(z.dtype)], 1)
            mask = torch.cat([mask, r_mask], 1)
        t2 = time.perf_counter()
        return z, mask, {"timing/vlm_fwd": t1 - t0, "timing/radio_fwd": t2 - t1}

    def _progress_of(self, examples) -> Optional[np.ndarray]:
        if getattr(getattr(self.action_model, "model", None), "progress_encoder", None) is None:
            return None
        return np.asarray([example["progress"] for example in examples], dtype=np.float32)

    # ------------------------------------------------------------------ training / inference
    def forward(self, examples: List[dict] = None, **kwargs):
        images = [_example_images(ex) for ex in examples]
        cond, mask, timing = self._condition(images, [ex["lang"] for ex in examples])
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None
        progress = self._progress_of(examples)

        t_head = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array([ex["action"] for ex in examples]), device=cond.device, dtype=cond.dtype)
            target = actions[:, -self.action_horizon:, :]
            r = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            state_t = None if state is None else torch.tensor(np.array(state), device=cond.device, dtype=cond.dtype)
            progress_t = None if progress is None else torch.as_tensor(progress, device=cond.device)
            loss = self.action_model(
                cond.repeat(r, 1, 1), target.repeat(r, 1, 1), None if state_t is None else state_t.repeat(r, 1, 1),
                encoder_attention_mask=mask.repeat(r, 1), progress=None if progress_t is None else progress_t.repeat(r),
            )
        head_time = time.perf_counter() - t_head

        with torch.no_grad():  # free per-step metric: num_inference_timesteps sampling on the same features
            was_training = self.action_model.training
            self.action_model.eval()
            with torch.autocast("cuda", dtype=torch.float32):
                pred = self.action_model.predict_action(cond.detach(), state_t, encoder_attention_mask=mask, progress=progress_t)
            if was_training:
                self.action_model.train()
            diff = pred.float() - target.float()
        return {"action_loss": loss, **timing, "timing/head_fwd": head_time,
                "mse_score": (torch.linalg.norm(diff) / pred.numel()).item(),
                "action_rmse": diff.pow(2).mean().sqrt().item()}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        examples = examples if isinstance(examples, list) else [examples]
        images = [_example_images(ex) for ex in examples]
        size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if size:  # obs_image_size [H, W] -> PIL (W, H)
            images = resize_images(images, target_size=(int(size[1]), int(size[0])))
        cond, mask, _ = self._condition(images, [ex["lang"] for ex in examples])
        state = None
        if "state" in examples[0]:
            state = torch.from_numpy(np.array([ex["state"] for ex in examples])).to(cond.device, dtype=cond.dtype)
        progress = self._progress_of(examples)
        progress = None if progress is None else torch.as_tensor(progress, device=cond.device)
        with torch.autocast("cuda", dtype=torch.float32):
            pred = self.action_model.predict_action(cond, state, encoder_attention_mask=mask, progress=progress)
        return {"normalized_actions": pred.detach().cpu().numpy()}
