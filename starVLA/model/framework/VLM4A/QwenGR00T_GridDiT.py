# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""

import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import time

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_GridDiTHeader import (
    GridFlowmatchingActionHead,
    get_grid_dit_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenGR00T
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenGR00TGridDiTDefaultConfig:
    """QwenGR00T framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier ---
    name: str = "QwenGR00T_GridDiT"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (used for cross-attention alignment)
            "vl_hidden_dim": 2048,
        }
    )

    # # === DINO encoder (optional multi-view spatial tokens) === Dino is not used in this QwenGR00T version, we can add it later when we want to use it
    # dino: dict = field(default_factory=lambda: {
    #     # DINO backbone variant: "dinov2_vits14" | "dinov2_vitb14" | ...
    #     "dino_backbone": "dinov2_vits14",
    # })

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            # DiT model size: "DiT-B" | "DiT-L" | "DiT-XL"
            "action_model_type": "DiT-B",
            # Hidden dim for action model (auto-aligned at runtime)
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            # Whether to add positional embeddings in the action head
            "add_pos_embed": True,
            "max_seq_len": 1024,
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 8,
            # Repeat factor for flow-matching loss (more noise samples per batch)
            "repeated_diffusion_steps": 8,
            # Beta distribution params for noise schedule
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # Inference denoising steps
            "num_inference_timesteps": 4,
            # Number of vision tokens fed to action head
            "num_target_vision_tokens": 32,
            # === action-grid knobs (see GridFlowmatchingActionHead) ===
            "action_groups": [],     # per-group action dims, e.g. [7, 7, 6, 6, 3]; REQUIRED unless action_group_indices
            "state_groups": [],      # per-group state dims; auto = action_groups (or 2x for sin/cos state)
            # Explicit (possibly non-contiguous) dim index sets per group, e.g. ["0:7,14:20", "7:14,20:26", "26:29"]
            # for a left / right / waist grid. Overrides action_groups; needs state_group_indices for sin/cos state.
            "action_group_indices": [],
            "state_group_indices": [],
            "group_names": [],       # optional per-group names, used in the per-group L1 logs
            # How many consecutive timesteps one grid ROW covers: 1 = a token per (timestep, group),
            # 2 / 4 = coarser rows, action_horizon = one row per group (time-axis layout).
            # Token count = (action_horizon / time_patch_size) * num_groups.
            "time_patch_size": 1,
            # Keyframe ActionMap: predict only temporal anchors, interpolate the rest at inference.
            # keyframe_stride s -> anchors [s-1, 2s-1, ..., H-1]; keyframe_indices overrides explicitly.
            # 0 / [] (default) = Dense ActionMap (every timestep). Requires time_patch_size 1.
            "keyframe_stride": 0,
            "keyframe_indices": [],
            # Auxiliary dense-reconstruction loss for keyframe heads: MSE between the trajectory
            # interpolated from the CLEAN keyframe estimate and the full ground truth, added with
            # this weight (0 = off). Requires keyframes.
            "dense_recon_weight": 0.0,
            # Flow-matching loss weight; 0 + dense_recon_weight>0 = recon-only training (keyframes).
            "fm_loss_weight": 1.0,
            # Groups that share one encoder/decoder/state-encoder, e.g. [[0, 1]] to tie left and
            # right. Shared groups must have equal action and state dims.
            "share_groups": [],
            "state_as_kv": True,     # group state tokens appended to the cross-attention KV
            # True = whole 58-D state -> ONE token via a single MLP (vanilla-DiT style, no group split).
            # Combine with state_as_kv: false to also match the vanilla placement (leading sequence token).
            "state_single": False,
            "use_rope": True,        # axial (time, group) RoPE in the DiT self-attention
            "rope_axes": "time_group",  # "time_group" (2-axis) | "time" (1-axis over time only)
            # Group identity via AdaLN: concat a learned group embedding onto the diffusion-time
            # conditioning (per-token). Pair with add_pos_embed: false + rope_axes: time.
            "group_adaln_cond": False,
            "group_cond_dim": 64,
            "group_cond_mode": "proj",  # "proj" (768+64->proj768) | "split" (384||384 직결, proj 없음)
            "rope_theta": 10000.0,
            # === DiT Transformer sub-config ===
            "diffusion_model_cfg": {
                # Cross-attention dim (aligned to VLM hidden_size at runtime)
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    # # === Training precision flag === This is unnecessary, unused parameter
    # reduce_in_full_precision: bool = True


@FRAMEWORK_REGISTRY.register("QwenGR00T_GridDiT")
class Qwen_GR00T_GridDiT(baseframework):
    """
    Multimodal vision-language-action model (GR00T variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - ACTION-GRID flow-matching DiT head: the action chunk is a (time x joint-group) grid,
        each cell has its own input projection and decoder, with axial (time, group) RoPE

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenGR00TGridDiTDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: GridFlowmatchingActionHead = get_grid_dit_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """Training forward.

        `examples` is a list of example dicts, or — with
        datasets.vla_data.preprocess_in_workers — a dict
        {"qwen_inputs": <processor output>, "examples": ExampleList} built by
        QwenVLPreprocessCollate (the dataset already resized the frames).
        Returns the flow-matching loss plus per-step timing and a free eval
        metric (4-step sampling with the head in eval mode on the same VLM
        features), logged by the trainer like the regressor framework.
        """
        prebuilt = None
        if isinstance(examples, dict) and "qwen_inputs" in examples:
            prebuilt, examples = examples["qwen_inputs"], examples["examples"]
            examples = getattr(examples, "items", examples)  # ExampleList holder -> list

        batch_images = [example["image"] for example in examples]  #  [B, [PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B, len, action_dim]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Step 1: QWenVL input format
        t_pre = time.perf_counter()
        if prebuilt is not None:
            device = self.qwen_vl_interface.model.device
            qwen_inputs = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in prebuilt.items()}
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        preprocess_time = time.perf_counter() - t_pre
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)

        t_vlm = time.perf_counter()
        with torch.autocast("cuda", dtype=getattr(self.qwen_vl_interface, "autocast_dtype", torch.bfloat16)):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]
        torch.cuda.synchronize()
        vlm_time = time.perf_counter() - t_vlm

        # Step 4: Action Expert Forward and Loss
        t_head = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            mask_repeated = None
            if backbone_attention_mask is not None:
                mask_repeated = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(dtype=torch.bool)

            state_t = state_repeated = None
            if state is not None:
                state_t = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state_t.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated,
                encoder_attention_mask=mask_repeated,
            )  # (B, chunk_len, action_dim)
        torch.cuda.synchronize()
        head_time = time.perf_counter() - t_head

        # Free per-step eval metric on the SAME VLM features: sample the head
        # in eval mode and score like TrainerUtils.eval_action_model (L2 / N).
        with torch.no_grad():
            was_training = self.action_model.training
            self.action_model.eval()
            with torch.autocast("cuda", dtype=torch.float32):
                mask = backbone_attention_mask.to(dtype=torch.bool) if backbone_attention_mask is not None else None
                pred_eval = self.action_model.predict_action(last_hidden.detach(), state_t, encoder_attention_mask=mask)
            if was_training:
                self.action_model.train()
            diff = pred_eval.float() - actions_target.float()
            mse_score = (torch.linalg.norm(diff) / pred_eval.numel()).item()  # legacy: L2/N, scales with 1/sqrt(batch)
            action_rmse = diff.pow(2).mean().sqrt().item()  # batch-size invariant

        return {
            "action_loss": action_loss,
            "timing/preprocess": preprocess_time,
            "timing/vlm_fwd": vlm_time,
            "timing/head_fwd": head_time,
            "mse_score": mse_score,
            "action_rmse": action_rmse,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B, [PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        with torch.autocast("cuda", dtype=getattr(self.qwen_vl_interface, "autocast_dtype", torch.bfloat16)):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden, state, encoder_attention_mask=backbone_attention_mask
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model: Qwen_GR00T_GridDiT = Qwen_GR00T_GridDiT(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
