# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Qwen-GR00T-Regressor Framework
Qwen-VL + GR00T-style DiT regression head (no flow matching):
no noised-action input, no timestep embedding, single-pass action regression
with optional winner-takes-all multiple hypotheses (see GR00T_RegressorHeader).
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List, Optional, Tuple

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.modules.action_model.GR00T_TimeAxisRegressorHeader import (
    TimeAxisRegressionActionHead,
    get_time_axis_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@dataclass
class QwenGR00TTimeAxisRegressorDefaultConfig:
    """QwenGR00T_TimeAxisRegressor framework default parameters.

    Same VLM/DiT shape as QwenGR00T, minus every flow-matching knob
    (noise schedule, timestep buckets, inference steps, repeat factor),
    plus the WTA multiple-hypothesis fields of the regression head.
    """

    name: str = "QwenGR00T_TimeAxisRegressor"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "hidden_size": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 8,
            "num_target_vision_tokens": 32,
            # Time-axis knobs (see TimeAxisRegressionActionHead)
            "loss_type": "l1",
            "action_groups": [],     # per-group action dims, e.g. [7, 7, 6, 6, 3]; REQUIRED
            "state_groups": [],      # per-group state dims; auto = action_groups (or 2x for sin/cos state)
            "decoder_mode": "per_dim",  # per_dim | per_group | shared trajectory decoder
            "state_as_kv": True,     # group state tokens concatenated to the cross-attention KV
            "use_rope": True,        # axial (joint group, index in group) RoPE in the DiT self-attention
            "rope_theta": 10000.0,
            "diffusion_model_cfg": {
                "cross_attention_dim": 2048,  # aligned to VLM hidden_size at runtime
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
                # learnable LayerNorm scale/shift (GR00T's DiT default is False because
                # AdaLN supplied them; the regressor has no AdaLN)
                "norm_elementwise_affine": True,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00T_TimeAxisRegressor")
class Qwen_GR00T_TimeAxisRegressor(Qwen_GR00T):
    """QwenGR00T with the TIME-AXIS regression head: one query token per ACTION DIMENSION (29 for GR1),
    each decoded into that dimension's whole 16-step trajectory; axial (joint group, index) RoPE and
    per-group state tokens concatenated to the cross-attention KV.

    Inherits predict_action from Qwen_GR00T (the regression head exposes the
    same predict_action signature). forward drops the repeated_diffusion_steps
    batch expansion — regression is deterministic, repeating inputs is waste.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        baseframework.__init__(self)
        self.config = merge_framework_config(QwenGR00TTimeAxisRegressorDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # Frozen-VLM fast path: keep VLM weights in bf16 storage so autocast
        # doesn't re-cast ~850M fp32 params on every forward. If you ever
        # unfreeze the VLM in this framework, disable this (bf16 params have
        # no fp32 master and small optimizer updates would be lost).
        if self.config.framework.qwenvl.get("load_bf16", True):
            self.qwen_vl_interface.to(torch.bfloat16)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: TimeAxisRegressionActionHead = get_time_axis_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        # Batch may arrive prebuilt from QwenVLPreprocessCollate (processor ran in
        # the dataloader workers): {"qwen_inputs": ..., "examples": [...]}.
        prebuilt = None
        if isinstance(examples, dict) and "qwen_inputs" in examples:
            prebuilt, examples = examples["qwen_inputs"], examples["examples"]
            examples = getattr(examples, "items", examples)  # ExampleList holder -> list

        batch_images = [example["image"] for example in examples]  # [B, [PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B, len, action_dim]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        t_pre = time.perf_counter()
        if prebuilt is not None:
            device = self.qwen_vl_interface.model.device
            qwen_inputs = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in prebuilt.items()}
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        preprocess_time = time.perf_counter() - t_pre
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        t_vlm = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]
        torch.cuda.synchronize()
        vlm_time = time.perf_counter() - t_vlm

        t_head = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)
            assert actions_target.shape[1] == self.action_horizon, (
                f"dataset returned {actions.shape[1]} action steps < action_horizon {self.action_horizon}; "
                "check the data registry action_indices"
            )

            state_t = None
            if state is not None:
                state_t = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)

            action_loss = self.action_model(
                last_hidden, actions_target, state_t,
                encoder_attention_mask=backbone_attention_mask,
            )

        torch.cuda.synchronize()
        head_time = time.perf_counter() - t_head

        # Free per-step eval metric: re-run only the head in eval mode (dropout
        # off, score-head selection) on the SAME VLM features, and score it
        # exactly like TrainerUtils.eval_action_model (L2 norm / num points).
        with torch.no_grad():
            was_training = self.action_model.training
            self.action_model.eval()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_eval = self.action_model.predict_action(
                    last_hidden.detach(), state_t, encoder_attention_mask=backbone_attention_mask
                )
            if was_training:
                self.action_model.train()
            tgt_eval = self.action_model.expand_targets(actions_target) if hasattr(self.action_model, "expand_targets") else actions_target
            diff = pred_eval.float() - tgt_eval.float()
            mse_score = (torch.linalg.norm(diff) / pred_eval.numel()).item()  # legacy: L2/N, scales with 1/sqrt(batch)
            action_rmse = diff.pow(2).mean().sqrt().item()  # batch-size invariant

        out = {
            "action_loss": action_loss,
            "timing/preprocess": preprocess_time,
            "timing/vlm_fwd": vlm_time,
            "timing/head_fwd": head_time,
            "mse_score": mse_score,
            "action_rmse": action_rmse,
        }
        out.update(getattr(self.action_model, "last_wta_stats", {}))
        return out


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/realRobots/EgoDex/train_files/starvla_qwengr00t_regressor_egodex.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    model: Qwen_GR00T_Regressor = Qwen_GR00T_Regressor(cfg)
    print(model)

    action_dim = int(cfg.framework.action_model.action_dim)
    state_dim = int(cfg.framework.action_model.state_dim)
    horizon = int(cfg.framework.action_model.action_horizon)
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(horizon, action_dim)).astype(np.float32),
        "state": np.random.uniform(-1, 1, size=(1, state_dim)).astype(np.float32),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = dict(sample)
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    print(f"Action Loss: {forward_output['action_loss'].item()}")

    predict_output = model.predict_action(examples=[sample])
    print(f"Normalized Action shape: {predict_output['normalized_actions'].shape}")

    print("Finished")
