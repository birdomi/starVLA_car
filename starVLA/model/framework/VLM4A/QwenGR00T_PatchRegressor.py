"""QwenGR00T with the patch-based regression head (see GR00T_PatchRegressorHeader.py).

Differences from QwenGR00T_Regressor:
  - the dataset window is past+future (action_indices = range(-P, H)); the head receives the whole window,
    uses the past P steps as patch-embedded context and regresses the future H steps
  - predict_action accepts example["past_action"]: (P, action_dim) NORMALIZED per-step deltas of the executed
    action history (the server builds it from raw actions, see PolicyServerWrapper)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.QwenGR00T_Regressor import Qwen_GR00T_Regressor, QwenGR00TRegressorDefaultConfig
from starVLA.model.modules.action_model.GR00T_PatchRegressorHeader import PatchRegressionActionHead
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenGR00TPatchRegressorDefaultConfig(QwenGR00TRegressorDefaultConfig):
    name: str = "QwenGR00T_PatchRegressor"
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "hidden_size": 1024,
            "action_dim": 29,
            "state_dim": 58,
            "action_horizon": 16,   # future steps (regression target)
            "num_past_steps": 16,   # past steps fed as patch tokens
            "patch_t": 4,           # time patch size
            "patch_groups": [7, 7, 6, 6, 3],  # joint groups (columns of the T x D grid)
            "past_repr": "from_start",  # past deltas re-anchored at the start of the past window
            "target_mode": "delta",     # loader action_mode of the targets: delta | rel
            "use_rope": True,
            "rope_theta": 10000.0,
            "loss_type": "l1",
            "diffusion_model_cfg": {
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
                "norm_elementwise_affine": True,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00T_PatchRegressor")
class Qwen_GR00T_PatchRegressor(Qwen_GR00T_Regressor):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        baseframework.__init__(self)
        self.config = merge_framework_config(QwenGR00TPatchRegressorDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        if self.config.framework.qwenvl.get("load_bf16", True):
            self.qwen_vl_interface.to(torch.bfloat16)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
        self.action_model = PatchRegressionActionHead(full_config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.num_past_steps = int(self.config.framework.action_model.num_past_steps)

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        prebuilt = None
        if isinstance(examples, dict) and "qwen_inputs" in examples:
            prebuilt, examples = examples["qwen_inputs"], examples["examples"]
            examples = getattr(examples, "items", examples)
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]  # (B, P+H, D)
        state = [example["state"] for example in examples] if "state" in examples[0] else None
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
            qwenvl_outputs = self.qwen_vl_interface(**qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
            last_hidden = qwenvl_outputs.hidden_states[-1]
        torch.cuda.synchronize()
        vlm_time = time.perf_counter() - t_vlm
        t_head = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            # past_source "state": the past grid comes from the state window, the action window is future-only
            need = self.action_horizon + (self.num_past_steps if getattr(self.action_model, "past_source", "action") == "action" else 0)
            assert actions.shape[1] >= need, (
                f"dataset returned {actions.shape[1]} action steps < past+future {need}; use a pf DataConfig "
                "(action_indices = range(-num_past_steps, action_horizon))"
            )
            state_t = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
            action_loss = self.action_model(last_hidden, actions, state_t, encoder_attention_mask=backbone_attention_mask)
        torch.cuda.synchronize()
        head_time = time.perf_counter() - t_head
        with torch.no_grad():
            was_training = self.action_model.training
            self.action_model.eval()
            past, target = self.action_model.split_window(actions)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_eval = self.action_model.predict_action(
                    last_hidden.detach(), state_t, encoder_attention_mask=backbone_attention_mask, past_actions=past
                )
            if was_training:
                self.action_model.train()
            diff = pred_eval.float() - target.float()
            mse_score = (torch.linalg.norm(diff) / pred_eval.numel()).item()
            action_rmse = diff.pow(2).mean().sqrt().item()
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

    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        if "past_action" in examples[0]:
            past = [example["past_action"] for example in examples]
        elif "action" in examples[0] and np.asarray(examples[0]["action"]).shape[0] >= self.num_past_steps + self.action_horizon:
            # training-time eval on loader samples: the window already contains the past steps
            past = [np.asarray(example["action"])[-self.num_past_steps - self.action_horizon : -self.action_horizon] for example in examples]
        else:
            past = None
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(**qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
            last_hidden = qwenvl_outputs.hidden_states[-1]
        dev, dt = last_hidden.device, last_hidden.dtype
        state_t = torch.from_numpy(np.array(state, dtype=np.float32)).to(dev, dtype=dt) if state is not None else None
        past_t = torch.from_numpy(np.array(past, dtype=np.float32)).to(dev, dtype=dt) if past is not None else None
        if past_t is not None and past_t.ndim == 2:
            past_t = past_t[None]
        with torch.autocast("cuda", dtype=torch.float32):
            pred = self.action_model.predict_action(last_hidden, state_t, encoder_attention_mask=backbone_attention_mask, past_actions=past_t)
        return {"normalized_actions": pred.detach().float().cpu().numpy()}
