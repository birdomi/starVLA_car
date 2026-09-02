# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Qwen-GR00T-Context Framework

QwenGR00T variant with learnable context-embedding tokens:
  1. N learnable "context embeddings" are appended to the VLM input sequence
     (injected at placeholder-token positions via an embedding forward hook,
     same mechanism as QwenAdapter's action queries).
  2. Only the VLM output hidden states at those N context positions are fed to
     the DiT flow-matching action head — the image/text token outputs are used
     solely inside the VLM to contextualize the context embeddings.

Works with any Qwen interface exposed by `get_vlm_model` (Qwen2.5-VL /
Qwen3-VL / Qwen3.5); select the backbone via `framework.qwenvl.base_vlm`.
"""

import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenGR00T_Context
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenGR00TContextDefaultConfig:
    """QwenGR00T_Context framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.
    """

    # --- Registry identifier ---
    name: str = "QwenGR00T_Context"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL / Qwen3.5) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3.5-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (auto-aligned at runtime)
            "vl_hidden_dim": 2048,
        }
    )

    # === Context embedding (learnable tokens appended to VLM input) ===
    context_embedding: dict = field(
        default_factory=lambda: {
            # Number of learnable context tokens; only their VLM outputs
            # are passed to the DiT action head.
            "num_context_tokens": 32,
            # Placeholder string reserving one sequence position per context
            # token. MUST encode to exactly one token id in the backbone's
            # tokenizer ("<|fim_pad|>" works for Qwen3.5 / Qwen3-VL; the
            # emoji "🔍" used by QwenAdapter is single-token only on Qwen3-VL).
            "placeholder_token": "<|fim_pad|>",
        }
    )

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 8,
            "repeated_diffusion_steps": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
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


@FRAMEWORK_REGISTRY.register("QwenGR00T_Context")
class Qwen_GR00T_Context(baseframework):
    """
    Multimodal vision-language-action model (GR00T variant, context-embedding conditioned).

    Components:
      - Qwen-VL backbone (Qwen2.5 / Qwen3 / Qwen3.5) for fused language/vision encoding
      - N learnable context-embedding tokens appended to the VLM input sequence
      - Flow-matching (DiT) head conditioned ONLY on the context tokens' outputs
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenGR00TContextDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        vl_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        # align cross-attention dim of DiT to VLM hidden size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = vl_hidden_dim

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # === Learnable context embeddings ===
        self.num_context_tokens = int(self.config.framework.context_embedding.get("num_context_tokens", 32))
        self.context_embedding = nn.Parameter(torch.empty(self.num_context_tokens, vl_hidden_dim))
        nn.init.normal_(self.context_embedding, mean=0.0, std=0.02)

        # Placeholder token used to reserve sequence positions for the context
        # embeddings; its input embedding is replaced by `context_embedding`
        # via a forward hook before the transformer runs.
        self.context_placeholder = str(self.config.framework.context_embedding.get("placeholder_token", "<|fim_pad|>"))
        placeholder_ids = self.qwen_vl_interface.processor.tokenizer(
            self.context_placeholder, add_special_tokens=False
        )["input_ids"]
        assert len(placeholder_ids) == 1, (
            f"Context placeholder {self.context_placeholder!r} must map to exactly one token, "
            f"got {placeholder_ids}. Pick a different single-token placeholder for this tokenizer."
        )
        self.context_placeholder_id = placeholder_ids[0]
        self.context_prompt = self.context_placeholder * self.num_context_tokens

    # ──────────────────────────────────────────────────────────────
    #  Shared encoding: VLM forward with context injection
    # ──────────────────────────────────────────────────────────────
    def _encode_context(self, batch_images, instructions) -> torch.Tensor:
        """Run the VLM with appended context placeholders and return ONLY the
        hidden states at the context-token positions.

        Returns:
            context_hidden: [B, num_context_tokens, H]
        """
        # Reserve positions for the context embeddings at the end of the prompt
        instructions = [instruction + self.context_prompt for instruction in instructions]

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        input_ids = qwen_inputs["input_ids"]  # [B, L]
        batch_size, device = input_ids.shape[0], input_ids.device

        # Locate the context placeholder positions per sample
        context_mask = input_ids == self.context_placeholder_id  # [B, L]
        counts = context_mask.sum(dim=1)
        if not bool((counts == self.num_context_tokens).all()):
            raise RuntimeError(
                f"Expected {self.num_context_tokens} context placeholder tokens per sample, "
                f"got counts={counts.tolist()}. Instructions must not contain "
                f"{self.context_placeholder!r} themselves."
            )
        # [B, N] positions (nonzero returns row-major order → per-sample ascending)
        context_positions = context_mask.nonzero(as_tuple=False)[:, 1].view(batch_size, self.num_context_tokens)
        batch_indices = (
            torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.num_context_tokens)
        )  # [B, N]

        def inject_context_hook(module, inputs, output):
            """Replace placeholder embeddings with the learnable context embeddings."""
            context_embed = self.context_embedding.to(dtype=output.dtype, device=output.device)  # [N, H]
            output = output.clone()
            output[batch_indices, context_positions, :] = context_embed.unsqueeze(0)
            return output

        embedding_layer = self.qwen_vl_interface.model.get_input_embeddings()
        hook_handle = embedding_layer.register_forward_hook(inject_context_hook)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            hook_handle.remove()

        last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]
        # Keep ONLY the context-token outputs as the condition for the DiT head
        context_hidden = last_hidden[batch_indices, context_positions, :]  # [B, N, H]
        return context_hidden

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]  #  [B, [PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B, len, 7]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Step 1: VLM forward → context-token hidden states only
        context_hidden = self._encode_context(batch_images, instructions)  # [B, N, H]

        # Step 2: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=context_hidden.device, dtype=context_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

            repeated_diffusion_steps = self.config.framework.action_model.get("repeated_diffusion_steps", 4)
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            context_hidden_repeated = context_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=context_hidden.device, dtype=context_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            # All N context tokens are valid → no encoder attention mask needed
            action_loss = self.action_model(
                context_hidden_repeated, actions_target_repeated, state_repeated
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Returns:
            dict:
                normalized_actions (np.ndarray): [B, T, action_dim] diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B, [PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: VLM forward → context-token hidden states only
        context_hidden = self._encode_context(batch_images, instructions)  # [B, N, H]

        state = (
            torch.from_numpy(np.array(state)).to(context_hidden.device, dtype=context_hidden.dtype)
            if state is not None
            else None
        )

        # Step 2: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(context_hidden, state)  # (B, chunk_len, action_dim)

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
        default="examples/simBenchmarks/LIBERO/train_files/starvla_qwen3_5_context_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model: Qwen_GR00T_Context = Qwen_GR00T_Context(cfg)
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
    print(f"Normalized Action: {normalized_actions}")

    print("Finished")
