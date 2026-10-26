# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").
"""
Molmo2 VL Interface for starVLA.

Mirrors `_QWen3_VL_Interface` (./QWen3.py) and `_Gemma4_VL_Interface` (./Gemma4.py)
so framework files calling `qwen_vl_interface.{forward, generate, build_qwenvl_inputs}`
work unchanged when the dispatcher in `__init__.py` returns this class.

Backed by the AllenAI Molmo2 collection:
    https://huggingface.co/collections/allenai/molmo2

    | Model               | Text backbone           | Vision backbone        |
    |---------------------|-------------------------|------------------------|
    | allenai/Molmo2-4B   | Qwen3-4B-Instruct-2507  | SigLIP 2 so400m/14-384 |
    | allenai/Molmo2-8B   | Qwen3-8B                | SigLIP 2 so400m/14-384 |
    | allenai/Molmo2-O-7B | OLMo-2-7B               | SigLIP 2 so400m/14-384 |
    | allenai/Molmo2-VideoPoint-4B | Qwen3-4B-Instruct-2507 | SigLIP 2        |

All checkpoints ship custom modeling code, so loading goes through
`AutoModelForImageTextToText` with `trust_remote_code=True`.

Notable feature for VLA: Molmo2 natively emits 2D pointing tokens in the form
`<points coords="x:y, ..."/>` (and `<tracks .../>` for the VideoPoint variant),
with coordinates normalized to [0, 1000]. This is useful as an auxiliary grounding
signal when training action heads on top of the last hidden state.
"""

import contextlib
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.processing_utils import ProcessorMixin

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100


# --- transformers>=5 compat for the Molmo2 remote code (written against 4.57) ---
def _default_rope_init(config, device=None, **_):
    # transformers 5 dropped ROPE_INIT_FUNCTIONS["default"]; Molmo2RotaryEmbedding still looks it up.
    base = getattr(config, "rope_theta", None) or config.rope_parameters["rope_theta"]
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return inv_freq, 1.0


ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope_init)


@contextlib.contextmanager
def _lenient_processor_init():
    """transformers 5 ProcessorMixin rejects non-attribute kwargs, which Molmo2Processor passes
    (image_use_col_tokens, ...). Set them as plain attributes after init, as MolmoAct2's processor does."""
    orig = ProcessorMixin.__init__

    def _init(self, *args, **kwargs):
        keep = set(self.get_attributes()) | {"chat_template", "audio_tokenizer"}
        extra = {k: kwargs.pop(k) for k in list(kwargs) if k not in keep}
        orig(self, *args, **kwargs)
        for k, v in extra.items():
            setattr(self, k, v)

    ProcessorMixin.__init__ = _init
    try:
        yield
    finally:
        ProcessorMixin.__init__ = orig


class _Molmo2_VL_Interface(nn.Module):
    """
    Lightweight wrapper around `allenai/Molmo2-*` checkpoints.

    Purpose:
        - Match the interface of `_QWen3_VL_Interface` exactly so framework files
          can be VLM-agnostic.
        - Centralize Molmo2-specific preprocessing (chat template + multimodal packing).
        - Surface `model.config.hidden_size = text_config.hidden_size` so downstream
          DiT / flow-matching heads can read it the same way they read it for Qwen3-VL.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "allenai/Molmo2-4B")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        enable_grad_ckpt = bool(qwenvl_config.get("enable_gradient_checkpointing", False))

        # Fall back to sdpa if flash_attention_2 is requested but flash_attn isn't installed.
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[Molmo2][WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            # same knob as QWen3_5.py: float32 when the VLM is trained without an fp32 master copy (ZeRO-0 / Muon)
            dtype=getattr(torch, str(qwenvl_config.get("torch_dtype", "bfloat16"))),
        )
        # transformers 5 builds the model on the meta device and only materializes checkpoint tensors;
        # Molmo2RotaryEmbedding.inv_freq is a non-persistent buffer, so it comes out as uninitialized
        # memory (-> NaN hidden states). Recompute it. (MolmoAct2 instead stores inv_freq in the ckpt.)
        for m in model.modules():
            if hasattr(m, "inv_freq") and hasattr(m, "rope_init_fn"):
                inv_freq, m.attention_scaling = m.rope_init_fn(m.config, m.inv_freq.device)
                m.inv_freq = inv_freq
                m.original_inv_freq = inv_freq.clone()

        with _lenient_processor_init():
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"

        if enable_grad_ckpt:
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                print("[Molmo2] gradient_checkpointing ENABLED (use_reentrant=False)", flush=True)
            except Exception as e:
                print(f"[Molmo2] failed to enable gradient_checkpointing: {e}", flush=True)

        self.model = model
        self.processor = processor
        self.config = config

        # Surface the text-tower hidden size at the top level so downstream code that
        # reads `model.config.hidden_size` keeps working without a Molmo2 special case.
        text_cfg = getattr(self.model.config, "text_config", None)
        if text_cfg is None or not hasattr(text_cfg, "hidden_size"):
            raise RuntimeError(
                f"[Molmo2] could not locate text_config.hidden_size on `{model_id}`. "
                f"Check that the checkpoint exposes a Qwen3/OLMo-style text_config."
            )
        # Molmo2Config already exposes `hidden_size` as a read-only property on recent revisions.
        if getattr(self.model.config, "hidden_size", None) != text_cfg.hidden_size:
            self.model.config.hidden_size = text_cfg.hidden_size
        # transformers 5 no longer gives every config a `use_cache` default; Molmo2Model.forward reads it.
        if not hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        """
        Forward pass through the underlying Molmo2 backbone.

        Action / flow-matching heads consume only `hidden_states`, so we cap
        `logits_to_keep=1` by default to avoid materializing a huge (B, L, vocab)
        bf16 tensor. Callers can override.
        """
        kwargs.setdefault("logits_to_keep", 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**kwargs)
        return outputs

    def generate(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.float16):
            return self.model.generate(**kwargs)

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build a model-ready batch from raw (images, instructions). Same name and
        signature as `_QWen3_VL_Interface.build_qwenvl_inputs`.

        Args:
            images:       list[list[PIL.Image]] — multi-view images per sample.
            instructions: list[str]            — one instruction per sample.
            solutions:    optional list[str]   — assistant turn for SFT-style training.

        Returns:
            BatchFeature on `self.model.device`.
        """
        assert len(images) == len(instructions), "images and instructions must batch-align"

        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:
                cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                msg.append(
                    {"role": "assistant", "content": [{"type": "text", "text": solutions[len(messages)]}]}
                )
            messages.append(msg)

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=(solutions is None),
            return_dict=True,
            return_tensors="pt",
        )
        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse

    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="allenai/Molmo2-4B",
        help="HF model id or local path (any allenai/Molmo2-* checkpoint).",
    )
    parser.add_argument("--attn", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

    cfg = OmegaConf.create(
        {
            "framework": {
                "qwenvl": {
                    "base_vlm": args.model_id,
                    "attn_implementation": args.attn,
                }
            },
            "datasets": {"vla_data": {}},
        }
    )

    iface = _Molmo2_VL_Interface(cfg)
    print(f"[Molmo2] hidden_size = {iface.model.config.hidden_size}")
    print(f"[Molmo2] device = {next(iface.model.parameters()).device}")

    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    batch = iface.build_qwenvl_inputs(
        images=[[img, img], [img, img]],
        instructions=["Pick up the red block.", "Stack the green cube on the blue cube."],
    )
    print(f"[Molmo2] input_ids shape = {batch['input_ids'].shape}")
    out = iface(**batch, output_hidden_states=True, return_dict=True)
    print(f"[Molmo2] last hidden state shape = {tuple(out.hidden_states[-1].shape)}")
    print("[Molmo2] OK")
