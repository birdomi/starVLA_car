# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Qwen-GR00T Memory Framework (HAMLET-style history)

QwenGR00T plus a history memory after HAMLET (Koo et al., "Switch your Vision-Language-Action
Model into a History-Aware Policy", arXiv 2510.00695):

  * Moment tokens: N learnable embeddings appended at the very END of every VLM input
    sequence, after the chat template's generation prompt. The VLM is causal, so no
    image/text position can attend to them. Mathematically the image/text hidden states are
    then those of plain QwenGR00T; numerically (bf16) the longer sequence changes the GEMM
    shapes and the outputs drift the way a different batch size or padded length (e.g. B=1
    at deployment vs B=32 in training) already makes them drift. So the DiT's vl_embs come from QwenGR00T's own unmodified call: memory.moment_pass
    "kv_cache" runs it once and feeds the moment tokens as an N-token continuation on its
    cache; "append" (+ exact_context_pass) runs a second, appended pass for the moments.
  * Every frame of the history window (data config video delta_indices, oldest first, the
    last one is the current observation) gets its own VLM pass. The moment-token outputs of
    all frames go through MomentMemory, a small block-causal Transformer; its rows for the
    current frame are the memory features.
  * The DiT reads the memory through a residual cross-attention in each of its
    cross-attention blocks whose output projection starts at zero
    (diffusion_model_cfg.memory_cond), so a QwenGR00T checkpoint loads into this framework
    and predicts bit-identically until training moves that projection.

Differences from the paper: no time-contrastive pre-stage (moment tokens start random and
train end to end with the action loss, the official code's single-stage default); the moment
tokens stay trainable while the VLM is frozen (trainer.freeze_modules: qwen_vl_interface);
past frames get gradient only with memory.past_moment_grad (the official code always does).

Inference with a cache: predict_action returns the current frame's moment outputs under
"moments" ([B, N, H]); passing example["past_moments"] ([history-1, N, H], oldest first)
skips the VLM passes over past images. At an episode start the cache must repeat the first
frame, as the training loader clamps negative frame indices to frame 0.
"""

import contextlib
import sys
import time
import types
from pathlib import Path
from typing import List, Optional

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


def _gdn_continue_forward(self, hidden_states, cache_params=None, attention_mask=None):
    """Qwen3.5 GatedDeltaNet over a multi-token continuation of a cached prefix.

    HF only resumes from the cache for single-token decoding; a longer input restarts from a zero
    state. This is HF's forward with the cached conv inputs and recurrent state as the starting
    point. The continuation tokens are all valid (no padding), so attention_mask is not needed.
    """
    if cache_params is None or not cache_params.has_previous_state(self.layer_idx):
        raise RuntimeError("moment continuation needs the prefill cache (gradient checkpointing drops it)")
    assert self.activation == "silu", self.activation
    layer_cache = cache_params.layers[self.layer_idx]
    batch_size, seq_len, _ = hidden_states.shape

    # depthwise causal conv over [last K cached inputs, new inputs]; output i covers conv_in[i : i + K],
    # so outputs 1..seq_len belong to the new tokens
    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    conv_in = torch.cat([layer_cache.conv_states.to(mixed_qkv.dtype), mixed_qkv], dim=-1)
    mixed_qkv = F.silu(F.conv1d(conv_in, self.conv1d.weight, self.conv1d.bias, groups=conv_in.shape[1])[..., 1:])
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    beta = self.in_proj_b(hidden_states).sigmoid()
    g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    core_attn_out, _ = self.chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=layer_cache.recurrent_states,
        output_final_state=False, use_qk_l2norm_in_kernel=True,
    )
    core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


@contextlib.contextmanager
def _gdn_continuation(vlm):
    """Route every GatedDeltaNet of `vlm` through _gdn_continue_forward (no-op for full-attention VLMs)."""
    mixers = [m for m in vlm.modules() if type(m).__name__.endswith("GatedDeltaNet")]
    for m in mixers:
        m.forward = types.MethodType(_gdn_continue_forward, m)
    try:
        yield
    finally:
        for m in mixers:
            del m.forward


class MomentMemory(nn.Module):
    """Moment-token embeddings + block-causal Transformer over [history x N] moment outputs.

    Frame i attends to the moment tokens of frames <= i. Returns the rows of the last
    (current) frame, [B, N, d_model].
    """

    def __init__(self, vl_dim, num_moment_tokens=4, history=4, d_model=1024, num_layers=2, num_heads=8,
                 dropout=0.1, moment_init_std=0.02):
        super().__init__()
        self.num_moment_tokens, self.history = num_moment_tokens, history
        self.moment_tokens = nn.Parameter(torch.randn(num_moment_tokens, vl_dim) * moment_init_std)
        self.in_norm = nn.LayerNorm(vl_dim)
        self.in_proj = nn.Linear(vl_dim, d_model)
        self.time_embed = nn.Parameter(torch.randn(history, d_model) * 0.02)
        self.token_embed = nn.Parameter(torch.randn(num_moment_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, 4 * d_model, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)
        frame = torch.arange(history).repeat_interleave(num_moment_tokens)
        self.register_buffer("block_causal_mask", frame[None, :] > frame[:, None], persistent=False)  # True = masked

    def forward(self, moments):
        """moments: [B, history, N, vl_dim], oldest first."""
        b, t, n, _ = moments.shape
        x = self.in_proj(self.in_norm(moments.to(self.in_proj.weight.dtype)))
        x = x + self.time_embed[None, :, None] + self.token_embed[None, None]
        x = self.encoder(x.reshape(b, t * n, -1), mask=self.block_causal_mask)
        return self.out_norm(x[:, -n:])


@FRAMEWORK_REGISTRY.register("QwenGR00T_Memory")
class Qwen_GR00T_Memory(Qwen_GR00T):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config)
        mcfg = self.config.framework.get("memory", {})
        d_model = int(mcfg.get("d_model", 1024))
        dit_cfg = self.config.framework.action_model.diffusion_model_cfg
        if not dit_cfg.get("memory_cond", False) or int(dit_cfg.get("memory_dim", 0) or 0) != d_model:
            raise ValueError("QwenGR00T_Memory needs framework.action_model.diffusion_model_cfg.memory_cond: true "
                             f"and memory_dim == framework.memory.d_model ({d_model})")
        vl_dim = self.qwen_vl_interface.model.config.hidden_size
        embed = self.qwen_vl_interface.model.get_input_embeddings().weight
        self.memory_module = MomentMemory(
            vl_dim,
            num_moment_tokens=int(mcfg.get("num_moment_tokens", 4)),
            history=int(mcfg.get("history", 4)),
            d_model=d_model,
            num_layers=int(mcfg.get("num_layers", 2)),
            num_heads=int(mcfg.get("num_heads", 8)),
                dropout=float(mcfg.get("dropout", 0.1)),
            moment_init_std=float(embed.detach().float().std()),  # same scale as the token embeddings
        )
        # How the moment outputs are computed:
        #   "append"   (default, what the first runs used): placeholders appended to the input ids, the
        #              embedding hook writes the moment tokens into them; see exact_context_pass.
        #   "kv_cache": ONE full VLM pass, exactly QwenGR00T's call (bit-identical vl_embs), then the N
        #              moment tokens run as a continuation on its cache. The backward only goes through
        #              those N tokens, so moment gradients (incl. past_moment_grad) are nearly free.
        self.moment_pass = str(mcfg.get("moment_pass", "append"))
        if self.moment_pass not in ("append", "kv_cache"):
            raise ValueError(f"framework.memory.moment_pass must be append or kv_cache, got {self.moment_pass!r}")
        # append only. True: vl_embs from an unmodified VLM pass (bit-identical to QwenGR00T), moments from
        # a second pass. False: one pass serves both. Measured 2026-10-07 (Qwen3.5-2B, bf16 autocast, FA2):
        # NOT bit-identical — the longer sequence changes the GEMM shapes, so cuBLAS rounds differently and
        # the network amplifies it (max |d| 1.5-2.5 on |h| <= 85 for most batch sizes) — the same kind of
        # drift a different batch size or padded length already causes.
        self.exact_context_pass = bool(mcfg.get("exact_context_pass", True))
        # True: the moment tokens also get the action-loss gradient through the past frames' VLM passes
        # (what the official HAMLET code does), i.e. they learn what is worth keeping for later steps.
        # False: past frames run without gradient (cheaper; moment tokens only learn via the current frame).
        self.past_moment_grad = bool(mcfg.get("past_moment_grad", False))
        # Any single-token id reserves the positions; the embedding hook overwrites them.
        placeholder = str(mcfg.get("placeholder_token", "<|fim_pad|>"))
        ids = self.qwen_vl_interface.processor.tokenizer(placeholder, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"moment placeholder {placeholder!r} must be one token, got {ids}"
        self.moment_placeholder_id = ids[0]

    # ------------------------------------------------------------------
    def _vlm_forward(self, qwen_inputs):
        with torch.autocast("cuda", dtype=getattr(self.qwen_vl_interface, "autocast_dtype", torch.bfloat16)):
            outputs = self.qwen_vl_interface(
                **qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True
            )
        return outputs.hidden_states[-1]

    def _vlm_encode(self, batch_images, instructions, moment_grad: bool, need_context: bool = True):
        """VLM pass(es) for one frame per sample.

        Returns (vl_embs [B, L, H] or None, attention mask [B, L], moment outputs [B, N, H]).
        vl_embs come from the unmodified QwenGR00T pass when exact_context_pass is on, else from the
        appended pass. The processor pads on the left, so every sequence ends at position L - 1 and
        the N appended positions are the last ones for all samples.
        """
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        input_ids, attention_mask = qwen_inputs["input_ids"], qwen_inputs["attention_mask"]
        b, length = input_ids.shape
        n = self.memory_module.num_moment_tokens
        assert bool(attention_mask[:, -1].all()), "expected left padding (all sequences end at the last position)"
        if self.moment_pass == "kv_cache":
            return self._vlm_encode_kv(qwen_inputs, moment_grad, need_context)

        vl_embs = None
        if need_context and self.exact_context_pass:
            with torch.no_grad():  # the VLM is frozen: exactly QwenGR00T's computation
                vl_embs = self._vlm_forward(qwen_inputs)

        moment_inputs = dict(qwen_inputs)
        moment_inputs["input_ids"] = torch.cat([input_ids, input_ids.new_full((b, n), self.moment_placeholder_id)], 1)
        moment_inputs["attention_mask"] = torch.cat([attention_mask, attention_mask.new_ones((b, n))], 1)
        for key, value in list(moment_inputs.items()):  # per-token type ids, if the processor emits any
            if "token_type" in key and torch.is_tensor(value):
                moment_inputs[key] = torch.cat([value, value.new_zeros((b, n) + tuple(value.shape[2:]))], 1)

        moment_tokens = self.memory_module.moment_tokens

        def inject_moments(module, inputs, output):
            output = output.clone()
            output[:, length:] = moment_tokens.to(dtype=output.dtype)[None]
            return output

        handle = self.qwen_vl_interface.model.get_input_embeddings().register_forward_hook(inject_moments)
        try:
            with contextlib.nullcontext() if moment_grad else torch.no_grad():
                last_hidden = self._vlm_forward(moment_inputs)
        finally:
            handle.remove()
        if need_context and vl_embs is None:
            # The image/text positions never see the moment tokens (causal), so their gradient w.r.t.
            # them is exactly zero: detach instead of back-propagating zeros through the frozen VLM.
            vl_embs = last_hidden[:, :length].detach()
        return vl_embs, attention_mask, last_hidden[:, length:]

    def _vlm_encode_kv(self, qwen_inputs, moment_grad: bool, need_context: bool):
        """moment_pass == "kv_cache": QwenGR00T's own VLM call (bit-identical vl_embs) with its cache kept,
        then the N moment tokens as a continuation on that cache, at the positions they would have at the
        end of the sequence. Same return values as _vlm_encode."""
        input_ids, attention_mask = qwen_inputs["input_ids"], qwen_inputs["attention_mask"]
        b, length = input_ids.shape
        n = self.memory_module.num_moment_tokens
        vlm = self.qwen_vl_interface.model
        autocast = torch.autocast("cuda", dtype=getattr(self.qwen_vl_interface, "autocast_dtype", torch.bfloat16))

        with torch.no_grad(), autocast:  # frozen VLM, and the context never sees the moment tokens
            prefill = self.qwen_vl_interface(
                **qwen_inputs, output_attentions=False, output_hidden_states=need_context, return_dict=True,
                use_cache=True, logits_to_keep=1,
            )
        cache = prefill.past_key_values
        if cache is None or cache.get_seq_length() != length:
            raise RuntimeError("moment_pass=kv_cache needs the prefill cache; gradient checkpointing on the VLM "
                               "drops it in train mode")
        vl_embs = prefill.hidden_states[-1] if need_context else None

        # Positions: exactly those of the appended sequence, i.e. what the "append" pass would use.
        ext_mask = torch.cat([attention_mask, attention_mask.new_ones((b, n))], 1)
        rope_kwargs = {k: qwen_inputs[k] for k in ("image_grid_thw", "video_grid_thw") if k in qwen_inputs}
        if "mm_token_type_ids" in qwen_inputs:
            tt = qwen_inputs["mm_token_type_ids"]
            rope_kwargs["mm_token_type_ids"] = torch.cat([tt, tt.new_zeros((b, n))], 1)
        ext_ids = torch.cat([input_ids, input_ids.new_full((b, n), self.moment_placeholder_id)], 1)
        position_ids, _ = vlm.model.get_rope_index(ext_ids, attention_mask=ext_mask, **rope_kwargs)

        embeds = self.memory_module.moment_tokens[None].expand(b, -1, -1)
        with (contextlib.nullcontext() if moment_grad else torch.no_grad()), autocast, _gdn_continuation(vlm):
            out = self.qwen_vl_interface(
                inputs_embeds=embeds, attention_mask=ext_mask, position_ids=position_ids[..., length:],
                past_key_values=cache, use_cache=True, output_hidden_states=True, return_dict=True,
                logits_to_keep=1,
            )
        return vl_embs, attention_mask, out.hidden_states[-1]

    def _encode_with_history(self, batch_images, instructions, moment_grad: bool, past_moments=None):
        """batch_images: per sample a list of history frames, oldest first (or just the current frame
        when past_moments [B, history-1, N, H] is given). Returns vl_embs, attention mask,
        all moments [B, history, N, H] and the current moments [B, N, H]."""
        history = self.memory_module.history
        current = [list(images)[-1:] for images in batch_images]
        vl_embs, attention_mask, moments_now = self._vlm_encode(current, instructions, moment_grad)
        if past_moments is None:
            counts = {len(images) for images in batch_images}
            if counts != {history}:
                raise ValueError(f"expected {history} frames per sample (oldest first), got {sorted(counts)}; "
                                 "use a *_hist30 data config or pass past_moments")
            past_images = [list(images)[s:s + 1] for s in range(history - 1) for images in batch_images]
            _, _, past = self._vlm_encode(past_images, list(instructions) * (history - 1),
                                          moment_grad=moment_grad and self.past_moment_grad, need_context=False)
            b, n, h = moments_now.shape
            past_moments = past.view(history - 1, b, n, h).transpose(0, 1)
        past_moments = past_moments.to(device=moments_now.device, dtype=moments_now.dtype)
        moments = torch.cat([past_moments, moments_now[:, None]], dim=1)
        return vl_embs, attention_mask, moments, moments_now

    # ------------------------------------------------------------------
    def forward(self, examples: List[dict] = None, **kwargs):
        if isinstance(examples, dict) and "qwen_inputs" in examples:
            raise NotImplementedError("QwenGR00T_Memory runs one VLM pass per history frame; "
                                      "preprocess_in_workers is not supported")
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        progress = self._progress_of(examples)

        t_vlm = time.perf_counter()
        last_hidden, backbone_attention_mask, moments, _ = self._encode_with_history(
            batch_images, instructions, moment_grad=True
        )
        torch.cuda.synchronize()
        vlm_time = time.perf_counter() - t_vlm

        t_head = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float32):
            memory = self.memory_module(moments)  # [B, N, d_model]
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = self.config.framework.action_model.get("repeated_diffusion_steps", 4)
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            memory_repeated = memory.repeat(repeated_diffusion_steps, 1, 1)
            mask_repeated = None
            if backbone_attention_mask is not None:
                mask_repeated = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(dtype=torch.bool)

            state_t = state_repeated = None
            if state is not None:
                state_t = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state_t.repeat(repeated_diffusion_steps, 1, 1)

            progress_t = progress_repeated = None
            if progress is not None:
                progress_t = torch.as_tensor(progress, device=last_hidden.device)
                progress_repeated = progress_t.repeat(repeated_diffusion_steps)

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated,
                encoder_attention_mask=mask_repeated, progress=progress_repeated, memory_states=memory_repeated,
            )
        torch.cuda.synchronize()
        head_time = time.perf_counter() - t_head

        # Same free eval metric as QwenGR00T: 4-step sampling with the head in eval mode.
        with torch.no_grad():
            was_training = self.action_model.training
            self.action_model.eval()
            with torch.autocast("cuda", dtype=torch.float32):
                mask = backbone_attention_mask.to(dtype=torch.bool) if backbone_attention_mask is not None else None
                pred_eval = self.action_model.predict_action(
                    last_hidden, state_t, encoder_attention_mask=mask, progress=progress_t,
                    memory_states=memory.detach(),
                )
            if was_training:
                self.action_model.train()
            diff = pred_eval.float() - actions_target.float()
            mse_score = (torch.linalg.norm(diff) / pred_eval.numel()).item()
            action_rmse = diff.pow(2).mean().sqrt().item()

        return {
            "action_loss": action_loss,
            "timing/preprocess": 0.0,  # build_qwenvl_inputs runs inside the VLM timing here
            "timing/vlm_fwd": vlm_time,
            "timing/head_fwd": head_time,
            "mse_score": mse_score,
            "action_rmse": action_rmse,
        }

    def predict_action(self, examples: List[dict], **kwargs):
        """example["image"]: the history frames, oldest first — or only the current frame together with
        example["past_moments"] ([history-1, N, H], oldest first). Also returns this call's moment
        outputs ("moments", [B, N, H]) for the caller's cache."""
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        progress = self._progress_of(examples)
        past_moments = None
        if "past_moments" in examples[0]:
            past_moments = torch.as_tensor(np.stack([np.asarray(e["past_moments"]) for e in examples]))

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        with torch.no_grad():
            last_hidden, backbone_attention_mask, moments, moments_now = self._encode_with_history(
                batch_images, instructions, moment_grad=False, past_moments=past_moments
            )
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
            state = (
                torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
                if state is not None
                else None
            )
            if progress is not None:
                progress = torch.as_tensor(progress, device=last_hidden.device)
            with torch.autocast("cuda", dtype=torch.float32):
                memory = self.memory_module(moments)
                pred_actions = self.action_model.predict_action(
                    last_hidden, state, encoder_attention_mask=backbone_attention_mask, progress=progress,
                    memory_states=memory,
                )
        return {
            "normalized_actions": pred_actions.detach().cpu().numpy(),
            "moments": moments_now.float().cpu().numpy(),
        }
