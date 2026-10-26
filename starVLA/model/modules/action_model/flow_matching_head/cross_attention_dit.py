# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class ProgressEncoder(nn.Module):
    """Episode progress p = t / T_episode in [0, 1] -> (B, D) embedding.

    Deliberately separate from TimestepEncoder (the flow-matching noise level): its own sinusoidal
    features over [0, 1] and its own MLP, read only by the per-block ProgressAdaLayerNorm in front
    of each FFN. Never mixed into temb.
    """

    def __init__(self, embedding_dim, num_freqs=64, max_freq=100.0):
        super().__init__()
        # half a period over the episode (pi) up to periods of 1/50 of the episode (100 pi)
        freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), num_freqs)) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_freqs, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, progress):
        x = progress.float()[:, None] * self.freqs.float()[None]
        feats = torch.cat([x.sin(), x.cos()], dim=-1).to(self.mlp[0].weight.dtype)
        return self.mlp(feats)


class ProgressAdaLayerNorm(nn.Module):
    """Progress-conditioned shift/scale on the (non-affine) FFN pre-norm output.

    The modulation Linear is zero-initialised, so a block loaded from a checkpoint trained without
    progress computes exactly what it did before; the conditioning only appears as training moves
    this Linear away from zero. Only this Linear starts at zero — ProgressEncoder keeps its default
    init, otherwise both ends of the product would be zero and neither would get a gradient.
    """

    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x_norm, progress_emb):
        scale, shift = self.linear(self.silu(progress_emb)).chunk(2, dim=-1)
        return x_norm * (1 + scale[:, None]) + shift[:, None]


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
        cond_dim: Optional[int] = None,  # width of the conditioning vector (default: embedding_dim)
        shift_only: bool = False,  # modulate the shift only: LN(x) + shift(cond), scale fixed at 1
        zero_init: bool = False,  # zero-init the modulation Linear -> starts as a plain LayerNorm
        bias: bool = True,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        self.shift_only = shift_only
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim or embedding_dim, embedding_dim if shift_only else output_dim, bias=bias)
        if zero_init:
            nn.init.zeros_(self.linear.weight)
            if bias:
                nn.init.zeros_(self.linear.bias)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        if self.shift_only:
            shift = temb[:, None] if temb.dim() == 2 else temb
            return self.norm(x) + shift
        scale, shift = temb.chunk(2, dim=-1)
        if scale.dim() == 2:  # per-sample (B, D) -> broadcast over tokens
            scale, shift = scale[:, None], shift[:, None]
        # per-token (B, N, D) conditioning (e.g. timestep + group identity) applies as-is
        x = self.norm(x) * (1 + scale) + shift
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        progress_cond: bool = False,
        memory_dim: Optional[int] = None,
        ffn_progress_shift_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.dropout = dropout
        self.cross_attention_dim = cross_attention_dim
        self.activation_fn = activation_fn
        self.attention_bias = attention_bias
        self.norm_elementwise_affine = norm_elementwise_affine
        self.positional_embeddings = positional_embeddings
        self.num_positional_embeddings = num_positional_embeddings
        self.norm_type = norm_type

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        # progress (t / T_episode) AdaLN in front of the FFN, separate from the temb AdaLN of norm1
        self.ffn_progress_norm = ProgressAdaLayerNorm(dim) if progress_cond else None
        # Temporal progress vector (prev vs curr VLM features) -> shift-only AdaLayerNorm as the FFN pre-norm.
        # Same LayerNorm as norm3 (shared when affine); bias-free zero-init modulation, so a zero progress
        # vector (no history) and a fresh module both give exactly norm3 -> pretrained behaviour.
        self.ffn_progress_adanorm = None
        if ffn_progress_shift_dim:
            self.ffn_progress_adanorm = AdaLayerNorm(
                dim, norm_elementwise_affine=norm_elementwise_affine, norm_eps=norm_eps,
                cond_dim=ffn_progress_shift_dim, shift_only=True, zero_init=True, bias=False,
            )
            if norm_elementwise_affine:
                self.ffn_progress_adanorm.norm = self.norm3

        # 2. Memory cross-attention (HAMLET-style history features), residual, between attn1 and the
        # FFN. Output projection zero-initialised: the block is unchanged until training moves it.
        if memory_dim:
            self.norm_mem = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
            self.attn_mem = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=memory_dim,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )
            nn.init.zeros_(self.attn_mem.to_out[0].weight)
            if self.attn_mem.to_out[0].bias is not None:
                nn.init.zeros_(self.attn_mem.to_out[0].bias)
        else:
            self.norm_mem = self.attn_mem = None
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        if final_dropout:
            self.final_dropout = nn.Dropout(dropout)
        else:
            self.final_dropout = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
        progress_emb: Optional[torch.Tensor] = None,
        memory_states: Optional[torch.Tensor] = None,
        progress_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # 0. Self-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,  # @JinhuiYE original attention_mask=attention_mask
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        if self.attn_mem is not None and memory_states is not None:
            hidden_states = hidden_states + self.attn_mem(self.norm_mem(hidden_states), encoder_hidden_states=memory_states)

        # 4. Feed-forward
        if self.ffn_progress_adanorm is not None and progress_vector is not None:
            norm_hidden_states = self.ffn_progress_adanorm(hidden_states, progress_vector)
        else:
            norm_hidden_states = self.norm3(hidden_states)
        if self.ffn_progress_norm is not None and progress_emb is not None:
            norm_hidden_states = self.ffn_progress_norm(norm_hidden_states, progress_emb)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    # register_to_config auto-registers constructor params into config, enabling access via self.config.xxx instead of self.xxx
    @register_to_config  # Registers passed params to config. TODO: replace with our singleton pattern, implement a mergeable @merge_param_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        build_timestep_encoder: bool = True,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        # This is the single forward-mode switch:
        #   False -> legacy all-cross attention; True -> canonical interleaving.
        # Released checkpoints may come from an older repository snapshot;
        # use the checkpoint-time code/config when exact reproduction matters.
        interleave_self_attention=False,
        # Deprecated compatibility alias. The effective mode is represented by
        # interleave_self_attention after this value is applied.
        use_canonical_forward: Optional[bool] = None,
        cross_attention_dim: Optional[int] = None,
        # Episode-progress conditioning (t / T_episode): a ProgressEncoder of its own plus a
        # zero-initialised AdaLN in front of every block's FFN. Off -> no extra parameters.
        progress_cond: bool = False,
        # History memory (HAMLET-style): every CROSS-attention block gets an extra residual cross-attention
        # to `memory_states` (B, N_mem, memory_dim), output projection zero-initialised. Off -> no extra
        # parameters.
        memory_cond: bool = False,
        memory_dim: Optional[int] = None,
        # Temporal progress vector width: every block's FFN pre-norm becomes a shift-only, zero-initialised
        # AdaLayerNorm conditioned on it (QwenGR00T_GridDiT_ProgressShift). None -> no extra parameters.
        ffn_progress_shift_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if memory_cond and not memory_dim:
            raise ValueError("memory_cond=True needs memory_dim (width of the memory features)")
        if use_canonical_forward is not None:
            interleave_self_attention = bool(use_canonical_forward)
        self._interleave_self_attention = bool(interleave_self_attention)
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        # Timestep encoder. build_timestep_encoder=False skips it entirely — for heads that ALWAYS
        # pass temb_override (e.g. split-mode group AdaLN, which owns a narrower TimestepEncoder);
        # building it anyway would leave an unused parameter block (breaks DDP/DeepSpeed grads).
        compute_dtype = getattr(self.config, "compute_dtype", torch.float32)
        if bool(getattr(self.config, "build_timestep_encoder", True)):
            self.timestep_encoder = (
                TimestepEncoder(  # TODO BUG: self.config.compute_dtype doesn't error during training but fails at eval
                    embedding_dim=self.inner_dim, compute_dtype=compute_dtype
                )
            )
        else:
            self.timestep_encoder = None
        self.progress_encoder = ProgressEncoder(self.inner_dim) if progress_cond else None

        all_blocks = []
        for idx in range(self.config.num_layers):

            use_self_attn = idx % 2 == 1 and self._interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

            all_blocks += [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                    progress_cond=progress_cond,
                    memory_dim=memory_dim if (memory_cond and not use_self_attn) else None,
                    ffn_progress_shift_dim=ffn_progress_shift_dim,
                )
            ]
        self.transformer_blocks = nn.ModuleList(all_blocks)

        # Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D) or list of layer-wise tensors
        timestep: Optional[torch.LongTensor] = None,
        return_all_hidden_states: bool = False,
        encoder_attention_mask=None,
        return_pre_output: bool = False,
        temb_override: Optional[torch.Tensor] = None,
        progress: Optional[torch.Tensor] = None,  # (B,) episode progress t / T_episode in [0, 1]
        memory_states: Optional[torch.Tensor] = None,  # (B, N_mem, memory_dim) history memory features
        progress_vector: Optional[torch.Tensor] = None,  # (B, ffn_progress_shift_dim) temporal progress vector
    ):
        # Encode timesteps. temb_override lets the caller pass a PRE-BUILT conditioning tensor,
        # per-sample (B, D) or per-token (B, N, D) — e.g. timestep embedding concatenated with a
        # per-token group identity (AdaLN group conditioning).
        if temb_override is not None:
            temb = temb_override
        else:
            assert self.timestep_encoder is not None, (
                "DiT was built with build_timestep_encoder=False; the caller must pass temb_override"
            )
            temb = self.timestep_encoder(timestep)

        progress_emb = None
        if self.progress_encoder is not None:
            if progress is None:
                raise ValueError("DiT built with progress_cond=True needs `progress` (t / T_episode) on every call")
            progress_emb = self.progress_encoder(progress.to(hidden_states.device))
        if self.config.get("memory_cond", False) and memory_states is None:
            raise ValueError("DiT built with memory_cond=True needs `memory_states` on every call")

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        is_layerwise_encoder = isinstance(encoder_hidden_states, (list, tuple))
        encoder_hidden_states = (
            [state.contiguous() for state in encoder_hidden_states]
            if is_layerwise_encoder
            else encoder_hidden_states.contiguous()
        )

        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self._interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                    progress_emb=progress_emb,
                    progress_vector=progress_vector,
                )
            else:
                if is_layerwise_encoder:
                    block_encoder_hidden_states = encoder_hidden_states[idx]
                else:
                    block_encoder_hidden_states = encoder_hidden_states
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=block_encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                    progress_emb=progress_emb,
                    memory_states=memory_states,
                    progress_vector=progress_vector,
                )
            all_hidden_states.append(hidden_states)

        if return_pre_output:
            if return_all_hidden_states:
                return hidden_states, all_hidden_states
            return hidden_states

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=-1)
        if scale.dim() == 2:
            shift, scale = shift[:, None], scale[:, None]
        hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)


class SelfAttentionTransformer(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.gradient_checkpointing = False

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        print(
            "Total number of SelfAttentionTransformer parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        return_all_hidden_states: bool = False,
    ):
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states)
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states
