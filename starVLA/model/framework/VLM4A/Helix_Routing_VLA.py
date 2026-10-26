"""
Helix_Routing_VLA: Helix_VLA (System 2 Qwen on the stale frame t - s2_lag, System 1 C-RADIO + flow DiT on frame t) with
DEPTH ROUTING of System 1's C-RADIO features: every cross-attention block of the System 1 DiT reads its own softmax
mixture of ALL C-RADIO layers instead of the last layer only.

  visual prompt (VPT-shallow): routing.n_prompt learnable tokens appended to C-RADIO's INPUT sequence only (after the
    patch embedder / CLS / registers, before block 1); they go through every block with the patches (and the patches
    attend to them), so their hidden state after block l = layer l's image-dependent representative ->
    key_l = W_k LN(mean prompt state after block l)                                          [M, L, d_k]
  block queries: one learnable query q_b per System 1 cross-attention block (DiT-B: 16 blocks, self / cross
    interleaved -> 8)                                                                         [n_blk, d_k]
  routing: alpha[m, b, :] = softmax_l(q_b . key_l / sqrt(d_k)) over the L = 27 C-RADIO blocks (per image)
  fusion: F_b = sum_l alpha[b, l] LN(H_l), H_l = [summary prefix ; patches] after block l (LayerNorm without affine
    per layer: the residual stream's scale grows with depth); artifact positions masked as in Helix (final-layer norm)
  condition of cross-attention block b = [System 2 latent tokens ; RadioCondition_b(F_b)] (own LN + Linear + position
    embedding per block); the mask is shared by all blocks.
Queries start near zero -> near-uniform mixtures at the start of training. Per-block routing entropy / top layer is
returned as routing/* training metrics.

Everything else (frames, System 2, state token, DiT-B, flow loss, sampling) is Helix_VLA's; the flow loss / Euler
sampling are FlowmatchingActionHead.forward / predict_action with a per-block (list) encoder condition, re-implemented
here so the shared action head stays untouched (no progress / memory conditioning).
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.Helix_VLA import Helix_VLA, HelixVLADefaultConfig
from starVLA.model.framework.VLM4A.QwenGR00T_RadioAdaLN import RadioCondition, _example_images, _images_to_tensor
from starVLA.model.modules.dual_system.backbone import VisualFeatures
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@dataclass
class HelixRoutingVLADefaultConfig(HelixVLADefaultConfig):
    name: str = "Helix_Routing_VLA"
    routing: dict = field(default_factory=lambda: {
        "n_prompt": 1,              # VPT-shallow prompt tokens at C-RADIO's input (key = their mean state per layer)
        "prompt_init": "cls_mean",  # cls_mean = mean of C-RADIO's summary CLS input tokens | randn (CLS-token scale)
        "key_dim": 256,             # d_k of the prompt keys / block queries
    })


class RadioDepthRouter(nn.Module):
    """VPT-shallow prompt -> per-layer keys; per-block queries -> softmax over layers -> fused C-RADIO tokens."""

    def __init__(self, d: int, n_layers: int, n_blocks: int, n_prompt: int, key_dim: int, prompt_init: torch.Tensor):
        super().__init__()
        self.n_layers, self.n_blocks, self.key_dim = n_layers, n_blocks, key_dim
        self.prompt = nn.Parameter(prompt_init.reshape(1, d).repeat(n_prompt, 1).clone())
        self.key_norm = nn.LayerNorm(d)
        self.key_proj = nn.Linear(d, key_dim)
        self.query = nn.Parameter(torch.randn(n_blocks, key_dim) * 0.02)

    def forward(self, layers: torch.Tensor, prompt_states: torch.Tensor, valid: torch.Tensor):
        """layers [M, L, T, D] (prefix + patches after each block), prompt_states [M, L, D], valid [M, T] bool
        -> fused [M, n_blk, T, D] float32, alpha [M, n_blk, L] float32."""
        k = self.key_proj(self.key_norm(prompt_states.to(self.key_proj.weight.dtype)))  # [M, L, d_k]
        alpha = torch.softmax(torch.einsum("bd,mld->mbl", self.query, k).float() / math.sqrt(self.key_dim), -1)
        h = F.layer_norm(layers.float(), (layers.shape[-1],)).masked_fill(~valid[:, None, :, None], 0.0)
        return torch.einsum("mbl,mltd->mbtd", alpha, h), alpha


@FRAMEWORK_REGISTRY.register("Helix_Routing_VLA")
class Helix_Routing_VLA(Helix_VLA):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        config = merge_framework_config(HelixRoutingVLADefaultConfig, config)
        super().__init__(config, **kwargs)
        fw = self.config.framework
        if self.radio_encoder is None:
            raise ValueError("Helix_Routing_VLA routes System 1's C-RADIO layers: helix.s1_vision must be true")
        if fw.radio.get("feature", "backbone") != "backbone" or str(fw.radio.get("backbone", "c_radio")) != "c_radio":
            raise ValueError("Helix_Routing_VLA needs radio.backbone c_radio + radio.feature backbone (C-RADIO ViT blocks)")
        rt = fw.get("routing", {})
        enc = self.radio_encoder.encoder
        vit = enc.model.radio_model.model
        d = int(self.radio_encoder.feat_dim)
        self.n_skip = int(vit.patch_generator.num_skip)  # CLS + register tokens in front of the patches
        self.n_prefix = int(self.radio_cond.n_prefix)    # summary tokens kept in the condition (helix.use_prefix)
        dit = self.action_model.model
        self.cross_blocks = [i for i, blk in enumerate(dit.transformer_blocks) if blk.cross_attention_dim is not None]

        init = str(rt.get("prompt_init", "cls_mean"))
        cls = vit.patch_generator.cls_token.token.detach().float()  # [n_skip, D] C-RADIO's CLS + register input tokens
        if init == "cls_mean":
            p0 = cls[list(enc.summary_idxs)].mean(0)
        elif init == "randn":
            p0 = torch.randn(d) * cls[: enc.num_cls].norm(dim=-1).mean() / math.sqrt(d)
        else:
            raise ValueError(f"routing.prompt_init must be cls_mean | randn, got {init!r}")
        self.n_prompt = int(rt.get("n_prompt", 1))
        if self.n_prompt < 1:
            raise ValueError(f"routing.n_prompt must be >= 1, got {self.n_prompt}")
        self.radio_router = RadioDepthRouter(d, len(vit.blocks), len(self.cross_blocks), self.n_prompt,
                                             int(rt.get("key_dim", 256)), p0)
        h = int(self.radio_cond.proj.out_features)
        self.radio_cond = nn.ModuleList([  # replaces Helix's single last-layer RadioCondition
            RadioCondition(d, int(self.radio_encoder.prefix_dim), h, self.radio_encoder.num_tokens, self.n_prefix,
                           self.num_views) for _ in self.cross_blocks])
        if dit.config.get("progress_cond", False) or dit.config.get("memory_cond", False):
            raise ValueError("Helix_Routing_VLA: the DiT's progress / memory conditioning is not supported")
        logger.info(f"[Helix_Routing_VLA] {len(vit.blocks)} C-RADIO layers -> {len(self.cross_blocks)} System 1 "
                    f"cross-attention blocks {self.cross_blocks}; VPT-shallow prompt x{self.n_prompt} ({init}), "
                    f"key_dim {self.radio_router.key_dim}")

    # ------------------------------------------------------------------ C-RADIO, every layer, with the prompt
    def _radio_layers(self, images: torch.Tensor):
        """images [M, 3, H, W] in [0, 1] -> layers [M, L, P + N, D] (summary prefix + patches after every block),
        prompt states [M, L, D], valid [M, N] bool (final-layer artifact mask, as SharedRadioBackbone)."""
        enc = self.radio_encoder.encoder
        rm = enc.model.radio_model
        vit = rm.model
        if tuple(images.shape[-2:]) != enc.input_size:
            images = F.interpolate(images.float(), enc.input_size, mode="bilinear", antialias=True, align_corners=False)
        keep = list(self.radio_encoder.prefix_idxs) if self.n_prefix else []  # radio.prefix_tokens, as Helix
        sel = torch.tensor(keep + list(range(self.n_skip, self.n_skip + self.radio_encoder.num_tokens)),
                           device=images.device)
        layers, prompts = [], []
        with torch.set_grad_enabled(torch.is_grad_enabled()):  # also when C-RADIO is frozen: the prompt's gradient
            with torch.autocast("cuda", dtype=enc.dtype, enabled=enc.dtype != torch.float32):
                x = vit.patch_generator(rm.input_conditioner(images))
                x = torch.cat([x, self.radio_router.prompt[None].expand(x.shape[0], -1, -1).to(x.dtype)], 1)
                for i, blk in enumerate(vit.blocks):
                    x = blk(x)
                    if i == len(vit.blocks) - 1:
                        x = vit.norm(x)
                    layers.append(x[:, :-self.n_prompt].index_select(1, sel))
                    prompts.append(x[:, -self.n_prompt:].mean(1))
        layers, prompts = torch.stack(layers, 1), torch.stack(prompts, 1)
        patches = layers[:, -1, len(keep):]
        if enc.mask_outliers:
            valid = patches.detach().float().norm(dim=-1) <= enc.outlier_norm
        else:
            valid = torch.ones(patches.shape[:2], dtype=torch.bool, device=patches.device)
        valid_all = torch.cat([torch.ones(valid.shape[0], len(keep), dtype=torch.bool, device=valid.device), valid], 1)
        return layers, prompts, valid, valid_all

    def _condition(self, images_per_example: List[list], instructions: List[str]):
        """-> per-cross-block conditions [n_blk x [B, L_s2 + V (N + P), H]], shared mask [B, ...], timings, routing
        weights alpha [B * V, n_blk, L]."""
        s2_images, s1_images = self._split_frames(images_per_example)
        b = len(images_per_example)
        t0 = time.perf_counter()
        z, mask = self._system2(s2_images, instructions)
        t1 = time.perf_counter()
        flat = [im for ims in s1_images for im in ims]
        layers, prompts, valid, valid_all = self._radio_layers(_images_to_tensor(flat, self.radio_size, self._device))
        fused, alpha = self.radio_router(layers, prompts, valid_all)  # [M, n_blk, P + N, D]
        p = valid_all.shape[1] - valid.shape[1]
        conds, r_mask = [], None
        for j, cond_j in enumerate(self.radio_cond):
            f = VisualFeatures(fused[:, j, p:], valid, self.radio_encoder.grid, fused[:, j, :p] if p else None)
            r_tok, r_mask = cond_j(f, b)
            conds.append(torch.cat([z, r_tok.to(z.dtype)], 1))
        t2 = time.perf_counter()
        return conds, torch.cat([mask, r_mask], 1), {"timing/vlm_fwd": t1 - t0, "timing/radio_fwd": t2 - t1}, alpha

    def _layerwise(self, conds: List[torch.Tensor]) -> List[torch.Tensor]:
        """Per-cross-block conditions -> one entry per DiT block (self-attention blocks ignore theirs)."""
        out = [conds[0]] * len(self.action_model.model.transformer_blocks)
        for j, i in enumerate(self.cross_blocks):
            out[i] = conds[j]
        return out

    # ------------------------------------------------------------------ flow head with a per-block condition
    def _sa_embs(self, action_features, state_features):
        head = self.action_model
        if head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=action_features.device)
            action_features = action_features + head.position_embedding(pos_ids).unsqueeze(0)
        parts = [] if state_features is None else [state_features]
        if head.future_tokens is not None:
            parts.append(head.future_tokens.weight.unsqueeze(0).expand(action_features.shape[0], -1, -1))
        return torch.cat(parts + [action_features], dim=1)

    def _flow_loss(self, conds, actions, state, mask):
        """FlowmatchingActionHead.forward with encoder_hidden_states = per-block list."""
        head = self.action_model
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = head.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        t_discretized = (t[:, 0, 0] * head.num_timestep_buckets).long()
        sa_embs = self._sa_embs(head.action_encoder(noisy_trajectory, t_discretized),
                                head.state_encoder(state) if state is not None else None)
        out = head.model(hidden_states=sa_embs, encoder_hidden_states=self._layerwise(conds),
                         encoder_attention_mask=mask, timestep=t_discretized, return_all_hidden_states=False)
        pred_actions = head.action_decoder(out)[:, -actions.shape[1]:]
        return ((pred_actions - velocity) ** 2).mean()

    @torch.no_grad()
    def _flow_sample(self, conds, state, mask):
        """FlowmatchingActionHead.predict_action (Euler) with encoder_hidden_states = per-block list."""
        head = self.action_model
        b, device = conds[0].shape[0], conds[0].device
        actions = torch.randn(size=(b, head.action_horizon, head.action_dim), dtype=conds[0].dtype, device=device)
        num_steps = head.num_inference_timesteps
        dt = 1.0 / num_steps
        state_features = head.state_encoder(state) if state is not None else None
        enc = self._layerwise(conds)
        for t in range(num_steps):
            timesteps_tensor = torch.full(size=(b,), fill_value=int(t / float(num_steps) * head.num_timestep_buckets),
                                          device=device)
            sa_embs = self._sa_embs(head.action_encoder(actions, timesteps_tensor), state_features)
            out = head.model(hidden_states=sa_embs, encoder_hidden_states=enc, encoder_attention_mask=mask,
                             timestep=timesteps_tensor)
            actions = actions + dt * head.action_decoder(out)[:, -head.action_horizon:]
        return actions

    def _routing_metrics(self, alpha: torch.Tensor) -> dict:
        a = alpha.detach().float()
        ent = -(a * a.clamp_min(1e-12).log()).sum(-1).mean(0) / math.log(a.shape[-1])  # [n_blk], 1 = uniform
        mean = a.mean(0)  # [n_blk, L]
        out = {}
        for j in range(a.shape[1]):
            out[f"routing/b{self.cross_blocks[j]}_entropy"] = ent[j].item()
            out[f"routing/b{self.cross_blocks[j]}_top_layer"] = float(mean[j].argmax().item() + 1)
            out[f"routing/b{self.cross_blocks[j]}_depth"] = float((mean[j] * torch.arange(1, a.shape[-1] + 1,
                                                                                           device=a.device)).sum().item())
        return out

    # ------------------------------------------------------------------ training / inference
    def forward(self, examples: List[dict] = None, **kwargs):
        images = [_example_images(ex) for ex in examples]
        conds, mask, timing, alpha = self._condition(images, [ex["lang"] for ex in examples])
        dev, dt_ = conds[0].device, conds[0].dtype
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        t_head = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array([ex["action"] for ex in examples]), device=dev, dtype=dt_)
            target = actions[:, -self.action_horizon:, :]
            r = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            state_t = None if state is None else torch.tensor(np.array(state), device=dev, dtype=dt_)
            loss = self._flow_loss([c.repeat(r, 1, 1) for c in conds], target.repeat(r, 1, 1),
                                   None if state_t is None else state_t.repeat(r, 1, 1), mask.repeat(r, 1))
        head_time = time.perf_counter() - t_head

        with torch.no_grad():  # free per-step metric: num_inference_timesteps sampling on the same features
            was_training = self.action_model.training
            self.action_model.eval()
            with torch.autocast("cuda", dtype=torch.float32):
                pred = self._flow_sample([c.detach() for c in conds], state_t, mask)
            if was_training:
                self.action_model.train()
            diff = pred.float() - target.float()
        return {"action_loss": loss, **timing, "timing/head_fwd": head_time,
                "mse_score": (torch.linalg.norm(diff) / pred.numel()).item(),
                "action_rmse": diff.pow(2).mean().sqrt().item(), **self._routing_metrics(alpha)}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        examples = examples if isinstance(examples, list) else [examples]
        images = [_example_images(ex) for ex in examples]
        size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if size:  # obs_image_size [H, W] -> PIL (W, H)
            images = resize_images(images, target_size=(int(size[1]), int(size[0])))
        conds, mask, _, _ = self._condition(images, [ex["lang"] for ex in examples])
        state = None
        if "state" in examples[0]:
            state = torch.from_numpy(np.array([ex["state"] for ex in examples])).to(conds[0].device, dtype=conds[0].dtype)
        with torch.autocast("cuda", dtype=torch.float32):
            pred = self._flow_sample(conds, state, mask)
        return {"normalized_actions": pred.detach().cpu().numpy()}
