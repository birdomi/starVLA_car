"""QwenGR00T_GridDiT + temporal progress-conditioned AdaLayerNorm SHIFT (all pretrained weights frozen).

A parallel branch compares the frozen VLM's last-layer hidden states of a PREVIOUS observation
(t - offset, same episode, same instruction) with the CURRENT one and produces a progress vector
z [B, D_progress]. Every DiT block's FFN pre-norm becomes a shift-only AdaLayerNorm conditioned on z
(DiT ffn_progress_shift_dim = D_progress):  FFN input = LN(x) + Linear(SiLU(z)).
The LayerNorm is the block's own non-affine norm3; the scale is fixed at 1; the Linear is bias-free and
zero-initialised. The timestep AdaLNs (norm1, norm_out) and the VLM -> DiT cross-attention conditioning
(current hidden states) are unchanged.

Frozen: everything that exists in QwenGR00T_GridDiT (VLM, DiT, timestep AdaLN modulation, encoders,
decoders). Trainable: progress_encoder (projection + temporal cross-attention + MLP) and the per-block
ffn_progress_adanorm modulation Linears. Zero init -> the model reproduces the pretrained checkpoint
exactly at initialisation.

Data layout: sample["image"] holds, per camera, [frame(t - offset), frame(t)] (data config video
delta_indices [-offset, 0]; LeRobot packs camera-major, oldest first). Frames before the episode start
are clamped to frame 0 by the loader, so at t = 0 the previous image equals the current one; such
samples (and inference calls without history) get z = 0 -> zero shift (bias-free Linear) = baseline behaviour.
The progress vector is an observation-time quantity: computed once per observation and reused for
every flow-matching step of that chunk.
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T_GridDiT import Qwen_GR00T_GridDiT
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """x [B, N, D], mask [B, N] (True = valid) -> [B, D]."""
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(x.dtype)[..., None]
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class TemporalProgressEncoder(nn.Module):
    """(previous, current) VLM hidden states -> progress vector [B, d_progress].

    prev/curr are projected by one shared projection, current tokens attend to previous tokens
    (temporal cross-attention, key padding masked), delta = curr - aligned_prev; masked pools of
    curr (semantic / language-conditioned content), delta (signed change) and |delta| (change
    magnitude) are concatenated and mapped by an MLP to the progress vector. A linear head on the same
    pooled feature (3 * d_model) regresses the episode progress t / T_episode (auxiliary supervision).
    """

    def __init__(self, vl_dim: int, d_model: int = 256, d_progress: int = 256, num_heads: int = 4):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(vl_dim), nn.Linear(vl_dim, d_model))
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.encoder = nn.Sequential(
            nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_progress)
        )
        self.progress_head = nn.Linear(3 * d_model, 1)

    def forward(self, prev_h, curr_h, prev_mask=None, curr_mask=None):
        prev, curr = self.proj(prev_h), self.proj(curr_h)
        key_padding = None if prev_mask is None else ~prev_mask.bool()
        aligned_prev, _ = self.temporal_attn(curr, prev, prev, key_padding_mask=key_padding, need_weights=False)
        delta = curr - aligned_prev
        feat = torch.cat([_masked_mean(curr, curr_mask), _masked_mean(delta, curr_mask),
                          _masked_mean(delta.abs(), curr_mask)], dim=-1)
        return self.encoder(feat), self.progress_head(feat).squeeze(-1)  # z [B, d_progress], p_hat [B]


class ProgressObservationBuffer:
    """Inference-side history for ONE episode: returns the observation `offset` steps back.

    Mirrors the training loader: before `offset` steps exist, the earliest observation of the episode
    stands in (frame 0); at the very first step the current observation is used, which the framework
    detects (identical images) and maps to a zero shift residual. Call reset() at every episode start.
    """

    def __init__(self, offset: int):
        self.offset = int(offset)
        self.reset()

    def reset(self):
        self.frames = {}
        self.t = 0

    def pair(self, images: List):
        """images: current per-camera frames -> [prev_cam0, curr_cam0, prev_cam1, curr_cam1, ...]."""
        self.frames[self.t] = list(images)
        prev = self.frames[max(self.t - self.offset, 0)]
        for k in [k for k in self.frames if k < self.t - self.offset]:  # keep only what can still be needed
            del self.frames[k]
        self.t += 1
        return [im for p, c in zip(prev, images) for im in (p, c)]


@FRAMEWORK_REGISTRY.register("QwenGR00T_GridDiT_ProgressShift")
class Qwen_GR00T_GridDiT_ProgressShift(Qwen_GR00T_GridDiT):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        pcfg = self.config.framework.get("progress_shift", {}) or {}
        self.num_cameras = int(pcfg.get("num_cameras", 1))
        self.prev_offset = int(pcfg.get("prev_offset", 30))
        # weight of the auxiliary MSE between p_hat and t / T_episode (needs datasets.vla_data.emit_progress)
        self.progress_loss_weight = float(pcfg.get("progress_loss_weight", 0.1))
        dit = self.action_model.model
        blocks = list(dit.transformer_blocks)
        d_progress = dit.config.get("ffn_progress_shift_dim", None)
        if not d_progress or any(b.ffn_progress_adanorm is None for b in blocks):
            raise ValueError("set framework.action_model.diffusion_model_cfg.ffn_progress_shift_dim (progress vector width)")
        vl_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        self.progress_encoder = TemporalProgressEncoder(
            vl_dim, d_model=int(pcfg.get("d_model", 256)), d_progress=int(d_progress), num_heads=int(pcfg.get("num_heads", 4))
        )

        # Freeze every pretrained parameter; train only the progress encoder and the FFN progress AdaLNs.
        for p in self.parameters():
            p.requires_grad = False
        for m in [self.progress_encoder] + [b.ffn_progress_adanorm.linear for b in blocks]:
            for p in m.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------ helpers
    def _split_frames(self, batch_images):
        """Per sample: camera-major [prev, curr] pairs -> (prev images, curr images, has_prev[B])."""
        prev, curr, has_prev = [], [], []
        for imgs in batch_images:
            if len(imgs) == 2 * self.num_cameras:
                p, c = list(imgs[0::2]), list(imgs[1::2])
                has_prev.append(not all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(p, c)))
            elif len(imgs) == self.num_cameras:  # no history supplied -> baseline behaviour
                p, c = list(imgs), list(imgs)
                has_prev.append(False)
            else:
                raise ValueError(f"expected {2 * self.num_cameras} (prev+curr) or {self.num_cameras} images, got {len(imgs)}")
            prev.append(p)
            curr.append(c)
        return prev, curr, has_prev

    def _encode(self, images, instructions):
        """Frozen VLM pass -> last hidden states [B, L, H] and attention mask [B, L] (bool)."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
        mask = qwen_inputs.get("attention_mask", None)
        with torch.no_grad(), torch.autocast("cuda", dtype=getattr(self.qwen_vl_interface, "autocast_dtype", torch.bfloat16)):
            out = self.qwen_vl_interface(**qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True)
            last_hidden = out.hidden_states[-1]
        return last_hidden.detach(), (mask.to(dtype=torch.bool) if mask is not None else None)

    def progress_vector(self, prev_h, curr_h, prev_mask, curr_mask, has_prev):
        """-> (progress vector z [B, D_progress], exactly 0 for samples without history (-> zero shift);
        predicted episode progress p_hat [B] ~ t / T_episode)."""
        z, p_hat = self.progress_encoder(prev_h.float(), curr_h.float(), prev_mask, curr_mask)
        return z * torch.as_tensor(has_prev, device=z.device, dtype=z.dtype)[:, None], p_hat

    def _ffn_shift_rms(self, z):
        ada = self.action_model.model.transformer_blocks[0].ffn_progress_adanorm
        return ada.linear(ada.silu(z)).detach().pow(2).mean().sqrt().item()

    # ------------------------------------------------------------------ training
    def forward(self, examples: List[dict] = None, **kwargs):
        if isinstance(examples, dict) and "qwen_inputs" in examples:
            raise ValueError("QwenGR00T_GridDiT_ProgressShift needs raw examples (preprocess_in_workers: false)")
        import time

        prev_imgs, curr_imgs, has_prev = self._split_frames([ex["image"] for ex in examples])
        instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        t_vlm = time.perf_counter()
        curr_h, curr_mask = self._encode(curr_imgs, instructions)  # also the cross-attention conditioning
        prev_h, prev_mask = self._encode(prev_imgs, instructions)
        vlm_time = time.perf_counter() - t_vlm

        t_head = time.perf_counter()
        rep = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
        with torch.autocast("cuda", dtype=torch.float32):
            z, p_hat = self.progress_vector(prev_h, curr_h, prev_mask, curr_mask, has_prev)
            actions = torch.tensor(np.array(actions), device=curr_h.device, dtype=curr_h.dtype)
            actions_target = actions[:, -self.action_horizon :, :]
            state_t = torch.tensor(np.array(state), device=curr_h.device, dtype=curr_h.dtype) if state is not None else None
            action_loss = self.action_model(
                curr_h.repeat(rep, 1, 1), actions_target.repeat(rep, 1, 1),
                state_t.repeat(rep, 1, 1) if state_t is not None else None,
                encoder_attention_mask=curr_mask.repeat(rep, 1) if curr_mask is not None else None,
                progress_vector=z.repeat(rep, 1),
            )
        head_time = time.perf_counter() - t_head

        progress_out = {}
        if "progress" in examples[0]:
            p_gt = torch.tensor([float(ex["progress"]) for ex in examples], device=p_hat.device, dtype=torch.float32)
            progress_loss = torch.nn.functional.mse_loss(p_hat.float(), p_gt)
            progress_out = {
                "loss": action_loss + self.progress_loss_weight * progress_loss,
                "progress/loss": progress_loss.item(),
                "progress/mae": (p_hat.detach().float() - p_gt).abs().mean().item(),
            }

        with torch.no_grad():
            was_training = self.action_model.training
            self.action_model.eval()
            with torch.autocast("cuda", dtype=torch.float32):
                pred_eval = self.action_model.predict_action(curr_h, state_t, encoder_attention_mask=curr_mask,
                                                             progress_vector=z.detach())
            if was_training:
                self.action_model.train()
            diff = pred_eval.float() - actions_target.float()
            shift_rms = self._ffn_shift_rms(z.detach())
        return {
            "action_loss": action_loss,
            **progress_out,
            "timing/vlm_fwd": vlm_time,
            "timing/head_fwd": head_time,
            "mse_score": (torch.linalg.norm(diff) / pred_eval.numel()).item(),
            "action_rmse": diff.pow(2).mean().sqrt().item(),
            "progress/has_prev_frac": float(np.mean(has_prev)),
            "progress/z_norm": z.detach().norm(dim=-1).mean().item(),
            "progress/block0_shift_rms": shift_rms,
        }

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs):
        """examples[i]["image"]: camera-major [prev, curr] pairs (ProgressObservationBuffer.pair), or just the
        current frames (-> no history, zero residual)."""
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(ex["image"]) for ex in examples]
        size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if size:
            batch_images = resize_images(batch_images, target_size=size)
        prev_imgs, curr_imgs, has_prev = self._split_frames(batch_images)
        instructions = [ex["lang"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None
        curr_h, curr_mask = self._encode(curr_imgs, instructions)
        prev_h, prev_mask = self._encode(prev_imgs, instructions)
        with torch.autocast("cuda", dtype=torch.float32):
            z, p_hat = self.progress_vector(prev_h, curr_h, prev_mask, curr_mask, has_prev)
            state_t = torch.from_numpy(np.array(state)).to(curr_h.device, dtype=curr_h.dtype) if state is not None else None
            pred = self.action_model.predict_action(curr_h, state_t, encoder_attention_mask=curr_mask, progress_vector=z)
        return {"normalized_actions": pred.detach().cpu().numpy(), "progress": p_hat.float().cpu().numpy()}
