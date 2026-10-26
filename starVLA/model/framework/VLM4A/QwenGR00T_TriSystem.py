# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
QwenGR00T-TriSystem: System 2 -> System 1.5 -> System 1 on top of QwenGR00T.

    current image + instruction ─┬─ System 2 : Qwen VLM (QwenGR00T's qwen_vl_interface, unmodified, Qwen3.5-0.8B)
                                 │   one forward pass; S_t = last hidden states of EVERY token (vision + text),
                                 │   exactly QwenGR00T's VLM features                          [B, L_s, D_s]
    current image ───────────────┴─ C-RADIOv4 : all spatial patch tokens Z_t (no summary token) [B, N, D_c]
    System 1.5  "how should the visual world change": conditional flow-matching Flow Transformer,
                p(dZ | Z_t, S_t), dZ = Z_f - Z_t; self-attention over noisy change tokens, separate
                cross-attention to Z_t and to S_t, adaLN(tau); ODE sampling -> dZ_hat         [B, N, D_c]
    System 1    "which actions realise that change": GR00T-style flow-matching Action DiT; self-attention over
                [state, noisy actions], cross-attention to dZ_hat only (system1.cond_streams, default ["dz"]),
                adaLN(tau_a); ODE sampling -> action chunk A                                 [B, H, D_a]

Every module is its own nn.Module (qwen_vl_interface, radio_encoder, system15, system1), frozen / swapped /
ablated from the yaml:
  * stage-wise training = framework.tri_system.loss_weights (0 = loss not computed) + trainer.freeze_modules
  * System 1 dZ condition: system1.delta_source = gt (teacher forcing) | pred | pred_grad | mix | zero
  * system15.target (delta | future), system1.cond_streams / cond_mode,
    radio.feature (backbone | penultimate | <teacher head>)

Data: each example's "image" list is [frame t, frame t + o_1, ..., frame t + o_K] (data config video
delta_indices [0, *framework.tri_system.future_offsets]), plus "lang", "state" [1, D_st] and "action"
[H, D_a]. The dZ target is the last frame; System 2 and System 1 only ever see frame t.
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig, Qwen_GR00T
from starVLA.model.modules.tri_system.radio_encoder import PatchFeatureStats, RadioPatchEncoder
from starVLA.model.modules.tri_system.shapes import PatchState, ShapeChecker, System2Tokens, TriSystemOutput
from starVLA.model.modules.tri_system.system1_action import ActionExpert
from starVLA.model.modules.tri_system.system15_flow import LatentChangePredictor
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@dataclass
class QwenGR00TTriSystemDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_TriSystem"
    # NOTE: give every key the model reads a default here. The run's config.yaml (what from_pretrained rebuilds
    # from) only records keys seen by this default merge; a yaml-only key read later is silently dropped.
    tri_system: dict = field(default_factory=lambda: {
        "frame_size": [368, 640],  # (H, W) of the loader frames (System 2 input); == obs_image_size [W, H]
        "future_offsets": [30],  # frames after t, == data config video delta_indices[1:]; dZ target = last
        # Stage-wise training: a loss with weight 0 is not computed (and its module should be frozen).
        "loss_weights": {"system15": 1.0, "system1": 1.0},
        "check_shapes": True,
        # eval hook: metrics over num_samples fixed val examples, PNG grid of the first grid_samples of them
        "eval": {"num_samples": 8, "grid_samples": 8, "system15_steps": 4, "system1_steps": 4, "seed": 0},
    })
    radio: dict = field(default_factory=lambda: {
        "path": "playground/Pretrained_models/C-RADIOv4-SO400M",
        "teacher_ckpt": None,  # *_half.pth.tar, only for feature = siglip2-g / dino_v3_7b / sam3
        "feature": "backbone",
        "input_size": [256, 256],  # (H, W) the frames are resized to for C-RADIO (multiple of 16) -> 16x16 tokens
        "outlier_norm": 100.0,
        "mask_outliers": True,
        "dtype": "bfloat16",
        "stats_path": None,  # from starVLA/training/trisystem_prepare_stats.py
    })
    system15: dict = field(default_factory=lambda: {
        "target": "delta", "d_model": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0,
        "tau_schedule": {"schedule": "uniform"},  # tau ~ U(0, 1)
        "num_noise_samples": 1, "s2_drop_prob": 0.0,  # > 0 trains an S_t-free branch, needed only for cfg_scale > 1
        "sample_steps": 4, "sample_method": "euler", "cfg_scale": 1.0, "grad_ckpt": False,
    })
    system1: dict = field(default_factory=lambda: {
        "d_model": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0,
        "cond_streams": ["dz"],  # cross-attention streams; options "s2" (S_t), "zt" (Z_t), "zf" (Z_t + dZ)
        "cond_mode": "separate",  # separate: one cross-attention per stream | joint: one over all streams
        "tau_schedule": {"schedule": "beta", "beta_a": 1.5, "beta_b": 1.0, "beta_s": 0.999},  # GR00T
        "num_noise_samples": 4,  # GR00T repeated_diffusion_steps
        "sample_steps": 4, "sample_method": "euler",
        "delta_source": "gt",  # training-time dZ condition: gt (teacher forcing) | pred | pred_grad | mix | zero
        "delta_noise_std": 0.0,  # Gaussian noise on the GT dZ condition (scaled units)
        "pred_prob": 0.5, "pred_steps": 4,  # mix probability / ODE steps of the System 1.5 sample
        "pred_grad_scale": 1.0,  # pred_grad: multiplier on the action-loss gradient that reaches System 1.5
        "delta_drop_prob": 0.0,  # replace the change streams by learned null tokens with this probability
        "grad_ckpt": False,
    })


def _as_list(images):
    return list(images) if isinstance(images, (list, tuple)) else [images]


class _GradScale(torch.autograd.Function):
    """Identity forward; multiplies the incoming gradient by `scale` (balances how strongly the System 1 action
    loss pushes System 1.5 in delta_source: pred_grad)."""

    @staticmethod
    def forward(ctx, x, scale: float):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


@FRAMEWORK_REGISTRY.register("QwenGR00T_TriSystem")
class Qwen_GR00T_TriSystem(Qwen_GR00T):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        baseframework.__init__(self)  # not Qwen_GR00T.__init__: System 1 replaces its VLM-conditioned head
        self.config = merge_framework_config(QwenGR00TTriSystemDefaultConfig, config)
        fw = self.config.framework
        tri = fw.tri_system
        self.frame_size = tuple(int(v) for v in tri.frame_size)
        self.future_offsets = [int(v) for v in tri.future_offsets]
        self.loss_weights = {k: float(v) for k, v in dict(tri.loss_weights).items()}
        self.shapes = ShapeChecker(bool(tri.get("check_shapes", True)))

        # System 2: QwenGR00T's VLM, unmodified
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.s2_dim = int(self.qwen_vl_interface.model.config.hidden_size)

        # visual world state: C-RADIOv4 patch tokens
        rc = fw.radio
        self.radio_encoder = RadioPatchEncoder(
            rc.path, feature=rc.get("feature", "backbone"), input_size=tuple(rc.get("input_size", self.frame_size)),
            outlier_norm=float(rc.get("outlier_norm", 100.0)), mask_outliers=bool(rc.get("mask_outliers", True)),
            dtype=rc.get("dtype", "bfloat16"), teacher_ckpt=rc.get("teacher_ckpt", None),
        )
        self.feat_dim = self.radio_encoder.feat_dim
        self.radio_grid = self.radio_encoder.grid
        self.feature_stats = PatchFeatureStats(self.feat_dim, rc.get("stats_path", None))

        # System 1.5 (latent change world model) and System 1 (action expert)
        self.system15 = LatentChangePredictor(self.feat_dim, self.s2_dim, self.radio_grid, fw.system15)
        am = fw.action_model  # action / state dims and chunk length keep QwenGR00T's keys
        self.action_horizon, self.action_dim = int(am.action_horizon), int(am.action_dim)
        self.state_dim = int(am.get("state_dim", 0) or 0)
        self.system1 = ActionExpert(self.action_dim, self.state_dim, self.action_horizon, self.feat_dim, self.s2_dim,
                                    self.radio_grid, fw.system1)
        self.use_action_head = True
        self.delta_source = str(fw.system1.get("delta_source", "gt"))
        assert self.delta_source in ("gt", "pred", "pred_grad", "mix", "zero"), self.delta_source
        self._eval_examples = None
        logger.info(self.describe_shapes())

    # ------------------------------------------------------------------ shapes
    def describe_shapes(self) -> str:
        (gh, gw), n = self.radio_grid, self.radio_grid[0] * self.radio_grid[1]
        net = self.system1.net
        return "\n".join([
            "QwenGR00T_TriSystem tensor shapes:",
            f"  frames             [B, 1 + {len(self.future_offsets)}, 3, {self.frame_size[0]}, {self.frame_size[1]}]  (t, t+{self.future_offsets})",
            f"  S_t (System 2)     [B, L_s, {self.s2_dim}]  L_s = all prompt tokens (vision + text), last hidden states",
            f"  Z_t, Z_f, dZ       [B, {n}, {self.feat_dim}]  ({gh}x{gw} C-RADIO patches of the {self.radio_encoder.input_size[0]}x"
            f"{self.radio_encoder.input_size[1]} resized frame, {self.radio_encoder.feature})",
            f"  System 1.5         X_tau [B, {n}, {self.feat_dim}] -> d={self.system15.net.x_proj.out_features} -> v [B, {n}, {self.feat_dim}]",
            f"  System 1           tokens [B, {net.n_prefix} + {self.action_horizon}, {net.action_in.out_features}]  "
            f"cross {list(net.streams)} ({net.cond_mode})",
            f"  actions            [B, {self.action_horizon}, {self.action_dim}]  state [B, 1, {self.state_dim}]",
        ])

    def _progress_of(self, examples):
        return None  # the action expert is not progress-conditioned

    # ------------------------------------------------------------------ encoders
    @property
    def _device(self):
        return self.feature_stats.z_mean.device

    def _frames(self, examples: List[dict], n_expected: Optional[int] = None) -> torch.Tensor:
        """examples' "image" lists -> [B, T, 3, H, W] float in [0, 1] on the model device."""
        h, w = self.frame_size
        batch = []
        for ex in examples:
            ims = _as_list(to_pil_preserve(ex["image"]))
            if n_expected is not None and len(ims) != n_expected:
                raise ValueError(f"expected {n_expected} frames per example (t, t+{self.future_offsets}), got "
                                 f"{len(ims)}; use a *_future data config")
            batch.append(np.stack([np.asarray(im.convert("RGB") if im.size == (w, h) else im.convert("RGB").resize((w, h)))
                                   for im in ims]))
        x = torch.from_numpy(np.stack(batch)).to(self._device, non_blocking=True)
        return x.permute(0, 1, 4, 2, 3).float().div_(255.0)

    def encode_system2(self, images: List[List[Image.Image]], instructions: List[str]) -> System2Tokens:
        """System 2: the unmodified Qwen VLM reads the current image + instruction in one forward pass (exactly
        QwenGR00T's); S_t = last hidden states of every token (vision + text) [B, L_s, D_s], padding masked.
        Gradient flows into the VLM only if it is trainable (not in trainer.freeze_modules)."""
        iface = self.qwen_vl_interface
        qwen_inputs = iface.build_qwenvl_inputs(images=images, instructions=instructions)
        with torch.autocast("cuda", dtype=getattr(iface, "autocast_dtype", torch.bfloat16)):
            out = iface(**qwen_inputs, output_attentions=False, output_hidden_states=True, return_dict=True,
                        logits_to_keep=1)  # hidden states only; skips the full-vocabulary logits
        tokens = self.shapes("S_t", out.hidden_states[-1], "B L_s D_s")
        return System2Tokens(tokens, qwen_inputs["attention_mask"].bool())

    def encode_patches(self, frames: torch.Tensor, with_grad: bool) -> PatchState:
        """frames [M, 3, H, W] -> normalized C-RADIO patch tokens [M, N, D_c] + valid mask [M, N]."""
        with torch.set_grad_enabled(with_grad and torch.is_grad_enabled()):
            z, valid = self.radio_encoder(frames)
        return PatchState(self.feature_stats.normalize(z, valid), valid, self.radio_grid)

    def _radio_trainable(self) -> bool:
        return any(p.requires_grad for p in self.radio_encoder.parameters())

    def _current_and_future(self, cur, fut):
        """-> (Z_t [B, N, D_c], Z_f [B, N, D_c], valid [B, N]). Z_f is a stop-gradient target."""
        if self._radio_trainable():
            zt = self.encode_patches(cur, with_grad=True)
            zf = self.encode_patches(fut, with_grad=False)
            z_t, z_f, v_t, v_f = zt.z, zf.z, zt.valid, zf.valid
        else:
            both = self.encode_patches(torch.cat([cur, fut]), with_grad=False)
            (z_t, z_f), (v_t, v_f) = both.z.chunk(2), both.valid.chunk(2)
        valid = v_t & v_f
        z_t = self.shapes("Z_t", z_t.masked_fill(~valid[..., None], 0.0), "B N D_c")
        z_f = self.shapes("Z_f", z_f.masked_fill(~valid[..., None], 0.0), "B N D_c")
        return z_t, z_f, valid

    def _system1_conds(self, s2: Optional[System2Tokens], z_t, dz_scaled, valid):
        """(tokens, mask) per System 1 cross-attention stream. dz_scaled = dZ / dz_rms (unit RMS)."""
        conds = {"zt": (z_t, valid), "dz": (dz_scaled, valid)}
        if s2 is not None:
            conds["s2"] = (s2.tokens, s2.mask)
        if "zf" in self.system1.streams:  # predicted future tokens Z_t + dZ (normalized-feature units)
            conds["zf"] = ((z_t + dz_scaled * self.feature_stats.dz_rms).masked_fill(~valid[..., None], 0.0), valid)
        return conds

    def _sample_delta(self, z_t, valid, s2: System2Tokens, steps=None, cfg_scale=None) -> torch.Tensor:
        """System 1.5 ODE sample, dZ_hat / dz_rms [B, N, D_c]."""
        dz = self.system15.sample(z_t, valid, s2.tokens, s2.mask, self.feature_stats.dz_rms, steps=steps, cfg_scale=cfg_scale)
        return self.shapes("dZ_hat", dz, "B N D_c")

    def _system1_delta(self, z_t, dz, valid, s2):
        """Training-time dZ condition of System 1: GT (teacher forcing) or System 1.5's prediction."""
        src, s1 = self.delta_source, self.config.framework.system1
        if src == "zero":
            return torch.zeros_like(dz)
        if src == "gt":
            std = float(s1.get("delta_noise_std", 0.0))
            return dz + std * torch.randn_like(dz) * valid[..., None] if std > 0 else dz
        steps = int(s1.get("pred_steps", 4))
        if src == "pred_grad":  # end-to-end: the action loss reaches System 1.5 through the ODE solver
            dz_hat = self._sample_delta(z_t, valid, s2, steps=steps)
            scale = float(s1.get("pred_grad_scale", 1.0))
            return dz_hat if scale == 1.0 else _GradScale.apply(dz_hat, scale)
        with torch.no_grad():
            pred = self._sample_delta(z_t, valid, s2, steps=steps)
        if src == "pred":
            return pred
        use_pred = torch.rand(dz.shape[0], device=dz.device) < float(s1.get("pred_prob", 0.5))
        return torch.where(use_pred[:, None, None], pred, dz)

    def _state_actions(self, examples, device):
        state = actions = None
        if self.state_dim and "state" not in examples[0]:
            raise KeyError(f"System 1 is state-conditioned (state_dim {self.state_dim}): every example needs 'state' "
                           "[1, D_st] (datasets.*.include_state: true)")
        if self.state_dim:
            state = self.shapes("state", torch.tensor(np.array([ex["state"] for ex in examples]), device=device,
                                                      dtype=torch.float32), "B 1 D_st")
        if "action" in examples[0]:
            actions = torch.tensor(np.array([ex["action"] for ex in examples]), device=device, dtype=torch.float32)
        return state, actions

    # ------------------------------------------------------------------ training
    def forward(self, examples: List[dict] = None, **kwargs) -> Dict:
        if isinstance(examples, dict):
            raise NotImplementedError("QwenGR00T_TriSystem builds its own inputs; preprocess_in_workers is not supported")
        t0 = time.perf_counter()
        w = self.loss_weights
        need15, need1 = w.get("system15", 0) > 0, w.get("system1", 0) > 0
        # S_t feeds System 1.5; System 1 needs it only to sample dZ_hat or as its own "s2" stream
        need_s2 = need15 or (need1 and (self.delta_source in ("pred", "pred_grad", "mix") or "s2" in self.system1.streams))
        self.shapes.reset()

        frames = self.shapes("frames", self._frames(examples, 1 + len(self.future_offsets)), "B T 3 H W")
        instructions = [ex["lang"] for ex in examples]
        current_pils = [_as_list(to_pil_preserve(ex["image"]))[:1] for ex in examples]
        out: Dict = {}

        s2 = None
        if need_s2:
            t = time.perf_counter()
            s2 = self.encode_system2(current_pils, instructions)
            out["timing/system2"] = time.perf_counter() - t
        t = time.perf_counter()
        z_t, z_f, valid = self._current_and_future(frames[:, 0], frames[:, -1])
        # GT dZ, unit RMS. Stop-gradient target: with a trainable C-RADIO, Z_t gets gradient only as a condition.
        dz = ((z_f - z_t.detach()) / self.feature_stats.dz_rms).masked_fill(~valid[..., None], 0.0)
        out["timing/radio"] = time.perf_counter() - t

        total = 0.0
        if need15:
            t = time.perf_counter()
            loss15, logs = self.system15.loss(z_t, dz, valid, s2.tokens, s2.mask, self.feature_stats.dz_rms)
            out.update(logs)
            out["timing/system15"] = time.perf_counter() - t
            total = total + w["system15"] * loss15
        if need1:
            t = time.perf_counter()
            state, actions = self._state_actions(examples, z_t.device)
            self.shapes("actions", actions[:, -self.action_horizon:], f"B {self.action_horizon} {self.action_dim}")
            conds = self._system1_conds(s2, z_t, self._system1_delta(z_t, dz, valid, s2), valid)
            loss1, logs = self.system1.loss(actions, state, conds)
            out.update(logs)
            out["action_loss"] = loss1  # logged by the trainer as action_dit_loss
            out["timing/system1"] = time.perf_counter() - t
            total = total + w["system1"] * loss1
        if not torch.is_tensor(total):
            raise ValueError("framework.tri_system.loss_weights: every loss is 0 / disabled")
        out["loss"] = total
        out["timing/forward"] = time.perf_counter() - t0
        out["outlier_frac"] = float((~valid).float().mean())
        return out

    # ------------------------------------------------------------------ inference
    def _current_inputs(self, examples):
        """Current frame (first image) of every example at the training frame size (= obs_image_size)."""
        pils = [_as_list(to_pil_preserve(ex["image"]))[:1] for ex in examples]
        return resize_images(pils, target_size=(self.frame_size[1], self.frame_size[0]))

    @torch.no_grad()
    def predict_future(self, examples: List[dict], system15_steps: Optional[int] = None,
                       system1_steps: Optional[int] = None, system15_cfg: Optional[float] = None,
                       delta_override: Optional[torch.Tensor] = None, predict_actions: bool = True) -> TriSystemOutput:
        """image + instruction -> [S_t, Z_t] -> System 1.5 ODE -> dZ_hat -> System 1 ODE -> action chunk.

        delta_override [B, N, D_c] (scaled units dZ / dz_rms, e.g. the GT change) replaces dZ_hat for System 1.
        Runs under bf16 autocast like training, so fp32 and bf16 (policy server use_bf16) weights both work."""
        examples = examples if isinstance(examples, list) else [examples]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
            return self._predict_future(examples, system15_steps, system1_steps, system15_cfg, delta_override, predict_actions)

    def _predict_future(self, examples, system15_steps, system1_steps, system15_cfg, delta_override, predict_actions):
        self.shapes.reset()
        current = self._current_inputs(examples)
        cur = self.shapes("current", self._frames([{"image": im} for im in current])[:, 0], "B 3 H W")
        s2 = self.encode_system2(current, [ex["lang"] for ex in examples])
        zt = self.encode_patches(cur, with_grad=False)
        dz = self._sample_delta(zt.z, zt.valid, s2, steps=system15_steps, cfg_scale=system15_cfg)
        actions = None
        if predict_actions:
            state, _ = self._state_actions(examples, zt.z.device)
            conds = self._system1_conds(s2, zt.z, dz if delta_override is None else delta_override, zt.valid)
            actions = self.shapes("A", self.system1.sample(state, conds, steps=system1_steps), "B H_a D_a")
        delta = dz * self.feature_stats.dz_rms  # normalized-feature units
        return TriSystemOutput(s2, zt, delta, zt.z + delta, actions)

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> Dict:
        """QwenGR00T-compatible action inference (current frame only): (S_t, Z_t) -> dZ_hat -> System 1."""
        out = self.predict_future(examples if isinstance(examples, list) else [examples])
        return {"normalized_actions": out.actions.float().cpu().numpy()}

    # ------------------------------------------------------------------ validation
    def evaluate_batch(self, examples: List[dict], seed: int = 0) -> Dict:
        """Seeded evaluation that leaves the global (training) RNG state untouched."""
        devices = [self._device] if self._device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad(), \
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
            torch.manual_seed(seed)
            return self._evaluate_batch(examples)

    def _evaluate_batch(self, examples: List[dict]) -> Dict:
        """Metrics on examples that carry the future frame and the action chunk, plus a visual grid.

        system15/rel_mse : MSE(dZ_hat, dZ) / MSE(0, dZ)  (< 1 = beats "nothing changes")
        system15/cos     : cosine(dZ_hat, dZ) over valid tokens, per sample
        system1/rmse_*   : normalized-action RMSE with dZ_hat (pred), with the GT dZ (gt_delta) and with the change
                           streams replaced by the learned null tokens (no_delta)
        grid rows        : current | future (GT) | |dZ| GT | |dZ_hat|   (per-patch change magnitude)
        """
        ev = self.config.framework.tri_system.get("eval", {})
        s1_steps = ev.get("system1_steps")
        frames = self._frames(examples, 1 + len(self.future_offsets))
        self.shapes.reset()
        z_t, z_f, valid = self._current_and_future(frames[:, 0], frames[:, -1])
        dz_gt = ((z_f - z_t) / self.feature_stats.dz_rms).masked_fill(~valid[..., None], 0.0)
        pred = self.predict_future(examples, system15_steps=ev.get("system15_steps"), system1_steps=s1_steps)
        rms = self.feature_stats.dz_rms
        d_hat, d_gt = pred.delta_z, dz_gt * rms  # normalized-feature units
        v = valid[..., None].float()
        mse_pred = ((d_hat - d_gt).pow(2) * v).sum() / (v.sum() * d_gt.shape[-1])
        mse_zero = (d_gt.pow(2) * v).sum() / (v.sum() * d_gt.shape[-1])
        cos = F.cosine_similarity((d_hat * v).flatten(1), (d_gt * v).flatten(1), dim=1)
        metrics = {"system15/mse": mse_pred.item(), "system15/mse_zero_baseline": mse_zero.item(),
                   "system15/rel_mse": (mse_pred / mse_zero.clamp_min(1e-12)).item(), "system15/cos": cos.mean().item()}
        state, actions = self._state_actions(examples, z_t.device)
        if actions is not None:
            target = actions[:, -self.action_horizon:]
            rmse = lambda a: (a.float() - target).pow(2).mean().sqrt().item()
            conds_gt = self._system1_conds(pred.s2, z_t, dz_gt, valid)
            act_gt = self.system1.sample(state, conds_gt, steps=s1_steps)
            no_delta = torch.ones(target.shape[0], dtype=torch.bool, device=target.device)
            act_none = self.system1.sample(state, conds_gt, steps=s1_steps, drop_delta=no_delta)
            metrics.update({"system1/rmse_pred": rmse(pred.actions), "system1/rmse_gt_delta": rmse(act_gt),
                            "system1/rmse_no_delta": rmse(act_none)})
        h, w = self.frame_size
        gh, gw = self.radio_grid
        n_vis = min(int(ev.get("grid_samples", 8)), frames.shape[0])  # metrics use all examples, the grid n_vis
        mag = lambda d: (d[:n_vis].norm(dim=-1) * valid[:n_vis]).reshape(-1, 1, gh, gw)
        m_gt, m_hat = mag(d_gt), mag(d_hat)
        scale = m_gt.flatten(1).quantile(0.99, dim=1).clamp_min(1e-6)[:, None, None, None]
        heat = lambda m: F.interpolate((m / scale).clamp(0, 1), (h, w), mode="nearest").expand(-1, 3, -1, -1)
        to8 = lambda x: (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        cols = [to8(frames[:n_vis, 0]), to8(frames[:n_vis, -1]), to8(heat(m_gt)), to8(heat(m_hat))]
        grid = np.concatenate([np.concatenate([c[i] for c in cols], 1) for i in range(n_vis)], 0)
        return {"metrics": metrics, "grid": grid,
                "actions_pred": None if pred.actions is None else pred.actions.float().cpu().numpy()}

    def _load_eval_examples(self, get_batch):
        if self._eval_examples is not None:
            return self._eval_examples
        n = int(self.config.framework.tri_system.get("eval", {}).get("num_samples", 8))
        val_cfg = self.config.datasets.get("val_data", None) if hasattr(self.config.datasets, "get") else None
        if val_cfg is not None:
            from starVLA.dataloader.lerobot_datasets import get_vla_dataset

            ds = get_vla_dataset(data_cfg=val_cfg)
            rng = np.random.default_rng(0)
            self._eval_examples = [ds[int(i)] for i in rng.choice(len(ds), size=n, replace=False)]
        else:
            self._eval_examples = list(get_batch())[:n]
        return self._eval_examples

    def trainer_eval_hook(self, get_batch, step: int, output_dir: str, is_main: bool = True) -> Dict[str, float]:
        """Called by the trainer on every rank each eval_interval: fixed validation examples (datasets.val_data,
        else the first train batch; built on all ranks, the dataset constructor has a barrier), then metrics +
        a PNG grid under <output_dir>/samples on the main process."""
        examples = self._load_eval_examples(get_batch)
        if not is_main:
            return {}
        modes = {m: m.training for m in (self.system15, self.system1)}
        for m in modes:
            m.eval()
        try:
            res = self.evaluate_batch(examples, seed=int(self.config.framework.tri_system.get("eval", {}).get("seed", 0)))
        finally:
            for m, was in modes.items():
                m.train(was)
        out_dir = Path(output_dir) / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(res["grid"]).save(out_dir / f"step_{step:07d}.png")
        with open(out_dir / f"step_{step:07d}.json", "w") as f:
            json.dump({"metrics": res["metrics"], "lang": [ex["lang"] for ex in examples]}, f, indent=1)
        return {f"eval/{k}": v for k, v in res["metrics"].items()}
