# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
QwenGR00T-DualSystem: a slow long-horizon keyframe planner (System 2) and a fast short-horizon flow-matching policy
(System 1) on top of QwenGR00T, sharing ONE frozen C-RADIOv4 backbone.

    observation ── frozen C-RADIOv4 (radio_encoder, shared) ── summary [B, 2, D_c] ─────┬─ radio_(prefix_)projector_s2 ──┐
                                                               + patch tokens [B, N, D_c] └─ radio_(prefix_)projector_s1 ─┼┐
    image + instruction ── Qwen VLM (qwen_vl_interface, fully trainable in stage 1) ── last hidden states ──┐      │  │
                                                                     vlm_projector ─────────────────────────┤      │  │
    state (optional) ── state_projector ────────────────────────────────────────────────────────────────────┤      │  │
    System 2  s2_context = concat([vlm, radio prefix, radio, (state)], dim=1) -> keyframe DiT (flow matching)┘      │  │
              -> 61 keyframes [B, 61, D_a] on t_i = -2.0 .. +4.0 s (0.1 s), relative to the System 2 observation ◄─┘  │
              loss = MSE(interp(v_hat) at the dense GT timestamps, dense GT - interp(eps))  (GR00T velocity target) │
    System 1  TrajectoryTokenizer(61 keyframes) -> [B, 61, d1] (past / current / future type embedding)               │
              X = concat([noisy 1 s action tokens, 61 trajectory tokens]) -> full self-attention with temporal RoPE   │
              on timestamps relative to the CURRENT System 1 time (trajectory: t_i - delay), cross-attention to the  ◄┘
              current frame's C-RADIO [prefix, patch] tokens + its state token, adaLN(flow step); GR00T flow matching
              -> next 1 s of actions [B, N_a, D_a]

Stages (framework.dual_system.train_systems + trainer.freeze_modules):
    stage 1  train_systems [system2]: Qwen fully unfrozen + System 2 projectors / DiT; radio_encoder, system1 frozen.
    stage 2  train_systems [system1]: tokenizer + radio_projector_s1 + System 1 DiT; Qwen, System 2 frozen. System 1
             is trained on System 2's own sampled plans (system1.traj_source: pred | gt | mix) with a random plan age
             delay in {video_offsets} / action_rate (the System 1 frame is that many frames after the System 2 frame).
             system1.traj_noise adds structured noise to those plans (System 2 overfits its training episodes, so
             its plans there are far better than at test time; PlanNoise in dual_system/system1.py).

What is stale at inference (dual_system.stale_input; t = the current frame = video offset 0):
    plan (default)  System 2 (VLM + C-RADIO + state) runs on frame t; System 1 runs on frame t + o, o in video_offsets
                    (>= 0), with that plan aged o / rate (AsyncDualSystemRunner: System 2 in a background thread).
    vlm             only the VLM is slow: System 2's DiT sees the VLM tokens of frame t + o, o in video_offsets (<= 0),
                    next to C-RADIO / state of frame t (+ system2.vlm_age_embed of age -o / rate), and System 1 runs on
                    frame t with that fresh plan (plan age 0). AsyncVLMRunner: the VLM in a background thread, System 2's
                    DiT + System 1 on every control step.

Plan space (dual_system.relative_plan): absolute by default; with relative_plan System 2 plans deltas
(a[t + tau] - ref) * scale from a per-dim reference built from the current (augmented) state, System 1 takes that plan
as-is and still predicts absolute actions (plan_to_absolute converts a plan for display). dual_system.delta_plan
instead plans per-step deltas anchored at the current state (d[0] = a[t] - s[t], then a[t+k] - a[t+k-1] forward /
a[t+k] - a[t+k+1] backward, z-normalised; System 1 again takes the plan as-is; validation integrates it).
dual_system.s1_residual: System 1 predicts (a - A_sys2) * scale, A_sys2 = the plan in absolute units on System 1's
window, and outputs A_sys2 + residual / scale (warmup_scale instead for the first warmup_steps trainer steps, via
set_training_step; inference always uses scale). dual_system.
state_prev_action_dims prepends the previous command a[t - 1][dims] to the state (example["prev_action"] at inference).

Data (examples from the LeRobot loader, data config `*_dual*`):
    "image"  [frame(t_S2 + o) for o in dual_system.video_offsets]   (o in frames; video fps == action rate), for
             dual_system.num_views cameras camera-major (all offsets of camera 0, then camera 1, ...; the loader order).
             Every camera's frame goes to Qwen (System 2) and to C-RADIO; the views' C-RADIO tokens are concatenated
             on the token dim (+ a learned per-view embedding) for both systems.
    "action" [Q, D_a]  dense window a[t_S2 + k], k = k_min .. k_max covering [t_min, t_max] s (121 steps at 20 Hz)
    "action_valid" [Q], "frame_valid" [T_v]  (datasets.*.emit_action_valid: true) — False outside the episode
    "state"  [T_v, D_st] at the same offsets (System 2 uses row 0), "lang".
Inference: predict_action (synchronous, plan age 0, one shared C-RADIO pass) for the policy server, or
starVLA.model.modules.dual_system.AsyncDualSystemRunner (System 2 in a background thread) for decoupled control.
"""

import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig, Qwen_GR00T
from starVLA.model.modules.dual_system import (EpisodeMemory, KeyframeTimeline, PlanNoise, SharedRadioBackbone,
                                               System1ActionPolicy, System2KeyframePlanner, VisualFeatures)
from starVLA.model.modules.dual_system.system1 import smooth_time
from starVLA.model.modules.tri_system.shapes import ShapeChecker
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

_SYSTEMS = ("system2", "system1")
_TRAJ_SOURCES = ("pred", "gt", "mix")


@dataclass
class QwenGR00TDualSystemDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_DualSystem"
    # NOTE: every key the model reads has a default here (the run's config.yaml, which from_pretrained rebuilds from,
    # only records keys present in this default merge).
    dual_system: dict = field(default_factory=lambda: {
        "action_rate_hz": 20.0,  # dense action (and video) rate of the dataset; NOT the System 2 call rate
        "keyframes": {"t_min": -2.0, "t_max": 4.0, "dt": 0.1},  # -> 61 keyframes (20 past, 1 current, 40 future)
        "s1_horizon_s": 1.0,  # System 1 predicts round(s1_horizon_s * action_rate_hz) dense actions
        "video_offsets": [0],  # == data config video delta_indices (frames after the System 2 frame, first = 0)
        "num_views": 1,  # cameras per time step (== the data config's video keys), e.g. 3 for RoboTwin head + 2 wrists
        # plan: System 1 frame t + o uses the plan of frame t (video_offsets >= 0, first 0); vlm: System 2 uses the VLM
        # tokens of frame t + o and C-RADIO / state of frame t, System 1 frame t (video_offsets <= 0, last 0)
        "stale_input": "plan",
        "s1_offset_probs": None,  # sampling probabilities of video_offsets for the stale input's frame (None = uniform)
        "frame_size": [256, 256],  # (H, W) of the loader frames (== datasets.*.obs_image_size [W, H]); Qwen input
        "train_systems": ["system2"],  # stage 1: [system2] | stage 2: [system1] | joint: [system2, system1]
        "loss_weights": {"system2": 1.0, "system1": 1.0},
        "mask_out_of_episode": True,  # drop dense targets whose index falls outside the episode (needs action_valid)
        # previous command a[t - 1][dims] (action-normalised) prepended to the loader state of both systems
        # (framework.action_model.state_dim = loader state dim + len(dims)); training / validation take it from the
        # dense action window, inference from example["prev_action"]. [] = off
        "state_prev_action_dims": [],
        # System 2 plans DELTAS: keyframes = (a[t + tau] - ref) * scale, ref[d] = a slot of the (augmented) current
        # state in action-normalised units; System 1 takes that plan as-is and still predicts ABSOLUTE actions.
        # {stats_path: dataset_statistics-format json (loader order: state / action q01, q99 of the train split),
        #  ref: [[action_start, action_end, state_start], ...] covering every action dim (state index into the
        #  augmented state: the first len(state_prev_action_dims) slots are the previous command, already
        #  action-normalised), scale: 1.0}. None = absolute plan
        "relative_plan": None,
        # System 2 plans PER-STEP DELTAS of the ref dims (action-rate steps, anchored at the current state s[t]):
        # d[0] = a[t] - s[t], d[k] = a[t+k] - a[t+k-1] (k > 0), d[k] = a[t+k] - a[t+k+1] (k < 0), z-normalised per
        # window slot and clipped at +-clip; the other dims stay absolute. System 1 takes that plan as-is and still
        # predicts ABSOLUTE actions. {stats_path: s2_delta_stats.py json (anchor / step mean, std; the loader's
        # min_max action_min / action_max; k_range == the dense window), ref: [[action_start, action_end,
        # sin_start, cos_start], ...] (s[t] = atan2 of the loader's sin / cos state slots), clip: 10.0, min_std: 1e-6
        # (floor of the z-normalisation std: a rarely moving dim otherwise turns its few jumps into 100+ sigma)}. None = off
        "delta_plan": None,
        # System 1 predicts the RESIDUAL of System 2's plan on its window: target (a - A_sys2) * scale, A_sys2 = the plan
        # as absolute normalised actions at the System 1 times (plan_to_absolute of the interpolated plan); output
        # A_sys2 + prediction / scale. {scale: 1.0, warmup_scale: None, warmup_steps: 0} (warmup_scale for trainer steps
    # < warmup_steps, e.g. while System 2's plan and so the residual are still large). None = System 1 predicts the
    # actions directly
        "s1_residual": None,
        # stale_input vlm only: recurrent episode memory over the VLM's last hidden states (dual_system/memory.py),
        # its 16 tokens + a progress token appended to System 2's cross-attention context. Training unrolls it from the
        # initial memory over the VLM frames at update_offsets (video offsets < 0, oldest first, e.g. every 0.5 s over
        # the last 2 s; frames outside the episode skip their update); deployment updates it on every new VLM frame
        # (memory_init / memory_update / plan_keyframes(memory=...)). Progress = t / (T - 1) (loader emit_progress)
        # classified into progress_bins bins (cross-entropy x progress_loss_weight, trained with System 2).
        # {update_offsets: [...], num_tokens: 16, d_model: 768, depth: 2, num_heads: 12, mlp_ratio: 4.0,
        #  progress_bins: 50, progress_loss_weight: 0.1, stream: None, attention: standard}. None = off
        # attention: standard (nn.MultiheadAttention, what the memory checkpoints up to 10-19 use) | gated (sigmoid
        # output gate per head channel from the memory tokens, as in Qwen3.5's full-attention layers)
        # stream: {tbptt: 4} trains the memory as deployed: the loader streams episodes from their start every
        # update interval (datasets.vla_data.stream, sample["stream"]), each batch position carries its memory across
        # training steps (continuous updates from the episode start, one new VLM frame per step = System 2's frame
        # t - stride, update_offsets == [that offset]); gradient through the last tbptt updates (truncated BPTT over
        # cached VLM tokens, the older memory detached); samples whose stream lost its history (first episode joined
        # mid-way, out-of-order batches) are masked from every loss until their next episode starts. Eval: continuous
        # memory over whole val episodes (eval.stream_episodes). Needs a frozen VLM.
        "memory": None,
        # train only the LAST n decoder layers of the VLM's language model (+ its final norm), cast to float32 (the
        # ZeRO-0 path keeps no fp32 master copy; e.g. Qwen3-VL loads bf16); everything else in qwen_vl_interface frozen
        # (vision tower, embeddings, the other layers). trainer.freeze_modules must then NOT list qwen_vl_interface.
        # None = leave qwen_vl_interface as loaded (trainer.freeze_modules decides)
        "vlm_train_last_layers": None,
        "check_shapes": True,
        # None = the systems' diffusion.sample_steps; s1_num_samples: predict_action / act_from_plan output the MEAN of
        # that many System 1 samples on the same plan (env STARVLA_DUAL_S1_NUM_SAMPLES overrides it, e.g. a sim eval);
        # action_source "s2_plan" = ablation: output System 2's plan itself on the System 1 window (A_sys2, System 1
        # unused; env STARVLA_DUAL_ACTION_SOURCE overrides it)
        "inference": {"s2_steps": None, "s1_steps": None, "s1_num_samples": 1, "action_source": "s1"},
        # train_stats: normalise the validation split with the TRAIN split's statistics (as the model sees data at
        # deploy); False = the val split's own statistics (what get_vla_dataset(val_data) gives; older runs' numbers)
        "eval": {"num_samples": 64, "seed": 0, "s2_steps": None, "s1_steps": None, "train_stats": False},
    })
    radio: dict = field(default_factory=lambda: {
        "path": "playground/Pretrained_models/C-RADIOv4-SO400M",  # smallest released C-RADIOv4
        "teacher_ckpt": None,
        "feature": "backbone",
        "input_size": [256, 256],  # (H, W), multiple of 16 -> 16 x 16 spatial tokens
        "outlier_norm": 100.0,
        "mask_outliers": True,
        "dtype": "bfloat16",
        # prefix tokens given to both systems: summary (default: SigLIP2-g + DINOv3 CLS = RADIO's 2304-d summary as
        # 2 x 1152 tokens) | cls (3 CLS, + the near-constant SAM3 CLS) | all (3 CLS + 7 registers)
        "prefix_tokens": "summary",
        # False = frozen (default); True = fine-tune C-RADIO (patch embedder + position embedding + all blocks), n = only
        # its last n blocks; both systems' losses reach it. trainer.freeze_modules must then NOT list radio_encoder;
        # its lr is trainer.learning_rate.radio_encoder (else base)
        "trainable": False,
    })
    system2: dict = field(default_factory=lambda: {
        "d_model": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0, "qk_norm": True,
        "rope_time_scale": 20.0, "rope_base": 1000.0, "abs_time_embed": True, "grad_ckpt": False,
        "use_state": True, "num_noise_samples": 1,
        "use_radio_prefix": True,  # C-RADIO summary tokens (radio.prefix_tokens) in s2_context
        "vlm_age_embed": False,  # add an embedding of the VLM tokens' age (dual_system.stale_input vlm) to them
        # GR00T-style flow matching (type flow_matching: beta_*, noise_s, num_timestep_buckets, Euler sample_steps);
        # type x0 = x0-prediction DDPM / DDIM (num_train_timesteps, beta_schedule, beta_start / end, eta, clip_x0)
        "diffusion": {"type": "flow_matching", "beta_alpha": 1.5, "beta_beta": 1.0, "noise_s": 0.999,
                      "num_timestep_buckets": 1000, "sample_steps": 4,
                      "num_train_timesteps": 100, "beta_schedule": "squaredcos_cap_v2", "beta_start": 1e-4,
                      "beta_end": 0.02, "eta": 0.0, "clip_x0": None},
        # per action-dim interpolation, e.g. [{dims: [6, 7], mode: step}] for a binary gripper
        "interpolation": {"default_mode": "linear", "groups": [], "strict": True},
    })
    system1: dict = field(default_factory=lambda: {
        "d_model": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0, "qk_norm": True,
        "rope_time_scale": 20.0, "rope_base": 1000.0, "abs_time_embed": True, "grad_ckpt": False,
        "use_state": True,  # robot state at the System 1 time as one token
        "num_noise_samples": 1,
        "use_radio_prefix": True,  # current frame's C-RADIO summary tokens in the cross-attention context
        # where the plan / the state token go: self-attention sequence or cross-attention memory ("condition").
        # Default = [actions, plan] attend to [C-RADIO, state]; plan_input condition + state_input sequence =
        # [actions, state] attend to [plan (RoPE'd at its plan time), C-RADIO]
        # plan_input action (diffusion.type regression): no trajectory tokens; the first plan_chunk_s seconds of the plan
        # on System 1's window, Gaussian-smoothed in time (plan_smooth_sigma_s), are the action-token input
        "plan_input": "sequence", "state_input": "condition", "plan_chunk_s": 1.0, "plan_smooth_sigma_s": 0.0,
        # GR00T-style flow matching (type flow_matching: beta_*, noise_s, num_timestep_buckets, Euler sample_steps);
        # type x0 = x0-prediction DDPM / DDIM (num_train_timesteps, beta_schedule, beta_start / end, eta, clip_x0);
        # type regression = direct MSE regression (zero action input, one forward pass)
        "diffusion": {"type": "flow_matching", "beta_alpha": 1.5, "beta_beta": 1.0, "noise_s": 0.999,
                      "num_timestep_buckets": 1000, "sample_steps": 4,
                      "num_train_timesteps": 100, "beta_schedule": "squaredcos_cap_v2", "beta_start": 1e-4,
                      "beta_end": 0.02, "eta": 0.0, "clip_x0": None},
        # which keyframes System 1 is trained on: System 2's sampled plan (pred, as at test time), the GT keyframes
        # (gt, + traj_gt_noise_std) or a per-sample mix (traj_pred_prob of pred)
        "traj_source": "pred", "traj_gt_noise_std": 0.0, "traj_pred_prob": 0.5, "traj_pred_steps": None,
        # structured noise on System 2's sampled (pred) plan, modelled on its held-out plan error (PlanNoise in
        # dual_system/system1.py); scale_max 0 = off
        "traj_noise": {"scale_max": 0.0, "prob": 1.0, "white_frac": 0.35, "rho": 0.91, "std": 0.1,
                       "time_std": None, "dim_std": None, "shared_dims": []},
        # training-only keyframe token dropout: each trajectory token -> a learned mask token (type embedding and
        # plan time kept) with this probability; 0 = off
        "traj_token_dropout": 0.0,
        # state dims (indices into the loader's state vector) hidden from System 1 only (zeroed); [] = all
        "state_drop_dims": [],
    })


def _as_list(images):
    return list(images) if isinstance(images, (list, tuple)) else [images]


@FRAMEWORK_REGISTRY.register("QwenGR00T_DualSystem")
class Qwen_GR00T_DualSystem(Qwen_GR00T):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        baseframework.__init__(self)  # not Qwen_GR00T.__init__: the GR00T head is replaced by the two systems
        self.config = merge_framework_config(QwenGR00TDualSystemDefaultConfig, config)
        fw = self.config.framework
        ds = fw.dual_system

        # ---- time layout
        self.rate = float(ds.action_rate_hz)
        kc = ds.keyframes
        self.timeline = KeyframeTimeline(float(kc.t_min), float(kc.t_max), float(kc.dt))
        self.k_min = math.floor(self.timeline.t_min * self.rate + 1e-6)
        self.k_max = math.ceil(self.timeline.t_max * self.rate - 1e-6)
        self.window_len = self.k_max - self.k_min + 1
        self.s1_horizon = max(1, int(round(float(ds.s1_horizon_s) * self.rate)))
        self.video_offsets = [int(v) for v in ds.video_offsets]
        self.stale_input = str(ds.get("stale_input", "plan"))
        if self.stale_input not in ("plan", "vlm"):
            raise ValueError(f"dual_system.stale_input must be plan or vlm, got {self.stale_input!r}")
        end = 0 if self.stale_input == "plan" else -1
        if not self.video_offsets or self.video_offsets[end] != 0 or sorted(set(self.video_offsets)) != self.video_offsets:
            raise ValueError(f"dual_system.video_offsets must be ascending, unique and {'start' if end == 0 else 'end'} with 0 "
                             f"(stale_input {self.stale_input}), got {self.video_offsets}")
        self.cur = self.video_offsets.index(0)  # index of the current frame t (C-RADIO / state / dense window reference)
        if self.video_offsets[-1] + self.s1_horizon - 1 > self.k_max:
            raise ValueError(f"System 1 window of offset {self.video_offsets[-1]} exceeds the action window end k_max={self.k_max}")
        probs = ds.get("s1_offset_probs", None)
        self.s1_offset_probs = None if probs is None else torch.tensor([float(p) for p in probs])
        if self.s1_offset_probs is not None and self.s1_offset_probs.numel() != len(self.video_offsets):
            raise ValueError("dual_system.s1_offset_probs must have one entry per video offset")
        self.frame_size = tuple(int(v) for v in ds.frame_size)
        self.num_views = int(ds.get("num_views", 1))
        if self.num_views < 1:
            raise ValueError(f"dual_system.num_views must be >= 1, got {self.num_views}")
        self.train_systems = [str(s) for s in ds.train_systems]
        if not self.train_systems or not set(self.train_systems) <= set(_SYSTEMS):
            raise ValueError(f"dual_system.train_systems must be a non-empty subset of {_SYSTEMS}, got {self.train_systems}")
        self.loss_weights = {k: float(v) for k, v in dict(ds.loss_weights).items()}
        self.mask_out_of_episode = bool(ds.get("mask_out_of_episode", True))
        # read here so the run's config.yaml (accessed keys only) records them for from_pretrained
        self.inference_steps = {k: ds.inference.get(k, None) for k in ("s2_steps", "s1_steps")}
        self.s1_num_samples = int(os.environ.get("STARVLA_DUAL_S1_NUM_SAMPLES", "") or ds.inference.get("s1_num_samples", 1) or 1)
        if self.s1_num_samples > 1:
            warnings.warn(f"QwenGR00T_DualSystem: System 1 outputs the mean of {self.s1_num_samples} samples per call")
        self.s1_iters = int(os.environ.get("STARVLA_DUAL_S1_ITERS", "") or ds.inference.get("s1_iters", 1) or 1)
        if self.s1_iters > 1:
            warnings.warn(f"QwenGR00T_DualSystem: System 1 runs {self.s1_iters} times per call, each pass on the plan with its "
                          "in-window keyframes replaced by the previous pass's chunk")
        self.action_source = str(os.environ.get("STARVLA_DUAL_ACTION_SOURCE", "") or ds.inference.get("action_source", "s1") or "s1")
        if self.action_source not in ("s1", "s2_plan"):
            raise ValueError(f"dual_system.inference.action_source must be 's1' or 's2_plan', got {self.action_source!r}")
        if self.action_source == "s2_plan":
            warnings.warn("QwenGR00T_DualSystem: actions = System 2's plan on the System 1 window (System 1 unused)")
        self.eval_cfg = {k: ds.eval.get(k, None) for k in ("num_samples", "seed", "s2_steps", "s1_steps", "train_stats", "stream_episodes")}
        self.check_shapes = bool(ds.get("check_shapes", True))
        dense_times = torch.arange(self.k_min, self.k_max + 1, dtype=torch.float64) / self.rate
        self.register_buffer("dense_times", dense_times.float(), persistent=False)  # [Q], System 2 reference
        eps = 1e-6
        self.register_buffer("dense_in_span", (dense_times >= self.timeline.t_min - eps) & (dense_times <= self.timeline.t_max + eps),
                             persistent=False)

        # ---- modules
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.vlm_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        if ds.get("vlm_train_last_layers", None) is not None:
            self._train_vlm_last_layers(int(ds.vlm_train_last_layers))
        rc = fw.radio
        self.radio_encoder = SharedRadioBackbone(
            rc.path, input_size=tuple(rc.get("input_size", [256, 256])), feature=rc.get("feature", "backbone"),
            outlier_norm=float(rc.get("outlier_norm", 100.0)), mask_outliers=bool(rc.get("mask_outliers", True)),
            dtype=rc.get("dtype", "bfloat16"), teacher_ckpt=rc.get("teacher_ckpt", None),
            prefix_tokens=str(rc.get("prefix_tokens", "all")), trainable=rc.get("trainable", False),
        )
        if self.radio_encoder.trainable:
            tr = self.config.get("trainer", None)
            frozen = str(tr.get("freeze_modules", "") or "") if tr is not None and hasattr(tr, "get") else ""
            if "radio_encoder" in [p.strip() for p in frozen.split(",")]:
                raise ValueError("radio.trainable is set but trainer.freeze_modules lists radio_encoder "
                                 "(the trainer would freeze C-RADIO again)")
            n_tr = sum(p.numel() for p in self.radio_encoder.parameters() if p.requires_grad)
            logger.info(f"[DualSystem] C-RADIO {self.radio_encoder.trainable_desc}, {n_tr / 1e6:.1f}M parameters")
        am = fw.action_model
        self.action_dim = int(am.action_dim)
        # the policy server reads the chunk length from action_model.action_horizon: it must be System 1's horizon
        if int(am.get("action_horizon", -1)) != self.s1_horizon:
            raise ValueError(f"framework.action_model.action_horizon ({am.get('action_horizon', None)}) must equal the System 1 "
                             f"horizon round(s1_horizon_s * action_rate_hz) = {self.s1_horizon}")
        self.action_horizon = self.s1_horizon
        self.state_dim = int(am.get("state_dim", 0) or 0)
        self.prev_action_dims = [int(d) for d in (ds.get("state_prev_action_dims", None) or [])]
        if self.prev_action_dims and (self.state_dim <= len(self.prev_action_dims)
                                      or not all(0 <= d < self.action_dim for d in self.prev_action_dims)):
            raise ValueError(f"dual_system.state_prev_action_dims {self.prev_action_dims} needs action dims < {self.action_dim} "
                             f"and state_dim ({self.state_dim}) = loader state dim + {len(self.prev_action_dims)}")
        rp = ds.get("relative_plan", None)
        self.relative_plan = rp is not None
        self.rel_scale = 1.0
        if self.relative_plan:
            self._init_relative_plan(rp)
        dp = ds.get("delta_plan", None)
        self.delta_plan = dp is not None
        if self.delta_plan:
            if self.relative_plan:
                raise ValueError("dual_system.relative_plan and dual_system.delta_plan are exclusive")
            self._init_delta_plan(dp)
        sr = ds.get("s1_residual", None)
        self.s1_residual = sr is not None
        self.res_scale = float(sr.get("scale", 1.0)) if self.s1_residual else 1.0
        # warmup_scale for the first warmup_steps trainer steps (set_training_step); a fresh model (inference) uses scale
        self.res_scale_final = self.res_scale
        ws = sr.get("warmup_scale", None) if self.s1_residual else None
        self.res_scale_warmup = None if ws is None else float(ws)
        self.res_warmup_steps = int(sr.get("warmup_steps", 0) or 0) if self.s1_residual else 0
        rb = self.radio_encoder
        self.system2 = System2KeyframePlanner(self.vlm_dim, rb.feat_dim, rb.grid, self.state_dim, self.action_dim,
                                              self.timeline, fw.system2, num_prefix=rb.num_prefix, prefix_dim=rb.prefix_dim,
                                              num_views=self.num_views)
        self.system1 = System1ActionPolicy(rb.feat_dim, rb.grid, self.state_dim, self.action_dim, self.s1_horizon, self.rate,
                                           self.timeline, fw.system1, num_prefix=rb.num_prefix, prefix_dim=rb.prefix_dim,
                                           num_views=self.num_views)
        if (self.relative_plan or self.delta_plan) and not (self.system2.uses_state or self.system1.uses_state):
            raise ValueError("dual_system.relative_plan / delta_plan need the state (the reference): set system2/system1.use_state")
        mc = ds.get("memory", None)
        self.episode_memory = None
        self.mem_idx: List[int] = []  # video offset indices of the memory updates in training, oldest first
        self.mem_progress_weight = 0.0
        self.mem_tbptt = 0  # dual_system.memory.stream: truncated-BPTT length of the streamed memory (0 = window training)
        self._mem_streams: Dict[int, dict] = {}  # stream slot -> carried memory state (training)
        if mc is not None:
            if self.stale_input != "vlm":
                raise ValueError("dual_system.memory needs stale_input vlm")
            offs = [int(o) for o in (mc.get("update_offsets", None) or [])]
            if not offs or sorted(set(offs)) != offs or any(o >= 0 or o not in self.video_offsets for o in offs):
                raise ValueError(f"dual_system.memory.update_offsets must be ascending video offsets < 0, got {offs} "
                                 f"(video_offsets {self.video_offsets})")
            self.mem_idx = [self.video_offsets.index(o) for o in offs]
            self.mem_progress_weight = float(mc.get("progress_loss_weight", 0.1))
            sm = mc.get("stream", None)
            self.mem_tbptt = int(sm.get("tbptt", 4)) if sm else 0
            self.episode_memory = EpisodeMemory(
                self.vlm_dim, int(self.system2.conditioner.vlm_projector[1].out_features),
                num_tokens=int(mc.get("num_tokens", 16)), d_model=int(mc.get("d_model", 768)), depth=int(mc.get("depth", 2)),
                num_heads=int(mc.get("num_heads", 12)), mlp_ratio=float(mc.get("mlp_ratio", 4.0)),
                progress_bins=int(mc.get("progress_bins", 50)), attention=str(mc.get("attention", "standard")))
        self._warned_no_memory = False
        self.last_progress = None  # plan_keyframes with memory: expected progress [B] of the last call
        s1 = fw.system1
        self.traj_source = str(s1.get("traj_source", "pred"))
        if self.traj_source not in _TRAJ_SOURCES:
            raise ValueError(f"system1.traj_source must be one of {_TRAJ_SOURCES}, got {self.traj_source!r}")
        self.plan_noise = PlanNoise(s1.get("traj_noise", None), self.timeline.num, self.action_dim)
        # the trainer's generic action eval scores a [B, H, D] chunk against example["action"][-H:], which is not the
        # System 1 window of the dense [-2 s, +4 s] window -> own validation in trainer_eval_hook
        self.use_action_head = False
        self.max_plan_age_s = self.timeline.t_max - self.s1_horizon / self.rate
        self.train_delay_range_s = (0.0, self.video_offsets[-1] / self.rate)  # plan ages (0, 0 in vlm mode)
        self.train_vlm_age_range_s = (0.0, -self.video_offsets[0] / self.rate)  # VLM token ages (0, 0 in plan mode)
        # vlm mode: the VLM frame of the main validation metrics (the most likely training offset; the oldest if uniform)
        self.vlm_main_idx = 0 if self.s1_offset_probs is None else int(self.s1_offset_probs.argmax())
        # vlm mode with one VLM offset of nonzero probability: the fixed VLM lag (s) the model was trained with
        fixed = self.s1_offset_probs is not None and int((self.s1_offset_probs > 0).sum()) == 1
        self.fixed_vlm_age_s = -self.video_offsets[self.vlm_main_idx] / self.rate if self.stale_input == "vlm" and fixed else None
        if self.mem_tbptt and (len(self.mem_idx) != 1 or self.mem_idx[0] != self.vlm_main_idx or self.cur == self.vlm_main_idx):
            raise ValueError("dual_system.memory.stream: update_offsets must be the one VLM frame offset of System 2 "
                             f"(video_offsets[vlm_main_idx] = {self.video_offsets[self.vlm_main_idx]}), got "
                             f"{[self.video_offsets[j] for j in self.mem_idx]}")
        self._eval_examples = None
        self._plan_noise_rms = 0.0
        self._warned_no_valid = False
        logger.info(self.describe_shapes())

    # ------------------------------------------------------------------ bookkeeping
    def describe_shapes(self) -> str:
        v = self.num_views
        n, (gh, gw) = v * self.radio_encoder.num_tokens, self.radio_encoder.grid
        s2s = self.system2.conditioner.num_prefix
        tl = self.timeline
        return "\n".join([
            "QwenGR00T_DualSystem tensor shapes:",
            f"  frames           [B, {len(self.video_offsets)}, {v}, 3, {self.frame_size[0]}, {self.frame_size[1]}]  video offsets "
            f"{self.video_offsets}, {v} view(s)",
            f"  C-RADIO (shared) [B, {n}, {self.radio_encoder.feat_dim}]  ({v} x {gh}x{gw} patches, {self.radio_encoder.trainable_desc}) + prefix "
            f"[B, {v * self.radio_encoder.num_prefix}, {self.radio_encoder.prefix_dim}] {v} x {self.radio_encoder.prefix_names}",
            f"  VLM tokens       [B, L_v, {self.vlm_dim}]  (Qwen last hidden states, vision + text)",
            f"  s2_context       [B, L_v + {s2s} + {n}{' + 1' if self.system2.uses_state else ''}, "
            f"{self.system2.conditioner.vlm_projector[1].out_features}]  (vlm, radio prefix, radio patches, state)",
            f"  keyframes        [B, {tl.num}, {self.action_dim}]  {tl}",
            f"  dense window     [B, {self.window_len}, {self.action_dim}]  k = {self.k_min}..{self.k_max} @ {self.rate:g} Hz",
            f"  System 1 seq     {self.system1.describe_sequence()}  cross-attn ctx {self.system1.describe_context(n)}"
            f"  -> actions [B, {self.s1_horizon}, "
            f"{self.action_dim}] ({self.s1_horizon / self.rate:g} s), stale input {self.stale_input}: plan age in training "
            f"{self.train_delay_range_s} s, VLM token age {self.train_vlm_age_range_s} s",
            f"  interpolation    {self.system2.interpolator.describe()}",
        ] + ([f"  episode memory   [B, {self.episode_memory.num_tokens}, {self.episode_memory.d_model}] "
              f"({self.episode_memory.attention} attention) updated on the VLM frames at "
              f"{[self.video_offsets[j] for j in self.mem_idx]} in training (every new VLM frame at deployment) -> "
              f"+{self.episode_memory.num_ctx_tokens} s2_context tokens (memory + progress, {self.episode_memory.num_bins} bins)"]
             if self.episode_memory is not None else []))

    def _shapes(self) -> ShapeChecker:
        return ShapeChecker(self.check_shapes)  # per call: the async runner calls the systems from two threads

    @property
    def _device(self):
        return self.dense_times.device

    def _progress_of(self, examples):
        return None

    def _vlm_trainable(self) -> bool:
        return any(p.requires_grad for p in self.qwen_vl_interface.parameters())

    def _train_vlm_last_layers(self, n: int) -> None:
        """dual_system.vlm_train_last_layers: freeze qwen_vl_interface except the last n decoder layers of its language
        model and the final norm, which are cast to float32."""
        tr = self.config.get("trainer", None)
        frozen = str(tr.get("freeze_modules", "") or "") if tr is not None and hasattr(tr, "get") else ""
        if "qwen_vl_interface" in [p.strip() for p in frozen.split(",")]:
            raise ValueError("dual_system.vlm_train_last_layers is set but trainer.freeze_modules lists qwen_vl_interface "
                             "(the trainer would freeze the whole VLM again)")
        iface = self.qwen_vl_interface
        found = [(name, mod) for name, mod in iface.named_modules()
                 if name.endswith("language_model") and isinstance(getattr(mod, "layers", None), torch.nn.ModuleList)]
        if len(found) != 1:
            raise ValueError(f"vlm_train_last_layers: expected one '*language_model' with .layers, found {[n_ for n_, _ in found]}")
        lm_name, lm = found[0]
        if not 0 < n <= len(lm.layers):
            raise ValueError(f"vlm_train_last_layers must be in 1..{len(lm.layers)}, got {n}")
        for p in iface.parameters():
            p.requires_grad_(False)
        train = list(lm.layers[-n:]) + ([lm.norm] if getattr(lm, "norm", None) is not None else [])
        for mod in train:
            mod.float()
            for p in mod.parameters():
                p.requires_grad_(True)
        n_train = sum(p.numel() for p in iface.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in iface.parameters())
        logger.info(f"[DualSystem] VLM: training {lm_name}.layers[{len(lm.layers) - n}:{len(lm.layers)}] + norm (float32), "
                    f"{n_train / 1e6:.1f}M of {n_all / 1e6:.1f}M parameters; the rest frozen")

    # ------------------------------------------------------------------ inputs
    def _frames(self, examples: List[dict], n_time: int) -> torch.Tensor:
        """examples' "image" lists (n_time frames of each of the num_views cameras, camera-major) -> [B, T, V, 3, H, W]
        float in [0, 1] on the model device."""
        h, w = self.frame_size
        v = self.num_views
        batch = []
        for ex in examples:
            ims = _as_list(to_pil_preserve(ex["image"]))
            if len(ims) != v * n_time:
                raise ValueError(f"expected {n_time} frames x {v} views per example (video offsets {self.video_offsets}), got "
                                 f"{len(ims)} images: the data config video delta_indices must equal dual_system.video_offsets "
                                 "and its video keys dual_system.num_views")
            x = np.stack([np.asarray(im.convert("RGB") if im.size == (w, h) else im.convert("RGB").resize((w, h)))
                          for im in ims])  # [V * T, H, W, 3]
            batch.append(x.reshape(v, n_time, h, w, 3).swapaxes(0, 1))
        x = torch.from_numpy(np.stack(batch)).to(self._device, non_blocking=True)  # [B, T, V, H, W, 3]
        return x.permute(0, 1, 2, 5, 3, 4).float().div_(255.0)

    def _current_images(self, ex: dict, j: Optional[int] = None) -> list:
        """Every camera's frame at video offset index j (default: the current frame t): images [V * T] camera-major ->
        [V]. Inference examples carry one frame per camera (T = 1)."""
        ims = _as_list(to_pil_preserve(ex["image"]))
        if len(ims) % self.num_views:
            raise ValueError(f"{len(ims)} images is not a multiple of dual_system.num_views={self.num_views}")
        n_t = len(ims) // self.num_views
        if j is None:
            j = self.cur if n_t == len(self.video_offsets) else 0
        return ims[j::n_t]

    def _vlm_images(self, examples: List[dict], j=None):
        """The VLM sees every camera's frame at video offset index j[b] (default: the current frame) at frame_size."""
        pils = [self._current_images(ex, None if j is None else int(j[i])) for i, ex in enumerate(examples)]
        return resize_images(pils, target_size=(self.frame_size[1], self.frame_size[0]))

    def _cur_state(self, state: torch.Tensor) -> torch.Tensor:
        """state [B, T_s, D_st] -> the current frame's [B, 1, D_st] (T_s = 1 at inference)."""
        return state[:, self.cur : self.cur + 1] if state.shape[1] == len(self.video_offsets) else state[:, :1]

    def _state(self, examples: List[dict], sc: ShapeChecker) -> Optional[torch.Tensor]:
        """-> [B, T_s, D_st] or None."""
        if not (self.system2.uses_state or self.system1.uses_state):
            return None
        if "state" not in examples[0]:
            raise KeyError(f"state-conditioned model (state_dim {self.state_dim}): every example needs 'state' "
                           "(datasets.*.include_state: true)")
        st = torch.tensor(np.array([np.asarray(ex["state"], dtype=np.float32) for ex in examples]), device=self._device)
        st = st[:, None] if st.dim() == 2 else st
        if self.prev_action_dims:
            st = torch.cat([self._prev_actions(examples, st.shape[1]), st], dim=-1)
        return sc("state", st, f"B T_s {self.state_dim}")

    def _prev_actions(self, examples: List[dict], n_t: int) -> torch.Tensor:
        """dual_system.state_prev_action_dims: the previous command a[t + o - 1][dims] (action-normalised) for every
        state row (video offset o; one row at inference) -> [B, T_s, P]. From example["prev_action"] ([P] or [T_s, P],
        the last command sent at deploy) if given, else from the dense window example["action"] (training /
        validation; at an episode's first frame the loader pads a[-1] with a[0])."""
        dims = self.prev_action_dims
        offs = self.video_offsets if n_t == len(self.video_offsets) else [0]
        rows = []
        for ex in examples:
            if "prev_action" in ex:
                pa = np.asarray(ex["prev_action"], dtype=np.float32).reshape(-1, len(dims))
                pa = np.repeat(pa, n_t, 0) if pa.shape[0] == 1 else pa
            elif "action" in ex:
                a = np.asarray(ex["action"], dtype=np.float32)
                if a.shape[0] != self.window_len:
                    raise ValueError(f"example['action'] has {a.shape[0]} rows, expected the dense window ({self.window_len}); "
                                     "pass example['prev_action'] at inference")
                pa = a[[max(o - 1 - self.k_min, 0) for o in offs]][:, dims]
            else:
                raise KeyError(f"dual_system.state_prev_action_dims {dims}: every example needs 'prev_action' (the previous "
                               "command, action-normalised) or the dense 'action' window")
            if pa.shape != (n_t, len(dims)):
                raise ValueError(f"prev_action of shape {pa.shape}, expected ({n_t}, {len(dims)})")
            rows.append(pa)
        return torch.tensor(np.stack(rows), device=self._device)

    def _init_relative_plan(self, rp) -> None:
        """dual_system.relative_plan: per action dim d, ref[d] = ref_mul[d] * state[ref_index[d]] + ref_add[d] maps the
        normalised (augmented) state slot to action-normalised units (q99: [q01, q99] -> [-1, 1], identity where
        q01 == q99, as the loader's normaliser); previous-command slots are already action-normalised (identity).
        Persistent buffers: a checkpoint carries the map it was trained with."""
        p, d_a = len(self.prev_action_dims), self.action_dim
        idx = torch.full((d_a,), -1, dtype=torch.long)
        for a0, a1, s0 in rp["ref"]:
            a0, a1, s0 = int(a0), int(a1), int(s0)
            idx[a0:a1] = torch.arange(s0, s0 + a1 - a0)
        if bool((idx < 0).any()) or int(idx.max()) >= self.state_dim:
            raise ValueError(f"dual_system.relative_plan.ref must map every action dim to a state slot < {self.state_dim}, "
                             f"got index {idx.tolist()}")
        with open(rp["stats_path"]) as f:
            stats = json.load(f)
        stats = stats.get("new_embodiment", stats)
        aq01, aq99 = (np.asarray(stats["action"][k], dtype=np.float64) for k in ("q01", "q99"))
        sq01, sq99 = (np.asarray(stats["state"][k], dtype=np.float64) for k in ("q01", "q99"))
        if len(aq01) != d_a or len(sq01) != self.state_dim - p:
            raise ValueError(f"relative_plan.stats_path: action / state stats of {len(aq01)} / {len(sq01)} dims, expected "
                             f"{d_a} / {self.state_dim - p} (loader state, without the previous-command slots)")
        mul, add = np.ones(d_a), np.zeros(d_a)
        for d in range(d_a):
            j = int(idx[d])
            if j < p:
                if self.prev_action_dims[j] != d:
                    raise ValueError(f"relative_plan.ref: action dim {d} -> previous-command slot {j} = action dim "
                                     f"{self.prev_action_dims[j]}")
                continue
            k = j - p
            rs, ra = sq99[k] - sq01[k], aq99[d] - aq01[d]
            m1, c1 = (rs / 2, sq01[k] + rs / 2) if rs != 0 else (1.0, 0.0)  # state-normalised -> physical
            m2, c2 = (2 / ra, -2 * aq01[d] / ra - 1) if ra != 0 else (1.0, 0.0)  # physical -> action-normalised
            mul[d], add[d] = m2 * m1, m2 * c1 + c2
        self.register_buffer("ref_index", idx)
        self.register_buffer("ref_mul", torch.tensor(mul, dtype=torch.float32))
        self.register_buffer("ref_add", torch.tensor(add, dtype=torch.float32))
        self.rel_scale = float(rp.get("scale", 1.0))

    def action_reference(self, state_cur: torch.Tensor) -> torch.Tensor:
        """relative_plan: augmented current state [B, 1, D_st] -> the plan reference [B, D_a] (action-normalised)."""
        s = state_cur[:, 0].float()
        return s[:, self.ref_index] * self.ref_mul + self.ref_add

    def _init_delta_plan(self, dp) -> None:
        """dual_system.delta_plan: buffers of the per-step delta plan space (persistent: a checkpoint carries the
        statistics it was trained with). Slot k0 = -k_min of the dense window is the anchor d[0] = a[t] - s[t]."""
        d_a, q = self.action_dim, self.window_len
        with open(dp["stats_path"]) as f:
            st = json.load(f)
        if [int(v) for v in st["k_range"]] != [self.k_min, self.k_max]:
            raise ValueError(f"delta_plan.stats_path covers k {st['k_range']}, the dense window is [{self.k_min}, {self.k_max}]")
        mask = torch.zeros(d_a, dtype=torch.bool)
        sin_idx, cos_idx = torch.zeros(d_a, dtype=torch.long), torch.zeros(d_a, dtype=torch.long)
        for a0, a1, s0, c0 in dp["ref"]:
            a0, a1, s0, c0 = int(a0), int(a1), int(s0), int(c0)
            mask[a0:a1] = True
            sin_idx[a0:a1], cos_idx[a0:a1] = torch.arange(s0, s0 + a1 - a0), torch.arange(c0, c0 + a1 - a0)
        if int(max(sin_idx.max(), cos_idx.max())) >= self.state_dim:
            raise ValueError(f"delta_plan.ref: state slots must be < state_dim {self.state_dim}")
        amin, amax = (torch.tensor(st[k], dtype=torch.float32) for k in ("action_min", "action_max"))
        if amin.numel() != d_a or bool((amax[mask] <= amin[mask]).any()):
            raise ValueError(f"delta_plan.stats_path: action_min / action_max must have {d_a} dims with max > min on the delta dims")
        mean, std = torch.zeros(q, d_a), torch.ones(q, d_a)
        k0 = -self.k_min
        for rows, g in ((slice(None), "step"), (slice(k0, k0 + 1), "anchor")):
            m, s = torch.tensor(st[g]["mean"], dtype=torch.float32), torch.tensor(st[g]["std"], dtype=torch.float32)
            mean[rows] = torch.where(mask, m, mean[rows])
            std[rows] = torch.where(mask, s.clamp_min(float(dp.get("min_std", 1e-6))), std[rows])
        self.register_buffer("delta_mask", mask)
        self.register_buffer("delta_sin_idx", sin_idx)
        self.register_buffer("delta_cos_idx", cos_idx)
        self.register_buffer("delta_amin", amin)
        self.register_buffer("delta_amax", amax)
        self.register_buffer("delta_mean", mean)
        self.register_buffer("delta_std", std)
        self.delta_clip = float(dp.get("clip", 10.0))

    def _delta_from_abs(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """delta_plan: absolute normalised dense window [B, Q, D_a] + raw s[t] [B, D_a] -> the delta plan space."""
        if x.shape[1] != self.window_len:
            raise ValueError(f"delta_plan converts the dense window ({self.window_len} steps), got {x.shape[1]}")
        k0 = -self.k_min
        a = (x.float() + 1) / 2 * (self.delta_amax - self.delta_amin) + self.delta_amin
        d = torch.empty_like(a)
        d[:, k0] = a[:, k0] - ref
        d[:, k0 + 1:] = a[:, k0 + 1:] - a[:, k0:-1]
        d[:, :k0] = a[:, :k0] - a[:, 1:k0 + 1]
        z = ((d - self.delta_mean) / self.delta_std).clamp(-self.delta_clip, self.delta_clip)
        return torch.where(self.delta_mask, z, x.float())

    def _delta_to_abs(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """delta_plan: dense delta plan [B, Q, D_a] -> absolute normalised actions (a[t] = s[t] + d[0], then the
        cumulative steps forward / backward)."""
        if x.shape[1] != self.window_len:
            raise ValueError(f"delta_plan converts the dense window ({self.window_len} steps), got {x.shape[1]}")
        k0 = -self.k_min
        d = x.float() * self.delta_std + self.delta_mean
        fut = ref[:, None] + torch.cumsum(d[:, k0:], dim=1)
        past = fut[:, :1] + torch.flip(torch.cumsum(torch.flip(d[:, :k0], [1]), dim=1), [1])
        a = torch.cat([past, fut], dim=1)
        return torch.where(self.delta_mask, 2 * (a - self.delta_amin) / (self.delta_amax - self.delta_amin) - 1, x.float())

    def _plan_to_rel(self, x: torch.Tensor, ref: Optional[torch.Tensor]) -> torch.Tensor:
        """absolute normalised actions [B, N, D_a] -> System 2's plan space (identity without relative_plan /
        delta_plan; delta_plan needs the dense window, N = Q)."""
        if ref is None:
            return x
        return self._delta_from_abs(x, ref) if self.delta_plan else (x - ref[:, None]) * self.rel_scale

    def _plan_to_abs(self, x: torch.Tensor, ref: Optional[torch.Tensor]) -> torch.Tensor:
        if ref is None:
            return x
        return self._delta_to_abs(x, ref) if self.delta_plan else x / self.rel_scale + ref[:, None]

    def _plan_reference(self, state: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.delta_plan:  # raw s[t] of the delta dims (atan2 of the sin / cos state; other dims unused)
            s = self._cur_state(state)[:, 0].float()
            return torch.atan2(s[:, self.delta_sin_idx], s[:, self.delta_cos_idx])
        return self.action_reference(self._cur_state(state)) if self.relative_plan else None

    def set_training_step(self, step: int) -> None:
        """Called by the trainer before every training step (completed optimizer steps, resume-safe):
        s1_residual.warmup_scale for steps < warmup_steps, scale afterwards."""
        if self.res_scale_warmup is not None:
            self.res_scale = self.res_scale_warmup if int(step) < self.res_warmup_steps else self.res_scale_final

    def _plan_base(self, kf: torch.Tensor, ref: Optional[torch.Tensor], delay_s: torch.Tensor,
                   steps: Optional[int] = None) -> torch.Tensor:
        """s1_residual: System 2 plan kf [B, K, D_a] (plan space) as absolute normalised actions at the System 1 times
        delay_s[b] + k / rate, k < N_a (or < steps) -> A_sys2 [B, N_a, D_a] (no gradient). ref = the plan's reference
        (its frame)."""
        n = self.s1_horizon if steps is None else int(steps)
        with torch.no_grad():
            b = kf.shape[0]
            times = self.dense_times[None].expand(b, -1)
            dense, _ = self.system2.interpolator(kf.float(), self.system2.key_times, times)
            dense = self._plan_to_abs(dense, ref)
            q = (torch.as_tensor(delay_s, dtype=torch.float32, device=kf.device).reshape(-1, 1).expand(b, 1)
                 + torch.arange(n, device=kf.device)[None].float() / self.rate)
            win, _ = self.system2.interpolator(dense, times, q)
        return win.float()

    def _plan_chunk(self, kf: torch.Tensor, ref: Optional[torch.Tensor], delay_s: torch.Tensor) -> Optional[torch.Tensor]:
        """system1.plan_input action: the first plan_chunk_s seconds of the plan on System 1's window (absolute
        normalised), Gaussian-smoothed in time inside the chunk -> [B, n_in, D_a] (no gradient); else None."""
        if not self.system1.plan_as_input:
            return None
        win = self._plan_base(kf, ref, delay_s, steps=self.system1.plan_chunk_steps)
        return smooth_time(win, self.system1.plan_smooth_sigma_s * self.rate)

    def _s1_actions(self, kf, delay, radio, st, ref, steps=None) -> torch.Tensor:
        """System 1 sample -> absolute normalised actions (A_sys2 + residual / scale with s1_residual)."""
        a = self.system1.sample(kf, delay, radio, st, steps=None if steps is None else int(steps),
                                plan_chunk=self._plan_chunk(kf, ref, delay)).float()
        return self._plan_base(kf, ref, delay) + a / self.res_scale if self.s1_residual else a

    @torch.no_grad()
    def plan_to_absolute(self, examples: List[dict], keyframes: torch.Tensor) -> torch.Tensor:
        """System 2 plan (plan_keyframes) -> absolute normalised actions [B, K, D_a], with the examples' current state
        as the reference (identity without relative_plan / delta_plan). For visualisation; System 1 takes the plan
        as-is."""
        if not (self.relative_plan or self.delta_plan):
            return keyframes
        ref = self._plan_reference(self._state(examples, self._shapes()))
        if self.delta_plan:  # keyframes -> dense deltas -> integrate -> back to the keyframe times
            times = self.dense_times[None].expand(keyframes.shape[0], -1)
            dense, _ = self.system2.interpolator(keyframes.float(), self.system2.key_times, times)
            return self.system2.extract_keyframes(self._plan_to_abs(dense, ref), times)
        return self._plan_to_abs(keyframes.float(), ref)

    def _dense_targets(self, examples: List[dict], sc: ShapeChecker):
        """-> dense GT [B, Q, D_a], valid [B, Q] bool (inside the episode and inside the keyframe span)."""
        a = torch.tensor(np.array([np.asarray(ex["action"], dtype=np.float32) for ex in examples]), device=self._device)
        sc("dense actions", a, f"B {self.window_len} {self.action_dim}")
        valid = self.dense_in_span[None].expand(a.shape[0], -1).clone()
        if self.mask_out_of_episode:
            if "action_valid" in examples[0]:
                av = torch.tensor(np.array([np.asarray(ex["action_valid"], dtype=bool) for ex in examples]), device=self._device)
                valid &= sc("action_valid", av, f"B {self.window_len}")
            elif not self._warned_no_valid:
                warnings.warn("examples carry no 'action_valid' (set datasets.*.emit_action_valid: true): dense targets "
                              "outside the episode (padded with the first / last action) are NOT masked")
                self._warned_no_valid = True
        return a, valid

    def _frame_valid(self, examples: List[dict]) -> torch.Tensor:
        t, v = len(self.video_offsets), self.num_views
        if "frame_valid" in examples[0]:
            # the loader emits one entry per (camera, video delta), camera-major; the cameras share the deltas
            fv = torch.tensor(np.array([np.asarray(ex["frame_valid"], dtype=bool) for ex in examples]), device=self._device)
            if fv.shape[1] != t * v:
                raise ValueError(f"frame_valid has {fv.shape[1]} entries, expected {t} offsets x {v} views")
            return fv.view(-1, v, t).all(1)
        return torch.ones(len(examples), t, dtype=torch.bool, device=self._device)

    def _sample_s1_offsets(self, examples, dense_valid) -> torch.Tensor:
        """Per sample, the index j of the stale input's frame. plan: the System 1 frame (plan age = video_offsets[j] /
        rate), drawn among offsets whose frame lies inside the episode and whose first System 1 target step is valid.
        vlm: the VLM frame (token age = -video_offsets[j] / rate), drawn among offsets whose frame lies inside the
        episode. The current frame is the fallback."""
        offs = torch.tensor(self.video_offsets, device=self._device)
        ok = self._frame_valid(examples)  # [B, T_v]
        if self.stale_input == "plan":
            ok = ok & dense_valid[:, offs - self.k_min]
        ok[:, self.cur] = True
        w = ok.float() * (self.s1_offset_probs.to(self._device)[None] if self.s1_offset_probs is not None else 1.0)
        w[w.sum(1) == 0, self.cur] = 1.0
        return torch.multinomial(w, 1).squeeze(1)

    def _s1_window(self, dense, dense_valid, offset_frames: torch.Tensor):
        """Dense GT a[t_S1 .. t_S1 + N_a) for t_S1 = t_S2 + offset -> ([B, N_a, D_a], [B, N_a])."""
        idx = (offset_frames - self.k_min)[:, None] + torch.arange(self.s1_horizon, device=dense.device)[None]
        a = dense.gather(1, idx[..., None].expand(-1, -1, dense.shape[-1]))
        return a, dense_valid.gather(1, idx)

    # ------------------------------------------------------------------ encoders
    def encode_vlm(self, images, instructions, sc: ShapeChecker, with_grad: bool):
        """Qwen last hidden states of every prompt token (vision + text) -> ([B, L_v, D_v], mask [B, L_v])."""
        iface = self.qwen_vl_interface
        inputs = iface.build_qwenvl_inputs(images=images, instructions=instructions)
        with torch.set_grad_enabled(with_grad and torch.is_grad_enabled()), \
                torch.autocast("cuda", dtype=getattr(iface, "autocast_dtype", torch.bfloat16)):
            out = iface(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True, logits_to_keep=1)
        return sc("vlm_tokens", out.hidden_states[-1], "B L_v D_v"), inputs["attention_mask"].bool()

    def encode_radio(self, frames: torch.Tensor, sc: Optional[ShapeChecker] = None) -> VisualFeatures:
        """frames [M, V, 3, H, W] ([M, 3, H, W] for one view) -> shared C-RADIOv4 tokens (without gradient unless
        radio.trainable), the views concatenated on the token dim (view-major): tokens [M, V * N, D_c], prefix [M, V * P, D_c]."""
        frames = frames[:, None] if frames.dim() == 4 else frames
        m, v = frames.shape[:2]
        if v != self.num_views:
            raise ValueError(f"expected {self.num_views} views, got frames {tuple(frames.shape)}")
        f = self.radio_encoder(frames.flatten(0, 1))
        if v > 1:
            f = VisualFeatures(f.tokens.reshape(m, -1, f.tokens.shape[-1]), f.valid.reshape(m, -1), f.grid,
                               None if f.prefix is None else f.prefix.reshape(m, -1, f.prefix.shape[-1]))
        if sc is not None:
            sc("radio_tokens", f.tokens, f"M {v * self.radio_encoder.num_tokens} {self.radio_encoder.feat_dim}")
        return f

    def _s2_context(self, examples, radio_s2: VisualFeatures, state, sc, train_s2: bool, vlm_idx=None, vlm_age_s=None,
                    vlm=None, mem_ctx=None):
        """vlm_idx [B]: video offset index of the VLM frame (default: the current frame); vlm_age_s [B]: its age;
        vlm: precomputed (tokens, mask) from encode_vlm_cache (then examples' images are not encoded); mem_ctx: episode
        memory context tokens [B, N_m, d2] (EpisodeMemory.readout), appended."""
        if vlm is None:
            vlm = self.encode_vlm(self._vlm_images(examples, vlm_idx), [ex["lang"] for ex in examples], sc,
                                  with_grad=train_s2 and self._vlm_trainable())
        vlm, vmask = vlm
        with torch.set_grad_enabled(train_s2 and torch.is_grad_enabled()):
            st = None if (state is None or not self.system2.uses_state) else self._cur_state(state)
            ctx, cmask = self.system2.conditioner(vlm, vmask, radio_s2, st, vlm_age_s=vlm_age_s)
            if mem_ctx is not None:
                ctx = torch.cat([ctx, mem_ctx.to(ctx.dtype)], dim=1)
                cmask = torch.cat([cmask, torch.ones(mem_ctx.shape[:2], dtype=torch.bool, device=cmask.device)], dim=1)
        return sc("s2_context", ctx, "B L d2"), cmask

    @staticmethod
    def _pad_vlm(tok: torch.Tensor, mask: torch.Tensor, length: int):
        """VLM (tokens [B, L, D], mask [B, L]) right-padded to length (zero tokens, mask False)."""
        if tok.shape[1] >= length:
            return tok, mask
        pad = length - tok.shape[1]
        return (torch.cat([tok, tok.new_zeros(tok.shape[0], pad, tok.shape[2])], 1),
                torch.cat([mask, mask.new_zeros(mask.shape[0], pad)], 1))

    def _memory_context(self, examples, sc, s2_idx=None, train: bool = False):
        """Episode memory unrolled from its initial state over the VLM frames at video offset indices mem_idx (oldest
        first; a frame outside the episode skips its update) -> (memory context [B, N + 1, d2], progress logits
        [B, bins], VLM (tokens, mask) for System 2 or None). The VLM frames are encoded without gradient in one batch;
        with a frozen VLM the System 2 VLM tokens of samples whose VLM frame (s2_idx) is the last memory frame are
        reused from it (the others, e.g. frame t where t - 15 lies before the episode, are encoded separately)."""
        b, k = len(examples), len(self.mem_idx)
        fv = self._frame_valid(examples)
        reps = [ex for _ in self.mem_idx for ex in examples]
        js = [j for j in self.mem_idx for _ in examples]
        # own shape checkers: the batch here is k * B (and a subset below), not the forward's B
        tok, mask = self.encode_vlm(self._vlm_images(reps, js), [ex["lang"] for ex in reps], self._shapes(), with_grad=False)
        tok, mask = tok.view(k, b, *tok.shape[1:]), mask.view(k, b, -1)
        with torch.set_grad_enabled(train and torch.is_grad_enabled()):
            m = self.episode_memory.init(b).to(self._device)
            for i, j in enumerate(self.mem_idx):
                m = self.episode_memory.update(m, tok[i], mask[i], fv[:, j])
            ctx, logits = self.episode_memory.readout(m)
        vlm = None
        if s2_idx is not None and not self._vlm_trainable():
            s2_idx = torch.as_tensor(s2_idx).reshape(-1).cpu()
            reuse = s2_idx == self.mem_idx[-1]
            vt, vm = tok[-1], mask[-1]
            other = (~reuse).nonzero(as_tuple=True)[0].tolist()
            if other:
                t2, m2 = self.encode_vlm(self._vlm_images([examples[i] for i in other], [int(s2_idx[i]) for i in other]),
                                         [examples[i]["lang"] for i in other], self._shapes(), with_grad=False)
                n = max(vt.shape[1], t2.shape[1])
                vt, vm = self._pad_vlm(vt, vm, n)
                t2, m2 = self._pad_vlm(t2, m2, n)
                vt, vm = vt.clone(), vm.clone()
                vt[other], vm[other] = t2.to(vt.dtype), m2
            vlm = (vt, vm)
        return ctx, logits, vlm

    def _memory_stream_context(self, examples, sc, s2_idx, train: bool = False):
        """dual_system.memory.stream (training): every sample continues the memory of its stream (sample["stream"] =
        [slot, step, episode counter, step in the episode, ...]) by one update on the new VLM frame t - stride (System
        2's VLM frame, encoded once without gradient and reused for System 2); gradient through the last mem_tbptt
        updates (cached VLM tokens) from the detached older memory. A stream restarts from the initial memory at its
        episode's step 0; a stream that does not continue its previous step (first episode joined mid-way, reordered
        batches) is invalid until then. -> (memory context [B, N + 1, d2], progress logits [B, bins], System 2 VLM
        (tokens, mask), valid [B] bool)."""
        if self._vlm_trainable():
            raise ValueError("dual_system.memory.stream needs a frozen VLM (it caches the VLM tokens across steps)")
        if "stream" not in examples[0]:
            raise KeyError("dual_system.memory.stream: examples need 'stream' (datasets.vla_data.stream)")
        b, kmax, jm = len(examples), self.mem_tbptt, self.mem_idx[0]
        fv = self._frame_valid(examples)
        s2_idx = torch.as_tensor(s2_idx).reshape(-1).cpu()
        tok, mask = self.encode_vlm(self._vlm_images(examples, s2_idx.tolist()), [ex["lang"] for ex in examples],
                                    self._shapes(), with_grad=False)
        new = (fv[:, jm].cpu() & (s2_idx == jm)).tolist()  # the frame t - stride lies inside the episode
        info = [np.asarray(ex["stream"], dtype=np.int64) for ex in examples]
        states, seqs = [], []
        for i in range(b):
            slot, k, j, es = (int(x) for x in info[i][:4])
            st = self._mem_streams.get(slot)
            if es == 0 or st is None or not st["valid"] or st["key"] != (k - 1, j, es - 1):
                st = {"base": None, "frames": [], "valid": es == 0}
            st["key"] = (k, j, es)
            seq = st["frames"] + ([(tok[i], mask[i])] if new[i] else [])
            states.append(st)
            seqs.append(seq[-kmax:])
        n = max(len(q) for q in seqs)
        init = self.episode_memory.init(1).to(self._device)[0]
        m = torch.stack([init if st["base"] is None else st["base"] for st in states])
        first = m
        with torch.set_grad_enabled(train and torch.is_grad_enabled()):
            if n:  # right-aligned: the newest frame last; missing slots skip their update
                length = max(t_.shape[0] for q in seqs for t_, _ in q)
                tk = tok.new_zeros(n, b, length, tok.shape[-1])
                mk = mask.new_zeros(n, b, length)
                vk = torch.zeros(n, b, dtype=torch.bool, device=self._device)
                for i, q in enumerate(seqs):
                    for s_, (t_, m_) in enumerate(q, start=n - len(q)):
                        tk[s_, i, : t_.shape[0]], mk[s_, i, : m_.shape[0]], vk[s_, i] = t_, m_, True
                for s_ in range(n):
                    m = self.episode_memory.update(m, tk[s_], mk[s_], vk[s_])
                    if s_ == 0:
                        first = m
            ctx, logits = self.episode_memory.readout(m)
        for i, (st, q) in enumerate(zip(states, seqs)):
            if len(q) == kmax:  # the oldest cached frame leaves the window -> folded into the detached memory
                st["base"], st["frames"] = first[i].detach(), q[1:]
            else:
                st["frames"] = q
            self._mem_streams[int(info[i][0])] = st
        ok = torch.tensor([st["valid"] for st in states], device=self._device)
        return ctx, logits, (tok, mask), ok

    def _progress_targets(self, examples) -> Optional[torch.Tensor]:
        if "progress" not in examples[0]:
            return None
        return torch.tensor([float(ex["progress"]) for ex in examples], device=self._device)

    def _s1_trajectory(self, ctx, cmask, dense):
        """System 1's keyframe input during training (System 2's sampled plan / GT keyframes / mix), no gradient."""
        s1 = self.config.framework.system1
        b = dense.shape[0]
        times = self.dense_times[None].expand(b, -1)
        with torch.no_grad():
            gt = pred = None
            if self.traj_source in ("gt", "mix"):
                gt = self.system2.extract_keyframes(dense, times)
                std = float(s1.get("traj_gt_noise_std", 0.0))
                if std > 0:
                    gt = gt + std * torch.randn_like(gt)
            if self.traj_source in ("pred", "mix"):
                steps = s1.get("traj_pred_steps", None)
                pred = self.system2.sample(ctx.detach(), cmask, steps=None if steps is None else int(steps)).float()
                if self.plan_noise.enabled:
                    noise = self.plan_noise.sample(b, pred.device)
                    self._plan_noise_rms = float(noise.pow(2).mean().sqrt())
                    pred = pred + noise
            if self.traj_source == "gt":
                return gt
            if self.traj_source == "pred":
                return pred
            use = torch.rand(b, device=dense.device) < float(s1.get("traj_pred_prob", 0.5))
            return torch.where(use[:, None, None], pred, gt)

    # ------------------------------------------------------------------ training
    def forward(self, examples: List[dict] = None, **kwargs) -> Dict:
        if isinstance(examples, dict):
            raise NotImplementedError("QwenGR00T_DualSystem builds its own inputs; preprocess_in_workers is not supported")
        t0 = time.perf_counter()
        sc = self._shapes()
        train2, train1 = "system2" in self.train_systems, "system1" in self.train_systems
        need_ctx = train2 or (train1 and self.traj_source in ("pred", "mix"))
        b = len(examples)
        frames = sc("frames", self._frames(examples, len(self.video_offsets)), f"B T_v {self.num_views} 3 H W")
        dense, dvalid = self._dense_targets(examples, sc)
        state = self._state(examples, sc)
        out: Dict = {}
        # System 2's targets / System 1's GT keyframes in the plan space (deltas from the reference with relative_plan);
        # System 1's action targets stay absolute
        dense_s2 = self._plan_to_rel(dense, self._plan_reference(state))

        t = time.perf_counter()
        ar = torch.arange(b, device=self._device)
        vlm_idx = vlm_age = None
        if self.stale_input == "vlm":
            # C-RADIO / state / System 1 at the current frame t; only the VLM frame t + o (o <= 0) is drawn per sample
            j = torch.full((b,), self.cur, device=self._device) if train1 else None
            if need_ctx:
                vlm_idx = self._sample_s1_offsets(examples, dvalid)
                vlm_age = -torch.tensor(self.video_offsets, device=self._device)[vlm_idx].float() / self.rate
                out["system2/mean_vlm_age_s"] = float(vlm_age.mean())
            radio_s2 = radio_s1 = self.encode_radio(frames[:, self.cur], sc)
        else:
            # one shared C-RADIO pass over the System 2 frames and the System 1 frames that differ from them
            j = self._sample_s1_offsets(examples, dvalid) if train1 else None
            imgs, extra = [], None
            if need_ctx:
                imgs.append(frames[:, 0])
            if train1:
                extra = (j != 0).nonzero(as_tuple=True)[0] if need_ctx else ar
                imgs.append(frames[extra, j[extra]])
            feats = self.encode_radio(torch.cat(imgs), sc)
            radio_s2 = feats.index(slice(0, b)) if need_ctx else None
            radio_s1 = None
            if train1:
                if need_ctx:
                    tok, val, pre = radio_s2.tokens.clone(), radio_s2.valid.clone(), radio_s2.prefix.clone()
                    tok[extra], val[extra], pre[extra] = feats.tokens[b:], feats.valid[b:], feats.prefix[b:]
                    radio_s1 = VisualFeatures(tok, val, feats.grid, pre)
                else:
                    radio_s1 = feats
        out["timing/radio"] = time.perf_counter() - t

        ctx = cmask = mem_ctx = mem_logits = vlm_pre = mem_ok = None
        if need_ctx and self.episode_memory is not None:
            t = time.perf_counter()
            if self.mem_tbptt:  # streamed memory: samples without their stream's history leave every loss
                mem_ctx, mem_logits, vlm_pre, mem_ok = self._memory_stream_context(examples, sc, vlm_idx, train=train2)
                dvalid = dvalid & mem_ok[:, None]
                out["memory/stream_valid_frac"] = float(mem_ok.float().mean())
                out["memory/stream_mean_step"] = float(np.mean([int(ex["stream"][3]) for ex in examples]))
            else:
                mem_ctx, mem_logits, vlm_pre = self._memory_context(examples, sc, s2_idx=vlm_idx, train=train2)
            out["timing/memory"] = time.perf_counter() - t
        if need_ctx:
            t = time.perf_counter()
            ctx, cmask = self._s2_context(examples, radio_s2, state, sc, train_s2=train2, vlm_idx=vlm_idx, vlm_age_s=vlm_age,
                                          vlm=vlm_pre, mem_ctx=mem_ctx)
            out["timing/system2_context"] = time.perf_counter() - t

        total = 0.0
        times = self.dense_times[None].expand(b, -1)
        if train2:
            t = time.perf_counter()
            loss2, logs = self.system2.loss(ctx, cmask, dense_s2, times, dvalid)
            out.update(logs)
            total = total + self.loss_weights.get("system2", 1.0) * loss2
            if mem_logits is not None:
                prog = self._progress_targets(examples)
                if prog is None:
                    raise KeyError("dual_system.memory: examples need 'progress' (datasets.vla_data.emit_progress: true)")
                ce, logs = self.episode_memory.progress_loss(mem_logits, prog, valid=mem_ok)
                out.update(logs)
                total = total + self.mem_progress_weight * ce
            out["timing/system2"] = time.perf_counter() - t
        if train1:
            t = time.perf_counter()
            kf = sc("keyframes", self._s1_trajectory(ctx, cmask, dense_s2), f"B {self.timeline.num} {self.action_dim}")
            offs = torch.tensor(self.video_offsets, device=self._device)[j]
            a1, v1 = self._s1_window(dense, dvalid, offs)
            if self.s1_residual:  # target = (a - A_sys2) * scale on the System 1 window
                res = a1.float() - self._plan_base(kf, self._plan_reference(state), offs.float() / self.rate)
                out["system1/residual_rms"] = float(((res ** 2) * v1[..., None]).sum() / (v1.sum() * res.shape[-1]).clamp_min(1)) ** 0.5
                out["system1/res_scale"] = self.res_scale
                a1 = res * self.res_scale
            st1 = None
            if self.system1.uses_state:
                if state.shape[1] != len(self.video_offsets):
                    raise ValueError("system1.use_state needs the state at every video offset (state delta_indices == video_offsets)")
                st1 = state[ar, j][:, None]
            loss1, logs = self.system1.loss(kf, offs.float() / self.rate, radio_s1, st1, a1, v1,
                                            plan_chunk=self._plan_chunk(kf, self._plan_reference(state), offs.float() / self.rate))
            out.update(logs)
            out["system1/mean_delay_s"] = float(offs.float().mean() / self.rate)
            if self.plan_noise.enabled and self.traj_source != "gt":
                out["system1/plan_noise_rms"] = self._plan_noise_rms
            out["action_loss"] = loss1  # logged by the trainer as action_dit_loss
            total = total + self.loss_weights.get("system1", 1.0) * loss1
            out["timing/system1"] = time.perf_counter() - t
        out["loss"] = total
        out["timing/forward"] = time.perf_counter() - t0
        return out

    # ------------------------------------------------------------------ inference (also the async runner's interface)
    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda")

    @torch.no_grad()
    def encode_visual(self, examples: List[dict]) -> VisualFeatures:
        """Current frame of every camera of every example -> shared C-RADIOv4 tokens."""
        return self.encode_radio(self._frames([{"image": self._current_images(ex)} for ex in examples], 1)[:, 0])

    @torch.no_grad()
    def encode_vlm_cache(self, examples: List[dict]):
        """VLM tokens of the examples' current frame + instruction -> (tokens [B, L_v, D_v], mask [B, L_v]), for
        plan_keyframes(vlm=...) on later frames (stale_input vlm; AsyncVLMRunner)."""
        with self._autocast():
            return self.encode_vlm(self._vlm_images(examples), [ex["lang"] for ex in examples], self._shapes(), with_grad=False)

    @torch.no_grad()
    def memory_init(self, b: int = 1) -> Optional[torch.Tensor]:
        """dual_system.memory: initial episode memory [B, N, d] for a new episode (None without memory)."""
        return None if self.episode_memory is None else self.episode_memory.init(b).to(self._device)

    @torch.no_grad()
    def memory_from_window(self, examples: List[dict]):
        """dual_system.memory, examples carrying one frame per video offset (camera-major, as in training; optional
        "frame_valid" per (camera, offset), False = the frame lies before the episode): the training unroll (initial
        memory, one update per memory frame, invalid frames skipped) -> (memory [B, N, d], System 2 VLM (tokens, mask)
        of the main VLM frame). A stateless client-side window, no per-episode server state."""
        b = len(examples)
        fv = self._frame_valid(examples)
        idx = list(self.mem_idx) + ([] if self.vlm_main_idx in self.mem_idx else [self.vlm_main_idx])
        reps = [ex for _ in idx for ex in examples]
        js = [j for j in idx for _ in examples]
        with self._autocast():
            tok, mask = self.encode_vlm(self._vlm_images(reps, js), [ex["lang"] for ex in reps], self._shapes(), with_grad=False)
            tok, mask = tok.view(len(idx), b, *tok.shape[1:]), mask.view(len(idx), b, -1)
            m = self.memory_init(b)
            for i, j in enumerate(self.mem_idx):
                m = self.episode_memory.update(m, tok[i], mask[i], fv[:, j])
        i = idx.index(self.vlm_main_idx)
        return m, (tok[i], mask[i])

    @torch.no_grad()
    def memory_update(self, memory: torch.Tensor, vlm, valid=None) -> torch.Tensor:
        """One memory step on a new VLM frame: vlm = (tokens, mask) of encode_vlm_cache -> the updated memory."""
        with self._autocast():
            return self.episode_memory.update(memory, vlm[0], vlm[1], valid)

    @torch.no_grad()
    def plan_keyframes(self, examples: List[dict], radio: Optional[VisualFeatures] = None,
                       steps: Optional[int] = None, vlm=None, vlm_age_s=0.0, memory=None) -> torch.Tensor:
        """System 2: observation + instruction (+ state) -> clean keyframes [B, K, D_a] (normalized actions; deltas from
        the reference with relative_plan, see plan_to_absolute), timed relative to the examples' frame. vlm: cached (tokens, mask) of an earlier frame (encode_vlm_cache), vlm_age_s old (scalar or
        [B]); None = encode the examples' frame (age 0). memory: the episode memory (memory_init / memory_update;
        dual_system.memory only, None = the initial memory); its expected progress is left in self.last_progress."""
        sc = self._shapes()
        radio = self.encode_visual(examples) if radio is None else radio
        steps = self.inference_steps["s2_steps"] if steps is None else steps
        mem_ctx = None
        if self.episode_memory is not None:
            if memory is None:
                if not self._warned_no_memory:
                    warnings.warn("QwenGR00T_DualSystem: plan_keyframes without memory -> the initial (episode start) memory")
                    self._warned_no_memory = True
                memory = self.memory_init(len(examples))
            with self._autocast():
                mem_ctx, logits = self.episode_memory.readout(memory)
            self.last_progress = self.episode_memory.expected_progress(logits)
        with self._autocast():
            ctx, cmask = self._s2_context(examples, radio, self._state(examples, sc), sc, train_s2=False,
                                          vlm_age_s=None if vlm is None else vlm_age_s, vlm=vlm, mem_ctx=mem_ctx)
            kf = self.system2.sample(ctx, cmask, steps=None if steps is None else int(steps))
        return sc("keyframes", kf.float(), f"B {self.timeline.num} {self.action_dim}")

    @torch.no_grad()
    def act_from_plan(self, examples: List[dict], keyframes: torch.Tensor, delay_s, radio: Optional[VisualFeatures] = None,
                      steps: Optional[int] = None) -> torch.Tensor:
        """System 1: current frame + a System 2 plan made delay_s seconds ago -> next N_a actions [B, N_a, D_a].
        s1_residual: A_sys2 + residual, with the examples' current state as the plan reference (exact at plan age 0:
        predict_action / stale_input vlm)."""
        sc = self._shapes()
        radio = self.encode_visual(examples) if radio is None else radio
        b = keyframes.shape[0]
        delay = torch.as_tensor(delay_s, dtype=torch.float32, device=keyframes.device).reshape(-1).expand(b)
        st = None
        plan_ref = self.s1_residual or self.system1.plan_as_input  # System 1 needs the plan in absolute units
        state = self._state(examples, sc) if (self.system1.uses_state or plan_ref or self.action_source == "s2_plan") else None
        if self.action_source == "s2_plan":  # inference.action_source: System 2's plan itself (A_sys2), System 1 unused
            a = self._plan_base(keyframes, self._plan_reference(state), delay)
            return sc("actions", a.float(), f"B {self.s1_horizon} {self.action_dim}")
        if self.system1.uses_state:
            st = self._cur_state(state)
        steps = self.inference_steps["s1_steps"] if steps is None else steps
        ref = self._plan_reference(state) if plan_ref else None
        with self._autocast():
            a = self._s1_actions(keyframes, delay, radio, st, ref, steps)
            for _ in range(self.s1_num_samples - 1):  # inference.s1_num_samples > 1: mean of independent samples
                a = a + self._s1_actions(keyframes, delay, radio, st, ref, steps)
        a = a / self.s1_num_samples if self.s1_num_samples > 1 else a
        for _ in range(self.s1_iters - 1):  # inference.s1_iters > 1: System 1 again on the plan carrying its own chunk
            kf_it = self._plan_with_chunk(keyframes, a.float(), delay)
            with self._autocast():
                a = self._s1_actions(kf_it, delay, radio, st, ref, steps)
                for _ in range(self.s1_num_samples - 1):
                    a = a + self._s1_actions(kf_it, delay, radio, st, ref, steps)
            a = a / self.s1_num_samples if self.s1_num_samples > 1 else a
        return sc("actions", a.float(), f"B {self.s1_horizon} {self.action_dim}")

    def _plan_with_chunk(self, kf: torch.Tensor, a: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
        """inference.s1_iters: plan keyframes kf [B, K, D_a] whose times fall on System 1's steps delay[b] + k / rate
        (k < N_a) replaced by System 1's chunk a [B, N_a, D_a] there; the other keyframes keep System 2's plan.
        Absolute plans only (the chunk is absolute normalised actions)."""
        if self.relative_plan or self.delta_plan:
            raise NotImplementedError("inference.s1_iters assumes an absolute plan")
        pos = (self.system2.key_times.to(kf.device).float()[None] - delay.float()[:, None]) * self.rate  # [B, K] steps
        k = pos.round()
        inside = (k >= 0) & (k <= a.shape[1] - 1) & ((pos - k).abs() < 1e-3)
        g = k.clamp(0, a.shape[1] - 1).long()[..., None].expand(-1, -1, a.shape[2])
        return torch.where(inside[..., None], torch.gather(a, 1, g).to(kf.dtype), kf)

    def predict_action(self, examples: List[dict], **kwargs) -> Dict:
        """Synchronous System 2 -> System 1 (plan age 0, one shared C-RADIO pass); QwenGR00T-compatible output.
        stale_input vlm: examples carrying one frame per video offset (camera-major, as in training, e.g. [previous
        call's frame, frame t]) give the VLM the stale frame (video offset index vlm_main_idx) and C-RADIO / state /
        System 1 frame t; one frame per camera = the VLM on frame t (age 0)."""
        examples = examples if isinstance(examples, list) else [examples]
        if os.environ.get("STARVLA_DUAL_FASTS1"):
            return self._predict_fast_s1(examples, os.environ["STARVLA_DUAL_FASTS1"])
        radio = self.encode_visual(examples)
        vlm = memory = None
        n_img = len(_as_list(to_pil_preserve(examples[0]["image"])))
        if self.stale_input == "vlm" and n_img == self.num_views * len(self.video_offsets):
            if self.episode_memory is not None:
                memory, vlm = self.memory_from_window(examples)
            else:
                j = torch.full((len(examples),), self.vlm_main_idx)
                with self._autocast():
                    vlm = self.encode_vlm(self._vlm_images(examples, j), [ex["lang"] for ex in examples], self._shapes(),
                                          with_grad=False)
        kf = self.plan_keyframes(examples, radio=radio, vlm=vlm, vlm_age_s=self.fixed_vlm_age_s or 0.0, memory=memory)
        actions = self.act_from_plan(examples, kf, 0.0, radio=radio)
        if os.environ.get("STARVLA_DUAL_CORRECTOR"):
            actions = self._apply_corrector(examples, actions, kf, radio)
        out = {"normalized_actions": actions.cpu().numpy()}
        if kwargs.get("return_keyframes", False):
            out["keyframes"] = self.plan_to_absolute(examples, kf).cpu().numpy()  # absolute normalised
        return out

    def _predict_fast_s1(self, examples: List[dict], mode: str) -> Dict:
        """Decoupled rates with a client-held plan (env STARVLA_DUAL_FASTS1, client ROBOCASA_S2_EVERY): examples carrying
        "plan_kf" (System 2 keyframes [K, D_a] this server returned earlier) and "plan_age_s" skip System 2; otherwise
        System 2 runs on the current frame (age 0) and its keyframes go back to the client as "plan_kf". System 1 on the
        current frame: mode <run>.pt = fast System 1 (S1Corrector trained by train_s1fast.py: the aged plan on its own
        window + C-RADIO [+ state]); "plan" = the aged plan itself (control); "s1" = this model's System 1, aged plan."""
        if self.relative_plan or self.delta_plan:
            raise NotImplementedError("STARVLA_DUAL_FASTS1 assumes an absolute plan")
        radio = self.encode_visual(examples)
        fresh = not all("plan_kf" in ex for ex in examples)
        if fresh:
            kf = self.plan_keyframes(examples, radio=radio)
            age = torch.zeros(len(examples), device=self._device)
        else:
            kf = torch.as_tensor(np.stack([np.asarray(ex["plan_kf"], dtype=np.float32) for ex in examples]), device=self._device)
            age = torch.as_tensor([float(ex.get("plan_age_s", 0.0)) for ex in examples], dtype=torch.float32, device=self._device)
        if mode == "plan":
            a = self._plan_base(kf, None, age)
        elif mode == "s1":
            a = self.act_from_plan(examples, kf, age, radio=radio)
        else:
            from starVLA.model.modules.dual_system.corrector import S1Corrector
            st = self._cur_state(self._state(examples, self._shapes()))
            if getattr(self, "_fast_s1", None) is None:
                self._fast_s1 = S1Corrector.load(mode, H=self.s1_horizon, d_state=int(np.prod(st.shape[1:])),
                                                 n_kf=self.timeline.num).to(self._device)
                print(f"[DualSystem] fast System 1 {mode}: H {self._fast_s1.H}, {self._fast_s1.cfg}", flush=True)
            net = self._fast_s1
            with torch.no_grad(), self._autocast():
                a = net(self._plan_base(kf, None, age, steps=net.H), radio, st if net.state is not None else None).float()
        out = {"normalized_actions": a.cpu().numpy()}
        if fresh:
            out["plan_kf"] = kf.cpu().numpy()
        return out

    def _apply_corrector(self, examples: List[dict], actions: torch.Tensor, kf: torch.Tensor, radio) -> torch.Tensor:
        """Test-time S1Corrector (dual_system/corrector.py) on System 1's output, env STARVLA_DUAL_CORRECTOR=<run>.pt:
        smoothed System 1 chunk + [state, System 2 keyframes, executed past (client ROBOCASA_NUM_PAST_STEPS)]."""
        from starVLA.model.modules.dual_system.corrector import S1Corrector, past_from_examples
        st = self._cur_state(self._state(examples, self._shapes()))
        if getattr(self, "_corrector", None) is None:
            path = os.environ["STARVLA_DUAL_CORRECTOR"]
            self._corrector = S1Corrector.load(path, H=self.s1_horizon, d_state=int(np.prod(st.shape[1:])),
                                               n_kf=self.timeline.num).to(self._device)
            print(f"[DualSystem] corrector {path}: {self._corrector.cfg}", flush=True)
        net = self._corrector
        pa = pv = None
        if net.pproj is not None:
            if "past_action" not in examples[0]:
                raise ValueError("the corrector needs the executed past: run the client with ROBOCASA_NUM_PAST_STEPS="
                                 f"{net.past_n} (abs actions)")
            pa, pv = past_from_examples(examples, net.past_n, self.action_dim, self._device)
        with torch.no_grad(), self._autocast():
            y = net(actions, radio, st, kf, pa, pv).float()
        out = actions.clone()
        out[:, :net.H] = y
        return out

    # ------------------------------------------------------------------ validation
    def evaluate_batch(self, examples: List[dict], seed: int = 0, mem=None) -> Dict[str, float]:
        devices = [self._device] if self._device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            return self._evaluate_batch(examples, mem=mem)

    def _evaluate_batch(self, examples: List[dict], mem=None) -> Dict[str, float]:
        """Normalized-action RMSE of System 2's sampled plan on the dense window (all / past / future) and of
        System 1 at every video offset (plan age), with System 2's plan (pred) and with the GT keyframes (gt_plan);
        s2_plan_on_s1_window = the interpolated System 2 plan scored on the same 1 s window (System 1 should beat it).
        stale_input vlm: the system2/* main metrics and system1/rmse_pred / s2_plan_on_s1_window use the VLM frame
        video_offsets[vlm_main_idx] (the current frame where it lies outside the episode, as in training); per VLM frame
        offset o the System 2 plan (dense / future_1s) and System 1 at t on that plan are also scored as
        *_vlmage{-o / rate}s (samples whose VLM frame lies inside the episode). mem: (memory context, progress logits)
        of a streamed memory (else the training unroll over the examples' memory frames)."""
        ev = self.eval_cfg
        sc = self._shapes()
        b = len(examples)
        frames = self._frames(examples, len(self.video_offsets))
        dense, dvalid = self._dense_targets(examples, sc)
        state = self._state(examples, sc)
        ref = self._plan_reference(state)  # relative_plan: plans are compared in absolute units (the same error)
        times = self.dense_times[None].expand(b, -1)
        rmse = lambda e, m: float(((e ** 2) * m[..., None]).sum() / (m.sum() * e.shape[-1]).clamp_min(1)) ** 0.5
        m: Dict[str, float] = {}
        vlm_mode = self.stale_input == "vlm"
        fv = self._frame_valid(examples)
        s2_steps = ev.get("s2_steps", None)
        plans = {}  # VLM frame index -> (keyframes, interpolated plan, sample inside the episode [B])
        with self._autocast():
            radio0 = self.encode_radio(frames[:, self.cur])
        mem_ctx = None
        jvs = range(len(self.video_offsets)) if vlm_mode else [self.cur]
        if self.episode_memory is not None:  # one memory per sample; System 2 scored at the main VLM frame and frame t
            if mem is None:
                with self._autocast():
                    mem_ctx, mem_logits, _ = self._memory_context(examples, sc)
            else:
                mem_ctx, mem_logits = mem
            prog = self._progress_targets(examples)
            if prog is not None:
                _, logs = self.episode_memory.progress_loss(mem_logits, prog)
                m.update(logs)
            jvs = sorted({self.vlm_main_idx, self.cur})
        for jv in jvs:
            age = -self.video_offsets[jv] / self.rate
            with self._autocast():
                kw = dict(vlm_idx=torch.full((b,), jv), vlm_age_s=torch.full((b,), age, device=self._device)) if vlm_mode else {}
                ctx, cmask = self._s2_context(examples, radio0, state, sc, train_s2=False, mem_ctx=mem_ctx, **kw)
                kf = self.system2.sample(ctx, cmask, steps=None if s2_steps is None else int(s2_steps)).float()
            plan, _ = self.system2.interpolator(kf, self.system2.key_times, times)  # keyframes -> dense timestamps
            plan = self._plan_to_abs(plan, ref)
            if vlm_mode:
                err = self.system2.dense_loss.error(plan, dense)
                ok = dvalid & fv[:, jv : jv + 1]
                m[f"system2/dense_rmse_vlmage{age:.2f}s"] = rmse(err, ok)
                m[f"system2/future_1s_rmse_vlmage{age:.2f}s"] = rmse(err, ok & (times > 1e-6) & (times <= 1.0 + 1e-6))
            plans[jv] = (kf, plan, fv[:, jv])
        if vlm_mode:  # main = the VLM frame of vlm_main_idx, the current frame where it lies outside the episode
            jm = self.vlm_main_idx
            use = (fv[:, jm] if jm != self.cur else torch.ones_like(fv[:, 0]))[:, None, None]
            kf_main = torch.where(use, plans[jm][0], plans[self.cur][0])
            plan_main = torch.where(use, plans[jm][1], plans[self.cur][1])
        else:
            kf_main, plan_main = plans[self.cur][0], plans[self.cur][1]
        err = self.system2.dense_loss.error(plan_main, dense)
        m["system2/dense_rmse"] = rmse(err, dvalid)
        m["system2/past_rmse"] = rmse(err, dvalid & (times < -1e-6))
        m["system2/future_rmse"] = rmse(err, dvalid & (times > 1e-6))
        m["system2/future_1s_rmse"] = rmse(err, dvalid & (times > 1e-6) & (times <= 1.0 + 1e-6))
        if "system1" not in self.train_systems:
            return m
        kf_gt = self.system2.extract_keyframes(self._plan_to_rel(dense, ref), times)
        s1_steps = ev.get("s1_steps", None)
        s1 = lambda kf_, delay_, radio_, st_: self._s1_actions(kf_, delay_, radio_, st_, ref, s1_steps)
        agg = {"pred": [], "gt_plan": [], "s2_plan": []}
        if vlm_mode:  # System 1 at t (plan age 0) on every VLM age's plan
            offs = torch.full((b,), 0, device=self._device)
            a1, v1 = self._s1_window(dense, dvalid, offs)
            v1 = v1 & fv[:, self.cur : self.cur + 1]
            st = state[:, self.cur : self.cur + 1] if self.system1.uses_state else None
            delay = offs.float()
            with self._autocast():
                a_gt = s1(kf_gt, delay, radio0, st)
                a_main = s1(kf_main, delay, radio0, st)
            p_main, _ = self._s1_window(plan_main, dvalid, offs)
            for k, a in (("gt_plan", a_gt), ("pred", a_main), ("s2_plan", p_main)):
                agg[k].append((self.system2.dense_loss.error(a, a1), v1))
            for jv, (kf, plan, ok) in plans.items():
                age = -self.video_offsets[jv] / self.rate
                vv = v1 & ok[:, None]
                with self._autocast():
                    a_pred = s1(kf, delay, radio0, st)
                p_win, _ = self._s1_window(plan, dvalid, offs)
                for k, a in (("pred", a_pred), ("s2_plan", p_win)):
                    m[f"system1/rmse_{k}_vlmage{age:.2f}s"] = rmse(self.system2.dense_loss.error(a, a1), vv)
        else:
            kf, plan = kf_main, plan_main
            for jj, o in enumerate(self.video_offsets):
                offs = torch.full((b,), o, device=self._device)
                a1, v1 = self._s1_window(dense, dvalid, offs)
                v1 = v1 & fv[:, jj : jj + 1]
                if not bool(v1.any()):
                    continue
                radio = radio0 if o == 0 else self.encode_radio(frames[:, jj])
                st = state[:, jj : jj + 1] if self.system1.uses_state else None
                delay = offs.float() / self.rate
                with self._autocast():
                    a_pred = s1(kf, delay, radio, st)
                    a_gt = s1(kf_gt, delay, radio, st)
                p_win, _ = self._s1_window(plan, dvalid, offs)
                for k, a in (("pred", a_pred), ("gt_plan", a_gt), ("s2_plan", p_win)):
                    e = self.system2.dense_loss.error(a, a1)
                    m[f"system1/rmse_{k}_delay{o / self.rate:.2f}s"] = rmse(e, v1)
                    agg[k].append((e, v1))
        for k, lst in agg.items():
            if lst:
                e = torch.cat([x for x, _ in lst]); v = torch.cat([y for _, y in lst])
                m[{"pred": "system1/rmse_pred", "gt_plan": "system1/rmse_gt_plan", "s2_plan": "system1/s2_plan_on_s1_window"}[k]] = rmse(e, v)
        return m

    def _load_eval_streams(self, is_main: bool):
        """dual_system.memory.stream: eval.stream_episodes (default 8) val episodes spread over the val split, every
        update interval from frame 0 (main process only; the val dataset is built on every rank)."""
        if self._eval_examples is not None:
            return self._eval_examples
        val_cfg = self.config.datasets.get("val_data", None) if hasattr(self.config.datasets, "get") else None
        if val_cfg is None:
            raise ValueError("dual_system.memory.stream: the streamed eval needs datasets.val_data")
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset

        ds = get_vla_dataset(data_cfg=val_cfg)
        if self.eval_cfg.get("train_stats"):
            tr = get_vla_dataset(data_cfg=self.config.datasets.vla_data)
            for d in ds.datasets:
                d.set_transforms_metadata(tr.merged_metadata[d.tag])
        stride = -self.video_offsets[self.mem_idx[0]]
        n_tr = len(ds.datasets[0].trajectory_ids)
        tis = np.unique(np.linspace(0, n_tr - 1, int(self.eval_cfg.get("stream_episodes") or 8)).round().astype(int))
        self._eval_examples = [ds.stream_episode_samples(0, int(ti), stride) for ti in tis] if is_main else []
        return self._eval_examples

    @torch.no_grad()
    def _stream_evaluate(self, episodes: List[List[dict]], seed: int = 0) -> Dict[str, float]:
        """The episodes in parallel from frame 0, as deployed: the memory updated on every new VLM frame t - stride,
        _evaluate_batch on each step with it. RMSE metrics pooled (sample-weighted RMS), memory/* averaged; plus the
        progress error |E[p] - p| per true-progress quarter (memory/progress_mae_q1..q4)."""
        jm = self.mem_idx[0]
        mems = [self.memory_init(1) for _ in episodes]
        parts, prog = [], []
        for s_ in range(max(len(e) for e in episodes)):
            ids = [i for i, e in enumerate(episodes) if s_ < len(e)]
            batch = [episodes[i][s_] for i in ids]
            m = torch.cat([mems[i] for i in ids])
            fv = self._frame_valid(batch)[:, jm]
            if bool(fv.any()):
                with self._autocast():
                    tok, mask = self.encode_vlm(self._vlm_images(batch, [jm] * len(batch)), [ex["lang"] for ex in batch],
                                                self._shapes(), with_grad=False)
                    m = self.episode_memory.update(m, tok, mask, fv)
            for r, i in enumerate(ids):
                mems[i] = m[r : r + 1]
            with self._autocast():
                ctx, logits = self.episode_memory.readout(m)
            parts.append((self.evaluate_batch(batch, seed=seed + s_, mem=(ctx, logits)), len(batch)))
            p = self._progress_targets(batch)
            if p is not None:
                prog.append(torch.stack([p.float(), self.episode_memory.expected_progress(logits)], 1))
        res = {}
        for k in parts[0][0]:
            vals = [(v[k], n) for v, n in parts if k in v and not math.isnan(v[k])]
            if not vals:
                res[k] = float("nan")
            elif k.startswith("memory/"):
                res[k] = float(sum(v * n for v, n in vals) / sum(n for _, n in vals))
            else:
                res[k] = float(np.sqrt(sum(v * v * n for v, n in vals) / sum(n for _, n in vals)))
        if prog:
            pe = torch.cat(prog).cpu().numpy()
            for q in range(4):
                sel = (pe[:, 0] >= q / 4) & ((pe[:, 0] < (q + 1) / 4) | (q == 3))
                res[f"memory/progress_mae_q{q + 1}"] = float(np.abs(pe[sel, 1] - pe[sel, 0]).mean()) if sel.any() else float("nan")
        return res

    def _load_eval_examples(self, get_batch):
        if self._eval_examples is not None:
            return self._eval_examples
        n = int(self.eval_cfg["num_samples"] or 64)
        val_cfg = self.config.datasets.get("val_data", None) if hasattr(self.config.datasets, "get") else None
        if val_cfg is not None:
            from starVLA.dataloader.lerobot_datasets import get_vla_dataset

            ds = get_vla_dataset(data_cfg=val_cfg)
            if self.eval_cfg.get("train_stats"):
                # the val mixture merges only the val split's statistics -> normalise it with the TRAIN mixture's
                tr = get_vla_dataset(data_cfg=self.config.datasets.vla_data)
                for d in ds.datasets:
                    d.set_transforms_metadata(tr.merged_metadata[d.tag])
            rng = np.random.default_rng(0)
            self._eval_examples = [ds[int(i)] for i in rng.choice(len(ds), size=min(n, len(ds)), replace=False)]
        else:
            self._eval_examples = list(get_batch())[:n]
        return self._eval_examples

    def trainer_eval_hook(self, get_batch, step: int, output_dir: str, is_main: bool = True) -> Dict[str, float]:
        """Every eval_interval on every rank (the val dataset constructor has a barrier); metrics on the main process
        only, also written to <output_dir>/samples/step_XXXXXXX.json."""
        examples = self._load_eval_streams(is_main) if self.mem_tbptt else self._load_eval_examples(get_batch)
        if not is_main:
            return {}
        mods = {mod: mod.training for mod in (self.system2, self.system1, self.qwen_vl_interface)}
        for mod in mods:
            mod.eval()
        try:
            ev = self.eval_cfg
            res = {}
            bs = 32
            if self.mem_tbptt:
                res = self._stream_evaluate(examples, seed=int(ev["seed"] or 0))
            else:
                parts = [self.evaluate_batch(examples[i:i + bs], seed=int(ev["seed"] or 0) + i) for i in range(0, len(examples), bs)]
                for k in parts[0]:
                    vals = [p[k] for p in parts if k in p and not math.isnan(p[k])]
                    res[k] = float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")
        finally:
            for mod, was in mods.items():
                mod.train(was)
        out_dir = Path(output_dir) / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"step_{step:07d}.json", "w") as f:
            json.dump({"metrics": res, "num_samples": sum(len(e) for e in examples) if self.mem_tbptt else len(examples)}, f, indent=1)
        return {f"eval/{k}": v for k, v in res.items()}
