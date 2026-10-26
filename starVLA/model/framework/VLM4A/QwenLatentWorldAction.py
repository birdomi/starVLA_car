# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
QwenLatentWorldAction: System 2 (Qwen VLM) -> System 1.5 (C-RADIOv4 latent-change flow DiT) -> System 1 (action
query regression decoder).

    current image(s) + instruction --> System 2: Qwen3.5-0.8B (repo VLM interface), one forward pass,
                                       S = last hidden states of EVERY token (vision + text), padding masked  [B, L, D_s]
    current image (C-RADIO view)  --> frozen C-RADIOv4, spatial patch tokens only (no summary)          Z_t [B, N, D_c]
    (Z_t, S) --> System 1.5 LatentFlowDiT: conditional flow matching of the latent change dZ = Z_f - Z_t,
                 p(dZ | Z_t, S); X_tau = (1 - tau) eps + tau X_1, u = X_1 - eps, L_FM = ||v - u||^2
                 inference: X_0 ~ N(0, I) -> ODE (euler | heun | ...) -> X_1 = dZ_hat (scaled)           [B, N, D_c]
             --> H_f = backbone(dZ_hat, tau = 1 | Z_t, S): a SEPARATE forward pass after the ODE, output of the last
                 block + final normalization, i.e. right before the velocity head                      [B, N, d]
    (Q_action, H_f) --> System 1 ActionQueryDecoder: learnable queries, self-attn + cross-attn to H_f, MLP,
                 regression head -> action chunk A_hat (one deterministic pass, no diffusion)       [B, H_a, D_a]

The future frame is a training target only (System 1.5 loss / the noisy FM input); predict_action never reads it,
nor the GT actions.

Training (framework.latent_world.stage), default = C with joint.action_path "endpoint", from scratch:
  C  end-to-end joint training of System 1.5 and System 1 (both randomly initialised; Qwen / C-RADIO frozen):
       FM pass      v^ = v_theta(X_tau, tau | Z_t, S),  L_FM = ||v^ - (X_1 - eps)||^2
       endpoint     dZ~ = X_tau + (1 - tau) v^          one-step projection of the SAME forward pass, differentiable
       hidden pass  H_f = backbone(dZ~, tau = 1 | Z_t, S)        (extract_final_hidden, = inference)
       System 1     A^ = decoder(Q_action, H_f),  L_action = SmoothL1(A^, A)
       L = lambda_FM L_FM + lambda_A(step) L_action, one backward; L_action -> H_f -> dZ~ -> v^ -> System 1.5.
     lambda_A ramps linearly from joint.action_loss_weight_start to loss_weights.action over
     joint.action_loss_warmup_steps. joint.action_grad: full (default) | final_pass (dZ~ detached: gradient only via
     the hidden pass) | none (H_f detached). joint.action_path "ode" = System 1 on ODE samples instead: one per
     joint.ode_steps count (e.g. [1, 4], same noise, euler shares the tau = 0 velocity) -> one tau = 1 hidden pass
     over the K samples -> System 1, L_action averaged over them; action_grad full = differentiable solver.
     joint.gt_prob_* (scheduled teacher forcing, per example) applies to both paths.
  A  (staged recipe) System 1.5 only, L_FM.   B  (staged) System 1 only on the frozen System 1.5's ODE sample
     (pred) | GT dZ (gt) | mix | an on-disk cache of predicted H_f.
Inference is always the full ODE: X_0 ~ N(0, I) -> sample_steps (euler | heun ...) -> dZ_hat -> H_f -> A^.

Data: each example's "image" list follows framework.latent_world.image_layout ("t" = current view, "f" = future
frame of the C-RADIO view at t + future_offset). With datasets.*.emit_frame_meta the loader also sends
frame_offsets (checked against the layout), future_valid (False when the future index was clamped at the episode
end -> excluded from L_FM / teacher forcing) and sample_key / frame_keys (cache keys).
"""

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
from starVLA.model.modules.latent_world_action import (
    ActionQueryDecoder,
    CRADIOVisualEncoder,
    FlowConditions,
    LatentFlowDiT,
    LatentFlowSampler,
    TensorCache,
    apply_lora,
    regression_loss,
)
from starVLA.model.modules.latent_world_action.latent_flow_dit import one_step_endpoint
from starVLA.model.modules.latent_world_action.layers import GradScale
from starVLA.model.modules.tri_system.shapes import ShapeChecker, System2Tokens
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

STAGES = ("A", "B", "C")
HIDDEN_SOURCES = ("pred", "gt", "mix", "cache")
ACTION_GRADS = ("none", "final_pass", "full")
ACTION_PATHS = ("endpoint", "ode")


@dataclass
class LatentWorldActionConfig:
    """Defaults of framework.* for QwenLatentWorldAction. YAML values override them.

    NOTE: every key the model reads must have a default here: the run's config.yaml (used by from_pretrained) only
    keeps keys of this merged tree, a yaml-only key read later would be lost."""

    name: str = "QwenLatentWorldAction"
    # System 2: the repo's VLM interface (starVLA/model/modules/vlm), unmodified
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "Qwen/Qwen3.5-0.8B",
        "attn_implementation": "flash_attention_2",
        "torch_dtype": "bfloat16",  # float32 when Qwen is fully fine-tuned with ZeRO-0
    })
    system2: dict = field(default_factory=lambda: {
        "finetune": "frozen",  # frozen | full | lora   (full / lora: drop qwen_vl_interface from trainer.freeze_modules)
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
    })
    radio: dict = field(default_factory=lambda: {
        "path": "playground/Pretrained_models/C-RADIOv4-SO400M",  # or a C-RADIOv4-H HF dir
        "input_size": [256, 256],  # (H, W), multiple of 16 -> (H/16) x (W/16) patch tokens, same for current / future
        "feature": "backbone",  # backbone | penultimate | siglip2-g | dino_v3_7b | sam3 (heads need teacher_ckpt)
        "teacher_ckpt": None,
        "dtype": "bfloat16",
        "outlier_norm": 100.0,  # C-RADIO's fixed high-norm artifact tokens are masked
        "mask_outliers": True,
        "normalize_features": True,  # per-channel (Z - mean) / std with stats_path
        "stats_path": None,  # starVLA/training/latent_world_action_tools.py stats
        "trainable": False,
        "cache_dir": None,  # on-disk cache of raw frozen features (needs datasets.*.emit_frame_meta)
    })
    system15: dict = field(default_factory=lambda: {
        "d_model": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0,
        "cond_dim": None,  # width of the projected Z_t / S cross-attention streams (None = d_model)
        "qk_norm": True, "dropout": 0.0, "attn_dropout": 0.0,
        "adaln_cond": "tau",  # tau | tau+s2 (also pooled S; = QwenGR00T_TriSystem System 1.5, convertible ckpts)
        "tau_schedule": {"schedule": "uniform"},  # uniform | logit_normal (mean, std) | beta
        "num_noise_samples": 1,  # (eps, tau) draws per example in L_FM
        "delta_norm": "rms",  # rms: X_1 = dZ / per-channel RMS(dZ) (unit scale) | none
        "sample_steps": 4, "solver": "euler",  # inference ODE
        "grad_ckpt": False,
    })
    system1: dict = field(default_factory=lambda: {
        "d_model": 512, "depth": 4, "num_heads": 8, "mlp_ratio": 4.0, "dropout": 0.1, "attn_dropout": 0.0,
        "qk_norm": True, "add_grid_pos": True,
        "state_conditioning": "token",  # token | add | none (ignored when action_model.state_dim is 0)
        "loss": "smooth_l1", "smooth_l1_beta": 0.1,  # smooth_l1 | l1 | mse
        "hidden_source": "pred",  # pred (= inference path) | gt (teacher forcing) | mix | cache (stage B only)
        "pred_prob": 0.5,  # mix: probability of the predicted path per example
        "hidden_cache_dir": None,
    })
    action_model: dict = field(default_factory=lambda: {"action_dim": 7, "state_dim": 0, "action_horizon": 8})
    latent_world: dict = field(default_factory=lambda: {
        "stage": "C",  # C = end-to-end joint training (default) | A / B = staged recipe
        # one entry per loader image: "t" current view (System 2 input), "f" future frame of the C-RADIO view
        "image_layout": ["t", "f"],
        "radio_view": 0,  # index among the "t" views encoded by C-RADIO (same camera as "f")
        # video frames between t and the "f" frame (= data config video delta). Action chunk = a_t .. a_{t+H_a-1};
        # with future_offset == action_horizon, Z_f is the observation right after the chunk was executed.
        "future_offset": 8,
        "mask_clamped_future": True,  # drop samples whose future index was clamped at the episode end from L_FM
        "loss_weights": {"fm": 1.0, "action": 1.0},  # lambda_FM, target lambda_action (stage C; A / B use one loss)
        "joint": {
            # endpoint: System 1 reads backbone(dZ~ = X_tau + (1 - tau) v^, tau = 1) of the FM pass (no ODE in training)
            # ode: System 1 reads backbone(ODE sample, tau = 1) (previous stage-C behaviour)
            "action_path": "endpoint",
            "endpoint_tau": "shared",  # shared: the FM draw (X_tau, tau) | zero: own pure-noise pass at tau 0 (no GT in X)
            # how far the action loss reaches into System 1.5: full | final_pass (only the tau = 1 pass) | none
            "action_grad": "full",
            "action_grad_scale": 1.0,  # multiplier on that gradient
            "full_grad_steps": None,  # action_path ode + full: differentiable ODE steps (None = sample_steps)
            # action_path ode (stage C): one ODE sample per listed step count from the same noise, each -> H_f ->
            # System 1, L_action averaged over them (e.g. [1, 4]); None = [full_grad_steps or sample_steps]
            "ode_steps": None,
            "action_loss_schedule": "linear",  # linear: start -> loss_weights.action over warmup_steps | fixed
            "action_loss_warmup_steps": 2000,
            "action_loss_weight_start": 0.1,
            # scheduled teacher forcing of the endpoint / ode paths (training only): per example, System 1 reads the GT
            # X_1 instead of dZ~ (or the ODE samples) with probability p_gt(step) = gt_prob_start -> gt_prob_end over gt_prob_decay_steps
            # (linear | cosine; fixed = gt_prob_end). Clamped futures always read dZ~; eval / inference never read GT.
            "gt_prob_schedule": "linear",
            "gt_prob_start": 1.0,
            "gt_prob_end": 0.05,
            "gt_prob_decay_steps": 10000,
            "grad_norm_log_interval": None,  # per-loss gradient norms every N steps (None = trainer.logging_frequency, 0 = off)
        },
        "check_shapes": True,
        "eval": {"num_samples": 64, "batch_size": 16, "seed": 0},
    })


@dataclass
class Observation:
    """encode_observation() output."""

    s2: System2Tokens  # S [B, L, D_s] + mask [B, L]
    z_t: torch.Tensor  # [B, N, D_c] current C-RADIO tokens (model latent units)
    valid: torch.Tensor  # [B, N] bool, False = C-RADIO artifact token
    state: Optional[torch.Tensor] = None  # [B, 1, D_st]
    z_f: Optional[torch.Tensor] = None  # [B, N, D_c] future tokens (training targets only)
    valid_f: Optional[torch.Tensor] = None  # [B, N]
    future_valid: Optional[torch.Tensor] = None  # [B] bool, False = future index clamped at the episode end


def _as_list(images):
    return list(images) if isinstance(images, (list, tuple)) else [images]


def _module_fingerprint(*modules) -> str:
    h = hashlib.sha1()
    for m in modules:
        for name, t in list(m.named_parameters()) + list(m.named_buffers()):
            t = t.detach().float()
            h.update(f"{name}{tuple(t.shape)}{t.sum().item():.6e}{t.abs().sum().item():.6e}".encode())
    return h.hexdigest()[:16]


@FRAMEWORK_REGISTRY.register("QwenLatentWorldAction")
class QwenLatentWorldAction(baseframework):
    def __init__(self, config: Optional[dict] = None, vlm_interface=None, radio_backbone=None, **kwargs) -> None:
        """vlm_interface / radio_backbone inject stand-ins (CPU unit tests); training / inference build the real ones."""
        super().__init__()
        self.config = merge_framework_config(LatentWorldActionConfig, config)
        fw = self.config.framework
        lw = fw.latent_world
        self.stage = str(lw.stage).upper()
        if self.stage not in STAGES:
            raise ValueError(f"framework.latent_world.stage must be one of {STAGES}, got {lw.stage!r}")
        self.layout = [str(v) for v in lw.image_layout]
        if not self.layout or set(self.layout) - {"t", "f"} or "t" not in self.layout:
            raise ValueError(f"latent_world.image_layout needs 't' (current) / 'f' (future) entries, got {self.layout}")
        self.t_idx = [i for i, v in enumerate(self.layout) if v == "t"]
        f_idx = [i for i, v in enumerate(self.layout) if v == "f"]
        self.f_idx = f_idx[-1] if f_idx else None  # several "f": the last one is the target
        self.radio_view = int(lw.get("radio_view", 0))
        if not 0 <= self.radio_view < len(self.t_idx):
            raise ValueError(f"latent_world.radio_view {self.radio_view} out of range for {len(self.t_idx)} current views")
        self.future_offset = int(lw.future_offset)
        self.loss_weights = {k: float(v) for k, v in dict(lw.loss_weights).items()}
        joint = lw.get("joint", {})
        self.action_grad = str(joint.get("action_grad", "full"))
        if self.action_grad not in ACTION_GRADS:
            raise ValueError(f"latent_world.joint.action_grad must be one of {ACTION_GRADS}, got {self.action_grad!r}")
        self.action_path = str(joint.get("action_path", "endpoint"))
        if self.action_path not in ACTION_PATHS:
            raise ValueError(f"latent_world.joint.action_path must be one of {ACTION_PATHS}, got {self.action_path!r}")
        self.endpoint_tau = str(joint.get("endpoint_tau", "shared"))
        if self.endpoint_tau not in ("shared", "zero"):
            raise ValueError(f"latent_world.joint.endpoint_tau must be shared | zero, got {self.endpoint_tau!r}")
        if str(joint.get("gt_prob_schedule", "linear")) not in ("linear", "cosine", "fixed"):
            raise ValueError(f"latent_world.joint.gt_prob_schedule must be linear | cosine | fixed, "
                             f"got {joint.get('gt_prob_schedule')!r}")
        for k in ("gt_prob_start", "gt_prob_end"):
            if not 0.0 <= float(joint.get(k, 0.0)) <= 1.0:
                raise ValueError(f"latent_world.joint.{k} must be in [0, 1], got {joint.get(k)!r}")
        self.shapes = ShapeChecker(bool(lw.get("check_shapes", True)))
        # optimizer step of the trainer (set_training_step); fallback: count training forwards
        self._forward_count = 0
        self._trainer_step: Optional[int] = None

        # System 2
        self.qwen_vl_interface = vlm_interface if vlm_interface is not None else get_vlm_model(config=self.config)
        self.s2_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        self.s2_finetune = str(fw.system2.get("finetune", "frozen"))
        if self.s2_finetune not in ("frozen", "full", "lora"):
            raise ValueError(f"framework.system2.finetune must be frozen | full | lora, got {self.s2_finetune!r}")
        if self.s2_finetune == "lora":
            lc = fw.system2.lora
            n = apply_lora(self.qwen_vl_interface.model, r=int(lc.r), alpha=float(lc.alpha),
                           dropout=float(lc.get("dropout", 0.0)), target_modules=list(lc.target_modules))
            logger.info(f"System 2 LoRA: {n} Linear layers wrapped (r={lc.r}, alpha={lc.alpha})")

        # C-RADIOv4
        self.radio_encoder = CRADIOVisualEncoder.from_config(fw.radio, backbone=radio_backbone)
        self.feat_dim, self.grid = self.radio_encoder.feat_dim, self.radio_encoder.grid
        self.delta_norm = str(fw.system15.get("delta_norm", "rms"))
        if self.delta_norm not in ("rms", "none"):
            raise ValueError(f"system15.delta_norm must be rms | none, got {self.delta_norm!r}")

        # System 1.5
        s15 = fw.system15
        self.system15 = LatentFlowDiT(
            self.feat_dim, self.s2_dim, self.grid, d_model=int(s15.d_model), depth=int(s15.depth),
            num_heads=int(s15.num_heads), mlp_ratio=float(s15.mlp_ratio), cond_dim=s15.get("cond_dim", None),
            qk_norm=bool(s15.get("qk_norm", True)), dropout=float(s15.get("dropout", 0.0)),
            attn_dropout=float(s15.get("attn_dropout", 0.0)), adaln_cond=str(s15.get("adaln_cond", "tau")),
            grad_ckpt=bool(s15.get("grad_ckpt", False)),
        )
        self.sampler = LatentFlowSampler(int(s15.sample_steps), str(s15.solver))
        ode_steps = joint.get("ode_steps", None)
        if ode_steps is None:  # legacy: one sample with full_grad_steps (action_grad full) or sample_steps
            fgs = joint.get("full_grad_steps", None) if self.action_grad == "full" else None
            ode_steps = [int(fgs) if fgs else self.sampler.steps]
        ode_steps = [int(ode_steps)] if isinstance(ode_steps, (int, float, str)) else [int(n) for n in ode_steps]
        if not ode_steps or min(ode_steps) < 1 or len(set(ode_steps)) != len(ode_steps):
            raise ValueError(f"latent_world.joint.ode_steps must be distinct step counts >= 1, got {ode_steps}")
        self.ode_steps = ode_steps

        # System 1
        am, s1 = fw.action_model, fw.system1
        self.action_horizon, self.action_dim = int(am.action_horizon), int(am.action_dim)
        self.state_dim = int(am.get("state_dim", 0) or 0)
        self.system1 = ActionQueryDecoder(
            self.action_dim, self.action_horizon, int(s15.d_model), self.grid, state_dim=self.state_dim,
            d_model=int(s1.d_model), depth=int(s1.depth), num_heads=int(s1.num_heads), mlp_ratio=float(s1.mlp_ratio),
            dropout=float(s1.get("dropout", 0.0)), attn_dropout=float(s1.get("attn_dropout", 0.0)),
            qk_norm=bool(s1.get("qk_norm", True)), add_grid_pos=bool(s1.get("add_grid_pos", True)),
            state_conditioning=str(s1.get("state_conditioning", "token")),
        )
        self.hidden_source = str(s1.get("hidden_source", "pred"))
        if self.hidden_source not in HIDDEN_SOURCES:
            raise ValueError(f"system1.hidden_source must be one of {HIDDEN_SOURCES}, got {self.hidden_source!r}")
        if self.hidden_source == "cache" and self.stage != "B":
            raise ValueError("system1.hidden_source: cache needs a frozen System 1.5, i.e. latent_world.stage B")

        self.use_action_head = self.stage != "A"  # trainer: skip its predict_action eval while System 1 is untrained
        self._hidden_cache = None
        self._eval_examples = None
        self._warned = set()
        self._apply_trainability()
        logger.info(self.describe_shapes())

    # ------------------------------------------------------------------ setup helpers
    def _warn_once(self, key: str, msg: str):
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg)

    def _apply_trainability(self):
        """Stage-wise requires_grad (the trainer's freeze_modules should list the same frozen modules)."""
        plan = {
            "qwen_vl_interface": self.s2_finetune != "frozen",
            "radio_encoder": bool(self.config.framework.radio.get("trainable", False)),
            "system15": self.stage in ("A", "C"),
            "system1": self.stage in ("B", "C"),
        }
        for name, trainable in plan.items():
            if not trainable:
                getattr(self, name).requires_grad_(False)
        self.trainable_modules = [k for k, v in plan.items() if v]
        trainer = getattr(self.config, "trainer", None)
        frozen_cfg = str(trainer.get("freeze_modules", "") or "") if trainer is not None and hasattr(trainer, "get") else ""
        frozen_cfg = {p.strip() for p in frozen_cfg.split(",") if p.strip()}
        for name, trainable in plan.items():
            if not trainable and name not in frozen_cfg:
                logger.warning(f"[QwenLatentWorldAction] stage {self.stage}: '{name}' is frozen by the framework; "
                               "add it to trainer.freeze_modules so the optimizer does not hold it")
            if trainable and name in frozen_cfg:
                logger.warning(f"[QwenLatentWorldAction] stage {self.stage} trains '{name}' but trainer.freeze_modules "
                               "freezes it")

    def describe_shapes(self) -> str:
        gh, gw = self.grid
        n, s15, s1 = gh * gw, self.system15, self.system1
        return "\n".join([
            f"QwenLatentWorldAction stage {self.stage} (trainable: {self.trainable_modules}), layout {self.layout}, "
            f"future_offset {self.future_offset}",
            f"  System 2   S   [B, L, {self.s2_dim}]  last hidden states of all prompt tokens (finetune={self.s2_finetune})",
            f"  C-RADIO    Z_t [B, {n}, {self.feat_dim}]  ({gh}x{gw} patches of {self.radio_encoder.input_size}, "
            f"normalize={self.radio_encoder.normalize})",
            f"  System 1.5 X_tau [B, {n}, {self.feat_dim}] -> H [B, {n}, {s15.d_model}] -> v [B, {n}, {self.feat_dim}]  "
            f"(depth {len(s15.blocks)}, adaLN {s15.adaln_cond}, ODE {self.sampler.solver} x{self.sampler.steps})",
            f"  System 1   queries [B, {s1.horizon}, {s1.queries.shape[-1]}] x cross-attn H_f -> A [B, {self.action_horizon}, "
            f"{self.action_dim}]  (state {s1.state_conditioning}, "
            + (f"joint: action_path {self.action_path}, "
               + (f"endpoint_tau {self.endpoint_tau}, " if self.action_path == "endpoint" else
                  f"ode_steps {self.ode_steps}, ")
               + f"action_grad {self.action_grad}, "
               f"p_gt {self.endpoint_gt_prob(0):g} -> {self.endpoint_gt_prob(10 ** 9):g})"
               if self.stage == "C" else f"hidden_source {self.hidden_source})"),
        ])

    @property
    def _device(self) -> torch.device:
        return self.system15.x_proj.weight.device

    @property
    def s2_trainable(self) -> bool:
        return self.s2_finetune != "frozen"

    def _obs_size(self):
        """(W, H) of the loader frames (datasets.vla_data.obs_image_size), else None."""
        ds = getattr(self.config, "datasets", None)
        vd = ds.get("vla_data", None) if ds is not None and hasattr(ds, "get") else None
        size = vd.get("obs_image_size", None) if vd is not None and hasattr(vd, "get") else None
        return (int(size[0]), int(size[-1])) if size else None

    # ------------------------------------------------------------------ example parsing
    def _split_images(self, ex: dict):
        """-> (current views [V], future frame | None, frame keys of (radio current, future) | (None, None))."""
        ims = _as_list(to_pil_preserve(ex["image"]))
        keys = ex.get("frame_keys", None)
        if len(ims) == len(self.layout):
            cur = [ims[i] for i in self.t_idx]
            fut = ims[self.f_idx] if self.f_idx is not None else None
            if keys is not None and len(keys) == len(self.layout):
                kc = keys[self.t_idx[self.radio_view]]
                return cur, fut, (kc, keys[self.f_idx] if self.f_idx is not None else None)
            return cur, fut, (None, None)
        if len(ims) == len(self.t_idx):  # inference: the client sends the current views only
            return ims, None, (None, None)
        raise ValueError(f"example has {len(ims)} images; image_layout {self.layout} expects {len(self.layout)} "
                         f"(training) or {len(self.t_idx)} (current views only)")

    def _check_frame_meta(self, examples: List[dict], need_future: bool):
        if "frame_offsets" not in examples[0]:
            if need_future:
                self._warn_once("meta", "examples carry no frame_offsets / future_valid (datasets.*.emit_frame_meta: "
                                "false): the future offset cannot be verified and episode-end clamped futures are "
                                "not masked")
            return
        if "meta_ok" in self._warned:
            return
        offs = [int(o) for o in examples[0]["frame_offsets"]]
        if len(offs) != len(self.layout):
            raise ValueError(f"loader sends {len(offs)} frames (offsets {offs}), image_layout {self.layout} expects "
                             f"{len(self.layout)}")
        bad = [i for i in self.t_idx if offs[i] != 0]
        if bad or (self.f_idx is not None and offs[self.f_idx] != self.future_offset):
            raise ValueError(f"loader frame offsets {offs} do not match image_layout {self.layout} with current views "
                             f"at offset 0 and the target 'f' at future_offset {self.future_offset}; fix the data "
                             "config (video delta_indices) or framework.latent_world")
        self._warned.add("meta_ok")

    def _resize(self, im: Image.Image, size) -> Image.Image:
        im = im.convert("RGB")
        return im if size is None or im.size == size else im.resize(size)

    def _to_tensor(self, pils: Sequence[Image.Image]) -> torch.Tensor:
        """PIL list -> [M, 3, H, W] float in [0, 1] (all frames share the loader size; C-RADIO resizes after)."""
        size = self._obs_size() or pils[0].size
        arr = np.stack([np.asarray(self._resize(im, size)) for im in pils])
        return torch.from_numpy(arr).to(self._device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)

    def _state(self, examples: List[dict]) -> Optional[torch.Tensor]:
        if not self.system1.uses_state:
            return None
        if "state" not in examples[0]:
            raise KeyError(f"System 1 is state-conditioned (state_dim {self.state_dim}): every example needs 'state' "
                           "[1, D_st] (datasets.*.include_state: true)")
        state = torch.tensor(np.array([ex["state"] for ex in examples]), device=self._device, dtype=torch.float32)
        return self.shapes("state", state.reshape(len(examples), 1, -1), f"B 1 {self.state_dim}")

    def _actions(self, examples: List[dict]) -> torch.Tensor:
        a = torch.tensor(np.array([ex["action"] for ex in examples]), device=self._device, dtype=torch.float32)
        return self.shapes("A_gt", a[:, -self.action_horizon:], f"B {self.action_horizon} {self.action_dim}")

    # ------------------------------------------------------------------ System 2 / C-RADIO
    def encode_system2(self, images: List[List[Image.Image]], instructions: List[str]) -> System2Tokens:
        """One Qwen forward pass on the current view(s) + instruction; S = last hidden states of every token."""
        iface = self.qwen_vl_interface
        inputs = iface.build_qwenvl_inputs(images=images, instructions=instructions)
        with torch.set_grad_enabled(self.s2_trainable and torch.is_grad_enabled()), \
                torch.autocast("cuda", dtype=getattr(iface, "autocast_dtype", torch.bfloat16),
                               enabled=self._device.type == "cuda"):
            out = iface(**inputs, output_attentions=False, output_hidden_states=True, return_dict=True,
                        logits_to_keep=1)  # hidden states only, no full-vocabulary logits
        tokens = out.hidden_states[-1]
        if tokens.dim() != 3 or tokens.shape[-1] != self.s2_dim or tokens.shape[:2] != inputs["attention_mask"].shape:
            raise ValueError(f"System 2 hidden states {tuple(tokens.shape)} do not match [B, L, {self.s2_dim}] / the mask")
        return System2Tokens(tokens, inputs["attention_mask"].bool().to(tokens.device))

    def encode_observation(self, examples: List[dict], with_future: bool = False,
                           with_state: bool = True) -> Observation:
        """System 2 + C-RADIO features of the CURRENT observation (and, for training targets only, of the future
        frame) + the proprioceptive state if System 1 uses it. Never reads "action".
        Starts a new shape-check scope (B, L_s, N, ... are re-bound for this batch)."""
        self.shapes.reset()
        parsed = [self._split_images(ex) for ex in examples]
        views = [[self._resize(im, self._obs_size()) for im in cur] for cur, _, _ in parsed]
        s2 = self.encode_system2(views, [ex["lang"] for ex in examples])
        self.shapes("S", s2.tokens, "B L_s D_s")
        cur = self._to_tensor([v[self.radio_view] for v in views])
        keys_c = [k[0] for _, _, k in parsed]
        obs_kw = {}
        if with_future:
            if any(f is None for _, f, _ in parsed):
                raise ValueError(f"the future frame is needed (System 1.5 target / teacher forcing) but the examples "
                                 f"have no 'f' image (image_layout {self.layout}); use a *_future data config")
            fut = self._to_tensor([f for _, f, _ in parsed])
            keys_f = [k[1] for _, _, k in parsed]
            enc = self.radio_encoder
            if enc.trainable:  # gradient only through the current frame; the future frame is a stop-grad target
                pt = enc(cur, keys_c)
                with torch.no_grad():
                    pf = enc(fut, keys_f)
                z_t, v_t, z_f, v_f = pt.z, pt.valid, pf.z, pf.valid
            else:
                both = enc(torch.cat([cur, fut]), keys_c + keys_f)
                (z_t, z_f), (v_t, v_f) = both.z.chunk(2), both.valid.chunk(2)
            fv = torch.tensor([bool(ex.get("future_valid", True)) for ex in examples], device=self._device)
            obs_kw = {"z_f": self.shapes("Z_f", z_f, "B N D_c"), "valid_f": v_f, "future_valid": fv}
        else:
            pt = self.radio_encoder(cur, keys_c)
            z_t, v_t = pt.z, pt.valid
        state = self._state(examples) if with_state else None
        return Observation(s2, self.shapes("Z_t", z_t, "B N D_c"), v_t, state, **obs_kw)

    # ------------------------------------------------------------------ System 1.5
    def delta_scale(self) -> torch.Tensor:
        """Per-channel scale s of the flow data X_1 = dZ / s ([D_c]); unit RMS with delta_norm rms."""
        st = self.radio_encoder.stats
        if self.delta_norm == "none":
            return torch.ones_like(st.dz_rms)
        if not bool(st.loaded):
            self._warn_once("stats", "C-RADIO stats not loaded (radio.stats_path): identity feature / delta scale")
        return st.dz_rms if self.radio_encoder.normalize else st.dz_rms * st.z_std

    def latent_change_target(self, obs: Observation) -> torch.Tensor:
        """GT X_1 = (Z_f - Z_t) / s [B, N, D_c]; stop-gradient target (Z_t enters only as a condition)."""
        x1 = (obs.z_f - obs.z_t.detach()) / self.delta_scale()
        return x1.masked_fill(~(obs.valid & obs.valid_f)[..., None], 0.0)

    def latent_change_to_features(self, x1: torch.Tensor, obs: Observation):
        """X_1 -> (dZ_hat, Z_f_hat = Z_t + dZ_hat) in C-RADIO latent units (normalized features if normalize)."""
        delta = x1 * self.delta_scale()
        return delta, obs.z_t + delta

    def flow_conditions(self, obs: Observation) -> FlowConditions:
        return self.system15.encode_conditions(obs.z_t, obs.valid, obs.s2.tokens, obs.s2.mask)

    def sample_latent_change(self, obs: Observation, cond: Optional[FlowConditions] = None, steps: Optional[int] = None,
                             solver: Optional[str] = None, noise: Optional[torch.Tensor] = None,
                             generator: Optional[torch.Generator] = None, grad: bool = False,
                             record_last_hidden: bool = False):
        """System 1.5 ODE from X_0 ~ N(0, I) (tau = 0) to X_1 = dZ_hat (tau = 1), [B, N, D_c] in scaled units.
        grad=True keeps the graph through every solver step (joint stage, action_grad: full)."""
        cond = self.flow_conditions(obs) if cond is None else cond
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            out = self.sampler.sample(self.system15, cond, noise=noise, steps=steps, solver=solver,
                                      generator=generator, record_last_hidden=record_last_hidden)
        x1, last_h = out if record_last_hidden else (out, None)
        x1 = self.shapes("dZ_hat", x1.masked_fill(~obs.valid[..., None], 0.0), "B N D_c")
        return (x1, last_h) if record_last_hidden else x1

    def extract_final_hidden(self, obs: Observation, x1: torch.Tensor,
                             cond: Optional[FlowConditions] = None) -> torch.Tensor:
        """H_f = FlowDiT_Backbone(X_1, tau = 1 | Z_t, S): a separate forward pass AFTER the ODE, output of the last
        Transformer block + final normalization (before the velocity head), [B, N, d]. Training (every
        hidden_source) and inference use this same function."""
        cond = self.flow_conditions(obs) if cond is None else cond
        tau = torch.ones(x1.shape[0], device=x1.device)
        return self.shapes("H_f", self.system15.backbone(x1, tau, cond), "B N D_h")

    def _predicted_hidden(self, obs: Observation, cond: FlowConditions, generator=None) -> torch.Tensor:
        """The inference path: ODE sample -> tau = 1 hidden pass."""
        return self.extract_final_hidden(obs, self.sample_latent_change(obs, cond, generator=generator), cond)

    # ------------------------------------------------------------------ System 1 conditioning in training
    def _training_hidden(self, obs: Observation, cond: FlowConditions, out: Dict) -> torch.Tensor:
        src, s1 = self.hidden_source, self.config.framework.system1
        mode = self.action_grad if self.stage == "C" else "none"
        use_gt = None
        if src in ("gt", "mix"):
            use_gt = obs.future_valid.clone()  # clamped futures fall back to the predicted path
            if src == "mix":
                use_gt &= torch.rand(use_gt.shape[0], device=use_gt.device) >= float(s1.get("pred_prob", 0.5))
        need_pred = use_gt is None or not bool(use_gt.all())
        x_in = None
        if need_pred:
            steps = self.config.framework.latent_world.joint.get("full_grad_steps", None) if mode == "full" else None
            x_in = self.sample_latent_change(obs, cond, steps=steps, grad=mode == "full")
            if mode != "full":
                x_in = x_in.detach()
        if use_gt is not None and bool(use_gt.any()):
            x_gt = self.latent_change_target(obs)
            x_in = x_gt if x_in is None else torch.where(use_gt[:, None, None], x_gt, x_in)
            out["system1/gt_hidden_frac"] = use_gt.float().mean().item()
        with torch.set_grad_enabled(mode != "none" and torch.is_grad_enabled()):
            h = self.extract_final_hidden(obs, x_in, cond)
        if mode != "none":
            scale = float(self.config.framework.latent_world.joint.get("action_grad_scale", 1.0))
            h = GradScale.apply(h, scale) if scale != 1.0 else h
        return h

    def _hidden_cache_store(self) -> TensorCache:
        if self._hidden_cache is None:
            root = self.config.framework.system1.get("hidden_cache_dir", None)
            if not root:
                raise ValueError("system1.hidden_source: cache needs system1.hidden_cache_dir")
            fw = self.config.framework
            meta = {"kind": "system15_final_hidden", "system15": _module_fingerprint(self.system15, self.radio_encoder.stats),
                    "base_vlm": str(fw.qwenvl.base_vlm),
                    "radio": [str(fw.radio.path), list(self.radio_encoder.input_size), str(fw.radio.feature),
                              self.radio_encoder.normalize],
                    "steps": self.sampler.steps, "solver": self.sampler.solver, "layout": self.layout,
                    "radio_view": self.radio_view, "obs_size": self._obs_size()}
            self._hidden_cache = TensorCache(root, meta)
        return self._hidden_cache

    def cached_hidden(self, examples: List[dict]):
        """Predicted H_f per example from the on-disk cache; misses are computed with the inference path and stored.
        -> (H_f [B, N, d], valid [B, N]). Needs sample_key (datasets.*.emit_frame_meta)."""
        keys = [ex.get("sample_key", None) for ex in examples]
        if any(k is None for k in keys):
            raise KeyError("system1.hidden_source: cache needs sample_key in every example (datasets.*.emit_frame_meta: true)")
        store = self._hidden_cache_store()
        got = [store.get(k) for k in keys]
        miss = [i for i, g in enumerate(got) if g is None]
        if miss:
            with torch.no_grad():
                sub = [examples[i] for i in miss]
                obs = self.encode_observation(sub, with_state=False)
                h = self._predicted_hidden(obs, self.flow_conditions(obs))
            self.shapes.reset()  # the miss subset bound a smaller B
            for j, i in enumerate(miss):
                got[i] = {"h": h[j].to(torch.bfloat16), "valid": obs.valid[j]}
                store.put(keys[i], got[i])
        h = torch.stack([g["h"] for g in got]).to(self._device, torch.float32)
        valid = torch.stack([g["valid"] for g in got]).to(self._device)
        return h, valid, len(miss)

    # ------------------------------------------------------------------ joint training (stage C, endpoint path)
    def set_training_step(self, step: int) -> None:
        """Called by the trainer before every training step (completed optimizer steps, resume-safe)."""
        self._trainer_step = int(step)

    @property
    def training_step(self) -> int:
        if self._trainer_step is not None:
            return self._trainer_step
        trainer = getattr(self.config, "trainer", None)
        gas = int(trainer.get("gradient_accumulation_steps", 1) or 1) if trainer is not None and hasattr(trainer, "get") else 1
        return int(self._forward_count) // gas

    def action_loss_weight(self, step: Optional[int] = None) -> float:
        """lambda_A(step): linear ramp joint.action_loss_weight_start -> loss_weights.action over
        joint.action_loss_warmup_steps (schedule linear), or the constant loss_weights.action (fixed)."""
        target = self.loss_weights.get("action", 1.0)
        if self.stage != "C":
            return 1.0
        j = self.config.framework.latent_world.joint
        warm = int(j.get("action_loss_warmup_steps", 0) or 0)
        if str(j.get("action_loss_schedule", "linear")) == "fixed" or warm <= 0:
            return target
        start = float(j.get("action_loss_weight_start", 0.0))
        step = self.training_step if step is None else int(step)
        return start + (target - start) * min(1.0, max(step, 0) / warm)

    def endpoint_gt_prob(self, step: Optional[int] = None) -> float:
        """p_gt(step): probability that System 1 reads the GT X_1 instead of dZ~ (endpoint path, training only);
        joint.gt_prob_start -> gt_prob_end over joint.gt_prob_decay_steps (linear | cosine), or gt_prob_end (fixed)."""
        j = self.config.framework.latent_world.joint
        end = float(j.get("gt_prob_end", 0.0))
        decay = int(j.get("gt_prob_decay_steps", 0) or 0)
        schedule = str(j.get("gt_prob_schedule", "linear"))
        if schedule == "fixed" or decay <= 0:
            return end
        start = float(j.get("gt_prob_start", 1.0))
        step = self.training_step if step is None else int(step)
        r = max(step, 0) / decay
        if r >= 1.0:
            return end
        if schedule == "cosine":
            r = 0.5 * (1.0 - np.cos(np.pi * r))
        return float(start + (end - start) * r)

    def one_step_endpoint_hidden(self, obs: Observation, cond: FlowConditions, inter: Dict[str, torch.Tensor],
                                 out: Optional[Dict] = None, gt_prob: float = 0.0):
        """Training-time System 1 condition of the endpoint path.

        dZ~ = X_tau + (1 - tau) v^  from the FM pass (first DiT forward, graph kept: no detach / no_grad), then
        H_f = extract_final_hidden(obs, dZ~) (second DiT forward at tau = 1, the same function as inference).
        gt_prob > 0 (scheduled teacher forcing): each example with a valid future reads the GT X_1 instead of dZ~
        with that probability (no action gradient into v^ for those rows).
        inter = flow_matching_loss intermediates; their first B rows are the first noise draw.
        -> (H_f [B, N, d], dZ~ [B, N, D_c]); dZ~ is the endpoint itself, before the GT replacement."""
        b = obs.z_t.shape[0]
        x_tau, tau, v = inter["x_tau"][:b], inter["tau"][:b], inter["v"][:b]
        if self.endpoint_tau == "zero":  # own pass from pure noise: dZ~ = eps + v(eps, 0) = one Euler step, no GT in X
            x_tau = torch.randn_like(x_tau)
            tau = torch.zeros_like(tau)
            v = self.system15(x_tau, tau, cond)
        dz = self.shapes("dZ_endpoint", one_step_endpoint(x_tau, tau, v).masked_fill(~obs.valid[..., None], 0.0),
                         "B N D_c")
        mode = self.action_grad
        if mode != "full":  # final_pass / none: no action gradient through dZ~ -> v^
            dz = dz.detach()
        x_in = dz
        if gt_prob > 0:
            use_gt = obs.future_valid & (torch.rand(b, device=dz.device) < gt_prob)
            if bool(use_gt.any()):
                x_in = torch.where(use_gt[:, None, None], self.latent_change_target(obs).to(dz.dtype), dz)
            if out is not None:
                out["joint/gt_frac"] = use_gt.float().mean().item()
        with torch.set_grad_enabled(mode != "none" and torch.is_grad_enabled()):
            h = self.extract_final_hidden(obs, x_in, cond)
        if mode != "none":
            scale = float(self.config.framework.latent_world.joint.get("action_grad_scale", 1.0))
            h = GradScale.apply(h, scale) if scale != 1.0 else h
        if out is not None:
            out["joint/endpoint_tau_mean"] = float(tau.float().mean())
        return h, dz

    def ode_samples_hidden(self, obs: Observation, cond: FlowConditions, out: Optional[Dict] = None,
                           gt_prob: float = 0.0):
        """Training-time System 1 condition of action_path ode (stage C).

        One ODE sample per step count of joint.ode_steps, all from the SAME X_0 ~ N(0, I) (euler shares the tau = 0
        velocity, so [1, 4] costs 4 velocity evaluations; other solvers integrate each count separately), then one
        batched tau = 1 hidden pass over the K samples (= extract_final_hidden, as in inference). action_grad full:
        differentiable solver; final_pass: samples detached; none: H_f detached. gt_prob: scheduled teacher forcing as
        in the endpoint path, per example (all K rows of an example together).
        -> (H_f [K * B, N, d], dZ [K * B, N, D_c] before the GT replacement); rows k * B .. (k + 1) * B - 1 belong to
        ode_steps[k]."""
        b, k_n, dev = obs.z_t.shape[0], len(self.ode_steps), obs.z_t.device
        mode = self.action_grad
        x0 = self.sampler.initial_noise(cond, self.feat_dim, self.system15.num_tokens, dev)
        with torch.set_grad_enabled(mode == "full" and torch.is_grad_enabled()):
            if self.sampler.solver == "euler":
                v0 = self.system15(x0, torch.zeros(b, device=dev), cond)
                xs = []
                for n in self.ode_steps:
                    x = x0 + v0 / n
                    for i in range(1, n):
                        x = x + self.system15(x, torch.full((b,), i / n, device=dev), cond) / n
                    xs.append(x)
            else:
                xs = [self.sampler.sample(self.system15, cond, noise=x0, steps=n) for n in self.ode_steps]
        valid = obs.valid.repeat(k_n, 1)
        dz = self.shapes("dZ_ode", torch.cat(xs).masked_fill(~valid[..., None], 0.0), "KB N D_c")
        if mode != "full":
            dz = dz.detach()
        x_in = dz
        if gt_prob > 0 and obs.z_f is not None:
            use_gt = obs.future_valid & (torch.rand(b, device=dev) < gt_prob)
            if bool(use_gt.any()):
                x_gt = self.latent_change_target(obs).to(dz.dtype).repeat(k_n, 1, 1)
                x_in = torch.where(use_gt.repeat(k_n)[:, None, None], x_gt, dz)
            if out is not None:
                out["joint/gt_frac"] = use_gt.float().mean().item()
        with torch.set_grad_enabled(mode != "none" and torch.is_grad_enabled()):
            h = self.system15.backbone(x_in, torch.ones(k_n * b, device=dev), cond.repeat(k_n))
            h = self.shapes("H_f_ode", h, "KB N D_h")
        if mode != "none":
            scale = float(self.config.framework.latent_world.joint.get("action_grad_scale", 1.0))
            h = GradScale.apply(h, scale) if scale != 1.0 else h
        return h, dz

    def _grad_norm_due(self) -> bool:
        if not (self.training and torch.is_grad_enabled() and self.stage == "C"):
            return False
        n = self.config.framework.latent_world.joint.get("grad_norm_log_interval", None)
        if n is None:
            trainer = getattr(self.config, "trainer", None)
            n = trainer.get("logging_frequency", 100) if trainer is not None and hasattr(trainer, "get") else 100
        n = int(n or 0)
        return n > 0 and (self.training_step + 1) % n == 0  # the step the trainer logs (same rule as its grad_norm)

    def per_loss_grad_norms(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """L2 norm of d(loss_k)/d(params) per trainable module, for each (already weighted) loss, without touching
        .grad (torch.autograd.grad, graph retained for the real backward)."""
        groups = {"system15": [p for p in self.system15.parameters() if p.requires_grad],
                  "system15_velocity_head": [p for p in self.system15.velocity_head.parameters() if p.requires_grad],
                  "system1": [p for p in self.system1.parameters() if p.requires_grad]}
        params = list({id(p): p for g in groups.values() for p in g}.values())
        res = {}
        for name, loss in losses.items():
            if not torch.is_tensor(loss) or not loss.requires_grad or not params:
                continue
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            sq = {id(p): (g.detach().float().pow(2).sum() if g is not None else None) for p, g in zip(params, grads)}
            for gname, gp in groups.items():
                vals = [sq[id(p)] for p in gp if sq[id(p)] is not None]
                res[f"grad_norm/{name}->{gname}"] = float(torch.stack(vals).sum().sqrt()) if vals else 0.0
        return res

    # ------------------------------------------------------------------ training
    def forward(self, examples: List[dict] = None, **kwargs) -> Dict:
        """Training losses. Stage C (default, action_path endpoint):
            L = lambda_FM L_FM + lambda_A(step) L_action, with L_action -> H_f -> dZ~ -> v^ -> System 1.5."""
        if isinstance(examples, dict):
            raise NotImplementedError("QwenLatentWorldAction builds its own inputs; preprocess_in_workers is not supported")
        t0 = time.perf_counter()
        self.shapes.reset()
        w = self.loss_weights
        need_fm = self.stage == "A" or (self.stage == "C" and w.get("fm", 1.0) > 0)
        need_act = self.stage == "B" or (self.stage == "C" and w.get("action", 1.0) > 0)
        endpoint = need_act and self.stage == "C" and self.action_path == "endpoint"
        ode_multi = need_act and self.stage == "C" and self.action_path == "ode" and self.hidden_source == "pred"
        need_future = (need_fm or endpoint or (need_act and self.hidden_source in ("gt", "mix"))
                       or (ode_multi and self.training and self.endpoint_gt_prob() > 0))
        self._check_frame_meta(examples, need_future)
        out: Dict = {}
        losses: Dict[str, torch.Tensor] = {}

        if need_act and self.hidden_source == "cache":  # stage B only; System 2 / C-RADIO run only for cache misses
            t = time.perf_counter()
            h_f, h_valid, n_miss = self.cached_hidden(examples)
            out["system1/cache_miss_frac"] = n_miss / len(examples)
            out["timing/hidden"] = time.perf_counter() - t
            state = self._state(examples)
        else:
            t = time.perf_counter()
            obs = self.encode_observation(examples, with_future=need_future, with_state=need_act)
            out["timing/encode"] = time.perf_counter() - t
            cond = self.flow_conditions(obs)
            inter = None
            if need_fm or endpoint:  # first DiT forward: velocity at (X_tau, tau)
                t = time.perf_counter()
                mask = obs.valid & obs.valid_f
                if bool(self.config.framework.latent_world.get("mask_clamped_future", True)):
                    mask = mask & obs.future_valid[:, None]
                    out["system15/future_valid_frac"] = obs.future_valid.float().mean().item()
                s15 = self.config.framework.system15
                x1 = self.latent_change_target(obs)
                loss_fm, logs, inter = self.system15.flow_matching_loss(
                    x1, cond, mask, dict(s15.get("tau_schedule", {"schedule": "uniform"})),
                    int(s15.get("num_noise_samples", 1)), return_intermediates=True)
                out.update(logs)
                out["timing/system15"] = time.perf_counter() - t
                if need_fm:
                    out["fm_loss"] = loss_fm
                    losses["fm"] = (w.get("fm", 1.0) if self.stage == "C" else 1.0) * loss_fm
            if need_act:
                t = time.perf_counter()
                if endpoint:  # second DiT forward at tau = 1 on the one-step endpoint (or GT, p_gt(step))
                    p_gt = self.endpoint_gt_prob() if self.training else 0.0
                    out["joint/gt_prob"] = p_gt
                    h_f, dz = self.one_step_endpoint_hidden(obs, cond, inter, out, gt_prob=p_gt)
                    with torch.no_grad():
                        v = (obs.valid & obs.valid_f & obs.future_valid[:, None])[..., None].float()
                        out["system15/endpoint_rel_mse"] = float(((dz - x1).pow(2) * v).sum()
                                                                 / (x1.pow(2) * v).sum().clamp_min(1e-12))
                elif ode_multi:  # K ODE samples (joint.ode_steps, same noise) -> one tau = 1 pass over K * B rows
                    p_gt = self.endpoint_gt_prob() if self.training else 0.0
                    out["joint/gt_prob"] = p_gt
                    h_f, dz = self.ode_samples_hidden(obs, cond, out, gt_prob=p_gt)
                    if obs.z_f is not None:
                        with torch.no_grad():
                            x1g = self.latent_change_target(obs).float()
                            v = (obs.valid & obs.valid_f & obs.future_valid[:, None])[..., None].float()
                            den = (x1g.pow(2) * v).sum().clamp_min(1e-12)
                            for n, d in zip(self.ode_steps, dz.float().chunk(len(self.ode_steps))):
                                out[f"system15/ode{n}_rel_mse"] = float(((d - x1g).pow(2) * v).sum() / den)
                else:
                    h_f = self._training_hidden(obs, cond, out)
                h_valid, state = obs.valid, obs.state
                out["timing/hidden"] = time.perf_counter() - t
        if need_act:
            t = time.perf_counter()
            rep = h_f.shape[0] // len(examples)  # K rows per example on the ode path, else 1
            if rep > 1:
                h_valid = h_valid.repeat(rep, 1)
                state = None if state is None else state.repeat(rep, *([1] * (state.dim() - 1)))
            a_hat = self.shapes("A_hat", self.system1(h_f, h_valid, state),
                                f"{'KB' if rep > 1 else 'B'} {self.action_horizon} {self.action_dim}")
            s1 = self.config.framework.system1
            target = self._actions(examples)
            target = target.repeat(rep, 1, 1) if rep > 1 else target
            kind, beta = str(s1.get("loss", "smooth_l1")), float(s1.get("smooth_l1_beta", 0.1))
            loss_act, logs = regression_loss(a_hat, target, kind, beta)  # mean over rows = mean of the K losses
            out.update(logs)
            if rep > 1:
                with torch.no_grad():
                    for n, a_k, t_k in zip(self.ode_steps, a_hat.chunk(rep), target.chunk(rep)):
                        out[f"system1/action_loss_ode{n}"] = regression_loss(a_k, t_k, kind, beta)[0].item()
            out["action_loss"] = loss_act  # logged by the trainer as action_dit_loss
            out["timing/system1"] = time.perf_counter() - t
            lam = self.action_loss_weight()
            out["joint/action_weight"] = lam
            losses["action"] = lam * loss_act
        if self._grad_norm_due():
            t = time.perf_counter()
            out.update(self.per_loss_grad_norms(losses))
            out["timing/grad_norms"] = time.perf_counter() - t
        if not losses:
            raise ValueError("framework.latent_world.loss_weights: every loss is 0 / disabled")
        out["loss"] = sum(losses.values())
        out["timing/forward"] = time.perf_counter() - t0
        if self.training and torch.is_grad_enabled():
            self._forward_count += 1  # fallback step counter (completed training forwards)
        return out

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def predict_action(self, examples: List[dict], sample_steps: Optional[int] = None, solver: Optional[str] = None,
                       seed: Optional[int] = None, return_latent: bool = False, **kwargs) -> Dict:
        """Current view(s) + instruction (+ state) -> normalized action chunk [B, H_a, D_a].

        Reads only the current views (extra 'f' images of a training example are dropped by image_layout), "lang"
        and "state"; never "action". kwargs of other frameworks (use_ddim, ...) are ignored."""
        examples = examples if isinstance(examples, list) else [examples]
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self._device).manual_seed(int(seed))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
            self.shapes.reset()
            obs = self.encode_observation(examples, with_future=False)
            cond = self.flow_conditions(obs)
            x1 = self.sample_latent_change(obs, cond, steps=sample_steps, solver=solver, generator=gen)
            h_f = self.extract_final_hidden(obs, x1, cond)
            actions = self.system1(h_f, obs.valid, obs.state)
        out = {"normalized_actions": actions.float().cpu().numpy()}
        if return_latent:
            delta, z_future = self.latent_change_to_features(x1.float(), obs)
            out.update({"latent_change": delta.cpu().numpy(), "z_t": obs.z_t.float().cpu().numpy(),
                        "z_future": z_future.cpu().numpy(), "valid": obs.valid.cpu().numpy()})
        return out

    # ------------------------------------------------------------------ validation
    def evaluate_batch(self, examples: List[dict], seed: int = 0) -> Dict[str, float]:
        """Seeded metrics on examples with the future frame / actions; the training RNG state is untouched.

        system15/rel_mse : MSE(dZ_hat, dZ) / MSE(0, dZ) over valid tokens (< 1 beats "nothing changes")
        system15/cos     : per-sample cosine(dZ_hat, dZ)
        system15/fm_loss : L_FM with fixed (eps, tau)
        system1/rmse_pred: action RMSE of the inference path (predicted dZ_hat -> H_f)
        system1/rmse_gt_hidden: action RMSE with H_f from the GT dZ (teacher forcing upper bound, diagnostic)
        system1/rmse_endpoint  : (stage C endpoint) action RMSE of the training path, dZ~ = X_tau + (1 - tau) v^
        system15/endpoint_rel_mse: rel_mse of that one-step endpoint
        system1/rmse_pred_ode{n}, system15/cos_ode{n}: (stage C ode) the other joint.ode_steps counts than sample_steps
        """
        devices = [self._device] if self._device.type == "cuda" else []
        sums: Dict[str, float] = {}
        counts: Dict[str, float] = {}
        bs = int(self.config.framework.latent_world.get("eval", {}).get("batch_size", 16))
        with torch.random.fork_rng(devices=devices), torch.no_grad(), \
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
            torch.manual_seed(seed)
            for i in range(0, len(examples), bs):
                for k, (v, n) in self._evaluate_chunk(examples[i:i + bs]).items():
                    sums[k] = sums.get(k, 0.0) + v * n
                    counts[k] = counts.get(k, 0.0) + n
        return {k: sums[k] / max(counts[k], 1e-12) for k in sums}

    def _evaluate_chunk(self, examples: List[dict]) -> Dict[str, tuple]:
        self.shapes.reset()
        has_future = self.f_idx is not None and len(_as_list(examples[0]["image"])) == len(self.layout)
        obs = self.encode_observation(examples, with_future=has_future, with_state=self.stage != "A")
        cond = self.flow_conditions(obs)
        x1_hat = self.sample_latent_change(obs, cond)
        res, inter = {}, None
        b = len(examples)
        if has_future:
            x1_gt = self.latent_change_target(obs)
            m = (obs.valid & obs.valid_f & obs.future_valid[:, None])[..., None].float()
            s = self.delta_scale()
            d_hat, d_gt = x1_hat.float() * s * m, x1_gt.float() * s * m
            n_el = m.sum() * d_gt.shape[-1]
            if n_el > 0:
                mse, mse0 = (d_hat - d_gt).pow(2).sum() / n_el, d_gt.pow(2).sum() / n_el
                res["system15/rel_mse"] = ((mse / mse0.clamp_min(1e-12)).item(), 1)
                cos = F.cosine_similarity(d_hat.flatten(1), d_gt.flatten(1), dim=1)
                fv = obs.future_valid
                res["system15/cos"] = (cos[fv].mean().item(), int(fv.sum()))
                s15 = self.config.framework.system15
                loss, _, inter = self.system15.flow_matching_loss(
                    x1_gt, cond, m[..., 0].bool(), dict(s15.get("tau_schedule", {"schedule": "uniform"})),
                    return_intermediates=True)
                res["system15/fm_loss"] = (loss.item(), b)
        if self.stage != "A" and "action" in examples[0]:
            target = self._actions(examples)
            a_pred = self.system1(self.extract_final_hidden(obs, x1_hat, cond), obs.valid, obs.state).float()
            res["system1/rmse_pred"] = ((a_pred - target).pow(2).mean().sqrt().item(), b)
            res["system1/l1_pred"] = ((a_pred - target).abs().mean().item(), b)
            if has_future and bool(obs.future_valid.any()):
                fv = obs.future_valid
                a_gt = self.system1(self.extract_final_hidden(obs, self.latent_change_target(obs), cond), obs.valid,
                                    obs.state).float()
                res["system1/rmse_gt_hidden"] = ((a_gt[fv] - target[fv]).pow(2).mean().sqrt().item(), int(fv.sum()))
            if "system15/rel_mse" in res and self.stage == "C" and self.action_path == "endpoint":
                # the TRAINING path (one-step endpoint of the FM draw); its gap to rmse_pred (ODE) = the approximation
                # mismatch between training and inference
                h_e, dz_e = self.one_step_endpoint_hidden(obs, cond, inter)
                a_e = self.system1(h_e, obs.valid, obs.state).float()
                res["system1/rmse_endpoint"] = ((a_e - target).pow(2).mean().sqrt().item(), b)
                d_e = dz_e.float() * s * m
                res["system15/endpoint_rel_mse"] = (((d_e - d_gt).pow(2).sum() / n_el / mse0.clamp_min(1e-12)).item(), 1)
            if self.stage == "C" and self.action_path == "ode":
                # the other trained step counts (rmse_pred = sample_steps); drawn last so the metrics above keep their noise
                for n in self.ode_steps:
                    if n == self.sampler.steps:
                        continue
                    x_n = self.sample_latent_change(obs, cond, steps=n)
                    a_n = self.system1(self.extract_final_hidden(obs, x_n, cond), obs.valid, obs.state).float()
                    res[f"system1/rmse_pred_ode{n}"] = ((a_n - target).pow(2).mean().sqrt().item(), b)
                    if "system15/cos" in res:
                        d_n = x_n.float() * s * m
                        cos_n = F.cosine_similarity(d_n.flatten(1), d_gt.flatten(1), dim=1)
                        res[f"system15/cos_ode{n}"] = (cos_n[obs.future_valid].mean().item(), int(obs.future_valid.sum()))
        return res

    def _load_eval_examples(self, get_batch):
        if self._eval_examples is not None:
            return self._eval_examples
        n = int(self.config.framework.latent_world.get("eval", {}).get("num_samples", 64))
        datasets = getattr(self.config, "datasets", None)
        val_cfg = datasets.get("val_data", None) if datasets is not None and hasattr(datasets, "get") else None
        if val_cfg is not None:
            from starVLA.dataloader.lerobot_datasets import get_vla_dataset

            ds = get_vla_dataset(data_cfg=val_cfg)
            rng = np.random.default_rng(0)
            self._eval_examples = [ds[int(i)] for i in rng.choice(len(ds), size=min(n, len(ds)), replace=False)]
        else:
            self._eval_examples = list(get_batch())[:n]
        return self._eval_examples

    def trainer_eval_hook(self, get_batch, step: int, output_dir: str, is_main: bool = True) -> Dict[str, float]:
        """Called by the trainer on every rank each eval_interval (building the val dataset has a barrier);
        metrics on fixed validation examples on the main process -> <output_dir>/samples/step_*.json."""
        examples = self._load_eval_examples(get_batch)
        if not is_main:
            return {}
        modes = {m: m.training for m in (self.system15, self.system1)}
        for m in modes:
            m.eval()
        try:
            metrics = self.evaluate_batch(examples, seed=int(self.config.framework.latent_world.get("eval", {}).get("seed", 0)))
        finally:
            for m, was in modes.items():
                m.train(was)
        out_dir = Path(output_dir) / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"step_{step:07d}.json", "w") as f:
            json.dump({"metrics": metrics, "num_examples": len(examples)}, f, indent=1)
        return {f"eval/{k}": v for k, v in metrics.items()}
