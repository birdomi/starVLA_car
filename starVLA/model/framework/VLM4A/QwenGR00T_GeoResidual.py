"""
QwenGR00T_GeoResidual: a FROZEN System 2 (QwenGR00T = Qwen3.5 + flow-matching DiT, loaded from its checkpoint) plus a
lightweight geometry-guided LOCAL RESIDUAL policy (regression, no diffusion / flow) that corrects the next K steps of
System 2's trajectory from the CURRENT frame (user spec 10-22 "MoGe-2 Geometry-Guided Local Residual Policy").

    System 2 (frozen): frame t - d, state, language -> normalized 29-D joint trajectory a[t-d .. t-d+15] (the "plan")
    local chunk      : plan steps d .. d+K-1 = env steps t .. t+K-1 (K = 4 = 0.2 s at 20 Hz)
    action adapter   : 29-D absolute joints -> per hand wrist pose (torch FK of the GR1 MJCF, base frame) + 6 finger cmds
    geometry         : current ego_view frame t -> frozen MoGe-2 ViT-B -> its GEOMETRY NECK features (levels 1-4, pooled
                       to the 28 x 42 base grid) -> GeometryTokenizer -> geometry tokens
    queries          : TrajectoryEncoder(local chunk wrist poses / fingers, current wrist poses / fingers, plan age d)
                       -> 2K tokens (step x hand) -> ResidualDecoder (self-attn + cross-attn to geometry tokens)
    head             : 12-D per (step, hand) = 24-D per step x K: dp (3, m) | dw (3, rad, base frame) | dfinger (6,
                       normalized action units), zero-initialised
    composition      : p' = p + dp, R' = exp(dw) R (SE(3) residual), fingers f' = f + df; wrist pose -> arm joints by
                       damped-least-squares IK from the System 2 joints (waist kept) -> corrected 29-D joints.

Design facts (repo inspection 10-22): RoboCasa GR1 data has only the ego_view camera (no wrist camera, no extrinsics, no
EE pose), so the geometry frame is ego_view (user choice). Actions are absolute joint targets (min_max normalized,
stats = framework.geo_residual.action_stats); fingers have no kinematics and are corrected in normalized units.

Training: per sample a plan age d is drawn from geo_residual.s2_offsets (default 0 / 4 / 8 env steps = the residual
calls inside one 12-step System 2 period); the loader must deliver frames / states at deltas -max(d) .. 0 (yaml
datasets.vla_data.delta_indices_override for video.ego_view and every state key). System 2 runs (no_grad, eval) on frame
t - d; targets = ground-truth a[t .. t+K-1] mapped through the same adapter; loss = SmoothL1 on the scaled residuals.

Inference (policy server): with plan_kf / plan_age_s in the examples (client ROBOCASA_S2_EVERY=12, n_action_steps 4) the
cached plan is used at offset round(age * 20); otherwise System 2 runs on the current frame and its plan goes back as
"plan_kf". Returned chunk = K corrected steps + the rest of the plan (padded to 16). STARVLA_GEORES_OFF=1 disables the
correction (same schedule, System 2 only) for paired ablations.
"""

import json
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.framework.VLM4A.QwenGR00T_RadioAdaLN import _example_images, _images_to_tensor
from starVLA.model.modules.geo_residual.gr1_kinematics import ARM, HAND, SIDES, WAIST, GR1Kinematics, state_sincos_to_raw
from starVLA.model.modules.geo_residual.moge2_geometry import MoGe2NeckFeatures
from starVLA.model.modules.geo_residual.residual_policy import (GeometryTokenizer, ResidualDecoder, ResidualHead,
                                                                TrajectoryEncoder)
from starVLA.model.modules.geo_residual.se3 import axis_angle_to_matrix, matrix_to_rot6d, rotation_residual
from starVLA.model.tools import FRAMEWORK_REGISTRY

GEO_DEFAULTS = {
    "moge_path": "playground/Pretrained_models/moge-2-vitb-normal/model.pt",
    "moge_num_tokens": 1200,
    "moge_crop": [42, 214, 0, 256],     # letterbox content of the 256 x 256 ego_view (top, bottom, left, right)
    "neck_levels": [1, 2, 3, 4],
    "action_stats": None,               # dataset_statistics.json of the System 2 run (same mixture)
    "stats_key": "gr1",
    "horizon": 4,                       # K local steps (0.2 s at 20 Hz)
    "s2_offsets": [0, 4, 8],            # plan ages (env steps) drawn per training sample
    "rate": 20.0,
    "d_model": 256, "heads": 8, "n_layers": 3, "dropout": 0.0,
    "pos_scale": 0.02, "rot_scale": 0.05, "finger_scale": 0.1,   # residual units (m, rad, normalized cmd)
    "loss_weights": [1.0, 1.0, 1.0],    # pos, rot, finger
    "max_pos": 0.05, "max_rot": 0.3,    # inference clamps of |dp| (m) / |dw| (rad)
    "ik_iters": 8, "ik_damping": 0.02, "ik_rot_weight": 0.2,
    "train_system2": False,
}


@FRAMEWORK_REGISTRY.register("QwenGR00T_GeoResidual")
class QwenGR00T_GeoResidual(Qwen_GR00T):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)  # System 2: qwen_vl_interface + action_model
        gc_in = self.config.framework.get("geo_residual", None) or {}
        gc = {k: (gc_in.get(k, v) if hasattr(gc_in, "get") else v) for k, v in GEO_DEFAULTS.items()}
        unknown = set(gc_in.keys()) - set(GEO_DEFAULTS) if hasattr(gc_in, "keys") else set()
        if unknown:
            raise ValueError(f"framework.geo_residual: unknown keys {sorted(unknown)}")
        self.gc = gc
        self.K = int(gc["horizon"])
        self.offsets = sorted(int(v) for v in gc["s2_offsets"])
        if self.offsets[0] != 0 or self.offsets[-1] + self.K > self.action_horizon:
            raise ValueError(f"geo_residual.s2_offsets {self.offsets} must start at 0 and keep offset + K <= {self.action_horizon}")
        self.rate = float(gc["rate"])
        if not gc["action_stats"]:
            raise ValueError("framework.geo_residual.action_stats (dataset_statistics.json) is required")
        st = json.load(open(gc["action_stats"]))[gc["stats_key"]]["action"]
        a_min, a_max = torch.tensor(st["min"], dtype=torch.float32), torch.tensor(st["max"], dtype=torch.float32)
        if a_min.numel() != 29:
            raise ValueError(f"action stats have {a_min.numel()} dims, expected 29")
        self.register_buffer("a_min", a_min, persistent=False)
        self.register_buffer("a_max", a_max, persistent=False)

        if not bool(gc["train_system2"]):
            self.qwen_vl_interface.requires_grad_(False)
            self.action_model.requires_grad_(False)
        self.kin = GR1Kinematics()
        self.geo_backbone = MoGe2NeckFeatures(gc["moge_path"], num_tokens=gc["moge_num_tokens"], crop=gc["moge_crop"],
                                              levels=gc["neck_levels"])
        d = int(gc["d_model"])
        self.geo_tokenizer = GeometryTokenizer(self.geo_backbone.out_dim, d, self.geo_backbone.grid)
        self.traj_encoder = TrajectoryEncoder(d, self.K)
        self.residual_decoder = ResidualDecoder(d, int(gc["heads"]), int(gc["n_layers"]), float(gc["dropout"]))
        self.residual_head = ResidualHead(d, 12)
        self.scales = (float(gc["pos_scale"]), float(gc["rot_scale"]), float(gc["finger_scale"]))
        self.img_size = (256, 256)

    # ------------------------------------------------------------------ helpers
    @property
    def _device(self):
        return self.a_min.device

    def train(self, mode: bool = True):
        super().train(mode)
        if not bool(self.gc["train_system2"]):  # frozen System 2 samples exactly as at deployment (no dropout)
            self.qwen_vl_interface.eval()
            self.action_model.eval()
        self.geo_backbone.eval()
        return self

    def _unnorm(self, x):
        rng = self.a_max - self.a_min
        return torch.where(rng > 0, (x + 1) / 2 * rng + self.a_min, x)

    def _norm(self, x):
        rng = self.a_max - self.a_min
        return torch.where(rng > 0, 2 * (x - self.a_min) / rng.clamp_min(1e-12) - 1, x)

    def _wrists(self, raw):
        """[..., 29] raw joints -> p [..., 2, 3], R [..., 2, 3, 3] (left, right)."""
        w = self.kin.joints_to_wrists(raw)
        return torch.stack([w[s][0] for s in SIDES], -2), torch.stack([w[s][1] for s in SIDES], -3)

    @staticmethod
    def _hands(x):
        return torch.stack([x[..., HAND[s]] for s in SIDES], -2)  # [..., 2, 6]

    def _system2_plan(self, images: List, langs: List[str], states: np.ndarray) -> torch.Tensor:
        """Frozen System 2 on one frame per example -> normalized plan [B, 16, 29] float32."""
        exs = [{"image": [im], "lang": l, "state": s[None]} for im, l, s in zip(images, langs, states)]
        out = Qwen_GR00T.predict_action(self, exs)["normalized_actions"]
        return torch.as_tensor(np.asarray(out), dtype=torch.float32, device=self._device)

    def _residual(self, cur_images: List, seg_norm, cur_state58, age_s):
        """-> raw residual [B, K, 2, 12] (scaled units) and the plan's wrist poses / fingers."""
        seg_raw = self._unnorm(seg_norm)
        plan_p, plan_R = self._wrists(seg_raw)  # [B, K, 2, 3], [B, K, 2, 3, 3]
        plan_f = self._hands(seg_norm)
        cur_raw = state_sincos_to_raw(cur_state58)
        cur_p, cur_R = self._wrists(cur_raw)
        cur_f = self._hands(cur_raw)
        x = _images_to_tensor(cur_images, self.img_size, self._device)
        geo = self.geo_tokenizer(self.geo_backbone(x))
        q = self.traj_encoder(plan_p, matrix_to_rot6d(plan_R), plan_f, cur_p, matrix_to_rot6d(cur_R), cur_f, age_s)
        r = self.residual_head(self.residual_decoder(q, geo)).view(len(cur_images), self.K, 2, 12)
        return r, seg_raw, plan_p, plan_R, plan_f

    @staticmethod
    def _frames_and_states(ex, n):
        ims = _example_images(ex)
        st = np.asarray(ex["state"], dtype=np.float32).reshape(-1, 58)
        if len(ims) != n or st.shape[0] != n:
            raise ValueError(f"QwenGR00T_GeoResidual: {len(ims)} frames / {st.shape[0]} states per example, expected {n} "
                             "(datasets.vla_data.delta_indices_override must give video.ego_view and every state key the "
                             "deltas -max(s2_offsets) .. 0 matching geo_residual.s2_offsets)")
        return ims, st

    # ------------------------------------------------------------------ training
    def forward(self, examples: List[dict] = None, **kwargs):
        n, B, K = len(self.offsets), len(examples), self.K
        pick = np.random.randint(n, size=B)
        d = np.asarray([self.offsets[i] for i in pick])
        frames, states = zip(*[self._frames_and_states(ex, n) for ex in examples])
        # frame list is oldest first: deltas [-offsets[-1], ..., -offsets[0]] -> offset index i at position n - 1 - i
        s2_images = [frames[b][n - 1 - pick[b]] for b in range(B)]
        s2_states = np.stack([states[b][n - 1 - pick[b]] for b in range(B)])
        plan = self._system2_plan(s2_images, [ex["lang"] for ex in examples], s2_states)
        seg = torch.stack([plan[b, d[b]:d[b] + K] for b in range(B)])
        gt = torch.as_tensor(np.stack([np.asarray(ex["action"], dtype=np.float32)[:K] for ex in examples]), device=self._device)
        cur_state = torch.as_tensor(np.stack([s[-1] for s in states]), device=self._device)
        age = torch.as_tensor(d / self.rate, dtype=torch.float32, device=self._device)

        r, seg_raw, plan_p, plan_R, plan_f = self._residual([f[-1] for f in frames], seg, cur_state, age)
        gt_p, gt_R = self._wrists(self._unnorm(gt))
        gt_f = self._hands(gt)
        ps, rs, fs = self.scales
        t_dp, t_dw, t_df = (gt_p - plan_p) / ps, rotation_residual(gt_R, plan_R) / rs, (gt_f - plan_f) / fs
        w = self.gc["loss_weights"]
        l_pos = F.smooth_l1_loss(r[..., :3], t_dp)
        l_rot = F.smooth_l1_loss(r[..., 3:6], t_dw)
        l_fin = F.smooth_l1_loss(r[..., 6:], t_df)
        loss = w[0] * l_pos + w[1] * l_rot + w[2] * l_fin
        with torch.no_grad():
            e_p0 = (t_dp * ps).norm(dim=-1).mean().item() * 1e3
            e_p1 = ((t_dp - r[..., :3].detach()) * ps).norm(dim=-1).mean().item() * 1e3
            e_r0 = (t_dw * rs).norm(dim=-1).mean().item() * 57.2958
            e_r1 = ((t_dw - r[..., 3:6].detach()) * rs).norm(dim=-1).mean().item() * 57.2958
            e_f0 = (t_df * fs).abs().mean().item()
            e_f1 = ((t_df - r[..., 6:].detach()) * fs).abs().mean().item()
        return {
            "action_loss": loss,
            "geo/loss_pos": l_pos.item(), "geo/loss_rot": l_rot.item(), "geo/loss_finger": l_fin.item(),
            "geo/s2_pos_err_mm": e_p0, "geo/corr_pos_err_mm": e_p1,
            "geo/s2_rot_err_deg": e_r0, "geo/corr_rot_err_deg": e_r1,
            "geo/s2_finger_err": e_f0, "geo/corr_finger_err": e_f1,
        }

    # ------------------------------------------------------------------ inference
    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs):
        if type(examples) is not list:
            examples = [examples]
        B, K, T = len(examples), self.K, self.action_horizon
        cur_images = [_example_images(ex)[-1] for ex in examples]
        cur58 = np.stack([np.asarray(ex["state"], dtype=np.float32).reshape(-1, 58)[-1] for ex in examples])
        fresh = not all("plan_kf" in ex for ex in examples)
        if fresh:
            plan = self._system2_plan(cur_images, [ex["lang"] for ex in examples], cur58)
            d = np.zeros(B, dtype=np.int64)
        else:
            plan = torch.as_tensor(np.stack([np.asarray(ex["plan_kf"], dtype=np.float32) for ex in examples]), device=self._device)
            d = np.clip(np.rint(np.asarray([float(ex.get("plan_age_s", 0.0)) for ex in examples]) * self.rate), 0, T - K).astype(np.int64)
        seg = torch.stack([plan[b, d[b]:d[b] + K] for b in range(B)])
        out_seg = seg
        if os.environ.get("STARVLA_GEORES_OFF", "0") != "1":
            age = torch.as_tensor(d / self.rate, dtype=torch.float32, device=self._device)
            r, seg_raw, plan_p, plan_R, plan_f = self._residual(
                cur_images, seg, torch.as_tensor(cur58, device=self._device), age)
            ps, rs, fs = self.scales
            dp = r[..., :3] * ps
            dp = dp * (self.gc["max_pos"] / dp.norm(dim=-1, keepdim=True).clamp_min(self.gc["max_pos"]))
            dw = r[..., 3:6] * rs
            dw = dw * (self.gc["max_rot"] / dw.norm(dim=-1, keepdim=True).clamp_min(self.gc["max_rot"]))
            p_t, R_t = plan_p + dp, axis_angle_to_matrix(dw) @ plan_R
            new_raw = seg_raw.clone()
            for i, s in enumerate(SIDES):
                new_raw[..., ARM[s]] = self.kin.ik(s, seg_raw[..., WAIST], seg_raw[..., ARM[s]], p_t[..., i, :], R_t[..., i, :, :],
                                                   iters=int(self.gc["ik_iters"]), damping=float(self.gc["ik_damping"]),
                                                   rot_weight=float(self.gc["ik_rot_weight"]))
            out_seg = self._norm(new_raw)
            live = self.a_max > self.a_min  # constant dims (min == max, e.g. right_hand[5]) stay un-normalized: no residual
            for i, s in enumerate(SIDES):
                out_seg[..., HAND[s]] = torch.where(live[HAND[s]], (plan_f[..., i, :] + r[..., i, 6:] * fs).clamp(-1, 1),
                                                    plan_f[..., i, :])
            out_seg[..., WAIST] = seg[..., WAIST]
        chunks = []
        for b in range(B):
            rest = plan[b, d[b] + K:]
            c = torch.cat([out_seg[b], rest], 0)
            if c.shape[0] < T:
                c = torch.cat([c, c[-1:].expand(T - c.shape[0], -1)], 0)
            chunks.append(c[:T])
        out = {"normalized_actions": torch.stack(chunks).float().cpu().numpy()}
        if fresh:
            out["plan_kf"] = plan.float().cpu().numpy()
        return out
