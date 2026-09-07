"""Receding-horizon action inference on a val episode with the 20K checkpoint.

Predicts normalized actions chunk-by-chunk (execute the first STRIDE of each
16-step chunk, re-predict from the next GT frame), un-normalizes with the
TRAIN statistics (rel/q99 for head+arms, abs/min_max for hands), converts the
relative parts back to absolute joint targets, and saves (T, 42) preds.npy in
the raw units replay_mujoco.py expects.

Usage: rollout_val.py <task> <episode_idx> <out.npy>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/mnt/DATA2/yyg/starVLA")
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.framework.base_framework import build_framework

TASK = sys.argv[1]
EP = int(sys.argv[2])
OUT = sys.argv[3]
CKPT = sys.argv[4] if len(sys.argv) > 4 else "/mnt/DATA2/yyg/starVLA/playground/Checkpoints/starvla_qwengr00t_car3task_0904_v5/checkpoints/steps_20000_pytorch_model.pt"
ROOT = Path("/mnt/DATA1/yyg/lerobot/data/0820_car_data")
STRIDE = 16  # use the full predicted chunk

# action-vector slices (config order == replay_mujoco ACTION layout)
SL = {
    "head": (slice(0, 2), "rel"),
    "larm": (slice(2, 9), "rel"),
    "lhand": (slice(9, 22), "abs"),
    "rarm": (slice(22, 29), "rel"),
    "rhand": (slice(29, 42), "abs"),
}
STATE_REF = {"head": slice(0, 2), "larm": slice(15, 22), "rarm": slice(22, 29)}

cfg = OmegaConf.load(
    "/mnt/DATA2/yyg/starVLA/examples/realRobots/H1_2_car/train_files/starvla_qwengr00t_car3task.yaml"
)
cfg.datasets.vla_data.video_backend = "decord"
cfg.datasets.vla_data.video_backend_kwargs = {"num_threads": 2}

model = build_framework(cfg)
sd = torch.load(CKPT, map_location="cpu")
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
model = model.cuda().eval()

ds = make_LeRobotSingleDataset(ROOT, f"{TASK}/val", "h1_2_car", data_cfg=cfg.datasets.vla_data)

# ordered step indices of the chosen episode
ep_steps = [i for i, (tid, base) in enumerate(ds.all_steps) if tid == EP]
bases = [ds.all_steps[i][1] for i in ep_steps]
assert bases == sorted(bases)
T = len(ep_steps)
print(f"{TASK}/val ep{EP}: {T} frames", flush=True)

# raw absolute states from the parquet (for rel->abs reconstruction)
import pandas as pd

pq = sorted((ROOT / TASK / "val").glob(f"data/chunk-*/episode_{EP:06d}.parquet"))[0]
raw_state = np.stack(pd.read_parquet(pq)["observation.state"].to_numpy())  # (T, 63)

# train statistics (rel-mode cache): per-column arrays over the 42 action dims
stats = json.load(open(ROOT / TASK / "train/meta/stats_gr00t.rel.json"))["statistics"]["action"]
q01, q99 = np.array(stats["q01"]), np.array(stats["q99"])
amin, amax = np.array(stats["min"]), np.array(stats["max"])


def unnorm(pred_n):  # (16, 42) normalized -> physical (rel deltas / raw hand units)
    out = np.empty_like(pred_n, dtype=np.float64)
    for key, (sl, mode) in SL.items():
        x = pred_n[:, sl]
        if mode == "rel":  # q99 forward: 2*(x-q01)/(q99-q01)-1
            out[:, sl] = (x + 1.0) / 2.0 * (q99[sl] - q01[sl]) + q01[sl]
        else:  # min_max
            out[:, sl] = (x + 1.0) / 2.0 * (amax[sl] - amin[sl]) + amin[sl]
    return out


torch.manual_seed(0)
pred_traj = np.zeros((T, 42))
with torch.no_grad():
    for b in range(0, T, STRIDE):
        ex = ds[ep_steps[b]]
        pred_n = model.predict_action([ex])["normalized_actions"][0]  # (16, 42)
        phys = unnorm(np.asarray(pred_n, dtype=np.float64))
        # rel -> absolute against the GT state at the chunk start
        for key, ref in STATE_REF.items():
            sl, _ = SL[key]
            phys[:, sl] = phys[:, sl] + raw_state[b, ref]
        n = min(STRIDE, T - b)
        pred_traj[b : b + n] = phys[:n]
        if (b // STRIDE) % 5 == 0:
            print(f"  chunk @ {b}/{T}", flush=True)

np.save(OUT, pred_traj.astype(np.float32))
# quick numeric sanity vs GT actions
gt = np.stack(pd.read_parquet(pq)["action"].to_numpy())[:T]
arm_sl = np.r_[0:9, 22:29]
print(f"saved {OUT} shape={pred_traj.shape}")
print(f"arm/head MAE vs GT: {np.abs(pred_traj[:, arm_sl] - gt[:, arm_sl]).mean():.4f} rad")
hand_sl = np.r_[9:22, 29:42]
print(f"hand MAE vs GT: {np.abs(pred_traj[:, hand_sl] - gt[:, hand_sl]).mean():.1f} raw units")
