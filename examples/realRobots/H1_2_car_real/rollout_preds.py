"""Receding-horizon rollout of a carfew checkpoint on val episodes (one per task), for videos.

Every `--stride` frames the model gets the REAL observation at that frame (front image, state,
instruction) and predicts a chunk; the first `--stride` steps of each chunk are stitched into one
(T, 42) absolute action trajectory. This is open-loop w.r.t. the robot (observations come from
the recording), i.e. what the policy would command if the world followed the demo.

Per episode it writes <out>/<tag>/{pred.npy, gt_action.npy, state.npy, front.mp4, meta.json}.
render_rollout.py turns those into comparison videos.

Usage: CUDA_VISIBLE_DEVICES=4 python rollout_preds.py --ckpt <ckpt.pt> --out <dir> [--dtype float16]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/mnt/DATA2/yyg/starVLA")
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402
from starVLA.dataloader.gr00t_lerobot import video as video_io  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--config", default="/mnt/DATA2/yyg/starVLA/examples/realRobots/H1_2_car_real/train_files/"
                "starvla_qwengr00t_carfew_v1_2b_front_abs.yaml")
ap.add_argument("--dataset", default="car-few-tasks-v1")
ap.add_argument("--robot", default="h1_2_car_left_front")
ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
ap.add_argument("--stride", type=int, default=30)
ap.add_argument("--episodes", default="", help="comma-separated val episode ids; default: one median-length episode per task")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--stats-dataset", default="", help="dataset whose TRAIN split provides the normalization (default: --dataset)")
ap.add_argument("--stats-robot", default="", help="robot type for --stats-dataset (default: --robot)")
ap.add_argument("--rel-arms", action="store_true",
                help="model was trained with arms relative to state (mixed mode, e.g. leftturn_v2_mix): un-normalize "
                     "with the stats-dataset's stats_gr00t.rel.json and add the current arm state")
ap.add_argument("--front-cam", default="front_view", help="camera cut into front.mp4 for the videos")
ap.add_argument("--tag-prefix", default="")
ap.add_argument("--static_history", action="store_true",
                help="multi-frame configs: replace every history frame by the current one (memory ablation)")
args = ap.parse_args()

cfg = OmegaConf.load(args.config)
cfg.framework.qwenvl.autocast_dtype = args.dtype
root = Path(cfg.datasets.vla_data.data_root_dir)
split = root / args.dataset / "val"
train_cfg = OmegaConf.to_container(cfg.datasets.vla_data)
val_cfg = dict(train_cfg)
val_cfg.pop("exclude_episodes", None)
_corrupt = root / args.dataset / "val/meta/corrupted_episodes.json"
if _corrupt.exists():
    val_cfg["exclude_episodes"] = json.load(open(_corrupt))
stats_dataset = args.stats_dataset or args.dataset
stats_robot = args.stats_robot or args.robot
train_ds = make_LeRobotSingleDataset(root, f"{stats_dataset}/train", stats_robot, data_cfg=OmegaConf.create(train_cfg))
val_ds = make_LeRobotSingleDataset(root, f"{args.dataset}/val", args.robot, data_cfg=OmegaConf.create(val_cfg))
val_ds.set_transforms_metadata(train_ds.metadata)
step_index = {(int(t), int(b)): i for i, (t, b) in enumerate(val_ds._all_steps)}

stats = json.load(open(root / stats_dataset / f"train/meta/stats_gr00t.{'rel' if args.rel_arms else 'abs'}.json"))["statistics"]["action"]
q01, q99 = np.array(stats["q01"], np.float64), np.array(stats["q99"], np.float64)
const = q99 == q01


def unnorm(x):
    x = np.asarray(x, np.float64)
    return np.where(const, x, (x + 1.0) / 2.0 * (q99 - q01) + q01)


# Episodes: meta + raw rows (GT action / state come straight from parquet, un-normalized).
eps = pd.concat(pd.read_parquet(p) for p in sorted(split.glob("meta/episodes/*/*.parquet"))).set_index("episode_index")
tasks = pd.read_parquet(split / "meta/tasks.parquet")
task_of = {int(i): str(t) for t, i in zip(tasks.index, tasks["task_index"])}
for old, new in (cfg.datasets.vla_data.get("task_text_replace") or {}).items():
    task_of = {k: v.replace(old, new) for k, v in task_of.items()}
eps["task"] = [task_of[int(t[0])] for t in eps["tasks"]]
kept = set(int(t) for t in val_ds.trajectory_ids)
eps = eps[eps.index.isin(kept)]
if args.episodes:
    chosen = [int(e) for e in args.episodes.split(",")]
else:
    chosen = []
    for task, g in eps.groupby("task"):
        med = g["length"].median()
        chosen.append(int((g["length"] - med).abs().idxmin()))
data = pd.concat(pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state", "action"])
                 for p in sorted(split.glob("data/*/*.parquet")))

model = build_framework(cfg)
if args.ckpt != "none":  # "none": untrained head, pipeline smoke test only
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location="cpu", mmap=True), strict=False)
    assert not unexpected and not [k for k in missing if k.startswith(("qwen_vl_interface", "action_model"))], (missing[:3], unexpected[:3])
model.qwen_vl_interface.to(getattr(torch, args.dtype))
model = model.cuda().eval()

vkey = f"videos/observation.images.{args.front_cam}/"
out = Path(args.out)
for ep in chosen:
    row = eps.loc[ep]
    tag = args.tag_prefix + f"ep{ep:03d}_" + row["task"].split(" and ")[0].lower().replace(" ", "_")[:40]
    d = out / tag
    d.mkdir(parents=True, exist_ok=True)
    rows = data[data.episode_index == ep].sort_values("frame_index")
    gt = np.stack(rows["action"].to_numpy()).astype(np.float64)
    state = np.stack(rows["observation.state"].to_numpy()).astype(np.float64)
    T = len(gt)
    pred = np.zeros_like(gt)
    with torch.no_grad():
        for t0 in range(0, T, args.stride):
            torch.manual_seed(args.seed * 100003 + t0)
            # Fresh CUDA decoder per request: a reused torchcodec_cuda decoder returns wrong frames after a
            # backward seek into the first ~20 s of a file, and history windows (t-90..t) seek backward.
            video_io._CUDA_DECODER_CACHE.clear()
            ex = val_ds[step_index[(ep, t0)]]
            if args.static_history:
                ex = {**ex, "image": [ex["image"][-1]] * len(ex["image"])}
            chunk = unnorm(model.predict_action([ex])["normalized_actions"][0])
            if args.rel_arms:  # residual arms -> absolute, anchored on the arm state at t0
                chunk[:, 2:9] += state[t0, 15:22]
                chunk[:, 22:29] += state[t0, 22:29]
            n = min(args.stride, T - t0)
            pred[t0:t0 + n] = chunk[:n]
    np.save(d / "pred.npy", pred)
    np.save(d / "gt_action.npy", gt)
    np.save(d / "state.npy", state)
    src = split / f"videos/observation.images.{args.front_cam}/chunk-{int(row[vkey + 'chunk_index']):03d}/" \
                  f"file-{int(row[vkey + 'file_index']):03d}.mp4"
    a, b = float(row[vkey + "from_timestamp"]), float(row[vkey + "to_timestamp"])
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{a:.4f}", "-i", str(src),
                    "-t", f"{b - a:.4f}", "-vf", "scale=640:360", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-r", "30", str(d / "front.mp4")], check=True)
    arm = np.r_[2:9, 22:29]
    json.dump({"episode": ep, "task": row["task"], "frames": T, "fps": 30, "stride": args.stride,
               "ckpt": args.ckpt, "dtype": args.dtype, "robot": args.robot, "static_history": args.static_history,
               "dataset": args.dataset, "stats_dataset": stats_dataset, "rel_arms": args.rel_arms, "config": args.config,
               "arm_mae_deg": float(np.degrees(np.abs(pred[:, arm] - gt[:, arm]).mean())),
               "head_mae_deg": float(np.degrees(np.abs(pred[:, :2] - gt[:, :2]).mean()))},
              open(d / "meta.json", "w"), indent=2)
    print(f"{tag}: {T} frames, arm MAE {np.degrees(np.abs(pred[:, arm] - gt[:, arm]).mean()):.2f} deg, "
          f"head MAE {np.degrees(np.abs(pred[:, :2] - gt[:, :2]).mean()):.2f} deg", flush=True)
