"""Receding-horizon rollouts of a chunk-policy checkpoint on whole val episodes (all cameras of the data config).

Every `--stride` frames the model gets the RECORDED observation at that frame (the data config's cameras, state, instruction;
val normalized with the TRAIN statistics) and predicts a chunk; the first `--stride` steps of each chunk are stitched into one
(T, 42) absolute trajectory. Open-loop w.r.t. the robot: what the policy would command if the world followed the demo.
Episodes are visited in order and frames forward only, so the CUDA video decoders never seek backwards.

Writes <out>/ep{id:05d}/{pred.npy, gt_action.npy, meta.json} (the layout seq_check.py reads; gt_action = the recorded action).

Usage: CUDA_VISIBLE_DEVICES=4 python rollout_val_seq.py --ckpt <ckpt.pt> --config <yaml> --dataset car-1015-v1-ep \
           --robot h1_2_car_3cam_headstate --out <dir> --tasks 0,1
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/mnt/DATA2/yyg/starVLA")
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--robot", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tasks", default="", help="comma-separated task_index values (default: all)")
ap.add_argument("--stride", type=int, default=30)
ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_episodes", type=int, default=0, help="first N matching episodes only (0 = all)")
args = ap.parse_args()

cfg = OmegaConf.load(args.config)
assert str(cfg.datasets.vla_data.action_mode) == "abs", "this script un-normalizes abs actions only"
cfg.framework.qwenvl.autocast_dtype = args.dtype
root = Path(cfg.datasets.vla_data.data_root_dir)
horizon = int(cfg.framework.action_model.action_horizon)
assert args.stride <= horizon, (args.stride, horizon)

train_cfg = OmegaConf.to_container(cfg.datasets.vla_data)
val_cfg = dict(train_cfg)
val_cfg.pop("exclude_episodes", None)
val_cfg.pop("color_jitter", None)
train_ds = make_LeRobotSingleDataset(root, f"{args.dataset}/train", args.robot, data_cfg=OmegaConf.create(train_cfg))
val_ds = make_LeRobotSingleDataset(root, f"{args.dataset}/val", args.robot, data_cfg=OmegaConf.create(val_cfg))
val_ds.set_transforms_metadata(train_ds.metadata)

stats = json.load(open(root / args.dataset / "train/meta/stats_gr00t.abs.json"))["statistics"]["action"]
q01, q99 = np.array(stats["q01"], np.float64), np.array(stats["q99"], np.float64)
const = q99 == q01


def unnorm(x):
    x = np.asarray(x, np.float64)
    return np.where(const, x, (x + 1.0) / 2.0 * (q99 - q01) + q01)


meta = root / args.dataset / "val/meta"
episodes = pd.read_parquet(meta / "episodes/chunk-000/file-000.parquet")
task_text = {int(i): t for t, i in pd.read_parquet(meta / "tasks.parquet")["task_index"].items()}
ep_task = {int(e): int(t[0]) for e, t in zip(episodes["episode_index"], episodes["tasks"])}
want = {int(t) for t in args.tasks.split(",")} if args.tasks else set(task_text)
pos = defaultdict(dict)  # episode -> {frame: dataset index}
for j, (e, t) in enumerate(val_ds.all_steps):
    pos[int(e)][int(t)] = j
eps = sorted(e for e in pos if ep_task[e] in want)
if args.max_episodes:
    eps = eps[:args.max_episodes]
print(f"{len(eps)} val episodes of tasks {sorted(want)}, stride {args.stride}, horizon {horizon}", flush=True)

t0 = time.time()
model = build_framework(cfg)
if args.ckpt != "none":  # "none": untrained head, pipeline smoke test only
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location="cpu", mmap=True), strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert not [k for k in missing if "qwen_vl_interface" in k or "action_model" in k], f"missing: {missing[:5]}"
model.qwen_vl_interface.to(getattr(torch, args.dtype))
model = model.cuda().eval()
print(f"model ready in {time.time() - t0:.0f}s", flush=True)

out = Path(args.out)
t0 = time.time()
with torch.no_grad():
    for n, e in enumerate(eps):
        T = len(pos[e])
        assert sorted(pos[e]) == list(range(T)), f"episode {e}: frames not contiguous"
        frames = list(range(0, T, args.stride))
        chunks = []
        for b0 in range(0, len(frames), args.batch):
            exs = [val_ds[pos[e][t]] for t in frames[b0:b0 + args.batch]]
            torch.manual_seed(args.seed * 100003 + e * 1009 + b0)
            chunks.append(unnorm(model.predict_action(exs)["normalized_actions"]))
        chunks = np.concatenate(chunks)
        pred = np.concatenate([c[:args.stride] for c in chunks])[:T]
        gt = np.stack(val_ds.get_trajectory_data(e)["action"].values).astype(np.float64)
        assert gt.shape == pred.shape, (e, gt.shape, pred.shape)
        d = out / f"ep{e:05d}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "pred.npy", pred)
        np.save(d / "gt_action.npy", gt)
        json.dump({"episode": e, "task_index": ep_task[e], "task": task_text[ep_task[e]], "length": T, "stride": args.stride,
                   "ckpt": args.ckpt, "dtype": args.dtype}, open(d / "meta.json", "w"))
        if n % 10 == 0:
            print(f"  {n + 1}/{len(eps)} episodes  {time.time() - t0:.0f}s", flush=True)
print(f"done {len(eps)} episodes in {time.time() - t0:.0f}s", flush=True)
