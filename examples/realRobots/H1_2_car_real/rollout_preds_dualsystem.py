"""Receding-horizon rollout of a QwenGR00T_DualSystem checkpoint on car val episodes, for videos.

Same output layout as rollout_preds.py (render_rollout.py / render_compare.py read it), with a deployment-like
two-rate schedule on the recorded observations (open loop w.r.t. the robot):
  System 2 re-plans every --s2-stride frames from the real observation (61 keyframes, -2 .. +4 s);
  System 1 runs every --s1-stride frames on the current frame + that plan (plan age = frames since the plan / 30 Hz)
  and its first --s1-stride actions are executed.
Defaults 30 / 6 give plan ages {0, 0.2, 0.4, 0.6, 0.8} s = the stage-2 training offsets.
dual_system.stale_input vlm models: every --s1-stride frames System 2 re-plans on the current frame's C-RADIO / state
and the VLM tokens of frame t - --vlm-lag (default: the training VLM offset; frame t where t - lag < 0, as in training),
and System 1 runs on that plan (age 0); --s2-stride is unused and the System-2-only output is the per-step plan.

Per episode it writes <out>/<tag>/{pred.npy (System 1), gt_action.npy, state.npy, front.mp4, meta.json} and
<out_s2>/<tag>/ the same with pred.npy = the System 2 plan alone (keyframes linearly interpolated per frame), so
render_compare.py can show System 1 vs its plan. meta.json "stride" = --s2-stride (the re-plan ticks in the videos).

Usage: CUDA_VISIBLE_DEVICES=0 python rollout_preds_dualsystem.py --ckpt <ckpt.pt> --config <stage2.yaml> --out <dir>
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

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--out-s2", default="", help="dir for the System-2-only rollouts (default: <out>_s2only)")
ap.add_argument("--dataset", default="car-turn-signal-ep")
ap.add_argument("--robot", default="h1_2_car_turn_dual_s2", help="data config with video/state at offset 0 only")
ap.add_argument("--s2-stride", type=int, default=30)
ap.add_argument("--s1-stride", type=int, default=6)
ap.add_argument("--vlm-lag", type=int, default=None, help="stale_input vlm: VLM frame lag (default: training offset)")
ap.add_argument("--episodes", default="33,46,59,45")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--front-cam", default="head_left_2")
ap.add_argument("--tag-prefix", default="")
ap.add_argument("--memory-mode", default="continuous", choices=["continuous", "window", "init"],
                help="dual_system.memory: continuous = update on every VLM frame (every 0.5 s) from the episode start "
                     "(deployment); window = rebuild it before every System 2 call from the initial memory over the last "
                     "len(update_offsets) VLM frames (= training); init = always the initial memory (memory ablation)")
args = ap.parse_args()
assert args.s2_stride % args.s1_stride == 0, "System 1 calls must land on System 2 re-plans"

cfg = OmegaConf.load(args.config)
root = Path(cfg.datasets.vla_data.data_root_dir)
split = root / args.dataset / "val"
train_cfg = OmegaConf.to_container(cfg.datasets.vla_data)
val_cfg = dict(train_cfg)
val_cfg.pop("exclude_episodes", None)
val_cfg.pop("color_jitter", None)  # train-only augmentation
val_cfg["video_backend"] = "torchcodec"  # CPU decoding (no reused-CUDA-decoder seek bug)
_corrupt = split / "meta/corrupted_episodes.json"
if _corrupt.exists():
    val_cfg["exclude_episodes"] = json.load(open(_corrupt))
train_ds = make_LeRobotSingleDataset(root, f"{args.dataset}/train", args.robot, data_cfg=OmegaConf.create(train_cfg))
val_ds = make_LeRobotSingleDataset(root, f"{args.dataset}/val", args.robot, data_cfg=OmegaConf.create(val_cfg))
val_ds.set_transforms_metadata(train_ds.metadata)
step_index = {(int(t), int(b)): i for i, (t, b) in enumerate(val_ds._all_steps)}

stats = json.load(open(root / args.dataset / "train/meta/stats_gr00t.abs.json"))["statistics"]["action"]
q01, q99 = np.array(stats["q01"], np.float64), np.array(stats["q99"], np.float64)
const = q99 == q01


def unnorm(x):
    x = np.asarray(x, np.float64)
    return np.where(const, x, (x + 1.0) / 2.0 * (q99 - q01) + q01)


def norm(x):  # the loader's q99 normaliser (identity where q01 == q99, clamped at +-2.2)
    x = np.asarray(x, np.float64)
    return np.clip(np.where(const, x, 2.0 * (x - q01) / np.where(const, 1.0, q99 - q01) - 1.0), -2.2, 2.2)


eps = pd.concat(pd.read_parquet(p) for p in sorted(split.glob("meta/episodes/*/*.parquet"))).set_index("episode_index")
tasks = pd.read_parquet(split / "meta/tasks.parquet")
task_of = {int(i): str(t) for t, i in zip(tasks.index, tasks["task_index"])}
for old, new in (cfg.datasets.vla_data.get("task_text_replace") or {}).items():
    task_of = {k: v.replace(old, new) for k, v in task_of.items()}
eps["task"] = [task_of[int(t[0])] for t in eps["tasks"]]
chosen = [int(e) for e in args.episodes.split(",")]
data = pd.concat(pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state", "action"])
                 for p in sorted(split.glob("data/*/*.parquet")))

model = build_framework(cfg)
missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location="cpu", mmap=True), strict=False)
assert not unexpected and not missing, (missing[:3], unexpected[:3])
model = model.cuda().eval()
ds_cfg = cfg.framework.dual_system
rate = float(ds_cfg.action_rate_hz)
kf_times = model.timeline.times.cpu().numpy().astype(np.float64)  # [61] s, relative to the plan's observation

vkey = f"videos/observation.images.{args.front_cam}/"
out = Path(args.out)
out_s2 = Path(args.out_s2 or f"{args.out.rstrip('/')}_s2only")
arm = np.r_[2:9, 22:29]
vlm_mode = getattr(model, "stale_input", "plan") == "vlm"
vlm_lag = (args.vlm_lag if args.vlm_lag is not None else -model.video_offsets[model.vlm_main_idx]) if vlm_mode else 0
stride = args.s1_stride if vlm_mode else args.s2_stride
mem_on = getattr(model, "episode_memory", None) is not None
if mem_on:  # VLM frames on the grid 0, step, 2 step, ... (step = the training update interval); frame g is used from g + lag
    _mo = [model.video_offsets[j] for j in model.mem_idx]
    mem_step = _mo[1] - _mo[0] if len(_mo) > 1 else vlm_lag
    assert args.s1_stride % mem_step == 0 and vlm_lag % mem_step == 0, (args.s1_stride, vlm_lag, mem_step)
for ep in chosen:
    row = eps.loc[ep]
    tag = args.tag_prefix + f"ep{ep:03d}_" + row["task"].split(" and ")[0].lower().replace(" ", "_")[:40]
    rows = data[data.episode_index == ep].sort_values("frame_index")
    gt = np.stack(rows["action"].to_numpy()).astype(np.float64)
    state = np.stack(rows["observation.state"].to_numpy()).astype(np.float64)
    T = len(gt)
    pred, pred_s2 = np.zeros_like(gt), np.zeros_like(gt)
    progress = np.full(T, np.nan)
    if mem_on:
        memory, g_next, vcache = model.memory_init(1), 0, {}
    with torch.no_grad():
        for t0 in range(0, T, args.s1_stride):
            torch.manual_seed(args.seed * 100003 + t0)
            ex = val_ds[step_index[(ep, t0)]]
            if model.prev_action_dims and t0 > 0:  # the last command this rollout sent (else the recorded one)
                ex["prev_action"] = norm(pred[t0 - 1])[model.prev_action_dims].astype(np.float32)
            radio = model.encode_visual([ex])
            if vlm_mode:
                t_v = t0 - vlm_lag if t0 >= vlm_lag else t0
                mkw = {}
                if mem_on:
                    while g_next <= t0 - vlm_lag:
                        vcache[g_next] = model.encode_vlm_cache([val_ds[step_index[(ep, g_next)]]])
                        if args.memory_mode == "continuous":
                            memory = model.memory_update(memory, vcache[g_next])
                        g_next += mem_step
                    if args.memory_mode == "init":
                        memory = model.memory_init(1)
                    if args.memory_mode == "window":
                        memory = model.memory_init(1)
                        for g in sorted(vcache)[-len(model.mem_idx):]:
                            memory = model.memory_update(memory, vcache[g])
                    mkw["memory"] = memory
                vlm = vcache[t_v] if mem_on and t_v in vcache else model.encode_vlm_cache([val_ds[step_index[(ep, t_v)]]])
                kf = model.plan_keyframes([ex], radio=radio, vlm=vlm, vlm_age_s=(t0 - t_v) / rate, **mkw)
                if mem_on:
                    progress[t0:t0 + args.s1_stride] = float(model.last_progress[0])
                t_plan = t0
                plan = unnorm(model.plan_to_absolute([ex], kf)[0].cpu().numpy())
                n2 = min(args.s1_stride, T - t0)
                tt = np.arange(n2) / rate
                pred_s2[t0:t0 + n2] = np.stack([np.interp(tt, kf_times, plan[:, j]) for j in range(plan.shape[1])], 1)
            elif t0 % args.s2_stride == 0:
                kf = model.plan_keyframes([ex], radio=radio)
                t_plan = t0
                plan = unnorm(model.plan_to_absolute([ex], kf)[0].cpu().numpy())  # [61, D]
                n2 = min(args.s2_stride, T - t0)
                tt = np.arange(n2) / rate
                pred_s2[t0:t0 + n2] = np.stack([np.interp(tt, kf_times, plan[:, j]) for j in range(plan.shape[1])], 1)
            chunk = unnorm(model.act_from_plan([ex], kf, (t0 - t_plan) / rate, radio=radio)[0].cpu().numpy())
            n = min(args.s1_stride, T - t0)
            pred[t0:t0 + n] = chunk[:n]
    src = split / f"videos/observation.images.{args.front_cam}/chunk-{int(row[vkey + 'chunk_index']):03d}/" \
                  f"file-{int(row[vkey + 'file_index']):03d}.mp4"
    a, b = float(row[vkey + "from_timestamp"]), float(row[vkey + "to_timestamp"])
    for o, p, kind in ((out, pred, "system1"), (out_s2, pred_s2, "system2_plan")):
        d = o / tag
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "pred.npy", p)
        if mem_on:
            np.save(d / "progress.npy", progress)
        np.save(d / "gt_action.npy", gt)
        np.save(d / "state.npy", state)
        if not (d / "front.mp4").exists():
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{a:.4f}", "-i", str(src),
                            "-t", f"{b - a:.4f}", "-vf", "scale=640:360", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-r", "30", str(d / "front.mp4")], check=True)
        am, hm = float(np.degrees(np.abs(p[:, arm] - gt[:, arm]).mean())), float(np.degrees(np.abs(p[:, :2] - gt[:, :2]).mean()))
        json.dump({"episode": ep, "task": row["task"], "frames": T, "fps": 30, "stride": stride,
                   "s1_stride": args.s1_stride, "vlm_lag": vlm_lag, "pred": kind, "ckpt": args.ckpt, "robot": args.robot,
                   "dataset": args.dataset, "config": args.config, "arm_mae_deg": am, "head_mae_deg": hm,
                   **({"memory_mode": args.memory_mode} if mem_on else {})},
                  open(d / "meta.json", "w"), indent=2)
        print(f"{tag} [{kind}]: {T} frames, arm MAE {am:.2f} deg, head MAE {hm:.2f} deg", flush=True)
