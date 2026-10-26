"""'Push the turn-signal lever again' probe for a QwenGR00T_DualSystem checkpoint (car-turn-signal-ep val).

Deployment failure being mimicked: after the lever push the arm returns to the start pose while the head has not
turned yet, the observation looks like the episode start again and a memoryless policy pushes again.
Same scores as aliasing_probe.py / blink_probe.py (carfew models, front view only), here with the 3 cameras of the
DualSystem (head_left_2, wrist_left_2, wrist_right_2) and both turn-signal tasks (both push with the left arm).

Per episode (actions from the parquet, degrees): u = L-arm(push peak) - L-arm(pre-push frame f0), t* = head-turn onset.
  start      : real observation at f0                       (should push)
  post_push  : real observation at t*                       (should not push)
  natural    : frames after the push peak with the arm back within 5 deg of f0 and the head within 5 deg of its f0 yaw
               (= what the robot sees in the loop), real observation, plus swaps with the f0 observation:
               start_state (state of f0), start_head (head image of f0), start_wrists (wrist images of f0),
               start_images (all 3 images of f0).
Every condition gets identical noise (one seed before System 2 + System 1, plan age 0 = synchronous call).
Scores: push = <c[29] - c[0], u> / |u|^2 for the System 1 chunk (s1) and for the System 2 plan interpolated to
the same 0 .. 29/30 s (s2); past = max over the plan's past keyframes (-2 .. 0 s) of <kf(tau) - kf(0), u> / |u|^2,
i.e. whether System 2's reconstructed past contains the push; head = |yaw change| of the s1 chunk (deg).
Left-turn frames are labelled indicator on/off from the left side-mirror region of head_left_2 (as blink_probe.py).
dual_system.stale_input vlm (e.g. VLM 0.5 s behind): pass the matching --robot (video at t + video_offsets); the VLM gets
the frame of video_offsets[vlm_main_idx] (the current frame where that lies before the episode start, as in training),
C-RADIO / state / System 1 the current frame; the image swaps move every offset's frame of the swapped cameras.

Usage: CUDA_VISIBLE_DEVICES=6 python aliasing_probe_dual.py --ckpt <stage2.pt> --config <stage2.yaml> --out <json>
       v2 VLM-lag run: --dataset car-turn-signal-v2-ep --robot h1_2_car_turn_dual_vlm15_nohc
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/mnt/DATA2/yyg/starVLA")
from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps  # noqa: E402
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dataset", default="car-turn-signal-ep")
ap.add_argument("--robot", default="h1_2_car_turn_dual_s2", help="data config with video/state at offset 0 only")
ap.add_argument("--per_ep", type=int, default=8, help="natural aliased frames per episode")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

LARM, HEAD = np.r_[2:9], 0
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
    return np.degrees(np.where(const, x, (x + 1.0) / 2.0 * (q99 - q01) + q01))  # deg for joints (hands unused)


model = build_framework(cfg)
missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location="cpu", mmap=True), strict=False)
assert not unexpected and not missing, (missing[:3], unexpected[:3])
model = model.cuda().eval()
kf_times = model.timeline.times.cpu().numpy().astype(np.float64)  # [61] s
past = kf_times <= 1e-6
rate = float(cfg.framework.dual_system.action_rate_hz)
tt = np.array([0.0, 29.0 / rate])


vlm_mode = model.stale_input == "vlm"


def plan(ex, radio, t_img):
    """System 2 plan; stale_input vlm: VLM tokens of the frame t_img + video_offsets[vlm_main_idx] (t_img = frame of the
    example's images; the current frame where that lies before the episode start, as in training)."""
    if not vlm_mode:
        return model.plan_keyframes([ex], radio=radio)
    j = model.vlm_main_idx if t_img + model.video_offsets[model.vlm_main_idx] >= 0 else model.cur
    vlm = model.encode_vlm_cache([{**ex, "image": model._current_images(ex, j)}])
    return model.plan_keyframes([ex], radio=radio, vlm=vlm, vlm_age_s=-model.video_offsets[j] / rate)


def scores(ex, u, t_img):
    torch.manual_seed(args.seed)
    radio = model.encode_visual([ex])
    kf_n = plan(ex, radio, t_img)
    c = unnorm(model.act_from_plan([ex], kf_n, 0.0, radio=radio)[0].cpu().numpy())
    kf = unnorm(kf_n[0].cpu().numpy())
    p = np.stack([np.interp(tt, kf_times, kf[:, j]) for j in LARM], 1)  # [2, 7]
    k0 = np.stack([np.interp(0.0, kf_times, kf[:, j]) for j in LARM])
    uu = float(np.dot(u, u))
    return {"s1": float(np.dot(c[29, LARM] - c[0, LARM], u) / uu), "s2": float(np.dot(p[1] - p[0], u) / uu),
            "past": float(((kf[past][:, LARM] - k0) @ u).max() / uu), "head": float(abs(c[29, HEAD] - c[0, HEAD]))}


def example(ep, t):
    return val_ds[step_index[(ep, t)]]


def indicator_on(frames):
    """Orange pixels in the left side-mirror region of the 1280x720 head_left_2 (= old front_view) frame."""
    x = frames[:, 280:470, 0:150, :].astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return ((r > 200) & (g > 80) & (g < 220) & (b < 140) & (r - b > 90)).mean(axis=(1, 2)) > 0.01


eps = pd.concat(pd.read_parquet(p) for p in sorted(split.glob("meta/episodes/*/*.parquet"))).set_index("episode_index")
tasks = pd.read_parquet(split / "meta/tasks.parquet")
task_of = {int(i): str(t) for t, i in zip(tasks.index, tasks["task_index"])}
eps["task"] = [task_of[int(t[0])] for t in eps["tasks"]]
kept = sorted(int(t) for t in val_ds.trajectory_ids)
data = pd.concat(pd.read_parquet(p, columns=["episode_index", "frame_index", "action"]) for p in sorted(split.glob("data/*/*.parquet")))
vk = "videos/observation.images.head_left_2/"
rng = np.random.default_rng(args.seed)

rows, eprows = [], []
with torch.no_grad():
    for ep in kept:
        A = np.degrees(np.stack(data[data.episode_index == ep].sort_values("frame_index")["action"].to_numpy()))
        T = len(A)
        q, yaw = A[:, LARM], A[:, HEAD]
        d = np.abs(q - q[:10].mean(0)).max(1)
        mv = np.where(d > 10)[0]
        ht = np.where(np.abs(yaw - yaw[:10].mean()) > 10)[0]
        if not len(mv):
            continue
        ps = int(mv[0])
        f0 = max(ps - 10, 0)
        pk = ps + int(np.argmax(d[ps:ps + 150]))
        u = q[pk] - q[f0]
        ts = int(ht[0]) if len(ht) else T
        left = "left" in eps.loc[ep, "task"].lower()
        gt = lambda t: float(np.dot(q[min(t + 29, T - 1)] - q[t], u) / np.dot(u, u))
        ex0 = example(ep, f0)
        er = {"ep": ep, "task": eps.loc[ep, "task"], "f0": f0, "push_peak": pk, "t_star": ts, "gt_start": gt(f0),
              "start": scores(ex0, u, f0)}
        if ts + 30 < T:
            er["gt_post_push"] = gt(ts)
            er["post_push"] = scores(example(ep, ts), u, ts)
        eprows.append(er)
        print(ep, er["task"][:30], {k: v for k, v in er.items() if k in ("f0", "push_peak", "t_star", "gt_start")},
              "start", {k: round(v, 2) for k, v in er["start"].items()},
              "post", {k: round(v, 2) for k, v in er.get("post_push", {}).items()}, flush=True)
        dev0, ydev = np.abs(q - q[f0]).max(1), np.abs(yaw - yaw[f0])
        cand = np.array([t for t in range(pk, T - 31) if dev0[t] < 5 and ydev[t] < 5])
        if not len(cand):
            continue
        cand = np.sort(rng.choice(cand, size=min(args.per_ep, len(cand)), replace=False))
        on = [None] * len(cand)
        if left:
            e = eps.loc[ep]
            src = split / f"videos/observation.images.head_left_2/chunk-{int(e[vk + 'chunk_index']):03d}/file-{int(e[vk + 'file_index']):03d}.mp4"
            on = indicator_on(get_frames_by_timestamps(str(src), float(e[vk + "from_timestamp"]) + cand / 30, "torchcodec")).tolist()
        im0 = list(ex0["image"])
        nt = len(im0) // model.num_views  # frames per camera (camera-major image list)
        for t, o in zip(cand, on):
            ext = example(ep, int(t))
            imt = list(ext["image"])
            r = {"ep": ep, "t": int(t), "left": left, "indicator_on": o, "sec_after_peak": (int(t) - pk) / 30,
                 "sec_before_head_turn": (ts - int(t)) / 30, "arm_dev_deg": float(dev0[t]), "gt": gt(int(t)),
                 "gt_past": float(((q[max(int(t) - 60, 0):int(t) + 1] - q[t]) @ u).max() / np.dot(u, u)),
                 "real": scores(ext, u, int(t)),
                 "start_state": scores({**ext, "state": ex0["state"]}, u, int(t)),
                 "start_head": scores({**ext, "image": im0[:nt] + imt[nt:]}, u, f0),
                 "start_wrists": scores({**ext, "image": imt[:nt] + im0[nt:]}, u, f0),
                 "start_images": scores({**ext, "image": im0}, u, f0)}
            rows.append(r)
            print("  ", t, "on" if o else ("off" if o is not None else "-"), f"gt {r['gt']:.2f}",
                  {k: round(r[k]["s1"], 2) for k in ("real", "start_state", "start_head", "start_wrists", "start_images")}, flush=True)

json.dump({"episodes": eprows, "natural": rows, "ckpt": args.ckpt, "config": args.config}, open(args.out, "w"), indent=1)


def line(name, vals):
    v = np.asarray(vals, np.float64)
    return f"{name:28s} n={len(v):3d}  mean {v.mean():6.3f}  >0.1 {np.mean(v > 0.1):4.0%}"


print("\n=== push score = fraction of the way to the push peak within 1 s (re-push: > 0.1)")
print(line("gt start", [r["gt_start"] for r in eprows]))
for k in ("s1", "s2"):
    print(line(f"start {k}", [r["start"][k] for r in eprows]))
print(line("gt post_push", [r["gt_post_push"] for r in eprows if "post_push" in r]))
for k in ("s1", "s2"):
    print(line(f"post_push {k}", [r["post_push"][k] for r in eprows if "post_push" in r]))
print(f"\nnatural aliased frames (arm back <5 deg, head <5 deg): {len(rows)} frames / {len({r['ep'] for r in rows})} episodes")
print(line("gt", [r["gt"] for r in rows]))
for cond in ("real", "start_state", "start_head", "start_wrists", "start_images"):
    for k in ("s1", "s2"):
        print(line(f"{cond} {k}", [r[cond][k] for r in rows]))
for o in (True, False):
    sub = [r for r in rows if r["indicator_on"] is o]
    if sub:
        print(line(f"real s1 indicator {'on' if o else 'off'}", [r["real"]["s1"] for r in sub]))
print("\nSystem 2 reconstructed past (max push toward the peak within -2..0 s)")
for cond in ("real", "start_images"):
    print(line(f"natural {cond} past", [r[cond]["past"] for r in rows]))
print(line("natural gt past", [r["gt_past"] for r in rows]))
print(line("start past", [r["start"]["past"] for r in eprows]))
