"""Turn-signal blink vs. 'push again': is the repeat loop explained by the indicator's off phase?

After the lever push the left side-mirror repeater blinks (~0.4 s on / 0.4 s off). The only visual
evidence that the push already happened is that light, so a frame in the off phase looks like the
pre-push scene. For left-turn val episodes this samples frames between indicator activation and the
head-turn onset t*, labels each on/off from the mirror region, and scores the predicted 1-s chunk:
push = <pred[29]-pred[0], u>/|u|^2 with u = L-arm(push peak) - L-arm(pre-push), as in aliasing_probe.py.

Conditions (identical noise):
  base_real   : memoryless baseline, real state
  base_start  : baseline, state replaced by the pre-push state (arm back at the start pose)
  mem_start   : memory model, pre-push state, real history (t-90..t-30)
  mem_static  : memory model, pre-push state, history = current frame x4

Usage: CUDA_VISIBLE_DEVICES=4 python blink_probe.py --mem_ckpt <hamlet.pt> --base_ckpt <base.pt> --out <json>
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
from starVLA.dataloader.gr00t_lerobot import video as video_io  # noqa: E402
from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_timestamps  # noqa: E402
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402

TF = "/mnt/DATA2/yyg/starVLA/examples/realRobots/H1_2_car_real/train_files/"
ap = argparse.ArgumentParser()
ap.add_argument("--mem_ckpt", required=True)
ap.add_argument("--base_ckpt", required=True)
ap.add_argument("--mem_config", default=TF + "starvla_qwengr00t_dit_hamlet_carfew_v1_2b_front_abs.yaml")
ap.add_argument("--base_config", default=TF + "starvla_qwengr00t_carfew_v1_2b_front_abs.yaml")
ap.add_argument("--per_phase", type=int, default=3, help="frames per episode per blink phase")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

LARM, HEAD = np.r_[2:9], 0
cfg_m, cfg_b = OmegaConf.load(args.mem_config), OmegaConf.load(args.base_config)
for c in (cfg_m, cfg_b):
    c.framework.qwenvl.autocast_dtype = "float16"
root = Path(cfg_m.datasets.vla_data.data_root_dir)
split = root / "car-few-tasks-v1/val"
tcfg = OmegaConf.to_container(cfg_m.datasets.vla_data)
vcfg = dict(tcfg)
vcfg.pop("exclude_episodes", None)
train_ds = make_LeRobotSingleDataset(root, "car-few-tasks-v1/train", "h1_2_car_left_front_hist30", data_cfg=OmegaConf.create(tcfg))
val_ds = make_LeRobotSingleDataset(root, "car-few-tasks-v1/val", "h1_2_car_left_front_hist30", data_cfg=OmegaConf.create(vcfg))
val_ds.set_transforms_metadata(train_ds.metadata)
step_index = {(int(t), int(b)): i for i, (t, b) in enumerate(val_ds.all_steps)}
stats = json.load(open(root / "car-few-tasks-v1/train/meta/stats_gr00t.abs.json"))["statistics"]["action"]
q01, q99 = np.array(stats["q01"], np.float64), np.array(stats["q99"], np.float64)
const = q99 == q01


def unnorm(x):
    x = np.asarray(x, np.float64)
    return np.degrees(np.where(const, x, (x + 1.0) / 2.0 * (q99 - q01) + q01))


def example(ep, t):
    video_io._CUDA_DECODER_CACHE.clear()  # torchcodec_cuda backward-seek bug: fresh decoder per request
    return val_ds[step_index[(ep, t)]]


def load(cfg, ckpt):
    m = build_framework(cfg)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", mmap=True), strict=False)
    m.qwen_vl_interface.to(torch.float16)
    return m.cuda().eval()


def indicator_on(frames):
    """Orange pixels in the left side-mirror region of the 1280x720 front view."""
    x = frames[:, 280:470, 0:150, :].astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return ((r > 200) & (g > 80) & (g < 220) & (b < 140) & (r - b > 90)).mean(axis=(1, 2)) > 0.01


mem, base = load(cfg_m, args.mem_ckpt), load(cfg_b, args.base_ckpt)


def push_score(model, ex, u):
    torch.manual_seed(args.seed)
    with torch.no_grad():
        c = unnorm(model.predict_action([ex])["normalized_actions"][0])
    return float(np.dot(c[29, LARM] - c[0, LARM], u) / np.dot(u, u)), float(abs(c[29, HEAD] - c[0, HEAD]))


eps = pd.concat(pd.read_parquet(p) for p in sorted(split.glob("meta/episodes/*/*.parquet"))).set_index("episode_index")
tasks = pd.read_parquet(split / "meta/tasks.parquet")
task_of = {int(i): str(t) for t, i in zip(tasks.index, tasks["task_index"])}
eps["task"] = [t[0] if isinstance(t[0], str) else task_of[int(t[0])] for t in eps["tasks"]]
kept = set(int(t) for t in val_ds.trajectory_ids)
chosen = [int(e) for e, r in eps.iterrows() if r["task"].startswith("Activate the left turn signal") and int(e) in kept]
data = pd.concat(pd.read_parquet(p, columns=["episode_index", "frame_index", "action"]) for p in sorted(split.glob("data/*/*.parquet")))
vk = "videos/observation.images.front_view/"
rng = np.random.default_rng(args.seed)

rows = []
for ep in chosen:
    A = np.degrees(np.stack(data[data.episode_index == ep].sort_values("frame_index")["action"].to_numpy()))
    T = len(A)
    q, yaw = A[:, LARM], A[:, HEAD]
    d = np.abs(q - q[:10].mean(0)).max(1)
    mv = np.where(d > 10)[0]
    ht = np.where(np.abs(yaw - yaw[:10].mean()) > 10)[0]
    if not len(mv) or not len(ht):
        continue
    ps, ts = int(mv[0]), int(ht[0])
    f0 = max(ps - 10, 0)
    pk = ps + int(np.argmax(d[ps:ps + 150]))
    u = q[pk] - q[f0]
    e = eps.loc[ep]
    src = split / f"videos/observation.images.front_view/chunk-{int(e[vk + 'chunk_index']):03d}/file-{int(e[vk + 'file_index']):03d}.mp4"
    ts_frames = np.arange(0, min(ts, T - 31))
    on = indicator_on(get_frames_by_timestamps(str(src), float(e[vk + "from_timestamp"]) + ts_frames / 30, "torchcodec"))
    if not on.any():
        continue
    first_on = int(np.argmax(on))
    window = np.arange(first_on + 9, min(ts, T - 31))  # from ~0.3 s after activation to the head-turn onset
    if len(window) < 10:
        continue
    ex0 = example(ep, f0)
    for phase, mask in (("on", on[window]), ("off", ~on[window])):
        cand = window[mask]
        for t in rng.choice(cand, size=min(args.per_phase, len(cand)), replace=False) if len(cand) else []:
            ext = example(ep, int(t))
            aliased = {**ext, "state": ex0["state"]}  # pre-push state, real current image + history
            r = {"ep": ep, "t": int(t), "phase": phase, "arm_dev_from_start_deg": float(d[t]),
                 "base_real": push_score(base, {**ext, "image": ext["image"][-1:]}, u),
                 "base_start": push_score(base, {**aliased, "image": ext["image"][-1:]}, u),
                 "mem_start": push_score(mem, aliased, u),
                 "mem_static": push_score(mem, {**aliased, "image": [ext["image"][-1]] * len(ext["image"])}, u)}
            rows.append(r)
            print(ep, t, phase, {k: tuple(round(x, 2) for x in v) for k, v in r.items() if isinstance(v, tuple)}, flush=True)

json.dump(rows, open(args.out, "w"), indent=1)
print("\n=== push score (fraction of the way to the push peak in 1 s) and share of frames with push > 0.1")
for phase in ("on", "off"):
    sub = [r for r in rows if r["phase"] == phase]
    print(f"indicator {phase:3s}: {len(sub)} frames from {len({r['ep'] for r in sub})} episodes, "
          f"arm deviation from start median {np.median([r['arm_dev_from_start_deg'] for r in sub]):.1f} deg")
    for k in ("base_real", "base_start", "mem_start", "mem_static"):
        v = np.array([r[k][0] for r in sub])
        print(f"   {k:11s} push {v.mean():6.3f}  re-push (>0.1) {np.mean(v > 0.1):4.0%}")
