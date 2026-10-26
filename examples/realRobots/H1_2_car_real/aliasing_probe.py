"""Does the history memory stop a 'repeat the push' loop? Offline probe on left-turn val episodes.

Deployment failure being mimicked: after pushing the turn-signal lever and pulling the arm back, the
robot's observation looks like the episode start again and a memoryless policy pushes again.

Per episode (actions from the parquet, degrees):
  push direction u = L-arm(push peak) - L-arm(pre-push frame f0); t* = head-turn onset (push -> inspect).
Conditions, all with identical noise:
  A  start      : current image/state from f0, history = what the loader gives at f0 (memory model)
  B  aliased    : current image/state from f0, history = the REAL frames t*-90, t*-60, t*-30 (memory model)
  C  post-push  : real observation and history at t* (memory model)
  D  base start : current image/state from f0 (memoryless baseline)
  N  natural    : episodes whose arm really returns within 5 deg of f0 after the push: real obs at that
                  frame, real history vs static history (memory model)
Scores per predicted 1-s chunk: push = <pred[29]-pred[0], u>/|u|^2 (fraction of the way to the push
peak), head = |yaw[29]-yaw[0]| deg. GT scores at f0 / t* are printed for reference.

Usage: CUDA_VISIBLE_DEVICES=4 python aliasing_probe.py --mem_ckpt <hamlet.pt> --base_ckpt <base.pt> --out <json>
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
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402

TF = "/mnt/DATA2/yyg/starVLA/examples/realRobots/H1_2_car_real/train_files/"
ap = argparse.ArgumentParser()
ap.add_argument("--mem_ckpt", required=True)
ap.add_argument("--base_ckpt", required=True)
ap.add_argument("--mem_config", default=TF + "starvla_qwengr00t_dit_hamlet_carfew_v1_2b_front_abs.yaml")
ap.add_argument("--base_config", default=TF + "starvla_qwengr00t_carfew_v1_2b_front_abs.yaml")
ap.add_argument("--task_prefix", default="Activate the left turn signal")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

LARM, HEAD = np.r_[2:9], 0
cfg_m, cfg_b = OmegaConf.load(args.mem_config), OmegaConf.load(args.base_config)
for c in (cfg_m, cfg_b):
    c.framework.qwenvl.autocast_dtype = "float16"
root = Path(cfg_m.datasets.vla_data.data_root_dir)
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
    return np.degrees(np.where(const, x, (x + 1.0) / 2.0 * (q99 - q01) + q01))  # deg for joints (hands unused)


def example(ep, t):
    video_io._CUDA_DECODER_CACHE.clear()  # torchcodec_cuda backward-seek bug: fresh decoder per request
    return val_ds[step_index[(ep, t)]]


def load(cfg, ckpt):
    m = build_framework(cfg)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", mmap=True), strict=False)
    m.qwen_vl_interface.to(torch.float16)
    return m.cuda().eval()


mem, base = load(cfg_m, args.mem_ckpt), load(cfg_b, args.base_ckpt)


def predict(model, ex):
    torch.manual_seed(args.seed)
    with torch.no_grad():
        return unnorm(model.predict_action([ex])["normalized_actions"][0])


def scores(chunk, f0q, u):
    return float(np.dot(chunk[29, LARM] - chunk[0, LARM], u) / np.dot(u, u)), float(abs(chunk[29, HEAD] - chunk[0, HEAD]))


# Episodes of the task and their GT action trajectories
split = root / "car-few-tasks-v1/val"
eps = pd.concat(pd.read_parquet(p) for p in sorted(split.glob("meta/episodes/*/*.parquet"))).set_index("episode_index")
tasks = pd.read_parquet(split / "meta/tasks.parquet")
task_of = {int(i): str(t) for t, i in zip(tasks.index, tasks["task_index"])}
eps["task"] = [t[0] if isinstance(t[0], str) else task_of[int(t[0])] for t in eps["tasks"]]
kept = set(int(t) for t in val_ds.trajectory_ids)
chosen = [int(e) for e, r in eps.iterrows() if r["task"].startswith(args.task_prefix) and int(e) in kept]
data = pd.concat(pd.read_parquet(p, columns=["episode_index", "frame_index", "action"]) for p in sorted(split.glob("data/*/*.parquet")))

rows = []
for ep in chosen:
    A = np.degrees(np.stack(data[data.episode_index == ep].sort_values("frame_index")["action"].to_numpy()))
    T = len(A)
    q, yaw = A[:, LARM], A[:, HEAD]
    d = np.abs(q - q[:10].mean(0)).max(1)
    mv = np.where(d > 10)[0]
    ht = np.where(np.abs(yaw - yaw[:10].mean()) > 10)[0]
    if not len(mv) or not len(ht) or ht[0] < 100 or ht[0] + 30 >= T:
        continue
    ps, ts = int(mv[0]), int(ht[0])
    f0 = max(ps - 10, 0)
    pk = ps + int(np.argmax(d[ps:ps + 150]))
    u = q[pk] - q[f0]
    gt = lambda t: (float(np.dot(q[min(t + 29, T - 1)] - q[t], u) / np.dot(u, u)), float(abs(yaw[min(t + 29, T - 1)] - yaw[t])))
    ex0, ext = example(ep, f0), example(ep, ts)
    r = {"ep": ep, "f0": f0, "push_peak": pk, "t_star": ts, "gt_f0": gt(f0), "gt_tstar": gt(ts)}
    r["A_start"] = scores(predict(mem, ex0), q[f0], u)
    r["B_aliased_real_hist"] = scores(predict(mem, {**ex0, "image": ext["image"][:-1] + ex0["image"][-1:]}), q[f0], u)
    r["C_post_push"] = scores(predict(mem, ext), q[ts], u)
    r["D_base_start"] = scores(predict(base, {**ex0, "image": ex0["image"][-1:]}), q[f0], u)
    # natural aliasing: arm back within 5 deg of f0 after the push peak, before the episode ends
    back = np.where(np.abs(q[pk:] - q[f0]).max(1) < 5)[0]
    if len(back) and pk + back[0] + 30 < T:
        tn = pk + int(back[0])
        exn = example(ep, tn)
        r["t_natural"] = tn
        r["gt_natural"] = gt(tn)
        r["N_natural_real_hist"] = scores(predict(mem, exn), q[tn], u)
        r["N_natural_static_hist"] = scores(predict(mem, {**exn, "image": [exn["image"][-1]] * len(exn["image"])}), q[tn], u)
        r["N_natural_base"] = scores(predict(base, {**exn, "image": exn["image"][-1:]}), q[tn], u)
    rows.append(r)
    print(ep, {k: (tuple(round(x, 2) for x in v) if isinstance(v, tuple) else v) for k, v in r.items() if k != "ep"}, flush=True)

json.dump(rows, open(args.out, "w"), indent=1)


def summary(key):
    v = np.array([r[key] for r in rows if key in r])
    return f"n={len(v):3d}  push {v[:, 0].mean():6.3f} (>0.1: {np.mean(v[:, 0] > 0.1):4.0%})  head {v[:, 1].mean():5.2f} deg"


print("\n=== push = fraction of the way to the push peak within the 1-s chunk; head = |yaw change| in the chunk")
for key in ["gt_f0", "A_start", "D_base_start", "B_aliased_real_hist", "C_post_push", "gt_tstar",
            "gt_natural", "N_natural_real_hist", "N_natural_static_hist", "N_natural_base"]:
    print(f"{key:24s} {summary(key)}")
