"""Held-out val evaluation of an H1-2 car checkpoint, in physical units, per task.

For N random val steps it predicts one action chunk from the real observation (front image,
state, instruction) and compares it with the ground-truth chunk:
  head  [deg]  dims 0:2
  arms  [deg]  dims 2:9 (L) and 22:29 (R)
  hands [raw Inspire units] dims 9:22 (L) and 29:42 (R)
Reported as chunk means, per chunk step, and per task, next to two no-model baselines on the same steps:
  hold_cmd   = the current GT command a[t] held for the whole chunk (an oracle: a[t] is not observed),
  hold_state = the current measured state held (deployable; only for action keys the state also has).

Normalization: the val split is normalized with the TRAIN statistics (the loader would
otherwise use val's own stats, which shifts the model's state input and mis-scales GT).
abs action mode only: every group is un-normalized straight from train q01/q99.

Precision: --dtype float16|bfloat16|float32 sets the VLM weights and its autocast dtype; the
action head always runs in fp32. Noise for the flow-matching sampler is reseeded per batch,
so runs with different --dtype see identical noise.

Usage:
  CUDA_VISIBLE_DEVICES=4 python eval_val_chunks.py --ckpt <steps_N_pytorch_model.pt> \
      --out <result.json> [--dtype float16] [--n 1800]
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/mnt/DATA2/yyg/starVLA")
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--config", default="/mnt/DATA2/yyg/starVLA/examples/realRobots/H1_2_car_real/train_files/"
                "starvla_qwengr00t_carfew_v1_2b_front_abs.yaml")
ap.add_argument("--dataset", default="car-few-tasks-v1")
ap.add_argument("--robot", default="h1_2_car_left_front")
ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
ap.add_argument("--n", type=int, default=1800)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--static_history", action="store_true",
                help="multi-frame configs: replace every history frame by the current one (memory ablation)")
ap.add_argument("--curr_only", action="store_true",
                help="[prev, curr]-per-camera configs (ProgressShift): feed only the current frames -> no history, "
                     "zero progress shift = the frozen base model with the same noise (paired ablation)")
ap.add_argument("--sorted", action="store_true",
                help="visit the sampled val steps in (episode, frame) order: every video file is decoded forward only "
                     "(no reused-CUDA-decoder backward seeks)")
ap.add_argument("--video_backend", default=None,
                help="override datasets.vla_data.video_backend (e.g. torchcodec = CPU; avoids the reused-CUDA-decoder "
                     "backward-seek bug when a pass revisits the same files)")
args = ap.parse_args()

HEAD, LARM, RARM = np.r_[0:2], np.r_[2:9], np.r_[22:29]
LHAND, RHAND = np.r_[9:22], np.r_[29:42]
GROUPS = {"head": HEAD, "arm": np.r_[LARM, RARM], "l_arm": LARM, "r_arm": RARM, "hand": np.r_[LHAND, RHAND]}
ANGULAR = {"head", "arm", "l_arm", "r_arm"}
cfg = OmegaConf.load(args.config)
HORIZON = int(cfg.framework.action_model.action_horizon)
STEPS = sorted({s for s in (0, 9, 19, 29, HORIZON - 1) if s < HORIZON})
assert str(cfg.datasets.vla_data.action_mode) == "abs", "this script un-normalizes abs actions only"
cfg.framework.qwenvl.autocast_dtype = args.dtype
if args.video_backend:
    cfg.datasets.vla_data.video_backend = args.video_backend
root = Path(cfg.datasets.vla_data.data_root_dir)

# Train split only for its metadata (statistics); val with train stats.
train_cfg = OmegaConf.to_container(cfg.datasets.vla_data)
val_cfg = dict(train_cfg)
val_cfg.pop("exclude_episodes", None)  # those are train episode ids
val_cfg.pop("color_jitter", None)  # train-only augmentation
_corrupt = root / args.dataset / "val/meta/corrupted_episodes.json"  # written by make_episode_videos.py
if _corrupt.exists():
    val_cfg["exclude_episodes"] = json.load(open(_corrupt))
train_ds = make_LeRobotSingleDataset(root, f"{args.dataset}/train", args.robot, data_cfg=OmegaConf.create(train_cfg))
val_ds = make_LeRobotSingleDataset(root, f"{args.dataset}/val", args.robot, data_cfg=OmegaConf.create(val_cfg))
val_ds.set_transforms_metadata(train_ds.metadata)

stats = json.load(open(root / args.dataset / "train/meta/stats_gr00t.abs.json"))["statistics"]["action"]
q01, q99 = np.array(stats["q01"], np.float64), np.array(stats["q99"], np.float64)
# Dims where q01 == q99 are passed through un-normalized by the transform; keep them as-is.
const = q99 == q01


def unnorm(x):
    x = np.asarray(x, np.float64)
    y = (x + 1.0) / 2.0 * (q99 - q01) + q01
    return np.where(const, x, y)


# hold_state baseline: un-normalize the model's state input key by key (train stats, q99 like the transform) and put
# each state key into the action dims of the action key with the same name.
_mod = json.load(open(root / args.dataset / "train/meta/modality.json"))
_st_stats = train_ds.metadata.statistics.state
STATE_TO_ACTION, _off = [], 0
for _k in val_ds.modality_keys["state"]:
    _k = _k.split(".", 1)[1]
    _lo, _hi = np.asarray(_st_stats[_k].q01, np.float64), np.asarray(_st_stats[_k].q99, np.float64)
    if _k in _mod["action"]:
        _a = _mod["action"][_k]
        STATE_TO_ACTION.append((_off, _off + len(_lo), np.r_[_a["start"]:_a["end"]], _lo, _hi))
    _off += len(_lo)
HOLD_STATE_DIMS = np.concatenate([d for _, _, d, _, _ in STATE_TO_ACTION]) if STATE_TO_ACTION else np.zeros(0, int)


def hold_state(state):
    """state [1, D] (normalized model input) -> [42] action-layout vector (nan where the state has no such key)."""
    x, out = np.asarray(state, np.float64)[-1], np.full(len(q01), np.nan)
    for s0, s1, dims, lo, hi in STATE_TO_ACTION:
        v = x[s0:s1]
        out[dims] = np.where(hi == lo, v, (v + 1.0) / 2.0 * (hi - lo) + lo)
    return out


t0 = time.time()
model = build_framework(cfg)
if args.ckpt != "none":  # "none": untrained head, pipeline smoke test only
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location="cpu", mmap=True), strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert not [k for k in missing if "qwen_vl_interface" in k or "action_model" in k], f"missing: {missing[:5]}"
model.qwen_vl_interface.to(getattr(torch, args.dtype))
model = model.cuda().eval()
print(f"model ready in {time.time() - t0:.0f}s, VLM dtype {next(model.qwen_vl_interface.parameters()).dtype}, "
      f"autocast {model.qwen_vl_interface.autocast_dtype}, val steps {len(val_ds)}", flush=True)

rng = np.random.default_rng(args.seed)
idx = rng.choice(len(val_ds), size=min(args.n, len(val_ds)), replace=False)
if args.sorted:
    idx = np.sort(idx)  # all_steps is (episode, frame)-ordered

chunk = defaultdict(list)              # (task, group) -> per-sample chunk-mean error
base = defaultdict(list)               # (task, group, baseline) -> per-sample chunk-mean error of a no-model baseline
per_step = defaultdict(list)           # (group, step) -> per-sample error at that step
prog_err = defaultdict(list)           # task -> |p_hat - t/T| (frameworks that predict episode progress)
t0 = time.time()
with torch.no_grad():
    for b0 in range(0, len(idx), args.batch):
        exs = [val_ds[int(i)] for i in idx[b0:b0 + args.batch]]
        if args.static_history:
            exs = [{**e, "image": [e["image"][-1]] * len(e["image"])} for e in exs]
        if args.curr_only:
            exs = [{**e, "image": e["image"][1::2]} for e in exs]
        torch.manual_seed(args.seed * 100003 + b0)
        out = model.predict_action(exs)
        pred = unnorm(out["normalized_actions"])
        if "progress" in out and "progress" in exs[0]:  # ProgressShift: predicted vs true t / T_episode
            for e, p in zip(exs, out["progress"]):
                prog_err[e["lang"]].append(abs(float(p) - float(e["progress"])))
                prog_err["ALL"].append(abs(float(p) - float(e["progress"])))
        gt = unnorm(np.stack([e["action"] for e in exs]))
        d = np.abs(pred - gt)
        for j, e in enumerate(exs):
            held = {"hold_cmd": gt[j, :1], "hold_state": hold_state(e["state"])[None]}
            for g, cols in GROUPS.items():
                v = d[j][:, cols].mean()
                chunk[(e["lang"], g)].append(v)
                chunk[("ALL", g)].append(v)
                for s in STEPS:
                    per_step[(g, s)].append(d[j, s, cols].mean())
                for bn, h in held.items():
                    if np.isnan(h[0, cols]).any():
                        continue
                    bv = np.abs(gt[j][:, cols] - h[:, cols]).mean()
                    base[(e["lang"], g, bn)].append(bv)
                    base[("ALL", g, bn)].append(bv)
        if b0 // args.batch % 20 == 0:
            print(f"  {b0 + len(exs)}/{len(idx)}  {time.time() - t0:.0f}s", flush=True)


def fmt(g, v):
    return float(np.degrees(v)) if g in ANGULAR else float(v)


tasks = sorted({t for t, _ in chunk if t != "ALL"}) + ["ALL"]
res = {
    "ckpt": args.ckpt, "dtype": args.dtype, "n": len(idx), "seed": args.seed, "robot": args.robot,
    "static_history": args.static_history,
    "units": {"head/arm/l_arm/r_arm": "deg", "hand": "raw"},
    "per_task": {t: {"n": len(chunk[(t, "arm")]), **{g: fmt(g, np.mean(chunk[(t, g)])) for g in GROUPS},
                     **{f"{g}_{bn}": fmt(g, np.mean(base[(t, g, bn)])) for g in GROUPS for bn in ("hold_cmd", "hold_state")
                        if base[(t, g, bn)]}}
                 for t in tasks},
    "per_step": {g: {str(s): fmt(g, np.mean(per_step[(g, s)])) for s in STEPS} for g in GROUPS},
    "arm_p90_deg": float(np.degrees(np.quantile(chunk[("ALL", "arm")], 0.9))),
    "curr_only": args.curr_only,
    "sorted": args.sorted,
    "horizon": HORIZON,
    "video_backend": str(cfg.datasets.vla_data.video_backend),
}
if prog_err:
    res["progress_mae"] = {t: float(np.mean(v)) for t, v in prog_err.items()}
json.dump(res, open(args.out, "w"), indent=2)

print(f"\n{args.ckpt}  dtype={args.dtype}  n={len(idx)}  ({time.time() - t0:.0f}s)")
print(f"{'task':60s} {'n':>4s} {'head°':>6s} {'arm°':>6s} {'Larm°':>6s} {'Rarm°':>6s} {'hand':>6s}")
for t in tasks:
    r = res["per_task"][t]
    print(f"{t[:60]:60s} {r['n']:4d} {r['head']:6.2f} {r['arm']:6.2f} {r['l_arm']:6.2f} {r['r_arm']:6.2f} {r['hand']:6.1f}")
if prog_err:
    print("progress MAE (|p_hat - t/T|): " + "  ".join(f"{t[:30]} {np.mean(v):.3f}" for t, v in prog_err.items()))
a = res["per_task"]["ALL"]
for bn in ("hold_cmd", "hold_state"):
    print(f"{'baseline ' + bn:60s} {a['n']:4d} " + " ".join(f"{a.get(f'{g}_{bn}', float('nan')):6.2f}" if g != "hand"
                                                         else f"{a.get(f'{g}_{bn}', float('nan')):6.1f}" for g in GROUPS))
print("per chunk step  " + "  ".join(f"t{s}: head {res['per_step']['head'][str(s)]:.2f}° arm {res['per_step']['arm'][str(s)]:.2f}° "
                                      f"hand {res['per_step']['hand'][str(s)]:.1f}" for s in STEPS))
