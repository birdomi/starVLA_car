"""Behavioural analysis of an H1-2 car checkpoint on held-out val steps (abs actions only).

Complements eval_val_chunks.py (plain per-task error) with four questions:
  1. Hold baselines  - is the model better than "stay where you are"? Two versions: chunk = current
                       measured state, and chunk = previous commanded action a[t-1] (no tracking offset).
  2. Prompt swap     - every sample is predicted under all 6 task prompts with identical noise.
                       How far does the chunk move when the prompt changes, and which prompt's
                       chunk is closest to GT (does the model follow language or the image)?
  3. Noise spread    - extra flow-matching noise seeds under the true prompt: spread vs. error.
  4. Where the error is - by GT motion within the chunk and by episode phase (t / T).

Writes <out>.npz (raw per-sample arrays, physical units) and <out>.json (summary).
Units: head / arms in rad inside the npz, deg in the json; hands in raw Inspire units.

Usage:
  CUDA_VISIBLE_DEVICES=4 python analyze_val_behavior.py --ckpt <steps_N_pytorch_model.pt> --out <prefix>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/mnt/DATA2/yyg/starVLA")
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True, help="output prefix (.npz/.json appended)")
ap.add_argument("--config", default="/mnt/DATA2/yyg/starVLA/examples/realRobots/H1_2_car_real/train_files/"
                "starvla_qwengr00t_carfew_v1_2b_front_abs.yaml")
ap.add_argument("--dataset", default="car-few-tasks-v1")
ap.add_argument("--robot", default="h1_2_car_left_front")
ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
ap.add_argument("--n", type=int, default=900)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--extra_noise", type=int, default=3, help="extra noise seeds under the true prompt")
args = ap.parse_args()

PROMPTS = [
    "Activate the left turn signal and inspect its operation.",
    "Activate the right turn signal and inspect its operation.",
    "Turn on the hazard lights using the center hazard light button and inspect their operation.",
    "Turn off the hazard lights using the center hazard light button and inspect their operation.",
    "Operate the windshield wiper at speed setting 1 and inspect its operation.",
    "Operate the windshield wiper at speed setting 2 and inspect its operation.",
]
SHORT = ["left_turn", "right_turn", "hazard_on", "hazard_off", "wiper_1", "wiper_2"]
HEAD, LARM, RARM = np.r_[0:2], np.r_[2:9], np.r_[22:29]
LHAND, RHAND = np.r_[9:22], np.r_[29:42]
ARM = np.r_[LARM, RARM]
HAND = np.r_[LHAND, RHAND]
GROUPS = {"head": HEAD, "arm": ARM, "l_arm": LARM, "r_arm": RARM, "hand": HAND, "l_hand": LHAND, "r_hand": RHAND}
ANGULAR = {"head", "arm", "l_arm", "r_arm"}
# raw observation.state (65) -> action layout (42): head_cmd, L arm, L hand, R arm, R hand
STATE_TO_ACTION = np.r_[63:65, 15:22, 37:50, 22:29, 50:63]

cfg = OmegaConf.load(args.config)
assert str(cfg.datasets.vla_data.action_mode) == "abs", "abs actions only"
cfg.framework.qwenvl.autocast_dtype = args.dtype
root = Path(cfg.datasets.vla_data.data_root_dir)

train_cfg = OmegaConf.to_container(cfg.datasets.vla_data)
val_cfg = dict(train_cfg)
val_cfg.pop("exclude_episodes", None)
train_ds = make_LeRobotSingleDataset(root, f"{args.dataset}/train", args.robot, data_cfg=OmegaConf.create(train_cfg))
val_ds = make_LeRobotSingleDataset(root, f"{args.dataset}/val", args.robot, data_cfg=OmegaConf.create(val_cfg))
val_ds.set_transforms_metadata(train_ds.metadata)

stats = json.load(open(root / args.dataset / "train/meta/stats_gr00t.abs.json"))["statistics"]["action"]
q01, q99 = np.array(stats["q01"], np.float64), np.array(stats["q99"], np.float64)
const = q99 == q01


def unnorm(x):
    x = np.asarray(x, np.float64)
    return np.where(const, x, (x + 1.0) / 2.0 * (q99 - q01) + q01)


model = build_framework(cfg)
missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location="cpu", mmap=True), strict=False)
assert not unexpected, unexpected[:5]
assert not [k for k in missing if "qwen_vl_interface" in k or "action_model" in k], missing[:5]
model.qwen_vl_interface.to(getattr(torch, args.dtype))
model = model.cuda().eval()

rng = np.random.default_rng(args.seed)
idx = np.sort(rng.choice(len(val_ds), size=min(args.n, len(val_ds)), replace=False))
traj_len = dict(zip(val_ds.trajectory_ids.tolist(), val_ds.trajectory_lengths.tolist()))

N, K, E = len(idx), len(PROMPTS), args.extra_noise
gt = np.zeros((N, 30, 42))
hold = np.zeros((N, 42))
prev = np.zeros((N, 42))               # a[t-1] (a[0] at the first frame)
pred_p = np.zeros((N, K, 30, 42))       # every prompt, same noise
pred_e = np.zeros((N, E, 30, 42))       # true prompt, extra noise seeds
task = np.zeros(N, int)
ep = np.zeros(N, int)
frame = np.zeros(N, int)
eplen = np.zeros(N, int)

t0 = time.time()
with torch.no_grad():
    for b0 in range(0, N, args.batch):
        bi = idx[b0:b0 + args.batch]
        exs = [val_ds[int(i)] for i in bi]
        for j, (i, e) in enumerate(zip(bi, exs)):
            n = b0 + j
            tid, base = val_ds.all_steps[int(i)]
            task[n] = PROMPTS.index(e["lang"])
            ep[n], frame[n], eplen[n] = tid, base, traj_len[tid]
            gt[n] = unnorm(e["action"])
            df = val_ds.get_trajectory_data(tid)
            hold[n] = np.asarray(df["observation.state"].iloc[base], np.float64)[STATE_TO_ACTION]
            prev[n] = np.asarray(df["action"].iloc[max(base - 1, 0)], np.float64)
        sl = slice(b0, b0 + len(exs))
        for k, p in enumerate(PROMPTS):
            torch.manual_seed(args.seed * 100003 + b0)
            pred_p[sl, k] = unnorm(model.predict_action([{**e, "lang": p} for e in exs])["normalized_actions"])
        for s in range(E):
            torch.manual_seed(args.seed * 100003 + b0 + 7919 * (s + 1))
            pred_e[sl, s] = unnorm(model.predict_action(exs)["normalized_actions"])
        if b0 // args.batch % 10 == 0:
            print(f"  {b0 + len(exs)}/{N}  {time.time() - t0:.0f}s", flush=True)

pred = pred_p[np.arange(N), task]       # true prompt
np.savez_compressed(args.out + ".npz", gt=gt, hold=hold, prev=prev, pred_p=pred_p, pred_e=pred_e, task=task, ep=ep,
                    frame=frame, eplen=eplen, idx=idx, prompts=np.array(SHORT))


def err(a, b, cols):
    """mean |a-b| over chunk steps and cols -> per sample."""
    return np.abs(a - b)[..., cols].mean(axis=(-1, -2))


def fmt(g, v):
    return float(np.degrees(v)) if g in ANGULAR else float(v)


res = {"ckpt": args.ckpt, "dtype": args.dtype, "n": N, "seed": args.seed, "units": "deg (head/arm), raw (hand)"}

# 1. model vs hold baseline
tasks = [(SHORT[k], task == k) for k in range(K)] + [("ALL", np.ones(N, bool))]
res["vs_hold"] = {}
for name, m in tasks:
    r = {"n": int(m.sum())}
    for g, cols in GROUPS.items():
        r[g] = {"model": fmt(g, err(pred[m], gt[m], cols).mean()),
                "hold_state": fmt(g, err(hold[m][:, None], gt[m], cols).mean()),
                "hold_prev_action": fmt(g, err(prev[m][:, None], gt[m], cols).mean())}
    res["vs_hold"][name] = r

# 2. prompt swap: how much does the chunk move, and which prompt best explains GT
d_swap = {g: np.zeros((K, K)) for g in ("head", "l_arm", "r_arm", "hand")}
best_prompt = np.zeros((K, K), int)
for k in range(K):
    m = task == k
    for g in d_swap:
        for k2 in range(K):
            d_swap[g][k, k2] = fmt(g, err(pred_p[m, k2], pred[m], GROUPS[g]).mean())
    e_all = np.stack([err(pred_p[m, k2], gt[m], np.r_[ARM, HEAD]) for k2 in range(K)], 1)
    best_prompt[k] = np.bincount(e_all.argmin(1), minlength=K)
res["prompt_swap"] = {
    "rows_true_cols_given": SHORT,
    "mean_abs_change_vs_true_prompt": {g: v.round(3).tolist() for g, v in d_swap.items()},
    "argmin_gt_error_counts(arm+head)": best_prompt.tolist(),
}

# 3. noise spread under the true prompt: mean abs deviation of extra seeds from the main sample
spread = {g: fmt(g, np.mean([err(pred_e[:, s], pred, cols).mean() for s in range(E)])) for g, cols in GROUPS.items()}
mean_pred = np.concatenate([pred[:, None], pred_e], 1).mean(1)
res["noise"] = {"seed_to_seed_mean_abs": spread,
                "error_single_sample": {g: fmt(g, err(pred, gt, c).mean()) for g, c in GROUPS.items()},
                "error_mean_of_seeds": {g: fmt(g, err(mean_pred, gt, c).mean()) for g, c in GROUPS.items()}}

# 4. error by GT motion within the chunk (arm, deg) and by episode phase
motion = np.degrees(np.abs(gt[:, :, ARM] - gt[:, :1, ARM]).max(axis=(1, 2)))
e_arm, h_arm, p_arm = err(pred, gt, ARM), err(hold[:, None], gt, ARM), err(prev[:, None], gt, ARM)
bins = [0, 1, 3, 10, 30, 1e9]
res["by_motion"] = []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (motion >= lo) & (motion < hi)
    if m.any():
        res["by_motion"].append({"gt_arm_motion_deg": f"[{lo},{hi})", "n": int(m.sum()),
                                 "model_arm": fmt("arm", e_arm[m].mean()), "hold_state_arm": fmt("arm", h_arm[m].mean()),
                                 "hold_prev_arm": fmt("arm", p_arm[m].mean())})
phase = frame / np.maximum(eplen - 1, 1)
res["by_phase"] = []
for lo in np.arange(0, 1, 0.2):
    m = (phase >= lo) & (phase < lo + 0.2 + 1e-9)
    res["by_phase"].append({"phase": f"[{lo:.1f},{lo + 0.2:.1f})", "n": int(m.sum()),
                            "model_arm": fmt("arm", e_arm[m].mean()), "hold_state_arm": fmt("arm", h_arm[m].mean()),
                            "hold_prev_arm": fmt("arm", p_arm[m].mean()), "model_head": fmt("head", err(pred[m], gt[m], HEAD).mean()),
                            "gt_arm_motion": float(motion[m].mean())})

json.dump(res, open(args.out + ".json", "w"), indent=2)
print(json.dumps(res, indent=1))
print(f"done in {time.time() - t0:.0f}s")
