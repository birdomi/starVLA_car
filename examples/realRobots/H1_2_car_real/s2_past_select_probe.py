"""System 2 best-of-N by the known past: sample N plans (one Qwen / context pass, N noise draws), keep the plan whose
past keyframes (-2 .. <0 s) are closest to the actions actually executed, and give that plan to System 1.

At deployment the past actions are known (the commands already sent); System 2 still predicts them. Offline, the
executed past = the demo's actions a[t-60 .. t-1] (the loader's normalized dense window, padded slots masked).
Picks compared with identical System 1 noise:
  first    : sample 0 of the N (= the usual single-sample call)
  best_all : smallest past MSE over all 42 normalized action dims
  best_ah  : smallest past MSE over head + arms (dims 0:2, 2:9, 22:29), hands ignored
Frames: start f0 / head-turn onset t* / natural aliased frames of aliasing_probe_dual.py's json (same push score
u per episode), plus --acc_per_ep random frames per episode for accuracy (System 1 chunk MAE vs the demo, degrees).
Per frame it also stores how many of the N plans push in the next second / contain the push in their past.

Usage: CUDA_VISIBLE_DEVICES=6 python s2_past_select_probe.py --ckpt <stage2.pt> --config <stage2.yaml> \
           --probe <aliasing_probe_dual json> --out <json>
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
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--probe", required=True, help="aliasing_probe_dual.py output (episodes + natural frames)")
ap.add_argument("--out", required=True)
ap.add_argument("--dataset", default="car-turn-signal-ep")
ap.add_argument("--robot", default="h1_2_car_turn_dual_s2")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--acc_per_ep", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

LARM, HEAD, ARMS = np.r_[2:9], 0, np.r_[2:9, 22:29]
AH = np.r_[0:9, 22:29]
cfg = OmegaConf.load(args.config)
root = Path(cfg.datasets.vla_data.data_root_dir)
split = root / args.dataset / "val"
train_cfg = OmegaConf.to_container(cfg.datasets.vla_data)
val_cfg = dict(train_cfg)
val_cfg.pop("exclude_episodes", None)
val_cfg.pop("color_jitter", None)
val_cfg["video_backend"] = "torchcodec"
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
    return np.degrees(np.where(const, x, (x + 1.0) / 2.0 * (q99 - q01) + q01))


model = build_framework(cfg)
missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location="cpu", mmap=True), strict=False)
assert not unexpected and not missing, (missing[:3], unexpected[:3])
model = model.cuda().eval()
rate = float(cfg.framework.dual_system.action_rate_hz)
kf_times = model.timeline.times.cpu().numpy().astype(np.float64)
past_i = np.where(kf_times < -1e-6)[0]                                  # strictly past keyframes (-2.0 .. -0.1 s)
k_lo = int(np.floor(float(cfg.framework.dual_system.keyframes.t_min) * rate + 1e-6))   # dense window starts at a[t + k_lo]
past_dense = np.rint(kf_times[past_i] * rate).astype(int) - k_lo
cur_dense = -k_lo
tt = np.array([0.0, 29.0 / rate])
s2_steps = model.inference_steps["s2_steps"]

probe = json.load(open(args.probe))
data = pd.concat(pd.read_parquet(p, columns=["episode_index", "frame_index", "action"]) for p in sorted(split.glob("data/*/*.parquet")))
A_of = {e["ep"]: np.degrees(np.stack(data[data.episode_index == e["ep"]].sort_values("frame_index")["action"].to_numpy()))
        for e in probe["episodes"]}


def push(c_larm0, c_larm1, u):
    return float(np.dot(c_larm1 - c_larm0, u) / np.dot(u, u))


@torch.no_grad()
def run(ep, t, u):
    ex = val_ds[step_index[(ep, t)]]
    A = A_of[ep]
    gt_n = np.asarray(ex["action"], np.float64)              # normalized dense window a[t + k_lo ..]
    ok = np.abs(unnorm(gt_n[cur_dense])[ARMS] - A[t, ARMS]).max()
    assert ok < 0.05, f"loader action window misaligned at ep {ep} t {t}: {ok:.3f} deg"
    valid = np.asarray(ex.get("action_valid", np.ones(len(gt_n), bool)), bool)[past_dense]
    torch.manual_seed(args.seed)
    radio = model.encode_visual([ex])
    sc = model._shapes()
    with model._autocast():
        ctx, cmask = model._s2_context([ex], radio, model._state([ex], sc), sc, train_s2=False)
        rep = lambda x: None if x is None else x.repeat(args.n, *([1] * (x.dim() - 1)))
        kf_n = model.system2.sample(rep(ctx), rep(cmask), steps=s2_steps).float()   # [N, 61, D]
    kfn = kf_n.cpu().numpy().astype(np.float64)
    err = (kfn[:, past_i] - gt_n[past_dense][None]) ** 2                            # [N, P, D]
    w = valid[None, :, None].astype(np.float64)
    if w.sum() == 0:
        d_all = d_ah = np.zeros(args.n)
    else:
        d_all = (err * w).sum((1, 2)) / (w.sum() * err.shape[2])
        d_ah = (err[..., AH] * w).sum((1, 2)) / (w.sum() * len(AH))
    kf = unnorm(kfn)                                                                 # [N, 61, D] deg
    plan01 = np.stack([[np.interp(tt, kf_times, kf[s, :, j]) for j in LARM] for s in range(args.n)])  # [N, 7, 2]
    plan_push = np.array([push(p[:, 0], p[:, 1], u) for p in plan01])
    k0 = np.stack([[np.interp(0.0, kf_times, kf[s, :, j]) for j in LARM] for s in range(args.n)])     # [N, 7]
    vp = past_i[valid] if valid.any() else past_i[-1:]
    plan_past = np.array([((kf[s][vp][:, LARM] - k0[s]) @ u).max() / np.dot(u, u) for s in range(args.n)])
    T = len(A)
    r = {"ep": ep, "t": t, "n_valid_past": int(valid.sum()),
         "frac_plans_push": float(np.mean(plan_push > 0.1)), "frac_plans_past_push": float(np.mean(plan_past > 0.1)),
         "gt": push(A[t, LARM], A[min(t + 29, T - 1), LARM], u),
         "gt_past": float(((A[max(t - 60, 0):t + 1, LARM] - A[t, LARM]) @ u).max() / np.dot(u, u))}
    for name, k in (("first", 0), ("best_all", int(np.argmin(d_all))), ("best_ah", int(np.argmin(d_ah)))):
        torch.manual_seed(args.seed + 1)
        c = unnorm(model.act_from_plan([ex], kf_n[k:k + 1], 0.0, radio=radio)[0].cpu().numpy())
        n = min(30, T - t)
        r[name] = {"k": k, "past_mse": float(d_all[k]), "past_mse_ah": float(d_ah[k]), "s1": push(c[0, LARM], c[29, LARM], u),
                   "s2": float(plan_push[k]), "past": float(plan_past[k]), "head": float(abs(c[29, HEAD] - c[0, HEAD])),
                   "arm_mae": float(np.abs(c[:n, ARMS] - A[t:t + n, ARMS]).mean()),
                   "head_mae": float(np.abs(c[:n, :2] - A[t:t + n, :2]).mean())}
    return r


rng = np.random.default_rng(args.seed)
eps = {e["ep"]: e for e in probe["episodes"]}
u_of = {}
for ep, e in eps.items():
    q = A_of[ep][:, LARM]
    u_of[ep] = q[e["push_peak"]] - q[e["f0"]]
jobs = []
for ep, e in eps.items():
    jobs.append(("start", ep, e["f0"]))
    if "post_push" in e:
        jobs.append(("post_push", ep, e["t_star"]))
for r in probe["natural"]:
    jobs.append(("natural_before" if r["sec_before_head_turn"] > 0 else "natural_after", r["ep"], r["t"]))
for ep in eps:
    T = len(A_of[ep])
    jobs += [("acc", ep, int(t)) for t in np.sort(rng.choice(np.arange(0, T - 30), size=args.acc_per_ep, replace=False))]

rows = []
for i, (kind, ep, t) in enumerate(jobs):
    r = {"kind": kind, **run(ep, int(t), u_of[ep])}
    rows.append(r)
    print(f"[{i + 1}/{len(jobs)}] {kind:14s} ep {ep:2d} t {t:4d} gt {r['gt']:5.2f} | plans push {r['frac_plans_push']:4.0%} "
          f"past-push {r['frac_plans_past_push']:4.0%} | s1 first {r['first']['s1']:5.2f} best_all {r['best_all']['s1']:5.2f} "
          f"best_ah {r['best_ah']['s1']:5.2f}", flush=True)
json.dump({"rows": rows, "n": args.n, "ckpt": args.ckpt, "probe": args.probe}, open(args.out, "w"), indent=1)


def stat(sub, pick, key, thr=None):
    v = np.array([r[pick][key] for r in sub])
    return f"{np.mean(v > thr):4.0%}" if thr is not None else f"{v.mean():6.3f}"


print(f"\n=== N = {args.n} System 2 samples; push > 0.1 = pushes within 1 s")
for kind in ("start", "post_push", "natural_before", "natural_after", "acc"):
    sub = [r for r in rows if r["kind"] == kind]
    if not sub:
        continue
    gt = np.mean([r["gt"] > 0.1 for r in sub])
    print(f"{kind:14s} n={len(sub):3d} gt push {gt:4.0%} | plans push {np.mean([r['frac_plans_push'] for r in sub]):4.0%} "
          f"past-push {np.mean([r['frac_plans_past_push'] for r in sub]):4.0%} (gt past {np.mean([r['gt_past'] > 0.1 for r in sub]):4.0%})")
    for pick in ("first", "best_all", "best_ah"):
        print(f"   {pick:9s} s1 push {stat(sub, pick, 's1', 0.1)}  s2 push {stat(sub, pick, 's2', 0.1)}  past {stat(sub, pick, 'past', 0.1)}"
              f"  arm MAE {stat(sub, pick, 'arm_mae')}  head MAE {stat(sub, pick, 'head_mae')}  past MSE {stat(sub, pick, 'past_mse')}")
ep47 = [r for r in rows if r["ep"] == 47 and r["kind"] in ("natural_before", "post_push")]
print("\nep 47 re-push frames:")
for r in ep47:
    print(f"   t {r['t']} plans push {r['frac_plans_push']:4.0%} past-push {r['frac_plans_past_push']:4.0%} gt_past {r['gt_past']:.2f} | "
          + " ".join(f"{p} s1 {r[p]['s1']:5.2f} past {r[p]['past']:5.2f} head {r[p]['head']:4.1f}" for p in ("first", "best_all", "best_ah")))
