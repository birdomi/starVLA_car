"""System 2 past inpainting and the ground-truth plan, on the frames of aliasing_probe_dual.py's json.

At deployment the executed past is known; System 2 still generates it. Conditions (identical S2 / S1 noise):
  plain      : the usual System 2 sample -> System 1 (plan age 0)
  inpaint    : System 2 Euler sampling with the past keyframes (-2.0 .. -0.1 s, inside the episode) clamped at every
               step to the known past at that noise level, x_tau = (1 - tau) eps + tau x_known (same eps), and set to
               the known past after the last step; the future is generated -> System 1
  gt_plan    : no System 2: the demo's own 61 keyframes (past + current + future, normalized loader window) -> System 1
Offline the "known past" = the demo's actions a[t-60 .. t-1]. Scores as in s2_past_select_probe.py.

Usage: CUDA_VISIBLE_DEVICES=6 python s2_inpaint_probe.py --ckpt <stage2.pt> --config <stage2.yaml> \
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
ap.add_argument("--acc_per_ep", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

LARM, HEAD, ARMS = np.r_[2:9], 0, np.r_[2:9, 22:29]
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
cur_dense = -k_lo
tt = np.array([0.0, 29.0 / rate])
s2_steps = model.inference_steps["s2_steps"]

probe = json.load(open(args.probe))
data = pd.concat(pd.read_parquet(p, columns=["episode_index", "frame_index", "action"]) for p in sorted(split.glob("data/*/*.parquet")))
A_of = {e["ep"]: np.degrees(np.stack(data[data.episode_index == e["ep"]].sort_values("frame_index")["action"].to_numpy()))
        for e in probe["episodes"]}
kf_dense = np.rint(kf_times * rate).astype(int) - k_lo          # every keyframe's slot in the loader's dense window
sch = model.system2.scheduler
n_steps = int(s2_steps or sch.sample_steps)


def push(c_larm0, c_larm1, u):
    return float(np.dot(c_larm1 - c_larm0, u) / np.dot(u, u))


def sample_s2(ctx, cmask, noise, known=None, kmask=None):
    """FlowMatchingScheduler.sample with optional clamping of the kmask keyframes to `known` (normalized)."""
    x = noise.clone()
    for i in range(n_steps):
        tau = i / n_steps
        if known is not None:
            x = torch.where(kmask, (1 - tau) * noise + tau * known, x)
        t = torch.full((x.shape[0],), int(i / n_steps * sch.num_train_timesteps), device=x.device, dtype=torch.long)
        x = x + (1.0 / n_steps) * model.system2.dit(x, t, ctx, cmask).to(x.dtype)
    return x if known is None else torch.where(kmask, known, x)


checked = False


@torch.no_grad()
def run(ep, t, u):
    global checked
    ex = val_ds[step_index[(ep, t)]]
    A = A_of[ep]
    T = len(A)
    gt_n = np.asarray(ex["action"], np.float64)
    ok = np.abs(unnorm(gt_n[cur_dense])[ARMS] - A[t, ARMS]).max()
    assert ok < 0.05, f"loader action window misaligned at ep {ep} t {t}: {ok:.3f} deg"
    valid = np.asarray(ex.get("action_valid", np.ones(len(gt_n), bool)), bool)
    torch.manual_seed(args.seed)
    radio = model.encode_visual([ex])
    sc = model._shapes()
    dev = next(model.system2.parameters()).device
    noise = torch.randn(1, model.timeline.num, model.action_dim, device=dev)
    known = torch.tensor(gt_n[kf_dense], dtype=torch.float32, device=dev)[None]
    kmask_np = np.zeros((model.timeline.num, 1), bool)
    kmask_np[past_i, 0] = valid[kf_dense[past_i]]
    kmask = torch.tensor(kmask_np, device=dev)[None]
    with model._autocast():
        ctx, cmask = model._s2_context([ex], radio, model._state([ex], sc), sc, train_s2=False)
        plans = {"plain": sample_s2(ctx, cmask, noise), "inpaint": sample_s2(ctx, cmask, noise, known, kmask)}
        if not checked:
            ref = model.system2.sample(ctx, cmask, steps=s2_steps, noise=noise)
            assert torch.equal(ref.float(), plans["plain"].float()), "re-implemented sampler != System2.sample"
            checked = True
    plans = {k: v.float() for k, v in plans.items()}
    plans["gt_plan"] = known
    r = {"ep": ep, "t": t, "n_known_past": int(kmask_np.sum()),
         "gt": push(A[t, LARM], A[min(t + 29, T - 1), LARM], u),
         "gt_past": float(((A[max(t - 60, 0):t + 1, LARM] - A[t, LARM]) @ u).max() / np.dot(u, u))}
    for name, kfn in plans.items():
        kf = unnorm(kfn[0].cpu().numpy())
        p = np.stack([np.interp(tt, kf_times, kf[:, j]) for j in LARM], 1)
        k0 = np.array([np.interp(0.0, kf_times, kf[:, j]) for j in LARM])
        vp = past_i[kmask_np[past_i, 0]] if kmask_np.any() else past_i[-1:]
        torch.manual_seed(args.seed + 1)
        c = unnorm(model.act_from_plan([ex], kfn, 0.0, radio=radio)[0].cpu().numpy())
        n = min(30, T - t)
        r[name] = {"s1": push(c[0, LARM], c[29, LARM], u), "s2": push(p[0], p[1], u),
                   "past": float(((kf[vp][:, LARM] - k0) @ u).max() / np.dot(u, u)), "head": float(abs(c[29, HEAD] - c[0, HEAD])),
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
    print(f"[{i + 1}/{len(jobs)}] {kind:14s} ep {ep:2d} t {t:4d} gt {r['gt']:5.2f} gt_past {r['gt_past']:5.2f} | s1 plain "
          f"{r['plain']['s1']:5.2f} inpaint {r['inpaint']['s1']:5.2f} gt_plan {r['gt_plan']['s1']:5.2f}", flush=True)
json.dump({"rows": rows, "ckpt": args.ckpt, "probe": args.probe}, open(args.out, "w"), indent=1)


def stat(sub, pick, key, thr=None):
    v = np.array([r[pick][key] for r in sub])
    return f"{np.mean(v > thr):4.0%}" if thr is not None else f"{v.mean():6.3f}"


print("\n=== push > 0.1 = pushes within 1 s; MAE of the System 1 chunk vs the demo (deg)")
for kind in ("start", "post_push", "natural_before", "natural_after", "acc"):
    sub = [r for r in rows if r["kind"] == kind]
    if not sub:
        continue
    print(f"{kind:14s} n={len(sub):3d} gt push {np.mean([r['gt'] > 0.1 for r in sub]):4.0%}  (gt past push {np.mean([r['gt_past'] > 0.1 for r in sub]):4.0%})")
    for pick in ("plain", "inpaint", "gt_plan"):
        print(f"   {pick:8s} s1 push {stat(sub, pick, 's1', 0.1)}  s2 push {stat(sub, pick, 's2', 0.1)}  past {stat(sub, pick, 'past', 0.1)}"
              f"  arm MAE {stat(sub, pick, 'arm_mae')}  head MAE {stat(sub, pick, 'head_mae')}")
print("\nep 47 re-push frames:")
for r in [r for r in rows if r["ep"] == 47 and r["kind"] in ("natural_before", "post_push")]:
    print(f"   t {r['t']} gt_past {r['gt_past']:.2f} | " + " ".join(
        f"{p} s1 {r[p]['s1']:5.2f} past {r[p]['past']:5.2f} head {r[p]['head']:4.1f}" for p in ("plain", "inpaint", "gt_plan")))
