"""Re-score a QwenGR00T_DualSystem checkpoint on the exact frames of an aliasing_probe_dual.py json, optionally read from
another export of the same recordings (paired comparison of models trained on different exports / splits).

Frames, push direction u, conditions and scores are those of aliasing_probe_dual.py: start f0, post_push t*, natural
frames with real / start_state / start_head / start_wrists / start_images. With --ep_map (pickle {(split, ep): ((src_split,
src_ep), err, offset, n, len_src, len_probe)}, e.g. the v2 -> v1 body-state match) every probe episode is read from
<dataset>/<src_split> episode src_ep; only exact copies (err 0, offset 0, same length) are kept and the arm actions are
asserted equal to the probe export's. Without --ep_map the probe's own episodes are read (re-run check: must reproduce
the probe json).

Usage: CUDA_VISIBLE_DEVICES=6 python aliasing_probe_replay.py --ckpt <pt> --config <yaml> --probe <probe json> \
           --probe_dataset car-turn-signal-v2-ep --dataset car-turn-signal-ep --robot h1_2_car_turn_dual_s2 \
           --ep_map match.pkl --out <json>
"""
import argparse
import json
import pickle
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
ap.add_argument("--probe", required=True, help="aliasing_probe_dual.py output")
ap.add_argument("--probe_dataset", default="car-turn-signal-v2-ep", help="export the probe was run on (its val split)")
ap.add_argument("--dataset", default="car-turn-signal-ep", help="export to read the frames from")
ap.add_argument("--robot", default="h1_2_car_turn_dual_s2")
ap.add_argument("--ep_map", default=None, help="pickle {('val', probe_ep): ((src_split, src_ep), err, offset, n, len_src, len_probe)}")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

LARM, HEAD, ARMS = np.r_[2:9], 0, np.r_[2:9, 22:29]
cfg = OmegaConf.load(args.config)
root = Path(cfg.datasets.vla_data.data_root_dir)
probe = json.load(open(args.probe))

# probe episode -> (source split, source episode)
if args.ep_map:
    raw = pickle.load(open(args.ep_map, "rb"))
    src_of, skipped = {}, []
    for e in probe["episodes"]:
        b = raw[("val", e["ep"])]
        if b is not None and b[1] == 0 and b[2] == 0 and b[4] == b[5]:
            src_of[e["ep"]] = b[0]
        else:
            skipped.append((e["ep"], None if b is None else b[0]))
    print(f"ep map: {len(src_of)} exact copies, skipped {skipped}", flush=True)
else:
    src_of = {e["ep"]: ("val", e["ep"]) for e in probe["episodes"]}

train_cfg = OmegaConf.to_container(cfg.datasets.vla_data)
src_cfg = dict(train_cfg)
src_cfg.pop("exclude_episodes", None)
src_cfg.pop("color_jitter", None)
src_cfg["video_backend"] = "torchcodec"
train_ds = make_LeRobotSingleDataset(root, f"{args.dataset}/train", args.robot, data_cfg=OmegaConf.create(train_cfg))
ds = {}
for split in sorted({s for s, _ in src_of.values()}):
    c = dict(src_cfg)
    if split != "train":
        _corrupt = root / args.dataset / split / "meta/corrupted_episodes.json"
        if _corrupt.exists():
            c["exclude_episodes"] = json.load(open(_corrupt))
    d = make_LeRobotSingleDataset(root, f"{args.dataset}/{split}", args.robot, data_cfg=OmegaConf.create(c))
    d.set_transforms_metadata(train_ds.metadata)
    ds[split] = (d, {(int(t), int(b)): i for i, (t, b) in enumerate(d._all_steps)})
missing_src = [ep for ep, (s, e) in src_of.items() if (e, 0) not in ds[s][1]]
for ep in missing_src:
    print(f"probe ep {ep}: source {src_of[ep]} not loadable (excluded / corrupted), skipped", flush=True)
    src_of.pop(ep)


def actions(dataset, split, eps):
    p = root / dataset / split
    data = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "action"]) for f in sorted(p.glob("data/*/*.parquet")))
    data = data[data.episode_index.isin(eps)]
    return {int(e): np.degrees(np.stack(g.sort_values("frame_index")["action"].to_numpy())) for e, g in data.groupby("episode_index")}


A_probe = actions(args.probe_dataset, "val", list(src_of))
A_src = {}
for split in ds:
    A_src.update({(split, e): a for e, a in actions(args.dataset, split, [e for s, e in src_of.values() if s == split]).items()})
for ep, k in src_of.items():
    a, b = A_probe[ep], A_src[k]
    assert a.shape == b.shape and np.abs(a[:, ARMS] - b[:, ARMS]).max() < 1e-3, f"probe ep {ep} != source {k} (arm actions)"

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
kf_times = model.timeline.times.cpu().numpy().astype(np.float64)
past = kf_times <= 1e-6
rate = float(cfg.framework.dual_system.action_rate_hz)
tt = np.array([0.0, 29.0 / rate])
vlm_mode = model.stale_input == "vlm"


def plan(ex, radio, t_img):
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
    p = np.stack([np.interp(tt, kf_times, kf[:, j]) for j in LARM], 1)
    k0 = np.stack([np.interp(0.0, kf_times, kf[:, j]) for j in LARM])
    uu = float(np.dot(u, u))
    return {"s1": float(np.dot(c[29, LARM] - c[0, LARM], u) / uu), "s2": float(np.dot(p[1] - p[0], u) / uu),
            "past": float(((kf[past][:, LARM] - k0) @ u).max() / uu), "head": float(abs(c[29, HEAD] - c[0, HEAD]))}


def example(ep, t):
    s, e = src_of[ep]
    d, idx = ds[s]
    return d[idx[(e, t)]]


eprows, rows = [], []
nat = {}
for r in probe["natural"]:
    nat.setdefault(r["ep"], []).append(r)
with torch.no_grad():
    for e in probe["episodes"]:
        ep = e["ep"]
        if ep not in src_of:
            continue
        q = A_probe[ep][:, LARM]
        u = q[e["push_peak"]] - q[e["f0"]]
        f0, ts = e["f0"], e["t_star"]
        ex0 = example(ep, f0)
        er = {**{k: v for k, v in e.items() if k not in ("start", "post_push")}, "src": list(src_of[ep]),
              "start": scores(ex0, u, f0)}
        if "post_push" in e:
            er["post_push"] = scores(example(ep, ts), u, ts)
        eprows.append(er)
        print(ep, src_of[ep], "start", {k: round(v, 2) for k, v in er["start"].items()},
              "post", {k: round(v, 2) for k, v in er.get("post_push", {}).items()}, flush=True)
        im0 = list(ex0["image"])
        nt = len(im0) // model.num_views
        for r0 in nat.get(ep, []):
            t = r0["t"]
            ext = example(ep, t)
            imt = list(ext["image"])
            r = {**{k: v for k, v in r0.items() if k not in ("real", "start_state", "start_head", "start_wrists", "start_images")},
                 "real": scores(ext, u, t),
                 "start_state": scores({**ext, "state": ex0["state"]}, u, t),
                 "start_head": scores({**ext, "image": im0[:nt] + imt[nt:]}, u, f0),
                 "start_wrists": scores({**ext, "image": imt[:nt] + im0[nt:]}, u, f0),
                 "start_images": scores({**ext, "image": im0}, u, f0)}
            rows.append(r)
            print("  ", t, f"gt {r['gt']:.2f}", {k: round(r[k]["s1"], 2) for k in ("real", "start_state", "start_head", "start_wrists", "start_images")}, flush=True)

json.dump({"episodes": eprows, "natural": rows, "ckpt": args.ckpt, "config": args.config, "probe": args.probe,
           "dataset": args.dataset, "ep_map": args.ep_map}, open(args.out, "w"), indent=1)

# paired summary: this checkpoint vs the probe json's on the same frames
P = {e["ep"]: e for e in probe["episodes"]}
PN = {(r["ep"], r["t"]): r for r in probe["natural"]}
f = lambda v: f"{np.mean(np.asarray(v) > 0.1):4.0%}"
print(f"\n=== push > 0.1 within 1 s (s1); this ckpt vs probe ckpt on the same {len(eprows)} episodes")
print(f"start      gt {f([e['gt_start'] for e in eprows])}  this {f([e['start']['s1'] for e in eprows])}  "
      f"probe {f([P[e['ep']]['start']['s1'] for e in eprows])}")
pp = [e for e in eprows if "post_push" in e]
print(f"post_push  gt {f([e['gt_post_push'] for e in pp])}  this {f([e['post_push']['s1'] for e in pp])}  "
      f"probe {f([P[e['ep']]['post_push']['s1'] for e in pp])}")
for name, sub in (("natural before head turn", [r for r in rows if r["sec_before_head_turn"] > 0]),
                  ("natural after head turn", [r for r in rows if r["sec_before_head_turn"] <= 0])):
    print(f"{name}: n={len(sub)} eps={len({r['ep'] for r in sub})}  gt {f([r['gt'] for r in sub])}")
    for cond in ("real", "start_state", "start_head", "start_wrists", "start_images"):
        print(f"   {cond:13s} this {f([r[cond]['s1'] for r in sub])}  probe {f([PN[(r['ep'], r['t'])][cond]['s1'] for r in sub])}")
    print(f"   S2 past>0.1  this {f([r['real']['past'] for r in sub])}  probe {f([PN[(r['ep'], r['t'])]['real']['past'] for r in sub])}"
          f"  gt {f([r['gt_past'] for r in sub])}")
    for ep in sorted({r["ep"] for r in sub}):
        s = [r for r in sub if r["ep"] == ep]
        print(f"   ep {ep:2d} n={len(s)} gt {f([r['gt'] for r in s])} | this s1 {f([r['real']['s1'] for r in s])} past "
              f"{np.mean([r['real']['past'] for r in s]):5.2f} | probe s1 {f([PN[(r['ep'], r['t'])]['real']['s1'] for r in s])} past "
              f"{np.mean([PN[(r['ep'], r['t'])]['real']['past'] for r in s]):5.2f} | gt_past {np.mean([r['gt_past'] for r in s]):5.2f}")
