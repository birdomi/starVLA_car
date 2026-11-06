"""Streamed val evaluation on car-v3 (car_v3_val_3cam_headstate: car-1015-v3-ep/val, car-0901-ov/val, car-3rd-ov/val with
car_v3_episode_lists_val.json, 296 episodes, 21 tasks), the eval_val_stream.py protocol on the car-v3 mixture:
every val episode walked from frame 0, one query per --stride frames on the RECORDED observation; a
QwenGR00T_EpisodeMemory (episode_memory.stream) model carries its memory across the queries (--memory_reset: the
initial memory at every query instead, i.e. the memory sees the current frame only), any other framework is queried
statelessly; the same frames and noise seeds for every model (paired).

Normalization is the training run's: the train mixture of the run's config is built and its merged metadata is set on
the val datasets (checked against the run's dataset_statistics.json); errors are un-normalized with those statistics.
Per query: chunk-mean |pred - GT| for head / arms (deg) and hands (raw), per chunk step, the hold_cmd (GT a[t] held) and
hold_state (measured state held) baselines, and |p_hat - t / (T - 1)| for memory models. --part i/n runs every n-th
episode batch (same batches / seeds as the full run); --merge_parts n merges accum_part*.pkl into result.json.
Rollout (open loop): the first `stride` steps of every chunk stitched into a (T, 42) trajectory per episode, written to
<out>/rollout/ep<set>_<id>/{pred.npy, gt_action.npy, meta.json} (seqcheck/event_check.py input) and scored in result.json
"rollout": frame-mean |pred - GT| next to HOLD (the previous GT command held over each window, event_check.py's reference).

Usage: CUDA_VISIBLE_DEVICES=0 python eval_val_car_v3.py --ckpt <run>/checkpoints/steps_50000_pytorch_model.pt \
           --out <dir> [--stride 30] [--memory_reset] [--part 0/4]
"""
import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset, make_LeRobotSingleDataset
from starVLA.model.framework.base_framework import build_framework

HERE = Path(__file__).resolve().parent
VAL = [("car-1015-v3-ep/val", "h1_2_car_3cam_headstate"),
       ("car-0901-ov/val", "h1_2_car_3cam_headstate_ch2clip"),
       ("car-3rd-ov/val", "h1_2_car_3cam_headstate_ch2clip")]
HEAD, LARM, RARM = np.r_[0:2], np.r_[2:9], np.r_[22:29]
LHAND, RHAND = np.r_[9:22], np.r_[29:42]
GROUPS = {"head": HEAD, "arm": np.r_[LARM, RARM], "l_arm": LARM, "r_arm": RARM, "hand": np.r_[LHAND, RHAND]}
ANGULAR = {"head", "arm", "l_arm", "r_arm"}
# model state (42: head, L arm, R arm, L hand, R hand) -> action dims of the same joints
STATE_TO_ACTION = [(0, 2, np.r_[0:2]), (2, 9, LARM), (9, 16, RARM), (16, 29, LHAND), (29, 42, RHAND)]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--config", default="", help="default: <run>/config.full.yaml next to the checkpoint")
ap.add_argument("--out", required=True)
ap.add_argument("--stride", type=int, default=30)
ap.add_argument("--memory_reset", action="store_true", help="memory models: the initial memory at every query")
ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
ap.add_argument("--batch", type=int, default=16, help="episodes walked in parallel")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--part", default="")
ap.add_argument("--merge_parts", type=int, default=0)
ap.add_argument("--lists", default=str(HERE / "car_v3_episode_lists_val.json"))
args = ap.parse_args()
out_dir = Path(args.out)
STEPS = [0, 9, 19, 29, 49]
ACC_KEYS = ("chunk", "base", "per_step", "prog", "roll")


def fmt(g, v):
    return float(np.degrees(v)) if g in ANGULAR else float(v)


def summarize(acc, info, path):
    chunk, base, per_step, prog = acc["chunk"], acc["base"], acc["per_step"], acc["prog"]
    keys = sorted({k for k, _ in chunk if not k.startswith("set:") and k != "ALL"})
    sets = sorted({k for k, _ in chunk if k.startswith("set:")})

    def row(k):
        r = {"n": len(chunk[(k, "arm")]), **{g: fmt(g, np.mean(chunk[(k, g)])) for g in GROUPS}}
        for g in GROUPS:
            for bn in ("hold_cmd", "hold_state"):
                if base[(k, g, bn)]:
                    r[f"{g}_{bn}"] = fmt(g, np.mean(base[(k, g, bn)]))
        if prog[k]:
            r["progress_mae"] = float(np.mean(prog[k]))
        return r

    res = {**info, "units": {"head/arm/l_arm/r_arm": "deg", "hand": "raw"},
           "ALL": row("ALL"), "per_set": {k[4:]: row(k) for k in sets}, "per_task": {k: row(k) for k in keys},
           "per_step": {g: {str(s): fmt(g, np.mean(per_step[(g, s)])) for s in STEPS} for g in GROUPS}}
    roll = acc.get("roll")
    if roll:
        def rrow(k):
            return {"n_episodes": len(roll[(k, "arm", "pred")]),
                    **{f"{g}{'' if w == 'pred' else '_hold'}": fmt(g, np.mean(roll[(k, g, w)])) for g in GROUPS for w in ("pred", "hold")}}
        res["rollout"] = {"ALL": rrow("ALL"), "per_set": {k[4:]: rrow(k) for k in sets}, "per_task": {k: rrow(k) for k in keys}}
    json.dump(res, open(path, "w"), indent=2, ensure_ascii=False)
    a = res["ALL"]
    print(f"\n{info['ckpt']}  memory_reset={info['memory_reset']}  {info['episodes']} episodes  {a['n']} queries")
    print(f"ALL  head {a['head']:.2f}  arm {a['arm']:.2f}  hand {a['hand']:.1f}  progress {a.get('progress_mae', float('nan')):.3f}"
          f"   | hold_cmd head {a.get('head_hold_cmd', float('nan')):.2f} arm {a.get('arm_hold_cmd', float('nan')):.2f}"
          f" hand {a.get('hand_hold_cmd', float('nan')):.1f}")
    if "rollout" in res:
        r = res["rollout"]["ALL"]
        print(f"rollout ({r['n_episodes']} episodes)  head {r['head']:.2f}  arm {r['arm']:.2f}  hand {r['hand']:.1f}"
              f"   | HOLD head {r['head_hold']:.2f} arm {r['arm_hold']:.2f} hand {r['hand_hold']:.1f}")
    for k, r in res["per_set"].items():
        print(f"  {k:22s} n {r['n']:5d}  head {r['head']:.2f}  arm {r['arm']:.2f}  hand {r['hand']:.1f}  progress {r.get('progress_mae', float('nan')):.3f}")
    return res


if args.merge_parts:
    parts = [pickle.load(open(out_dir / f"accum_part{i}.pkl", "rb")) for i in range(args.merge_parts)]
    acc = {k: defaultdict(list) for k in ACC_KEYS}
    for pt in parts:
        for k in acc:
            for kk, v in pt.get(k, {}).items():
                acc[k][kk].extend(v)
    info = dict(parts[0]["info"], episodes=sum(pt["info"]["episodes"] for pt in parts), parts=args.merge_parts)
    summarize(acc, info, out_dir / "result.json")
    sys.exit(0)

ckpt = Path(args.ckpt)
run_dir = ckpt.parents[1]
cfg = OmegaConf.load(args.config or run_dir / "config.full.yaml")
cfg.framework.qwenvl.autocast_dtype = args.dtype
cfg.trainer.pretrained_checkpoint = None
mc = cfg.framework.get("episode_memory", None)
if mc is not None:  # the checkpoint holds every memory / decision tensor: no pretrained files at construction
    mc.init_from = None
    if mc.get("decision", None) is not None:
        mc.decision.init_from = None

# train mixture (as the run) -> merged normalization metadata for the val datasets
train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.datasets.vla_data))
for k in ("stream", "color_jitter"):
    train_cfg.pop(k, None)
mix = get_vla_dataset(data_cfg=train_cfg, balance_dataset_weights=bool(train_cfg.get("balance_dataset_weights", False)),
                      balance_trajectory_weights=bool(train_cfg.get("balance_trajectory_weights", False)))
tag = mix.datasets[0].tag
merged = mix.merged_metadata[tag]
saved = json.load(open(run_dir / "dataset_statistics.json"))
sv = next(iter(saved.values()))
q01, q99 = np.asarray(sv["action"]["q01"], np.float64), np.asarray(sv["action"]["q99"], np.float64)
s01, s99 = np.asarray(sv["state"]["q01"], np.float64), np.asarray(sv["state"]["q99"], np.float64)
mq99 = np.concatenate([np.asarray(getattr(merged.statistics.action[k.split(".", 1)[1]], "q99"), np.float64)
                       for k in mix.datasets[0].modality_keys["action"]])
assert np.allclose(mq99, q99, atol=1e-3), "merged train statistics differ from the run's dataset_statistics.json"
del mix

val_cfg = dict(OmegaConf.to_container(cfg.datasets.vla_data))
for k in ("stream", "color_jitter", "exclude_episodes"):
    val_cfg.pop(k, None)
val_cfg.update(emit_progress=True, episode_lists=args.lists)
root = Path(cfg.datasets.vla_data.data_root_dir)
vals = []
for name, robot in VAL:
    ds = make_LeRobotSingleDataset(root, name, robot, data_cfg=OmegaConf.create(val_cfg))
    ds.set_transforms_metadata(merged)
    pos = defaultdict(dict)
    for j, (e, t) in enumerate(ds.all_steps):
        pos[int(e)][int(t)] = j
    vals.append((name, ds, pos, sorted(pos)))
    print(f"{name}: {len(pos)} episodes", flush=True)
# one task id per instruction over the val sets (sorted text), the same in every part / model run
task_text = set()
for name, _ in VAL:
    meta = root / name / "meta"
    if (meta / "tasks.jsonl").exists():
        task_text |= {json.loads(line)["task"] for line in open(meta / "tasks.jsonl")}
    else:
        task_text |= set(pd.read_parquet(meta / "tasks.parquet").index)
TASK_ID = {t: i for i, t in enumerate(sorted(task_text))}


def unnorm(x, lo, hi):
    x = np.asarray(x, np.float64)
    return np.where(hi == lo, x, (x + 1.0) / 2.0 * (hi - lo) + lo)


def hold_prev(gt, stride):
    """event_check.py's HOLD: the previous GT command held over each `stride` window"""
    out = np.empty_like(gt)
    for a in range(0, len(gt), stride):
        out[a:a + stride] = gt[max(a - 1, 0)]
    return out


def hold_state(state):
    s = unnorm(np.asarray(state, np.float64)[-1], s01, s99)
    out = np.full(42, np.nan)
    for a, b, dims in STATE_TO_ACTION:
        out[dims] = s[a:b]
    return out


t0 = time.time()
model = build_framework(cfg)
missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu", mmap=True), strict=False)
assert not unexpected and not missing, f"checkpoint mismatch: missing {missing[:4]}, unexpected {unexpected[:4]}"
model.qwen_vl_interface.to(getattr(torch, args.dtype))
model = model.cuda().eval()
stream_mem = bool(getattr(model, "mem_tbptt", 0))
print(f"model ready in {time.time() - t0:.0f}s ({type(model).__name__}, streamed memory {stream_mem}, "
      f"memory_reset {args.memory_reset})", flush=True)

batches = [(vi, eps[g0:g0 + args.batch], g0) for vi, (_, _, _, eps) in enumerate(vals) for g0 in range(0, len(eps), args.batch)]
p_i, p_n = (int(v) for v in args.part.split("/")) if args.part else (0, 1)
acc = {k: defaultdict(list) for k in ACC_KEYS}
done = 0
out_dir.mkdir(parents=True, exist_ok=True)
assert args.stride <= int(cfg.framework.action_model.action_horizon), "rollout stitching needs stride <= horizon"
t0 = time.time()
with torch.no_grad():
    for bi, (vi, group, g0) in enumerate(batches):
        if bi % p_n != p_i:
            continue
        name, ds, pos, _ = vals[vi]
        done += len(group)
        T = {e: len(pos[e]) for e in group}
        mem = {}
        roll_pred, roll_gt, lang = defaultdict(list), defaultdict(list), {}
        k = 0
        while True:
            act = [e for e in group if k * args.stride < T[e]]
            if not act:
                break
            exs = [ds[pos[e][k * args.stride]] for e in act]
            torch.manual_seed(args.seed * 1000003 + bi * 1009 + k)
            if stream_mem:
                m = None if (k == 0 or args.memory_reset) else torch.stack([mem[e] for e in act])
                out = model.predict_action(exs, memory=m)
                for i, e in enumerate(act):
                    mem[e] = out["memory"][i]
            else:
                out = model.predict_action(exs)
            pred = unnorm(out["normalized_actions"], q01, q99)
            gt = unnorm(np.stack([x["action"] for x in exs]), q01, q99)
            d = np.abs(pred - gt)
            for i, (e, x) in enumerate(zip(act, exs)):
                keys = ("ALL", "set:" + name, x["lang"])
                roll_pred[e].append(pred[i, :args.stride])
                roll_gt[e].append(gt[i, :args.stride])
                lang[e] = x["lang"]
                held = {"hold_cmd": np.repeat(gt[i, :1], gt.shape[1], 0), "hold_state": np.repeat(hold_state(x["state"])[None], gt.shape[1], 0)}
                for g, cols in GROUPS.items():
                    v = d[i][:, cols].mean()
                    for kk in keys:
                        acc["chunk"][(kk, g)].append(v)
                    for s in STEPS:
                        acc["per_step"][(g, s)].append(d[i, s, cols].mean())
                    for bn, h in held.items():
                        if not np.isnan(h[0, cols]).any():
                            bv = np.abs(gt[i][:, cols] - h[:, cols]).mean()
                            for kk in keys:
                                acc["base"][(kk, g, bn)].append(bv)
                if "progress" in out:
                    pe = abs(float(out["progress"][i]) - float(x["progress"]))
                    for kk in keys:
                        acc["prog"][kk].append(pe)
            k += 1
        short = name.split("/")[0].replace("car-", "").replace("-ep", "").replace("-ov", "")
        for e in group:
            P, G = np.concatenate(roll_pred[e])[:T[e]], np.concatenate(roll_gt[e])[:T[e]]
            H = hold_prev(G, args.stride)
            keys = ("ALL", "set:" + name, lang[e])
            for g, cols in GROUPS.items():
                for w, X in (("pred", P), ("hold", H)):
                    v = np.abs(X[:, cols] - G[:, cols]).mean()
                    for kk in keys:
                        acc["roll"][(kk, g, w)].append(v)
            dd = out_dir / "rollout" / f"ep{short}_{e:05d}"
            dd.mkdir(parents=True, exist_ok=True)
            np.save(dd / "pred.npy", P.astype(np.float32))
            np.save(dd / "gt_action.npy", G.astype(np.float32))
            json.dump({"episode": f"{name}:{e}", "dataset": name, "episode_index": int(e), "task_index": TASK_ID[lang[e]],
                       "task": lang[e], "stride": args.stride, "T": T[e]}, open(dd / "meta.json", "w"), indent=1)
        print(f"  batch {bi + 1}/{len(batches)} ({name}, {len(group)} episodes)  {time.time() - t0:.0f}s", flush=True)

info = {"ckpt": str(ckpt), "framework": type(model).__name__, "streamed_memory": stream_mem, "memory_reset": args.memory_reset,
        "stride": args.stride, "dtype": args.dtype, "seed": args.seed, "episodes": done}
if args.part:
    pickle.dump({k: dict(v) for k, v in acc.items()} | {"info": info}, open(out_dir / f"accum_part{p_i}.pkl", "wb"))
summarize(acc, info, out_dir / (f"result_part{p_i}.json" if args.part else "result.json"))
print(f"({time.time() - t0:.0f}s)")
