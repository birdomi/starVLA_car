"""Streamed val evaluation of a car1015 checkpoint, as deployed: every val episode is walked from frame 0 in steps of
`--stride` frames; at every step the model gets the RECORDED observation (cameras, state, instruction; val normalized with
the TRAIN statistics) and predicts a chunk. A QwenGR00T_EpisodeMemory (episode_memory.stream) model carries its memory
across the steps (memory_init at the episode start, then the returned "memory"); any other framework is queried statelessly
at the same frames with the same noise, so two runs of this script are paired.

Outputs (--out):
  stream.json  per-task chunk error at every query step (head / arms in deg, hands raw; chunk mean over the horizon) next
               to the hold_cmd (GT a[t] held) / hold_state (measured state held) baselines, per-chunk-step error, and the
               memory's progress error |p_hat - t / (T - 1)| (memory models)
  rollout/ep{id:05d}/{pred.npy, gt_action.npy, meta.json} for the --rollout_tasks episodes: the first `stride` steps of every
               chunk stitched into a (T, 42) trajectory (seqcheck/seq_check.py input)
--exec N (> stride, <= horizon): the stitched trajectory executes N steps of a chunk queried every N frames instead; the
               memory still updates on the `stride` grid (as trained), and a chunk at a frame between two grid points is a side
               query on the last grid memory (its returned memory dropped); the same seeds per frame keep runs paired
Episodes are batched (--batch at a time, same step index) and decoded forward only.
--part i/n runs only every n-th batch from the i-th, with the batches and seeds of the full run (n processes, e.g. one
per GPU, give the full run's rollouts); each writes accum_part{i}.pkl + stream_part{i}.json, and --merge_parts n
(no model / data) merges them into stream.json.

Usage: CUDA_VISIBLE_DEVICES=4 STARVLA_CUDA_DECODER_CACHE=32 python eval_val_stream.py --ckpt <ckpt.pt> --config <yaml> \
           --robot h1_2_car_3cam_headstate --out <dir> [--stride 15] [--tasks ...] [--rollout_tasks 0,1]
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

sys.path.insert(0, "/mnt/DATA2/yyg/starVLA")
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset  # noqa: E402
from starVLA.model.framework.base_framework import build_framework  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--dataset", default="car-1015-v1-ep")
ap.add_argument("--robot", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stride", type=int, default=15, help="query interval in frames (= the memory update interval)")
ap.add_argument("--exec", type=int, default=0, help="steps executed per stitched chunk (default: stride)")
ap.add_argument("--tasks", default="", help="comma-separated task_index values to evaluate (default: all)")
ap.add_argument("--rollout_tasks", default="0,1", help="tasks whose stitched trajectories are written (seq_check, event_check); 'all' = every task")
ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
ap.add_argument("--batch", type=int, default=8, help="episodes walked in parallel")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_episodes", type=int, default=0)
ap.add_argument("--part", default="", help="i/n: every n-th episode batch from the i-th (same batches / seeds as the full run)")
ap.add_argument("--merge_parts", type=int, default=0, help="n: merge accum_part{0..n-1}.pkl of --out into stream.json")
ap.add_argument("--episodes", default="", help="comma-separated val episode ids only (e.g. a deploy bundle's samples)")
args = ap.parse_args()

HEAD, LARM, RARM = np.r_[0:2], np.r_[2:9], np.r_[22:29]
LHAND, RHAND = np.r_[9:22], np.r_[29:42]
GROUPS = {"head": HEAD, "arm": np.r_[LARM, RARM], "l_arm": LARM, "r_arm": RARM, "hand": np.r_[LHAND, RHAND]}
ANGULAR = {"head", "arm", "l_arm", "r_arm"}

cfg = OmegaConf.load(args.config)
assert str(cfg.datasets.vla_data.action_mode) == "abs", "this script un-normalizes abs actions only"
cfg.framework.qwenvl.autocast_dtype = args.dtype
root = Path(cfg.datasets.vla_data.data_root_dir)
horizon = int(cfg.framework.action_model.action_horizon)
assert args.stride <= horizon, (args.stride, horizon)
EXEC = args.exec or args.stride
assert args.stride <= EXEC <= horizon, (args.stride, EXEC, horizon)
STEPS = sorted({s for s in (0, 9, 14, 19, 29, horizon - 1) if s < horizon})

def fmt(g, v):
    return float(np.degrees(v)) if g in ANGULAR else float(v)


def summarize(chunk, base, per_step, prog, task_text, info, path):
    """accumulated per-query errors -> stream.json (+ the printed table)"""
    keys = sorted({t for t, _ in chunk if t != "ALL"}) + ["ALL"]
    res = {
        **info, "units": {"head/arm/l_arm/r_arm": "deg", "hand": "raw"},
        "per_task": {str(t): {"task": task_text.get(t, t) if t != "ALL" else "ALL", "n": len(chunk[(t, "arm")]),
                              **{g: fmt(g, np.mean(chunk[(t, g)])) for g in GROUPS},
                              **{f"{g}_{bn}": fmt(g, np.mean(base[(t, g, bn)])) for g in GROUPS
                                 for bn in ("hold_cmd", "hold_state") if base[(t, g, bn)]},
                              **({"progress_mae": float(np.mean(prog[t]))} if prog[t] else {})}
                     for t in keys},
        "per_step": {g: {str(s): fmt(g, np.mean(per_step[(g, s)])) for s in STEPS} for g in GROUPS},
    }
    json.dump(res, open(path, "w"), indent=2)
    print(f"\n{info['ckpt']}  {info['framework']}  streamed memory {info['streamed_memory']}  {info['episodes']} episodes")
    print(f"{'task':60s} {'n':>5s} {'head°':>6s} {'arm°':>6s} {'hand':>6s} {'prog':>6s}")
    for t in keys:
        r = res["per_task"][str(t)]
        print(f"{str(r['task'])[:60]:60s} {r['n']:5d} {r['head']:6.2f} {r['arm']:6.2f} {r['hand']:6.1f} {r.get('progress_mae', float('nan')):6.3f}")
    a = res["per_task"]["ALL"]
    for bn in ("hold_cmd", "hold_state"):
        print(f"{'baseline ' + bn:60s} {a['n']:5d} {a.get('head_' + bn, float('nan')):6.2f} {a.get('arm_' + bn, float('nan')):6.2f} "
              f"{a.get('hand_' + bn, float('nan')):6.1f}")
    print("per chunk step  " + "  ".join(f"t{s}: head {res['per_step']['head'][str(s)]:.2f}° arm {res['per_step']['arm'][str(s)]:.2f}°"
                                          for s in STEPS))
    return res


if args.merge_parts:
    parts = [pickle.load(open(Path(args.out) / f"accum_part{i}.pkl", "rb")) for i in range(args.merge_parts)]
    acc = {k: defaultdict(list) for k in ("chunk", "base", "per_step", "prog")}
    for pt in parts:
        for k in acc:
            for kk, v in pt[k].items():
                acc[k][kk].extend(v)
    info = dict(parts[0]["info"], episodes=sum(pt["info"]["episodes"] for pt in parts), parts=args.merge_parts)
    summarize(**acc, task_text=parts[0]["task_text"], info=info, path=Path(args.out) / "stream.json")
    sys.exit(0)

train_cfg = OmegaConf.to_container(cfg.datasets.vla_data)
val_cfg = dict(train_cfg)
for k in ("exclude_episodes", "color_jitter", "stream"):
    val_cfg.pop(k, None)
val_cfg["emit_progress"] = True
train_ds = make_LeRobotSingleDataset(root, f"{args.dataset}/train", args.robot, data_cfg=OmegaConf.create(train_cfg))
val_ds = make_LeRobotSingleDataset(root, f"{args.dataset}/val", args.robot, data_cfg=OmegaConf.create(val_cfg))
val_ds.set_transforms_metadata(train_ds.metadata)

stats = json.load(open(root / args.dataset / "train/meta/stats_gr00t.abs.json"))["statistics"]["action"]
q01, q99 = np.array(stats["q01"], np.float64), np.array(stats["q99"], np.float64)
const = q99 == q01


def unnorm(x):
    x = np.asarray(x, np.float64)
    return np.where(const, x, (x + 1.0) / 2.0 * (q99 - q01) + q01)


# hold_state baseline: the model's state input un-normalized key by key into the action dims of the same-named action key
_mod = json.load(open(root / args.dataset / "train/meta/modality.json"))
_st = train_ds.metadata.statistics.state
STATE_TO_ACTION, _off = [], 0
for _k in val_ds.modality_keys["state"]:
    _k = _k.split(".", 1)[1]
    _lo, _hi = np.asarray(_st[_k].q01, np.float64), np.asarray(_st[_k].q99, np.float64)
    if _k in _mod["action"]:
        STATE_TO_ACTION.append((_off, _off + len(_lo), np.r_[_mod["action"][_k]["start"]:_mod["action"][_k]["end"]], _lo, _hi))
    _off += len(_lo)


def hold_state(state):
    x, out = np.asarray(state, np.float64)[-1], np.full(len(q01), np.nan)
    for s0, s1, dims, lo, hi in STATE_TO_ACTION:
        out[dims] = np.where(hi == lo, x[s0:s1], (x[s0:s1] + 1.0) / 2.0 * (hi - lo) + lo)
    return out


meta = root / args.dataset / "val/meta"
episodes = pd.read_parquet(meta / "episodes/chunk-000/file-000.parquet")
task_text = {int(i): t for t, i in pd.read_parquet(meta / "tasks.parquet")["task_index"].items()}
ep_task = {int(e): int(t[0]) for e, t in zip(episodes["episode_index"], episodes["tasks"])}
want = {int(t) for t in args.tasks.split(",")} if args.tasks else set(task_text)
roll_tasks = (set(task_text) if args.rollout_tasks == "all" else
              {int(t) for t in args.rollout_tasks.split(",")} if args.rollout_tasks else set())
pos = defaultdict(dict)
for j, (e, t) in enumerate(val_ds.all_steps):
    pos[int(e)][int(t)] = j
eps = sorted(e for e in pos if ep_task[e] in want)
if args.max_episodes:
    eps = eps[:args.max_episodes]
if args.episodes:
    eps = [e for e in eps if e in {int(x) for x in args.episodes.split(",")}]

t0 = time.time()
model = build_framework(cfg)
missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location="cpu", mmap=True), strict=False)
assert not unexpected, f"unexpected keys: {unexpected[:5]}"
assert not [k for k in missing if "qwen_vl_interface" in k], f"missing VLM keys: {missing[:5]}"
if missing:
    print(f"WARNING {len(missing)} missing keys (e.g. {missing[:2]}): fine only for a smoke test", flush=True)
model.qwen_vl_interface.to(getattr(torch, args.dtype))
model = model.cuda().eval()
stream_mem = bool(getattr(model, "mem_tbptt", 0))
print(f"model ready in {time.time() - t0:.0f}s ({type(model).__name__}, streamed memory {stream_mem}); {len(eps)} val "
      f"episodes, stride {args.stride}, horizon {horizon}", flush=True)

chunk = defaultdict(list)     # (task, group) -> chunk-mean error per query
base = defaultdict(list)      # (task, group, baseline) -> same for the no-model baselines
per_step = defaultdict(list)  # (group, step) -> error at that chunk step
prog = defaultdict(list)      # task -> |p_hat - t / (T - 1)|
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
t0 = time.time()
with torch.no_grad():
    p_i, p_n = (int(v) for v in args.part.split("/")) if args.part else (0, 1)
    done = 0
    for g0 in range(0, len(eps), args.batch):
        if (g0 // args.batch) % p_n != p_i:
            continue
        group = eps[g0:g0 + args.batch]
        done += len(group)
        T = {e: len(pos[e]) for e in group}
        for e in group:
            assert sorted(pos[e]) == list(range(T[e])), f"episode {e}: frames not contiguous"
        mem = {e: None for e in group}
        stitched = {e: np.zeros((T[e], len(q01))) for e in group if ep_task[e] in roll_tasks}
        k = 0
        while True:
            act = [e for e in group if k * args.stride < T[e]]
            if not act:
                break
            exs = [val_ds[pos[e][k * args.stride]] for e in act]
            torch.manual_seed(args.seed * 1000003 + g0 * 1009 + k)
            if stream_mem:
                m = (model.memory_init(len(act)) if k == 0 else torch.stack([mem[e] for e in act]))
                out = model.predict_action(exs, memory=m)
                for i, e in enumerate(act):
                    mem[e] = out["memory"][i]
            else:
                out = model.predict_action(exs)
            pred = unnorm(out["normalized_actions"])
            gt = unnorm(np.stack([x["action"] for x in exs]))
            d = np.abs(pred - gt)
            for i, (e, x) in enumerate(zip(act, exs)):
                t_ = ep_task[e]
                held = {"hold_cmd": gt[i, :1], "hold_state": hold_state(x["state"])[None]}
                for gname, cols in GROUPS.items():
                    v = d[i][:, cols].mean()
                    chunk[(t_, gname)].append(v); chunk[("ALL", gname)].append(v)
                    for s in STEPS:
                        per_step[(gname, s)].append(d[i, s, cols].mean())
                    for bn, h in held.items():
                        if not np.isnan(h[0, cols]).any():
                            bv = np.abs(gt[i][:, cols] - h[:, cols]).mean()
                            base[(t_, gname, bn)].append(bv); base[("ALL", gname, bn)].append(bv)
                if "progress" in out:
                    pe = abs(float(out["progress"][i]) - float(x["progress"]))
                    prog[t_].append(pe); prog["ALL"].append(pe)
                if e in stitched and EXEC == args.stride:
                    a = k * args.stride
                    stitched[e][a:a + args.stride] = pred[i][: min(args.stride, T[e] - a)]
            a = k * args.stride
            for f in range(-(-a // EXEC) * EXEC, a + args.stride, EXEC) if EXEC != args.stride else []:
                sel = [i for i, e in enumerate(act) if e in stitched and f < T[e]]
                if not sel:
                    continue
                if f == a:
                    side = pred
                else:  # between grid points: the chunk from frame f on the memory of grid frame a
                    sexs = [val_ds[pos[act[i]][f]] for i in sel]
                    torch.manual_seed(args.seed * 1000003 + g0 * 1009 + 500009 + f)
                    so = (model.predict_action(sexs, memory=torch.stack([mem[act[i]] for i in sel])) if stream_mem
                          else model.predict_action(sexs))
                    side = np.zeros_like(pred); side[sel] = unnorm(so["normalized_actions"])
                for i in sel:
                    e = act[i]
                    stitched[e][f:f + EXEC] = side[i][: min(EXEC, T[e] - f)]
            k += 1
        for e, traj in stitched.items():
            dd = out_dir / "rollout" / f"ep{e:05d}"
            dd.mkdir(parents=True, exist_ok=True)
            np.save(dd / "pred.npy", traj)
            np.save(dd / "gt_action.npy", np.stack(val_ds.get_trajectory_data(e)["action"].values).astype(np.float64))
            json.dump({"episode": e, "task_index": ep_task[e], "task": task_text[ep_task[e]], "length": T[e],
                       "stride": args.stride, "exec": EXEC, "ckpt": args.ckpt, "streamed_memory": stream_mem}, open(dd / "meta.json", "w"))
        print(f"  {min(g0 + args.batch, len(eps))}/{len(eps)} episodes  {time.time() - t0:.0f}s", flush=True)


info = {"ckpt": args.ckpt, "framework": type(model).__name__, "streamed_memory": stream_mem, "stride": args.stride,
        "exec": EXEC, "horizon": horizon, "dtype": args.dtype, "seed": args.seed, "episodes": done}
if args.part:
    pickle.dump({"chunk": dict(chunk), "base": dict(base), "per_step": dict(per_step), "prog": dict(prog),
                 "task_text": task_text, "info": info}, open(out_dir / f"accum_part{p_i}.pkl", "wb"))
summarize(chunk, base, per_step, prog, task_text, info, out_dir / (f"stream_part{p_i}.json" if args.part else "stream.json"))
print(f"({time.time() - t0:.0f}s)")
