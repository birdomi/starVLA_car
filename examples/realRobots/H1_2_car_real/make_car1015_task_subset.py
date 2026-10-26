"""Self-contained download subset of the prepared car-1015-v1-ep: --per-task episodes of every one of the 34 tasks.

Picks, per task, --per-task episodes at random (--seed) among the prepared <src> episodes with length >= --min-length
(the training min_episode_length), and writes them as a standalone LeRobot v3.0 dataset <dst> (no split sub-folder):
  - episodes renumbered 0 .. N-1, task-major (task 0's episodes first), each sorted by source id; the source id per new
    id -> meta/source_episodes.json,
  - data/chunk-000/file-{i:03d}.parquet (one episode per file, `index` recomputed, state / action as prepared: 34 tasks,
    hand ch2 clipped),
  - videos/observation.images.<cam>/chunk-000/file-{i:03d}.mp4 = REAL copies of the source mp4s (the prepared set only
    holds symlinks), all 4 cameras,
  - meta: info.json / episodes parquet updated, tasks.parquet + modality.json copied, stats.json recomputed on the subset.

Usage: python make_car1015_task_subset.py playground/Datasets/car-1015-v1-ep/train playground/Datasets/car-1015-v1-20pertask
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path, help="prepared split, e.g. playground/Datasets/car-1015-v1-ep/train")
ap.add_argument("dst", type=Path)
ap.add_argument("--per-task", type=int, default=20)
ap.add_argument("--min-length", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--copy-threads", type=int, default=4)
args = ap.parse_args()
assert not args.dst.exists(), f"{args.dst} exists"

src, dst = args.src, args.dst
info = json.load(open(src / "meta/info.json"))
cams = [k for k in info["features"] if k.startswith("observation.images.")]
episodes = pd.read_parquet(src / "meta/episodes/chunk-000/file-000.parquet")
tasks = pd.read_parquet(src / "meta/tasks.parquet")
task_text = {int(i): t for t, i in tasks["task_index"].items()}
task_merge = json.load(open(src / "meta/task_merge.json"))
episodes["task"] = episodes["tasks"].map(lambda t: int(t[0]))

rng = np.random.default_rng(args.seed)
picked = []
for task in sorted(task_text):
    cand = episodes[(episodes["task"] == task) & (episodes["length"] >= args.min_length)]
    assert len(cand) >= args.per_task, f"task {task}: only {len(cand)} episodes of length >= {args.min_length}"
    picked.append(cand.iloc[np.sort(rng.choice(len(cand), args.per_task, replace=False))])
picked = pd.concat(picked, ignore_index=True)
print(f"{len(picked)} episodes ({args.per_task} x {len(task_text)} tasks), {int(picked['length'].sum())} frames", flush=True)

(dst / "data/chunk-000").mkdir(parents=True)
(dst / "meta/episodes/chunk-000").mkdir(parents=True)
schema = pa.schema([("episode_index", pa.int64()), ("frame_index", pa.int64()), ("timestamp", pa.float64()),
                    ("index", pa.int64()), ("task_index", pa.int64()), ("next.done", pa.bool_()),
                    ("observation.state", pa.list_(pa.float64())), ("action", pa.list_(pa.float64()))])
rows, source, all_state, all_action, copies = [], {}, [], [], []
gidx = 0
for i, ep in picked.iterrows():
    e, n = int(ep["episode_index"]), int(ep["length"])
    df = pd.read_parquet(src / ep["data_path"])
    assert len(df) == n and (df["episode_index"] == e).all() and (df["task_index"] == ep["task"]).all(), e
    df["episode_index"] = i
    df["index"] = np.arange(gidx, gidx + n, dtype=np.int64)
    pq.write_table(pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False),
                   dst / f"data/chunk-000/file-{i:03d}.parquet")
    all_state.append(np.stack(df["observation.state"].values))
    all_action.append(np.stack(df["action"].values))
    row = {k: v for k, v in ep.items() if k != "task"}
    row.update({"episode_index": i, "dataset_from_index": gidx, "dataset_to_index": gidx + n,
                "data_path": f"data/chunk-000/file-{i:03d}.parquet", "data/chunk_index": 0, "data/file_index": i})
    for cam in cams:
        s = src / info["video_path"].format(video_key=cam, chunk_index=ep[f"videos/{cam}/chunk_index"],
                                            file_index=ep[f"videos/{cam}/file_index"])
        copies.append((os.path.realpath(s), dst / info["video_path"].format(video_key=cam, chunk_index=0, file_index=i)))
        row.update({f"videos/{cam}/chunk_index": 0, f"videos/{cam}/file_index": i})
    rows.append(row)
    source[i] = {"source_episode_index": e, "task_index": int(ep["task"]), "task": task_text[int(ep["task"])],
                 "raw_instruction": task_merge[str(e)], "length": n}
    gidx += n

pd.DataFrame(rows).to_parquet(dst / "meta/episodes/chunk-000/file-000.parquet", index=False)
shutil.copy(src / "meta/tasks.parquet", dst / "meta/tasks.parquet")
shutil.copy(src / "meta/modality.json", dst / "meta/modality.json")
info.update({"splits": {"train": f"0:{len(picked)}"}, "total_episodes": len(picked), "total_frames": gidx})
json.dump(info, open(dst / "meta/info.json", "w"), indent=2)


def stats_of(x):
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "min": x.min(0).tolist(), "max": x.max(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(), "q99": np.quantile(x, 0.99, axis=0).tolist()}


json.dump({"observation.state": stats_of(np.concatenate(all_state)), "action": stats_of(np.concatenate(all_action))},
          open(dst / "meta/stats.json", "w"))
json.dump({str(k): v for k, v in source.items()}, open(dst / "meta/source_episodes.json", "w"), indent=1)
note = json.load(open(src / "meta/car1015_prep_note.json"))
note.update({"subset": {"source": str(src), "script": "examples/realRobots/H1_2_car_real/make_car1015_task_subset.py",
                        "per_task": args.per_task, "min_length": args.min_length, "seed": args.seed,
                        "order": "task-major, source id ascending within a task"},
             "videos": "real copies of the source mp4s (GOP 250)"})
json.dump(note, open(dst / "meta/car1015_prep_note.json", "w"), indent=1)
print(f"data + meta written, copying {len(copies)} videos", flush=True)


def copy(job):
    s, d = job
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(s, d)
    assert os.path.getsize(d) == os.path.getsize(s), d


with ThreadPoolExecutor(args.copy_threads) as pool:
    for n, _ in enumerate(pool.map(copy, copies)):
        if n % 500 == 0:
            print(f"{n + 1}/{len(copies)} videos", flush=True)
print("done", flush=True)
