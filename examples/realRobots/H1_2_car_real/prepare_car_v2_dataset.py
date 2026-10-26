"""Convert the car-turn-signal_v2 export (LeRobot v2.0) into the layout of car-turn-signal-ep.

car-turn-signal_v2 (2026-10-13) re-exports the same 1203 recordings as car-turn-signal with
  - action 0:2 = raw head [t] (operator head pose; == car-turn-signal state[t, 63:65]),
  - 63-dim state: head 2 | legs/waist/arms/pad 35 | RIGHT hand 13 | LEFT hand 13 (no head_cmd),
  - 21 previously frozen / misaligned episodes re-synced, one mp4 per episode (GOP 250),
  - camera keys without the `_2` suffix, a new split, and subtask_index = -1 everywhere.

This writes <dst>/{train,val} as a v3.0 set that the existing car data configs load unchanged:
  1. data      - per-episode parquet concatenated into data/chunk-000/file-XXX.parquet
                 (--eps-per-file episodes each), `index` recomputed, subtask_index and the
                 duplicate annotation.human.task_description column dropped.
  2. tasks     - " (fixed seat)" removed from every instruction: 4 tasks -> 2, the same index
                 in train and val (0 = left, 1 = right). Original instruction per episode ->
                 meta/task_merge.json.
  3. ch2       - hand action channel 2 (dims 11, 31) clipped to 1650 as in prepare_car_dataset.py
                 (the measured state saturates at ~1652). Pre-clip values -> meta/action_ch2_preclip.npz.
  4. hand rows - out-of-range hand-state rows (sensor garbage) linearly interpolated per episode.
                 Originals -> meta/state_row_precorrection.npz.
  5. videos    - every camera at videos/observation.images.<cam>_2/chunk-000/file-{episode}.mp4: with
                 --link-videos a symlink to the source mp4 (GOP 250; ~12x slower random access on CPU than
                 GOP 15), otherwise re-encoded (h264_nvenc, GOP --gop, no B-frames, cq 19, exactly `length` frames).
  6. meta      - info.json (v3.0 templates, joint names), modality.json (head_left_2 ... keys,
                 left_ee = state 50:63, right_ee = state 37:50), episodes parquet, stats.json.

Usage: python prepare_car_v2_dataset.py playground/Datasets/car-turn-signal_v2 \
           playground/Datasets/car-turn-signal-v2-ep --ref playground/Datasets/car-turn-signal-ep/train
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CAMS = ["head_left", "head_right", "wrist_left", "wrist_right"]
STATE_HANDS = [(37, 50), (50, 63)]  # v2: right hand | left hand
ACTION_CH2 = {"left_ch2": 9 + 2, "right_ch2": 29 + 2}
CH2_MAX = 1650.0
HAND_STATE_VALID = (-100.0, 2100.0)
TASK_SUFFIX = " (fixed seat)"
TASKS = ["Activate the left turn signal and inspect its operation.",
         "Activate the right turn signal and inspect its operation."]

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("dst", type=Path)
ap.add_argument("--ref", type=Path, required=True, help="car-turn-signal-ep/train (joint names, feature layout)")
ap.add_argument("--gpus", default="0,1,2,3")
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--gop", type=int, default=15)
ap.add_argument("--eps-per-file", type=int, default=300)
ap.add_argument("--skip-videos", action="store_true")
ap.add_argument("--link-videos", action="store_true", help="symlink the source mp4s instead of re-encoding")
ap.add_argument("--video-root", type=Path, default=None,
                help="write encoded videos under <video-root>/<split>/videos instead of <dst> (e.g. while <dst> holds symlinks in use)")
ap.add_argument("--skip-data", action="store_true", help="videos only (data / meta already written)")
args = ap.parse_args()
gpus = args.gpus.split(",")


def n_packets(f):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(f)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def encode(job):
    i, (src, dst, length) = job
    if dst.exists() and n_packets(dst) == length:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[i % len(gpus)])
    # decoded frames stay on the GPU (NVDEC -> NVENC, output yuv420p): ~0.3 s CPU per video, so a training run
    # sharing the GPUs / CPUs is not slowed down
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                    "-i", str(src), "-frames:v", str(length), "-map", "0:v:0", "-c:v", "h264_nvenc", "-preset", "p5",
                    "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0", "-g", str(args.gop), "-bf", "0",
                    "-forced-idr", "1", "-r", "30", "-an", str(tmp)], env=env, check=True)
    got = n_packets(tmp)
    if got != length:
        raise RuntimeError(f"{dst}: {got} frames, expected {length}")
    os.replace(tmp, dst)
    return "ok"


def stats_of(x):
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "min": x.min(0).tolist(), "max": x.max(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(), "q99": np.quantile(x, 0.99, axis=0).tolist()}


ref_info = json.load(open(args.ref / "meta/info.json"))
ref_state_names = ref_info["features"]["observation.state"]["names"]
# v2 state = ref 0:37 | right hand (ref 50:63) | left hand (ref 37:50); ref 63:65 (raw head) dropped
state_names = ref_state_names[:37] + ref_state_names[50:63] + ref_state_names[37:50]
action_names = ref_info["features"]["action"]["names"]
assert state_names[37] == "right_hand_00" and state_names[50] == "left_hand_00", state_names[37:51]

for split in ("train", "val"):
    s, d = args.src / split, args.dst / split
    (d / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    (d / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    src_info = json.load(open(s / "meta/info.json"))
    fps = src_info["fps"]
    src_tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(s / "meta/tasks.jsonl")}
    episodes = [json.loads(l) for l in open(s / "meta/episodes.jsonl")]
    episodes.sort(key=lambda e: e["episode_index"])

    if args.skip_data:
        assert (d / "meta/info.json").exists(), d
    if not args.skip_data:
        # ---- data ----
        task_merge, pre_ch2 = {}, {k: [] for k in ACTION_CH2}
        bad_rows, bad_orig = [], []
        ep_rows, all_state, all_action = [], [], []
        gidx = 0
        n_clipped = 0
        groups = [episodes[i:i + args.eps_per_file] for i in range(0, len(episodes), args.eps_per_file)]
        for file_index, group in enumerate(groups):
            frames = []
            for ep in group:
                e = ep["episode_index"]
                df = pd.read_parquet(s / src_info["data_path"].format(episode_chunk=e // src_info["chunks_size"], episode_index=e))
                assert len(df) == ep["length"] and (df["frame_index"].values == np.arange(len(df))).all(), e
                assert (df["task_index"] == df["annotation.human.task_description"]).all(), e
                assert df["task_index"].nunique() == 1, e
                text = src_tasks[int(df["task_index"].iloc[0])]
                task_merge[e] = text
                new_task = TASKS.index(text.replace(TASK_SUFFIX, ""))
                st = np.stack(df["observation.state"].values).astype(np.float64)
                act = np.stack(df["action"].values).astype(np.float64)
                assert st.shape[1] == 63 and act.shape[1] == 42, (e, st.shape, act.shape)
                for k, c in ACTION_CH2.items():
                    pre_ch2[k].append(act[:, c].copy())
                    n_clipped += int((act[:, c] > CH2_MAX).sum())
                    act[:, c] = np.minimum(act[:, c], CH2_MAX)
                lo, hi = HAND_STATE_VALID
                hands = np.concatenate([st[:, a:b] for a, b in STATE_HANDS], axis=1)
                bad = ((hands < lo) | (hands > hi)).any(axis=1)
                if bad.any():
                    good = np.where(~bad)[0]
                    bad_rows.append(gidx + np.where(bad)[0])
                    bad_orig.append(st[bad].copy())
                    for a_, b_ in STATE_HANDS:
                        for dd in range(a_, b_):
                            st[bad, dd] = np.interp(np.where(bad)[0], good, st[good, dd])
                n = len(df)
                frames.append(pd.DataFrame({
                    "episode_index": np.full(n, e, np.int64), "frame_index": df["frame_index"].values.astype(np.int64),
                    "timestamp": df["timestamp"].values.astype(np.float64), "index": np.arange(gidx, gidx + n, dtype=np.int64),
                    "task_index": np.full(n, new_task, np.int64), "next.done": df["next.done"].values.astype(bool),
                    "observation.state": list(st), "action": list(act)}))
                row = {"episode_index": e, "tasks": [new_task], "length": n, "dataset_from_index": gidx,
                       "dataset_to_index": gidx + n, "data_path": f"data/chunk-000/file-{file_index:03d}.parquet"}
                for cam in CAMS:
                    k = f"videos/observation.images.{cam}_2/"
                    row.update({k + "chunk_index": 0, k + "file_index": e, k + "from_timestamp": 0.0, k + "to_timestamp": n / fps})
                row.update({"data/chunk_index": 0, "data/file_index": file_index})
                ep_rows.append(row)
                all_state.append(st)
                all_action.append(act)
                gidx += n
            out = pd.concat(frames, ignore_index=True)
            schema = pa.schema([("episode_index", pa.int64()), ("frame_index", pa.int64()), ("timestamp", pa.float64()),
                                ("index", pa.int64()), ("task_index", pa.int64()), ("next.done", pa.bool_()),
                                ("observation.state", pa.list_(pa.float64())), ("action", pa.list_(pa.float64()))])
            pq.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False), d / f"data/chunk-000/file-{file_index:03d}.parquet")
        pd.DataFrame(ep_rows).to_parquet(d / "meta/episodes/chunk-000/file-000.parquet", index=False)
        pd.DataFrame({"task_index": np.arange(len(TASKS), dtype=np.int64)}, index=pd.Index(TASKS, name="task")).to_parquet(d / "meta/tasks.parquet")
        json.dump({str(k): v for k, v in task_merge.items()}, open(d / "meta/task_merge.json", "w"), indent=1)
        np.savez(d / "meta/action_ch2_preclip.npz", **{k: np.concatenate(v) for k, v in pre_ch2.items()})
        np.savez(d / "meta/state_row_precorrection.npz",
                 row=np.concatenate(bad_rows) if bad_rows else np.zeros(0, np.int64),
                 original=np.concatenate(bad_orig) if bad_orig else np.zeros((0, 63)))
        print(f"[{split}] {len(episodes)} episodes, {gidx} frames, {len(groups)} data files, ch2 clipped {n_clipped}, "
              f"hand-state rows fixed {sum(len(r) for r in bad_rows)}, tasks {np.bincount([r['tasks'][0] for r in ep_rows]).tolist()}", flush=True)

        # ---- meta ----
        info = {k: v for k, v in ref_info.items() if k not in ("source_episode_ids", "features")}
        info.update({"created_at": src_info["created_at"], "splits": {split: f"0:{len(episodes)}"},
                     "total_episodes": len(episodes), "total_frames": gidx, "total_tasks": len(TASKS)})
        feats = {k: v for k, v in ref_info["features"].items() if not k.startswith("observation.images.")}
        feats["observation.state"] = dict(ref_info["features"]["observation.state"], shape=[63], names=state_names)
        feats["action"] = dict(ref_info["features"]["action"], names=action_names)
        for cam in CAMS:
            v = json.loads(json.dumps(src_info["features"][f"observation.images.{cam}"]))
            v.setdefault("info", {})["video.channels"] = 3
            feats[f"observation.images.{cam}_2"] = v
        info["features"] = feats
        json.dump(info, open(d / "meta/info.json", "w"), indent=2)
        ref_mod = json.load(open(args.ref / "meta/modality.json"))
        mod = json.loads(json.dumps(ref_mod))
        del mod["state"]["head_cmd"]
        mod["state"]["right_ee"].update(start=37, end=50)
        mod["state"]["left_ee"].update(start=50, end=63)
        json.dump(mod, open(d / "meta/modality.json", "w"), indent=2)
        stats = {"observation.state": stats_of(np.concatenate(all_state)), "action": stats_of(np.concatenate(all_action))}
        json.dump(stats, open(d / "meta/stats.json", "w"))
        json.dump({"source": str(args.src / split), "script": "examples/realRobots/H1_2_car_real/prepare_car_v2_dataset.py",
                   "head_action": "action[t, 0:2] = raw head (operator head pose) at t, as exported",
                   "state": "63-D, 37:50 right hand, 50:63 left hand, no head_cmd", "tasks": TASKS,
                   "videos": "symlinks to the source mp4s (GOP 250)" if args.link_videos else f"re-encoded, GOP {args.gop}"},
                  open(d / "meta/v2_prep_note.json", "w"), indent=1)

    # ---- videos ----
    if args.skip_videos:
        continue
    jobs = []
    for cam in CAMS:
        for ep in episodes:
            e = ep["episode_index"]
            src = s / src_info["video_path"].format(episode_chunk=e // src_info["chunks_size"],
                                                    video_key=f"observation.images.{cam}", episode_index=e)
            vd = (args.video_root / split) if (args.video_root and not args.link_videos) else d
            jobs.append((src, vd / f"videos/observation.images.{cam}_2/chunk-000/file-{e:03d}.mp4", ep["length"]))
    if args.link_videos:
        for src, dst, _ in jobs:
            assert src.is_file(), src
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.is_symlink():
                os.symlink(src.resolve(), dst)
        print(f"[{split}] {len(jobs)} video symlinks", flush=True)
        continue
    print(f"[{split}] encoding {len(jobs)} videos", flush=True)
    failed = []
    with ThreadPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(encode, j) for j in enumerate(jobs)]
        for n, fu in enumerate(futs):
            try:
                fu.result()
            except Exception as e:
                failed.append(str(e)[:200])
            if n % 300 == 0:
                print(f"[{split}] {n + 1}/{len(jobs)}", flush=True)
    print(f"[{split}] videos done, {len(failed)} failed", *failed[:10], sep="\n  ", flush=True)
