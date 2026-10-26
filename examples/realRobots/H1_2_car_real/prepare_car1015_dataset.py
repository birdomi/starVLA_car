"""Convert the car-1015-v1 export (LeRobot v2.0, 2026-10-15, 41 raw instructions) into the v3.0 layout of
car-turn-signal-v2-ep, so the existing car data configs (h1_2_car_turn_dual*_nohc) load it unchanged.

car-1015-v1 (train 14559 / val 766 episodes, 30 Hz):
  - action 0:2 = raw head [t] (operator head pose), 2:9 L arm, 9:22 L hand, 22:29 R arm, 29:42 R hand,
  - 63-dim state: head 2 | legs/waist/arms/pad 35 | LEFT hand 37:50 | RIGHT hand 50:63 (the car-few-tasks-v1 order;
    only car-turn-signal_v2 had the two hands swapped), no head_cmd,
  - cameras head_left / head_right / wrist_left / wrist_right, one mp4 per episode (GOP 250), subtask_index = -1.

This writes <dst>/{train,val}:
  1. episodes - the ids listed in --exclude (car1015_exclude_episodes.json: short / no head look / dark video /
                sensor garbage / teleop reset / 1-frame command spikes) are dropped; the rest keep their source
                episode_index (so the ids are not contiguous). Dropped ids -> meta/excluded_episodes.json.
  2. tasks    - the 41 raw instructions mapped to the 34 normalized ones of --task-map (car1015_task_map.json), the
                same task_index in train and val. Raw instruction per episode -> meta/task_merge.json.
  3. ch2      - hand action channel 2 (dims 11, 31) clipped to 1650 as in prepare_car_dataset.py (the measured state
                saturates at ~1652). Pre-clip values of the clipped rows -> meta/action_ch2_preclip.npz.
  4. data     - --eps-per-file episodes per parquet (default 1: every sample reads one small file, so no multi-episode
                filtering and no need for data_cfg.cache_parquet_tables), `index` recomputed, subtask_index dropped.
                A kept episode with a hand-state row outside HAND_STATE_VALID raises (the exclusion list is stale).
  5. videos   - videos/observation.images.<cam>_2/chunk-{e // 1000:03d}/file-{e:03d}.mp4: with --link-videos a symlink
                to the source mp4 (GOP 250), otherwise re-encoded (h264_nvenc, GOP --gop, no B-frames, cq 19, exactly
                `length` frames).
  6. meta     - info.json (v3.0 templates, joint names of --ref), modality.json (head_left_2 ... keys,
                left_ee = state 37:50, right_ee = state 50:63), episodes parquet, tasks.parquet, stats.json.

Usage: python prepare_car1015_dataset.py playground/Datasets/car-1015-v1 playground/Datasets/car-1015-v1-ep \
           --ref playground/Datasets/car-turn-signal-ep/train --link-videos
"""

import argparse
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

HERE = Path(__file__).resolve().parent
CAMS = ["head_left", "head_right", "wrist_left", "wrist_right"]
STATE_HANDS = [(37, 50), (50, 63)]  # left hand | right hand
ACTION_CH2 = {"left_ch2": 9 + 2, "right_ch2": 29 + 2}
CH2_MAX = 1650.0
HAND_STATE_VALID = (-100.0, 2200.0)
CHUNK = 1000

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("dst", type=Path)
ap.add_argument("--ref", type=Path, required=True, help="car-turn-signal-ep/train (joint names, feature layout)")
ap.add_argument("--task-map", type=Path, default=HERE / "car1015_task_map.json")
ap.add_argument("--exclude", type=Path, default=HERE / "car1015_exclude_episodes.json")
ap.add_argument("--splits", default="train,val")
ap.add_argument("--gpus", default="0,1,2,3")
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--read-threads", type=int, default=8)
ap.add_argument("--gop", type=int, default=15)
ap.add_argument("--eps-per-file", type=int, default=1)
ap.add_argument("--skip-videos", action="store_true")
ap.add_argument("--link-videos", action="store_true", help="symlink the source mp4s instead of re-encoding")
ap.add_argument("--video-root", type=Path, default=None,
                help="write encoded videos under <video-root>/<split>/videos instead of <dst> (e.g. while <dst> holds symlinks in use)")
ap.add_argument("--skip-data", action="store_true", help="videos only (data / meta already written)")
ap.add_argument("--overwrite", action="store_true", help="delete <dst>/<split>/{data,meta} of an earlier run first")
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
    # decoded frames stay on the GPU (NVDEC -> NVENC, output yuv420p): ~0.3 s CPU per video
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


task_map = json.load(open(args.task_map))
TASKS = [t["task"] for t in sorted(task_map["tasks"], key=lambda t: t["task_index"])]
assert [t["task_index"] for t in task_map["tasks"]] == list(range(len(TASKS))), "task_index must be 0..N-1"
raw_to_task = task_map["raw_to_task"]
exclude = json.load(open(args.exclude))

ref_info = json.load(open(args.ref / "meta/info.json"))
state_names = ref_info["features"]["observation.state"]["names"][:63]  # ref 63:65 (raw head) dropped
action_names = ref_info["features"]["action"]["names"]
assert state_names[37] == "left_hand_00" and state_names[50] == "right_hand_00", state_names[36:51]

for split in args.splits.split(","):
    s, d = args.src / split, args.dst / split
    src_info = json.load(open(s / "meta/info.json"))
    fps = src_info["fps"]
    src_tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(s / "meta/tasks.jsonl")}
    all_episodes = sorted((json.loads(l) for l in open(s / "meta/episodes.jsonl")), key=lambda e: e["episode_index"])
    drop = set(int(e) for e in exclude[split])
    episodes = [ep for ep in all_episodes if ep["episode_index"] not in drop]
    assert drop <= {ep["episode_index"] for ep in all_episodes}, sorted(drop - {ep["episode_index"] for ep in all_episodes})[:10]

    if args.skip_data:
        assert (d / "meta/info.json").exists(), d
    else:
        for sub in ("data", "meta"):
            if (d / sub).exists():
                assert args.overwrite, f"{d / sub} exists (pass --overwrite to replace it)"
                shutil.rmtree(d / sub)
        (d / "meta/episodes/chunk-000").mkdir(parents=True)

        def load(ep):
            e = ep["episode_index"]
            df = pd.read_parquet(s / src_info["data_path"].format(episode_chunk=e // src_info["chunks_size"], episode_index=e))
            assert len(df) == ep["length"] and (df["frame_index"].values == np.arange(len(df))).all(), e
            assert df["task_index"].nunique() == 1 and src_tasks[int(df["task_index"].iloc[0])] == ep["tasks"][0], e
            st = np.stack(df["observation.state"].values).astype(np.float64)
            act = np.stack(df["action"].values).astype(np.float64)
            assert st.shape[1] == 63 and act.shape[1] == 42, (e, st.shape, act.shape)
            lo, hi = HAND_STATE_VALID
            hands = np.concatenate([st[:, a:b] for a, b in STATE_HANDS], axis=1)
            n_bad = int(((hands < lo) | (hands > hi)).any(axis=1).sum())
            if n_bad:
                raise RuntimeError(f"[{split}] episode {e}: {n_bad} hand-state rows outside {HAND_STATE_VALID}; "
                                   f"add it to {args.exclude}")
            # only plain arrays leave the reader threads (the object columns of df are ~10x the data in memory)
            return (df["frame_index"].values.astype(np.int64), df["timestamp"].values.astype(np.float64),
                    df["next.done"].values.astype(bool), st, act)

        task_merge, pre_ch2 = {}, {"row": [], **{k: [] for k in ACTION_CH2}}
        ep_rows, all_state, all_action = [], [], []
        gidx = n_clipped = 0
        groups = [episodes[i:i + args.eps_per_file] for i in range(0, len(episodes), args.eps_per_file)]
        with ThreadPoolExecutor(args.read_threads) as pool:
            for file_index, (group, loaded) in enumerate(zip(groups, pool.map(lambda g: [load(ep) for ep in g], groups))):
                chunk_index = file_index // CHUNK
                (d / f"data/chunk-{chunk_index:03d}").mkdir(parents=True, exist_ok=True)
                frames = []
                for ep, (fi, ts, done, st, act) in zip(group, loaded):
                    e, n = ep["episode_index"], ep["length"]
                    text = ep["tasks"][0]
                    task_merge[e] = text
                    new_task = TASKS.index(raw_to_task[text])
                    clip = (act[:, list(ACTION_CH2.values())] > CH2_MAX).any(axis=1)
                    if clip.any():
                        pre_ch2["row"].append(gidx + np.where(clip)[0])
                        for k, c in ACTION_CH2.items():
                            pre_ch2[k].append(act[clip, c].copy())
                    for c in ACTION_CH2.values():
                        n_clipped += int((act[:, c] > CH2_MAX).sum())
                        act[:, c] = np.minimum(act[:, c], CH2_MAX)
                    frames.append(pd.DataFrame({
                        "episode_index": np.full(n, e, np.int64), "frame_index": fi, "timestamp": ts,
                        "index": np.arange(gidx, gidx + n, dtype=np.int64), "task_index": np.full(n, new_task, np.int64),
                        "next.done": done,
                        "observation.state": list(st), "action": list(act)}))
                    row = {"episode_index": e, "tasks": [new_task], "length": n, "dataset_from_index": gidx,
                           "dataset_to_index": gidx + n,
                           "data_path": f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}
                    for cam in CAMS:
                        k = f"videos/observation.images.{cam}_2/"
                        row.update({k + "chunk_index": e // CHUNK, k + "file_index": e, k + "from_timestamp": 0.0,
                                    k + "to_timestamp": n / fps})
                    row.update({"data/chunk_index": chunk_index, "data/file_index": file_index})
                    ep_rows.append(row)
                    all_state.append(st)
                    all_action.append(act)
                    gidx += n
                out = pd.concat(frames, ignore_index=True)
                schema = pa.schema([("episode_index", pa.int64()), ("frame_index", pa.int64()), ("timestamp", pa.float64()),
                                    ("index", pa.int64()), ("task_index", pa.int64()), ("next.done", pa.bool_()),
                                    ("observation.state", pa.list_(pa.float64())), ("action", pa.list_(pa.float64()))])
                pq.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False),
                               d / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
                if file_index % 1000 == 0:
                    print(f"[{split}] {file_index + 1}/{len(groups)} data files", flush=True)
        pd.DataFrame(ep_rows).to_parquet(d / "meta/episodes/chunk-000/file-000.parquet", index=False)
        pd.DataFrame({"task_index": np.arange(len(TASKS), dtype=np.int64)}, index=pd.Index(TASKS, name="task")).to_parquet(d / "meta/tasks.parquet")
        json.dump({str(k): v for k, v in task_merge.items()}, open(d / "meta/task_merge.json", "w"), indent=1)
        json.dump({"episodes": sorted(drop), "rule": exclude.get("rule", ""),
                   "extra": {k: v.get(split, []) if isinstance(v, dict) else v for k, v in exclude.get("extra", {}).items()}},
                  open(d / "meta/excluded_episodes.json", "w"), indent=1)
        np.savez(d / "meta/action_ch2_preclip.npz",
                 **{k: np.concatenate(v) if v else np.zeros(0) for k, v in pre_ch2.items()})
        counts = np.bincount([r["tasks"][0] for r in ep_rows], minlength=len(TASKS))
        print(f"[{split}] {len(episodes)} episodes kept ({len(drop)} dropped), {gidx} frames, {len(groups)} data files, "
              f"ch2 clipped {n_clipped}, tasks {counts.tolist()}", flush=True)

        # ---- meta ----
        info = {k: v for k, v in ref_info.items() if k not in ("source_episode_ids", "features")}
        info.update({"created_at": src_info["created_at"], "splits": {split: f"0:{len(episodes)}"}, "chunks_size": CHUNK,
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
        mod = json.load(open(args.ref / "meta/modality.json"))
        del mod["state"]["head_cmd"]
        mod["state"]["left_ee"].update(start=37, end=50)
        mod["state"]["right_ee"].update(start=50, end=63)
        assert set(mod["video"]) == {f"{c}_2" for c in CAMS}, mod["video"]
        json.dump(mod, open(d / "meta/modality.json", "w"), indent=2)
        stats = {"observation.state": stats_of(np.concatenate(all_state)), "action": stats_of(np.concatenate(all_action))}
        json.dump(stats, open(d / "meta/stats.json", "w"))
        json.dump({"source": str(args.src / split), "script": "examples/realRobots/H1_2_car_real/prepare_car1015_dataset.py",
                   "task_map": str(args.task_map), "exclude": str(args.exclude),
                   "episodes": {"source": len(all_episodes), "kept": len(episodes), "dropped": len(drop)},
                   "head_action": "action[t, 0:2] = raw head (operator head pose) at t, as exported",
                   "state": "63-D, 37:50 left hand, 50:63 right hand, no head_cmd", "tasks": TASKS,
                   "videos": "symlinks to the source mp4s (GOP 250)" if args.link_videos else f"re-encoded, GOP {args.gop}"},
                  open(d / "meta/car1015_prep_note.json", "w"), indent=1)

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
            jobs.append((src, vd / f"videos/observation.images.{cam}_2/chunk-{e // CHUNK:03d}/file-{e:03d}.mp4", ep["length"]))
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
            if n % 1000 == 0:
                print(f"[{split}] {n + 1}/{len(jobs)}", flush=True)
    print(f"[{split}] videos done, {len(failed)} failed", *failed[:10], sep="\n  ", flush=True)
