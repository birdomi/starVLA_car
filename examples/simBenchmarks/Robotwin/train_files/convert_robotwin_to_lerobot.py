"""Convert raw RoboTwin 2.0 episodes (per-episode HDF5 + instructions JSON) into the
LeRobot v2.0 layout consumed by starVLA's gr00t_lerobot loader.

No `lerobot` package dependency: parquet via pyarrow, mp4 via PyAV (libx264).

Output layout (one dataset per <task>/<setting>):
    <out>/meta/{info.json,episodes.jsonl,tasks.jsonl,modality.json}
    <out>/data/chunk-000/episode_XXXXXX.parquet
    <out>/videos/chunk-000/observation.images.{cam_high,cam_left_wrist,cam_right_wrist}/episode_XXXXXX.mp4

State/action convention follows RoboTwin's own policy converters (GO1/pi0):
    state[t]  = joint_action/vector[t]      (t = 0 .. T-2)
    action[t] = joint_action/vector[t+1]
vector = [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]  -> 14-D, matches modality.json.

Usage:
    python convert_robotwin_to_lerobot.py --src <extracted_dir_with_data_and_instructions> \
        --out playground/Datasets/RoboTwin/Clean/adjust_bottle [--fps 15] [--workers 16]
    python convert_robotwin_to_lerobot.py --zip dataset/adjust_bottle/aloha-agilex_clean_50.zip --out ...
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = {  # lerobot key -> raw hdf5 camera group
    "observation.images.cam_high": "head_camera",
    "observation.images.cam_left_wrist": "left_camera",
    "observation.images.cam_right_wrist": "right_camera",
}
STATE_DIM = 14
MODALITY_JSON = Path(__file__).resolve().parent / "modality.json"


def write_video(frames: np.ndarray, path: Path, fps: int, crf: int = 18):
    """frames: (T, H, W, 3) RGB uint8 -> h264/yuv420p mp4 with constant fps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, _ = frames.shape
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = W, H
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "fast", "g": "10"}
        for f in frames:
            vf = av.VideoFrame.from_ndarray(f, format="rgb24")
            for pkt in stream.encode(vf):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)


def convert_episode(args):
    ep_idx, hdf5_path, instr_path, out_dir, fps, seed, task_offset = args
    rng = np.random.default_rng(seed + ep_idx)
    with open(instr_path) as f:
        instructions = json.load(f)["seen"]
    instruction = str(rng.choice(instructions))

    with h5py.File(hdf5_path, "r") as h:
        vec = h["joint_action/vector"][:].astype(np.float32)  # (T, 14)
        assert vec.shape[1] == STATE_DIM, vec.shape
        imgs = {}
        for lk, cam in CAMERAS.items():
            raw = h[f"observation/{cam}/rgb"][:]
            dec = [cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for b in raw]
            imgs[lk] = np.stack(dec)
    T = vec.shape[0] - 1  # drop last frame (no next action)
    state, action = vec[:-1], vec[1:]
    hw = None
    for lk, arr in imgs.items():
        arr = arr[:T]
        hw = arr.shape[1:3]
        write_video(arr, out_dir / "videos" / "chunk-000" / lk / f"episode_{ep_idx:06d}.mp4", fps)

    ts = np.arange(T, dtype=np.float64) / fps
    table = pa.table({
        "observation.state": pa.array(list(state), type=pa.list_(pa.float32(), STATE_DIM)),
        "action": pa.array(list(action), type=pa.list_(pa.float32(), STATE_DIM)),
        "timestamp": ts,
        "frame_index": np.arange(T, dtype=np.int64),
        "episode_index": np.full(T, ep_idx, dtype=np.int64),
        "index": np.zeros(T, dtype=np.int64),  # filled after global offset known
        "task_index": np.zeros(T, dtype=np.int64),  # filled by main
        "next.done": np.array([False] * (T - 1) + [True]),
    })
    pdir = out_dir / "data" / "chunk-000"
    pdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, pdir / f"episode_{ep_idx:06d}.parquet")
    return ep_idx, T, instruction, hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="extracted dir containing data/ and instructions/")
    ap.add_argument("--zip", help="raw RoboTwin zip; extracted to a temp dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robot_type", default="aloha-agilex")
    ap.add_argument("--limit", type=int, default=None, help="convert only first N episodes (debug)")
    a = ap.parse_args()

    out = Path(a.out)
    if (out / "meta" / "info.json").exists():
        print(f"[skip] {out} already converted")
        return
    tmp = None
    src = a.src
    if a.zip:
        tmp = tempfile.mkdtemp(prefix="robotwin_extract_", dir=os.environ.get("ROBOTWIN_EXTRACT_TMP", None))
        subprocess.run(["unzip", "-q", a.zip, "-d", tmp], check=True)
        subs = [p for p in Path(tmp).iterdir() if p.is_dir()]
        src = str(subs[0]) if len(subs) == 1 else tmp
    src = Path(src)
    hdf5s = sorted((src / "data").glob("episode*.hdf5"), key=lambda p: int(p.stem.replace("episode", "")))
    if a.limit:
        hdf5s = hdf5s[: a.limit]
    assert hdf5s, f"no episodes under {src}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    jobs = []
    for new_idx, hp in enumerate(hdf5s):
        raw_idx = int(hp.stem.replace("episode", ""))
        jobs.append((new_idx, str(hp), str(src / "instructions" / f"episode{raw_idx}.json"), out, a.fps, a.seed, 0))
    results = {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(convert_episode, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs)):
            ep, T, instr, hw = fu.result()
            results[ep] = (T, instr, hw)
            if (i + 1) % 25 == 0 or i + 1 == len(futs):
                print(f"  {i + 1}/{len(futs)} episodes", flush=True)

    # tasks / episodes meta + fill index & task_index columns
    tasks, task_to_idx = [], {}
    episodes, total = [], 0
    for ep in sorted(results):
        T, instr, hw = results[ep]
        if instr not in task_to_idx:
            task_to_idx[instr] = len(tasks)
            tasks.append(instr)
        ti = task_to_idx[instr]
        p = out / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        t = pq.read_table(p)
        t = t.set_column(t.schema.get_field_index("index"), "index", pa.array(np.arange(total, total + T, dtype=np.int64)))
        t = t.set_column(t.schema.get_field_index("task_index"), "task_index", pa.array(np.full(T, ti, dtype=np.int64)))
        pq.write_table(t, p)
        episodes.append({"episode_index": ep, "tasks": [instr], "length": T})
        total += T
    H, W = results[0][2]

    meta = out / "meta"
    meta.mkdir()
    with open(meta / "episodes.jsonl", "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
    with open(meta / "tasks.jsonl", "w") as f:
        for i, t in enumerate(tasks):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    shutil.copy(MODALITY_JSON, meta / "modality.json")
    feats = {}
    for lk in CAMERAS:
        feats[lk] = {"dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channel"],
                     "video_info": {"video.fps": float(a.fps), "video.codec": "h264", "video.pix_fmt": "yuv420p",
                                    "video.is_depth_map": False, "has_audio": False}}
    feats["observation.state"] = {"dtype": "float32", "shape": [STATE_DIM]}
    feats["action"] = {"dtype": "float32", "shape": [STATE_DIM]}
    for k, dt in [("timestamp", "float64"), ("frame_index", "int64"), ("episode_index", "int64"),
                  ("index", "int64"), ("task_index", "int64"), ("next.done", "bool")]:
        feats[k] = {"dtype": dt, "shape": [1]}
    info = {
        "codebase_version": "v2.0", "robot_type": a.robot_type,
        "total_episodes": len(episodes), "total_frames": total, "total_tasks": len(tasks),
        "total_videos": len(episodes) * len(CAMERAS), "total_chunks": 1, "chunks_size": 1000,
        "fps": float(a.fps), "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feats,
    }
    with open(meta / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"[done] {out}: {len(episodes)} episodes, {total} frames, {len(tasks)} tasks, {H}x{W}")


if __name__ == "__main__":
    main()
