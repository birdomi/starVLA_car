"""Convert RMBench demo_clean episodes (XPolicyLab-format HDF5) into the LeRobot v2.0 layout consumed by
starVLA's gr00t_lerobot loader. Same output layout and 14-D convention as the RoboTwin converter
(../../Robotwin/train_files/convert_robotwin_to_lerobot.py), so the `robotwin` data config applies unchanged.

Input (after RMBench `bash scripts/_download_data.sh`):
    <RMBench>/data/demo_clean/<task>/aloha_agilex/data/episode_XXXXXXX.hdf5
    <RMBench>/data/demo_clean/<task>/aloha_agilex/instruction/episode_XXXXXXX.json   ({"seen": [...], "unseen": [...]})

State/action: the HDF5 stores both groups per frame and action[t] == state[t+1] (checked on battery_try), so every
frame is kept:  state[t] = state/*[t],  action[t] = action/*[t];
vector = [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)] -> 14-D, matches modality.json.

Images are decoded ONLY through XPolicyLab's decode_image_bit (RGB for both legacy and XPL-RGB1 JPEG bits; no channel
swap afterwards — see XPolicyLab/AGENTS.md) and written as rgb24 h264 mp4s.

Usage (starVLA env; needs pyarrow, av, h5py, cv2):
    python convert_rmbench_to_lerobot.py --task battery_try --out /mnt/DATA2/yyg/RMBench/lerobot/battery_try
    python convert_rmbench_to_lerobot.py --all --out_root /mnt/DATA2/yyg/RMBench/lerobot
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RMBENCH_ROOT = Path(os.environ.get("RMBENCH_ROOT", "/mnt/DATA2/yyg/RMBench"))
sys.path.insert(0, str(RMBENCH_ROOT))
from XPolicyLab.utils.process_data import decode_image_bit  # noqa: E402

TASKS = ["battery_try", "blocks_ranking_try", "cover_blocks", "observe_and_pickup", "place_block_mat",
         "press_button", "put_back_block", "rearrange_blocks", "swap_T", "swap_blocks"]
CAMERAS = {  # lerobot key -> XPolicyLab camera group
    "observation.images.cam_high": "cam_head",
    "observation.images.cam_left_wrist": "cam_left_wrist",
    "observation.images.cam_right_wrist": "cam_right_wrist",
}
PARTS = ["left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states"]
STATE_DIM = 14
MODALITY_JSON = Path(__file__).resolve().parents[2] / "Robotwin" / "train_files" / "modality.json"


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
            for pkt in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)


def convert_episode(args):
    ep_idx, hdf5_path, instr_path, out_dir, fps, seed = args
    rng = np.random.default_rng(seed + ep_idx)
    with open(instr_path) as f:
        instruction = str(rng.choice(json.load(f)["seen"]))

    with h5py.File(hdf5_path, "r") as h:
        state = np.concatenate([h[f"state/{p}"][:] for p in PARTS], axis=1).astype(np.float32)
        action = np.concatenate([h[f"action/{p}"][:] for p in PARTS], axis=1).astype(np.float32)
        assert state.shape == action.shape and state.shape[1] == STATE_DIM, (state.shape, action.shape)
        imgs = {lk: np.stack([decode_image_bit(b) for b in h[f"vision/{cam}/colors"][:]]) for lk, cam in CAMERAS.items()}
    T = state.shape[0]
    hw = None
    for lk, arr in imgs.items():
        assert arr.shape[0] == T, (lk, arr.shape, T)
        hw = arr.shape[1:3]
        write_video(arr, out_dir / "videos" / "chunk-000" / lk / f"episode_{ep_idx:06d}.mp4", fps)

    table = pa.table({
        "observation.state": pa.array(list(state), type=pa.list_(pa.float32(), STATE_DIM)),
        "action": pa.array(list(action), type=pa.list_(pa.float32(), STATE_DIM)),
        "timestamp": np.arange(T, dtype=np.float64) / fps,
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


def convert_task(task: str, out: Path, fps: int, workers: int, seed: int, limit=None, robot_type="aloha-agilex"):
    if (out / "meta" / "info.json").exists():
        print(f"[skip] {out} already converted")
        return
    src = RMBENCH_ROOT / "data" / "demo_clean" / task / "aloha_agilex"
    hdf5s = sorted((src / "data").glob("episode_*.hdf5"))
    if limit:
        hdf5s = hdf5s[:limit]
    assert hdf5s, f"no episodes under {src}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    jobs = [(i, str(hp), str(src / "instruction" / f"{hp.stem}.json"), out, fps, seed) for i, hp in enumerate(hdf5s)]
    results = {}
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(convert_episode, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs)):
            ep, T, instr, hw = fu.result()
            results[ep] = (T, instr, hw)
            if (i + 1) % 25 == 0 or i + 1 == len(futs):
                print(f"  [{task}] {i + 1}/{len(futs)} episodes", flush=True)

    # tasks / episodes meta + fill index & task_index columns
    tasks, task_to_idx = [], {}
    episodes, total = [], 0
    for ep in sorted(results):
        T, instr, hw = results[ep]
        if instr not in task_to_idx:
            task_to_idx[instr] = len(tasks)
            tasks.append(instr)
        p = out / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        t = pq.read_table(p)
        t = t.set_column(t.schema.get_field_index("index"), "index", pa.array(np.arange(total, total + T, dtype=np.int64)))
        t = t.set_column(t.schema.get_field_index("task_index"), "task_index",
                         pa.array(np.full(T, task_to_idx[instr], dtype=np.int64)))
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
                     "video_info": {"video.fps": float(fps), "video.codec": "h264", "video.pix_fmt": "yuv420p",
                                    "video.is_depth_map": False, "has_audio": False}}
    feats["observation.state"] = {"dtype": "float32", "shape": [STATE_DIM]}
    feats["action"] = {"dtype": "float32", "shape": [STATE_DIM]}
    for k, dt in [("timestamp", "float64"), ("frame_index", "int64"), ("episode_index", "int64"),
                  ("index", "int64"), ("task_index", "int64"), ("next.done", "bool")]:
        feats[k] = {"dtype": dt, "shape": [1]}
    info = {
        "codebase_version": "v2.0", "robot_type": robot_type,
        "total_episodes": len(episodes), "total_frames": total, "total_tasks": len(tasks),
        "total_videos": len(episodes) * len(CAMERAS), "total_chunks": 1, "chunks_size": 1000,
        "fps": float(fps), "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feats,
    }
    with open(meta / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[done] {out}: {len(episodes)} episodes, {total} frames, {len(tasks)} instructions, {H}x{W}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="one RMBench task")
    ap.add_argument("--out", help="output dir for --task")
    ap.add_argument("--all", action="store_true", help="convert every task in TASKS into --out_root/<task>")
    ap.add_argument("--out_root", default=str(RMBENCH_ROOT / "lerobot"))
    ap.add_argument("--fps", type=int, default=15)  # additional_info/frequency in the HDF5
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="convert only first N episodes (debug)")
    a = ap.parse_args()
    if a.all:
        for task in TASKS:
            convert_task(task, Path(a.out_root) / task, a.fps, a.workers, a.seed, a.limit)
    else:
        assert a.task and a.out, "--task and --out (or --all)"
        convert_task(a.task, Path(a.out), a.fps, a.workers, a.seed, a.limit)


if __name__ == "__main__":
    main()
