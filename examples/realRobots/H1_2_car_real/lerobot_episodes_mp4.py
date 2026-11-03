#!/usr/bin/env python3
"""LeRobot v3 split (videos concatenated per camera in chunk files) -> one mp4 per episode for one camera, laid out as
EgoDex raw data (<out>/<split>/task<k>/<episode>.mp4), so examples/realRobots/EgoDex/tools/egodex_frames.py and
flora_subtasks.py read it as they read EgoDex. Also writes <out>/<split>_episodes.jsonl: {"path": "<split>/task<k>/<ep>",
"episode", "task_index", "description", "frames", "fps"} (the task text is the description; there is no hdf5).

  lerobot_episodes_mp4.py /appsdata/t509a04/data/car-1015-v3-ep/val /scratch/t509a04/data/car1015_val/raw \
      --split carval --camera observation.images.head_left_2 [--workers 32]
Frames are re-encoded (H.264, CRF 16, the source size and fps) from the episode's [from, to) timestamps; resumable.
"""
import argparse
import glob
import json
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd


def cut(args):
    src, dst, t0, t1, fps, n = args
    import av

    dst = Path(dst)
    if dst.exists():
        return dst.name, -1
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    written = 0
    with av.open(src) as cin, av.open(str(tmp), "w") as cout:
        sin = cin.streams.video[0]
        sin.codec_context.thread_count = 2
        cin.seek(max(0, int((t0 - 1.0) / sin.time_base)), stream=sin, backward=True)
        sout = cout.add_stream("libx264", rate=int(round(fps)))
        sout.width, sout.height, sout.pix_fmt = sin.codec_context.width, sin.codec_context.height, "yuv420p"
        sout.options = {"crf": "16", "preset": "fast", "threads": "2"}  # x264 takes every core by default
        sout.codec_context.thread_count = 2
        for fr in cin.decode(sin):
            t = float(fr.pts * sin.time_base)
            if t < t0 - 0.5 / fps:
                continue
            if t >= t1 - 0.5 / fps or written >= n:
                break
            fr.pts, fr.time_base = written, Fraction(1, int(round(fps)))  # renumbered from 0 at the output rate
            for p in sout.encode(fr):
                cout.mux(p)
            written += 1
        for p in sout.encode():
            cout.mux(p)
    tmp.rename(dst)
    return dst.name, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--split", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    info = json.loads((a.root / "meta/info.json").read_text())
    fps = float(info["fps"])
    tasks = pd.read_parquet(a.root / "meta/tasks.parquet")
    task_index = {t: int(i) for t, i in zip(tasks.index, tasks["task_index"])}
    task_text = {i: t for t, i in task_index.items()}
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(a.root / "meta/episodes/*/*.parquet")))])
    c = a.camera
    jobs, rows = [], []
    for _, e in eps.iterrows():
        task = list(e["tasks"])[0]  # the task text, or (this conversion) its task_index
        k = int(task) if not isinstance(task, str) else task_index[task]
        task = task_text[k]
        rel = f"{a.split}/task{k:02d}/{int(e['episode_index'])}"
        src = a.root / info["video_path"].format(video_key=c, chunk_index=int(e[f"videos/{c}/chunk_index"]),
                                                file_index=int(e[f"videos/{c}/file_index"]))
        jobs.append((str(src), str(a.out / (rel + ".mp4")), float(e[f"videos/{c}/from_timestamp"]),
                     float(e[f"videos/{c}/to_timestamp"]), fps, int(e["length"])))
        rows.append({"path": rel, "episode": int(e["episode_index"]), "task_index": k, "description": task,
                     "frames": int(e["length"]), "fps": fps})
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / f"{a.split}_episodes.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    bad = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for (name, written), j in zip(ex.map(cut, jobs), jobs):
            if written >= 0 and written != j[5]:
                bad += 1
                print(f"[warn] {name}: {written} frames, expected {j[5]}", flush=True)
    print(f"{len(jobs)} episodes -> {a.out}/{a.split} ({bad} with a frame count off)", flush=True)


if __name__ == "__main__":
    main()
