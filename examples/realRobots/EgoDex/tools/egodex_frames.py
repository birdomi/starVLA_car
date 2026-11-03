#!/usr/bin/env python3
"""Raw EgoDex (Apple; <split>/<task>/<N>.mp4 + <N>.hdf5, 1080p 30 fps) -> small per-episode frame files for the
episode-memory pretraining (starVLA/training/train_memory_pretrain.py).

Only images, the task description and time are needed there (no hand poses), so each episode becomes one
<dst>/<split>/<task>/<N>.npz:
  jpeg     uint8 [total bytes]  the frames' JPEGs back to back (every `--step`-th source frame, resized to --size)
  offsets  int64 [n + 1]        frame i = jpeg[offsets[i]:offsets[i + 1]]
and every converted episode one line of <dst>/index_<split>.jsonl:
  {"path": "<split>/<task>/<N>.npz", "split", "task", "episode", "n": frames kept, "fps": 30 / step,
   "src_frames": source frame count, "description": the hdf5's llm_description (fallback: the task name)}

Usage:
  egodex_frames.py <raw_root> <dst_root> --splits test part1 ... [--workers 64] [--step 3] [--size 448 256]
                   [--shard k --num-shards n]   (episode k, k + n, ... of every split; index_<split>_<k>of<n>.jsonl)
A split is read from <raw_root>/<split>/ or, when that is missing, straight from the download <raw_root>/<split>.zip
(no extraction). Resumable: episodes whose .npz exists are only re-indexed.
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import zipfile
from pathlib import Path

import numpy as np


_ZIPS = {}


def _src(zip_path, name: str):
    """A file of the episode: its path, or an in-memory copy of the zip member."""
    if zip_path is None:
        return name
    if zip_path not in _ZIPS:  # one handle per worker process
        _ZIPS[zip_path] = zipfile.ZipFile(zip_path)
    return io.BytesIO(_ZIPS[zip_path].read(name))


def _description(h5, task: str) -> str:
    try:
        import h5py

        with h5py.File(h5, "r") as f:
            attrs = dict(f.attrs)
        # EgoDex: llm_description (+ llm_description2 for reversible tasks, picked by which_llm_description)
        key = "llm_description2" if int(attrs.get("which_llm_description", 1)) == 2 and "llm_description2" in attrs \
            else "llm_description"
        d = attrs.get(key, None)
        if isinstance(d, bytes):
            d = d.decode()
        if d:
            return str(d).strip()
    except Exception as e:  # noqa: BLE001 - a broken hdf5 only loses the description
        print(f"[warn] {task}: {e}", file=sys.stderr)
    return task.replace("_", " ")


def convert_one(args):
    zip_path, src_mp4, dst_npz, rel, split, task, ep, step, size = args
    import av
    from PIL import Image

    dst_npz = Path(dst_npz)
    if dst_npz.exists():
        with np.load(dst_npz) as z:
            n = int(z["offsets"].shape[0] - 1)
            src_frames = int(z["src_frames"]) if "src_frames" in z else -1
    else:
        chunks, src_frames = [], 0
        try:
            with av.open(_src(zip_path, src_mp4)) as c:
                s = c.streams.video[0]
                # one decoder thread: the parallelism is the worker processes (many decoder threads per worker hit
                # the login node's virtual memory cap and, with ~200 workers, the thread limit: EAGAIN)
                s.codec_context.thread_count = 1
                for i, fr in enumerate(c.decode(s)):
                    src_frames += 1
                    if i % step:
                        continue
                    img = fr.to_image().resize(tuple(size), Image.BILINEAR)
                    b = io.BytesIO()
                    img.save(b, format="JPEG", quality=90)
                    chunks.append(np.frombuffer(b.getvalue(), dtype=np.uint8))
        except Exception as e:  # noqa: BLE001 - a broken video only loses its episode
            print(f"[warn] {src_mp4}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            return None
        if not chunks:
            return None
        n = len(chunks)
        offsets = np.zeros(n + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(c) for c in chunks])
        dst_npz.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst_npz.with_suffix(".tmp.npz")
        np.savez(tmp, jpeg=np.concatenate(chunks), offsets=offsets, src_frames=np.int64(src_frames))
        os.replace(tmp, dst_npz)
    return {"path": rel, "split": split, "task": task, "episode": ep, "n": n, "fps": 30.0 / step,
            "src_frames": src_frames, "description": _description(_src(zip_path, src_mp4[:-4] + ".hdf5"), task)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_root", type=Path)
    ap.add_argument("dst_root", type=Path)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--step", type=int, default=3, help="keep every step-th 30 fps frame (3 -> 10 fps)")
    ap.add_argument("--size", type=int, nargs=2, default=[448, 256], help="W H")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    for split in a.splits:
        if (a.raw_root / split).is_dir():
            zip_path, mp4s = None, [str(p) for p in sorted((a.raw_root / split).glob("*/*.mp4"))]
        else:
            zip_path = str(a.raw_root / f"{split}.zip")
            with zipfile.ZipFile(zip_path) as z:
                mp4s = sorted(n for n in z.namelist() if n.endswith(".mp4") and n.count("/") == 2)
        jobs = []
        for mp4 in mp4s:
            task, ep = Path(mp4).parent.name, int(Path(mp4).stem)
            rel = f"{split}/{task}/{ep}.npz"
            jobs.append((zip_path, mp4, str(a.dst_root / rel), rel, split, task, ep, a.step, a.size))
        jobs = jobs[a.shard::a.num_shards]
        if a.limit:
            jobs = jobs[: a.limit]
        print(f"[{split}] {len(jobs)} episodes", flush=True)
        rows, bad = [], 0
        # workers are recycled: their memory grows with the episodes decoded (~2 GB each after a few hundred)
        with mp.get_context("fork").Pool(a.workers, maxtasksperchild=64) as pool:
            for k, r in enumerate(pool.imap(convert_one, jobs, chunksize=4)):
                if r is None:
                    bad += 1
                else:
                    rows.append(r)
                if (k + 1) % 1000 == 0:
                    print(f"[{split}] {k + 1}/{len(jobs)}", flush=True)
        a.dst_root.mkdir(parents=True, exist_ok=True)
        tag = "" if a.num_shards == 1 else f"_{a.shard}of{a.num_shards}"
        with open(a.dst_root / f"index_{split}{tag}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        hours = sum(r["n"] for r in rows) / (30.0 / a.step) / 3600
        print(f"[{split}] done: {len(rows)} episodes ({bad} empty), {hours:.1f} h", flush=True)


if __name__ == "__main__":
    main()
