#!/usr/bin/env python3
"""AgiBot World 2026 episodes (agibot_index.py rows: AV1 top_head / hand_left / hand_right at 30 fps) -> per-episode
frame files in examples/realRobots/EgoDex/tools/egodex_frames.py's format, read by starVLA/dataloader/egodex_stream.py,
with the wrist cameras alongside:
  <dst>/<row path>.npz:
    jpeg, offsets                         top_head, every --step-th frame (10 fps), resized to --size (448x256)
    jpeg_left_wrist, offsets_left_wrist   hand_left, same frames, --wrist-size (312x256: the 640x528 aspect kept)
    jpeg_right_wrist, offsets_right_wrist hand_right
    src_frames                            source frame count
  <dst>/index_agibot_{train,test}_<k>of<n>.jsonl: {"path", "split", "theme", "task", "episode", "n": frames kept,
    "fps", "src_frames", "description", "cams": ["main", "left_wrist", "right_wrist"]}
The path is the index row's ("agibot/<dataset dir>/<episode>.npz"), the key of the Flora / native subtask files. Held
out: every episode whose path hashes to 0 mod --test-every (5% by default), the same split on every run.

  agibot_frames.py <index_agibot.jsonl> <dst> [--shard k --num-shards n] [--workers 64]
Resumable: episodes whose .npz exists are only re-indexed.
"""
import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

CAMS = (("video", ""), ("left_wrist", "_left_wrist"), ("right_wrist", "_right_wrist"))


def split_of(path: str, every: int) -> str:
    return "agibot_test" if int(hashlib.md5(path.encode()).hexdigest(), 16) % every == 0 else "agibot_train"


def convert(job):
    row, dst, step, size, wrist_size = job
    out = Path(dst) / row["path"]
    try:
        if out.exists():
            with np.load(out) as z:
                return row, int(z["offsets"].shape[0] - 1), int(z["src_frames"]), None
        import av
        from PIL import Image

        chunks = {k: [] for k, _ in CAMS}
        conts = [av.open(row[k]) for k, _ in CAMS]
        try:
            streams = []
            for c in conts:
                s = c.streams.video[0]
                s.codec_context.thread_count = 1
                streams.append(c.decode(s))
            src = 0
            for frames in zip(*streams):  # the three cameras share one clock (checked by the download)
                if src % step == 0:
                    for (k, _), fr in zip(CAMS, frames):
                        img = fr.to_image().resize(tuple(size if k == "video" else wrist_size), Image.BILINEAR)
                        b = io.BytesIO()
                        img.save(b, format="JPEG", quality=90)
                        chunks[k].append(np.frombuffer(b.getvalue(), dtype=np.uint8))
                src += 1
        finally:
            for c in conts:
                c.close()
        n = len(chunks["video"])
        if n == 0:
            return row, 0, src, "no frames"
        arrays = {"src_frames": np.int64(src)}
        for k, suffix in CAMS:
            off = np.zeros(n + 1, dtype=np.int64)
            off[1:] = np.cumsum([len(c) for c in chunks[k]])
            arrays["jpeg" + suffix], arrays["offsets" + suffix] = np.concatenate(chunks[k]), off
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, out)
        return row, n, src, None
    except Exception as err:  # noqa: BLE001 - logged, the episode left out of the index
        return row, 0, 0, f"{type(err).__name__}: {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--size", type=int, nargs=2, default=[448, 256])
    ap.add_argument("--wrist-size", type=int, nargs=2, default=[312, 256])
    ap.add_argument("--test-every", type=int, default=20)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.index)][a.shard::a.num_shards]
    a.dst.mkdir(parents=True, exist_ok=True)
    jobs = [(r, str(a.dst), a.step, a.size, a.wrist_size) for r in rows]
    tag = f"_{a.shard}of{a.num_shards}"
    out = {s: [] for s in ("agibot_train", "agibot_test")}
    errors = 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=64) as pool:
        for k, (r, n, src, err) in enumerate(pool.imap_unordered(convert, jobs, chunksize=1), 1):
            if err or n == 0:
                errors += 1
                print(f"[agibot_frames] {r['path']}: {err}", flush=True)
            else:
                s = split_of(r["path"], a.test_every)
                out[s].append({"path": r["path"], "split": s, "theme": r["theme"], "task": r["task"],
                               "episode": r["episode"], "n": n, "fps": float(r["fps"]) / a.step, "src_frames": src,
                               "description": r["description"], "cams": ["main", "left_wrist", "right_wrist"]})
            if k % 500 == 0 or k == len(jobs):
                print(f"[agibot_frames] shard {a.shard}: {k}/{len(jobs)} ({errors} errors)", flush=True)
    for s, rs in out.items():
        with open(a.dst / f"index_{s}{tag}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(rs, key=lambda r: r["path"]))
    print(f"[agibot_frames] shard {a.shard}: train {len(out['agibot_train'])}, test {len(out['agibot_test'])}, "
          f"{errors} errors -> {a.dst}", flush=True)


if __name__ == "__main__":
    main()
