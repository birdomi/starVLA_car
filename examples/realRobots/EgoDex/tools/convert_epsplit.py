#!/usr/bin/env python3
"""Repackage LeRobot v3.0 merged videos into per-episode mp4s.

Why: EgoDex merges ~150k frames into each mp4, so every random sample pays
~300ms of container open/index cost. Per-episode files (like LIBERO) make
open cost negligible.

Usage:
  convert_epsplit.py convert <src_root> <dst_root> [--workers 60] [--limit N]
  convert_epsplit.py verify  <src_root> <dst_root> [--samples 20]

convert:
  - videos/<key>/chunk-{ep//1000:03d}/file-{ep%1000:03d}.mp4, one per episode,
    re-encoded (libx264 veryfast crf20 g30) with exact frame counts
  - meta/episodes/*.parquet rewritten: video chunk/file indices remapped,
    from_timestamp=0, to_timestamp=length/fps
  - data/ symlinked; other meta files copied
verify:
  - random episodes: frame count check, open-time benchmark, PSNR old vs new
"""

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

VIDEO_PATH_TPL = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def load_episodes(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    assert files, f"no episodes parquet under {root}"
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["_src_parquet"] = np.repeat([str(f.relative_to(root)) for f in files], [len(d) for d in dfs])
    return df.sort_values("episode_index").reset_index(drop=True)


def video_keys(df: pd.DataFrame) -> list[str]:
    return sorted({c.split("/")[1] for c in df.columns if c.startswith("videos/") and c.endswith("/chunk_index")})


def cut_one(args):
    src, dst, from_ts, n_frames = args
    dst = Path(dst)
    if dst.exists():  # resumable
        return (str(dst), True, "exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{from_ts:.6f}", "-i", str(src),
        "-frames:v", str(int(n_frames)),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", "30",
        "-pix_fmt", "yuv420p", "-an", "-threads", "2",
        str(tmp),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return (str(dst), False, r.stderr[-300:])
    tmp.rename(dst)
    return (str(dst), True, "ok")


def cmd_convert(a):
    src_root, dst_root = Path(a.src_root), Path(a.dst_root)
    fps = json.load(open(src_root / "meta" / "info.json"))["fps"]
    df = load_episodes(src_root)
    keys = video_keys(df)
    if a.limit:
        df = df.iloc[: a.limit].copy()
    print(f"{src_root.name}: {len(df)} episodes, video keys {keys}, fps {fps}")

    # 1) build ffmpeg job list
    jobs = []
    for _, ep in df.iterrows():
        ep_idx = int(ep["episode_index"])
        for k in keys:
            src = src_root / VIDEO_PATH_TPL.format(
                video_key=k, chunk_index=int(ep[f"videos/{k}/chunk_index"]),
                file_index=int(ep[f"videos/{k}/file_index"]))
            dst = dst_root / VIDEO_PATH_TPL.format(
                video_key=k, chunk_index=ep_idx // 1000, file_index=ep_idx % 1000)
            jobs.append((str(src), str(dst), float(ep[f"videos/{k}/from_timestamp"]), int(ep["length"])))

    # 2) run ffmpeg pool
    done = fail = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(cut_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            _, ok, msg = f.result()
            done += ok
            fail += not ok
            if not ok:
                print("FAIL:", msg, file=sys.stderr)
            if i % 500 == 0 or i == len(jobs):
                print(f"  [{src_root.name}] {i}/{len(jobs)} cut ({fail} failed)", flush=True)
    if fail:
        sys.exit(f"{fail} episodes failed; not writing metadata")

    # 3) rewrite episodes parquet (per source file, preserving layout)
    if not a.limit:
        for rel, part in df.groupby("_src_parquet", sort=True):
            part = part.copy()
            for k in keys:
                part[f"videos/{k}/chunk_index"] = (part["episode_index"] // 1000).astype(part[f"videos/{k}/chunk_index"].dtype)
                part[f"videos/{k}/file_index"] = (part["episode_index"] % 1000).astype(part[f"videos/{k}/file_index"].dtype)
                part[f"videos/{k}/from_timestamp"] = 0.0
                part[f"videos/{k}/to_timestamp"] = part["length"] / fps
            out = dst_root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            part.drop(columns=["_src_parquet"]).to_parquet(out, index=False)

        # 4) data/ symlink + copy remaining meta files
        if not (dst_root / "data").exists():
            (dst_root / "data").symlink_to((src_root / "data").resolve())
        for f in (src_root / "meta").iterdir():
            if f.name == "episodes":
                continue
            tgt = dst_root / "meta" / f.name
            if not tgt.exists():
                shutil.copy2(f, tgt) if f.is_file() else shutil.copytree(f, tgt)
    print(f"DONE {src_root.name}: {done} cut, {fail} failed")


def cmd_verify(a):
    import decord
    import time, random
    src_root, dst_root = Path(a.src_root), Path(a.dst_root)
    fps = json.load(open(src_root / "meta" / "info.json"))["fps"]
    df = load_episodes(src_root)
    keys = video_keys(df)
    random.seed(0)
    rows = df.sample(min(a.samples, len(df)), random_state=0)
    open_times, psnrs, bad = [], [], 0
    for _, ep in rows.iterrows():
        ep_idx = int(ep["episode_index"])
        k = keys[0]
        src = src_root / VIDEO_PATH_TPL.format(video_key=k, chunk_index=int(ep[f"videos/{k}/chunk_index"]),
                                               file_index=int(ep[f"videos/{k}/file_index"]))
        dst = dst_root / VIDEO_PATH_TPL.format(video_key=k, chunk_index=ep_idx // 1000, file_index=ep_idx % 1000)
        if not dst.exists():
            print("missing:", dst); bad += 1; continue
        t0 = time.perf_counter(); vr_new = decord.VideoReader(str(dst)); open_times.append(time.perf_counter() - t0)
        n_new, n_exp = len(vr_new), int(ep["length"])
        if n_new != n_exp:
            print(f"frame count mismatch ep{ep_idx}: {n_new} vs {n_exp}"); bad += 1; continue
        # compare a middle frame old vs new
        fi = n_exp // 2
        vr_old = decord.VideoReader(str(src))
        old_f = vr_old[int(round(float(ep[f"videos/{k}/from_timestamp"]) * fps)) + fi].asnumpy().astype(np.float64)
        new_f = vr_new[fi].asnumpy().astype(np.float64)
        mse = ((old_f - new_f) ** 2).mean()
        psnrs.append(10 * np.log10(255**2 / mse) if mse > 0 else 99.0)
    print(f"open time: mean {np.mean(open_times)*1000:.1f}ms | PSNR: mean {np.mean(psnrs):.1f}dB min {np.min(psnrs):.1f}dB | bad: {bad}/{len(rows)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("convert", cmd_convert), ("verify", cmd_verify)):
        s = sub.add_parser(name)
        s.add_argument("src_root"); s.add_argument("dst_root")
        if name == "convert":
            s.add_argument("--workers", type=int, default=60)
            s.add_argument("--limit", type=int, default=0)
        else:
            s.add_argument("--samples", type=int, default=20)
        s.set_defaults(fn=fn)
    a = p.parse_args()
    a.fn(a)
