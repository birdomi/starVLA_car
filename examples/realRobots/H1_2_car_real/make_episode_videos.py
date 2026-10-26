"""Split a LeRobot v3 car dataset's concatenated videos into one mp4 per episode (short GOP).

The car exports concatenate many separately-encoded episodes into each file-XXX.mp4 with a
150-frame GOP; some files decode with errors from some episode onward (mismatched stream
parameters), and random access costs ~75 decoded frames. This writes, for each camera in --cams
and each episode NOT listed as corrupted (scan_video_episodes.py), exactly `length` frames
starting at the episode's from_timestamp into videos/<cam>/chunk-000/file-{episode_index}.mp4,
re-encoded with h264_nvenc (GOP --gop, no B-frames, cq 19 ~ 45 dB PSNR), then rewrites
meta/episodes so each episode points at its own file (from 0, to length/fps). Cameras not in
--cams keep their original files (symlinked). meta/ and data/ are copied.

Usage: python make_episode_videos.py <src_root> <dst_root> --cams a,b,c --bad-dir <dir with *_{split}_bad_ids.json>
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("dst", type=Path)
ap.add_argument("--cams", required=True)
ap.add_argument("--bad-dir", type=Path, required=True)
ap.add_argument("--gpus", default="0,1,2,3")
ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--gop", type=int, default=15)
args = ap.parse_args()
cams = args.cams.split(",")
gpus = args.gpus.split(",")
name = args.src.name


def n_packets(f):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(f)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def encode(job):
    i, (src, dst, start, length) = job
    if dst.exists() and n_packets(dst) == length:
        return "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[i % len(gpus)])
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-hwaccel", "cuda", "-ss", f"{start:.6f}", "-i", str(src),
                    "-frames:v", str(length), "-map", "0:v:0", "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
                    "-rc", "vbr", "-cq", "19", "-b:v", "0", "-g", str(args.gop), "-bf", "0", "-forced-idr", "1",
                    "-pix_fmt", "yuv420p", "-r", "30", "-an", str(tmp)], env=env, check=True)
    got = n_packets(tmp)
    if got != length:
        raise RuntimeError(f"{dst}: {got} frames, expected {length}")
    os.replace(tmp, dst)
    return "ok"


for split in ("train", "val"):
    s, d = args.src / split, args.dst / split
    if not s.exists():
        continue
    bad = set(json.load(open(args.bad_dir / f"{name}_{split}_bad_ids.json")))
    for sub in ("meta", "data"):
        if not (d / sub).exists():
            shutil.copytree(s / sub, d / sub, symlinks=True)
    info = json.load(open(s / "meta/info.json"))
    fps = info["fps"]
    for vdir in sorted((s / "videos").iterdir()):
        cam = vdir.name.replace("observation.images.", "")
        if cam not in cams:
            link = d / "videos" / vdir.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                os.symlink(vdir.resolve(), link)
    ep_files = sorted(glob.glob(str(d / "meta/episodes/*/*.parquet")))
    eps = pd.concat(pd.read_parquet(p) for p in ep_files)
    jobs = []
    for cam in cams:
        k = f"videos/observation.images.{cam}/"
        for r in eps.to_dict("records"):
            ep = int(r["episode_index"])
            if ep in bad:
                continue
            src = s / f"videos/observation.images.{cam}/chunk-{int(r[k + 'chunk_index']):03d}/file-{int(r[k + 'file_index']):03d}.mp4"
            dst = d / f"videos/observation.images.{cam}/chunk-000/file-{ep:03d}.mp4"
            jobs.append((src, dst, float(r[k + "from_timestamp"]), int(r["length"])))
    print(f"[{split}] {len(jobs)} episode videos ({len(bad)} corrupted episodes skipped)", flush=True)
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
    print(f"[{split}] done, {len(failed)} failed", *failed[:10], sep="\n  ", flush=True)
    # point every episode at its own file (corrupted episodes keep their old pointers; they are excluded)
    for p in ep_files:
        e = pd.read_parquet(p)
        good = ~e["episode_index"].isin(bad)
        for cam in cams:
            k = f"videos/observation.images.{cam}/"
            e.loc[good, k + "chunk_index"] = 0
            e.loc[good, k + "file_index"] = e.loc[good, "episode_index"].astype("int64")
            e.loc[good, k + "from_timestamp"] = 0.0
            e.loc[good, k + "to_timestamp"] = e.loc[good, "length"] / fps
        e.to_parquet(p)
    json.dump(sorted(bad), open(d / "meta/corrupted_episodes.json", "w"))
