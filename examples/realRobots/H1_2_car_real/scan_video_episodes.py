"""Find episodes whose video segment does not decode cleanly (LeRobot v3, concatenated mp4 files).

Every file is decoded once in full with ffmpeg + showinfo; each decoded frame's pts and each decoder
error line (attributed to the time of the last good frame) are mapped onto the episodes' [from, to)
ranges from meta/episodes. An episode is BAD for a camera if any error falls in its range or it
yields fewer frames than its length.

Usage: python scan_video_episodes.py <dataset_root> --cams a,b,c [--gpus 0,1,2,3] [--jobs 8] --out bad.json
"""
import argparse
import glob
import json
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("--cams", required=True)
ap.add_argument("--gpus", default="0,1,2,3")
ap.add_argument("--jobs", type=int, default=8)
ap.add_argument("--out", required=True)
args = ap.parse_args()
gpus = args.gpus.split(",")
PTS = re.compile(r"pts_time:\s*([0-9.]+)")


def decode(job):
    i, path = job
    p = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-hwaccel", "cuda", "-i", str(path),
                        "-vf", "showinfo", "-f", "null", "-"], capture_output=True, text=True,
                       env={"CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)], "PATH": "/usr/bin:/bin"})
    frames, errors, last = [], [], -1.0
    for line in p.stderr.splitlines():
        if "Parsed_showinfo" in line:
            m = PTS.search(line)
            if m:
                last = float(m.group(1))
                frames.append(last)
        elif line.startswith("[h264") or "error" in line.lower() or "corrupt" in line.lower():
            errors.append(last)
    return path, np.array(frames), np.array(errors)


report = {}
for split in ("train", "val"):
    sd = args.root / split
    if not sd.exists():
        continue
    eps = pd.concat(pd.read_parquet(p) for p in glob.glob(str(sd / "meta/episodes/*/*.parquet")))
    jobs, owners = [], defaultdict(list)
    for cam in args.cams.split(","):
        k = f"videos/observation.images.{cam}/"
        for fi, g in eps.groupby(k + "file_index"):
            f = sd / f"videos/observation.images.{cam}/chunk-{int(g[k + 'chunk_index'].iloc[0]):03d}/file-{int(fi):03d}.mp4"
            jobs.append(f)
            owners[f] = [(cam, int(r["episode_index"]), float(r[k + "from_timestamp"]), float(r[k + "to_timestamp"]), int(r["length"]))
                         for r in g.to_dict("records")]
    bad = defaultdict(list)
    with ThreadPoolExecutor(args.jobs) as ex:
        for n, (f, frames, errors) in enumerate(ex.map(decode, enumerate(jobs))):
            for cam, ep, a, b, length in owners[f]:
                nf = int(((frames >= a - 1e-6) & (frames < b - 1e-6)).sum())
                ne = int(((errors >= a - 1e-6) & (errors < b - 1e-6)).sum())
                if ne or nf < length:
                    bad[ep].append({"cam": cam, "errors": ne, "frames": nf, "length": length})
            if n % 40 == 0:
                print(f"[{split}] {n + 1}/{len(jobs)} files", flush=True)
    report[split] = {"n_episodes": int(len(eps)), "bad": {str(k): v for k, v in sorted(bad.items())}}
    print(f"[{split}] bad episodes: {len(bad)} / {len(eps)}", flush=True)
json.dump(report, open(args.out, "w"), indent=1)
