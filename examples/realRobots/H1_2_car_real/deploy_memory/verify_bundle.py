"""On-site sanity check of the bundle, no robot: stream the bundled val episodes (sample/<name>/) through
deploy_infer.CarPolicy as the robot would (raw mp4 frames decoded with PyAV, the recorded 63-D state, one query
per --interval frames from frame 0, t = frame / 30) and compare every chunk with the recorded actions.

    python verify_bundle.py [--interval 30] [--samples left-turn-signal,...]

Prints per sample: chunk error (head / arms deg, hands raw; chunk mean, steps past the episode end left out), progress
error |p_hat - f / (T - 1)| (memory models), and latency. Compare with the val numbers in offline_eval/ (single episodes
vary).
"""
import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd

from deploy_infer import BUNDLE, CAMERAS, CarPolicy

HEAD, ARM, HAND = np.r_[0:2], np.r_[2:9, 22:29], np.r_[9:22, 29:42]


def frames_at(path, wanted):
    """{frame index: RGB uint8} for the wanted indices, decoding the file forward once."""
    out, want = {}, set(wanted)
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                out[i] = fr.to_ndarray(format="rgb24")
                if len(out) == len(want):
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=30, help="frames between queries (= memory updates at 30 Hz)")
    ap.add_argument("--samples", default="", help="comma-separated sample names (default: all under sample/)")
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    names = args.samples.split(",") if args.samples else sorted(p.name for p in (BUNDLE / "sample").iterdir() if p.is_dir())
    policy = CarPolicy(ckpt=args.ckpt, update_interval_s=args.interval / 30.0)
    lat = []
    for name in names:
        d = BUNDLE / "sample" / name
        meta = json.load(open(d / "meta.json"))
        df = pd.read_parquet(d / "data.parquet")
        state63, gt = np.stack(df["observation.state"]), np.stack(df["action"]).astype(np.float64)
        T = len(df)
        qs = list(range(0, T, args.interval))
        t0 = time.time()
        imgs = {c: frames_at(d / f"{c}.mp4", qs) for c in CAMERAS}
        dec = time.time() - t0
        policy.reset(meta["task"])
        err, prog = {"head": [], "arm": [], "hand": []}, []
        for f in qs:
            out = policy.predict([imgs[c][f] for c in CAMERAS], state63[f], t=f / 30.0)
            lat.append(out["latency_ms"])
            n = min(policy.horizon, T - f)
            dd = np.abs(out["actions"][:n] - gt[f:f + n])
            err["head"].append(np.degrees(dd[:, HEAD].mean()))
            err["arm"].append(np.degrees(dd[:, ARM].mean()))
            err["hand"].append(dd[:, HAND].mean())
            if out["progress"] is not None:
                prog.append(abs(out["progress"] - f / (T - 1)))
        print(f"{name:24s} ep {meta['episode_index']:4d}  {T:4d} frames  {len(qs):3d} queries  head {np.mean(err['head']):5.2f} deg  "
              f"arm {np.mean(err['arm']):5.2f} deg  hand {np.mean(err['hand']):5.1f}  progress {np.mean(prog) if prog else float('nan'):.3f}  "
              f"(decode {dec:.0f}s)  [{meta['task']}]")
    lat = np.array(lat[1:]) if len(lat) > 1 else np.array(lat)
    print(f"predict latency: mean {lat.mean():.0f} ms, p95 {np.percentile(lat, 95):.0f} ms, max {lat.max():.0f} ms "
          f"(one 50-step chunk at 30 Hz lasts 1667 ms)")


if __name__ == "__main__":
    main()
