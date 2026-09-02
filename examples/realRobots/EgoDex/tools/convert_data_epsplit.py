#!/usr/bin/env python3
"""Split LeRobot v3.0 merged data parquets into one parquet per episode.

Companion to convert_epsplit.py (videos). The merged files (~90MB, ~170
episodes each) cost ~330ms per sample because the loader reads a whole file
and filters one episode. Per-episode files bring that to a few ms.

Usage:
  convert_data_epsplit.py convert <src_root> <dst_epsplit_root> [--workers N]
      writes <dst>/data_ep/chunk-{ep//1000:03d}/file-{ep%1000:03d}.parquet and
      <dst>/meta/episodes_ep/... (staging; nothing live is touched)
  convert_data_epsplit.py verify  <src_root> <dst_epsplit_root> [--samples 30]
  convert_data_epsplit.py swap    <dst_epsplit_root>
      data(symlink) -> removed, data_ep -> data ; episodes -> episodes_merged, episodes_ep -> episodes
"""
import argparse, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DATA_TPL = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def split_file(args):
    src_file, dst_data = args
    df = pd.read_parquet(src_file)
    n = 0
    for ep, g in df.groupby("episode_index", sort=True):
        ep = int(ep)
        out = Path(dst_data) / f"chunk-{ep // 1000:03d}" / f"file-{ep % 1000:03d}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        g.to_parquet(out, index=False)
        n += 1
    return n


def cmd_convert(a):
    src, dst = Path(a.src_root), Path(a.dst_root)
    files = sorted((src / "data").rglob("*.parquet"))
    dst_data = dst / "data_ep"
    print(f"{src.name}: {len(files)} merged parquet files -> {dst_data}", flush=True)
    total = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(split_file, (str(f), str(dst_data))) for f in files]
        for i, f in enumerate(as_completed(futs), 1):
            total += f.result()
            if i % 10 == 0 or i == len(files):
                print(f"  {i}/{len(files)} files, {total} episodes", flush=True)
    # episodes meta: remap data/chunk_index, data/file_index
    ep_files = sorted((dst / "meta" / "episodes").rglob("*.parquet"))
    for f in ep_files:
        df = pd.read_parquet(f)
        df["data/chunk_index"] = (df["episode_index"] // 1000).astype(df["data/chunk_index"].dtype)
        df["data/file_index"] = (df["episode_index"] % 1000).astype(df["data/file_index"].dtype)
        out = dst / "meta" / "episodes_ep" / f.relative_to(dst / "meta" / "episodes")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    print(f"DONE {src.name}: {total} episode parquets, meta staged in meta/episodes_ep", flush=True)


def cmd_verify(a):
    src, dst = Path(a.src_root), Path(a.dst_root)
    meta = pd.concat([pd.read_parquet(f) for f in sorted((dst / "meta" / "episodes_ep").rglob("*.parquet"))])
    orig = pd.concat([pd.read_parquet(f) for f in sorted((dst / "meta" / "episodes").rglob("*.parquet"))])
    rows = meta.sample(min(a.samples, len(meta)), random_state=0)
    bad, times = 0, []
    for _, ep in rows.iterrows():
        e = int(ep["episode_index"])
        new_p = dst / DATA_TPL.format(chunk_index=e // 1000, file_index=e % 1000)
        new_p = Path(str(new_p).replace("/data/", "/data_ep/"))
        o = orig[orig["episode_index"] == e].iloc[0]
        old_p = src / DATA_TPL.format(chunk_index=int(o["data/chunk_index"]), file_index=int(o["data/file_index"]))
        t0 = time.perf_counter(); new = pd.read_parquet(new_p); times.append(time.perf_counter() - t0)
        old = pd.read_parquet(old_p); old = old[old["episode_index"] == e]
        if len(new) != len(old) or len(new) != int(ep["length"]):
            print(f"row mismatch ep{e}: {len(new)} vs {len(old)} vs meta {ep['length']}"); bad += 1; continue
        for c in ("observation.state", "action", "timestamp", "frame_index", "index"):
            if not np.allclose(np.stack(new[c].values), np.stack(old[c].values)):
                print(f"value mismatch ep{e} col {c}"); bad += 1; break
    print(f"verify {dst.name}: read time mean {np.mean(times)*1000:.1f}ms | bad {bad}/{len(rows)}")
    sys.exit(1 if bad else 0)


def cmd_swap(a):
    dst = Path(a.dst_root)
    data, data_ep = dst / "data", dst / "data_ep"
    assert data_ep.is_dir(), "data_ep missing (run convert first)"
    if data.is_symlink():
        data.unlink()
    elif data.exists():
        data.rename(dst / "data_merged_backup")
    data_ep.rename(data)
    ep, ep_new = dst / "meta" / "episodes", dst / "meta" / "episodes_ep"
    assert ep_new.is_dir()
    ep.rename(dst / "meta" / "episodes_merged_backup")
    ep_new.rename(ep)
    print(f"swapped {dst.name}: data/ and meta/episodes now per-episode")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("convert"); s.add_argument("src_root"); s.add_argument("dst_root"); s.add_argument("--workers", type=int, default=32); s.set_defaults(fn=cmd_convert)
    s = sub.add_parser("verify"); s.add_argument("src_root"); s.add_argument("dst_root"); s.add_argument("--samples", type=int, default=30); s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("swap"); s.add_argument("dst_root"); s.set_defaults(fn=cmd_swap)
    a = p.parse_args(); a.fn(a)
