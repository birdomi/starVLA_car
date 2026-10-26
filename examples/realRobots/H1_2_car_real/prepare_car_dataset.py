"""Make a raw H1-2 car LeRobot v3.0 export loadable by starVLA, in place.

Replays, as one script, the fixes that were applied by hand to car-left-turn-signal
(2026-09-23), so a new export such as car-few-tasks-v1 ends up in the same state:

  1. meta/info.json   - the export writes `{episode_chunk}/{episode_file}` path templates
                        and null chunks_size / file sizes; the loader wants
                        `{chunk_index}/{file_index}`. Joint names are copied from the
                        reference set. The original is kept as info.json.bak.
  2. meta/tasks.parquet - the export stores `task` as a column; the loader expects it as
                        the index (it renames the reset index to `task`).
  3. meta/modality.json - not exported at all; copied from the reference set.
  3b. meta/episodes/*  - the loader reads data/chunk_index + data/file_index, which the export
                        omits; parsed from data_path. Original kept as <file>.parquet.bak.
  3c. data/*.parquet   - the export also carries per-row observation.images.* {path, timestamp}
                        pointers; with them the loader treats the set as image-only and
                        PIL-opens the mp4. Checked against meta/episodes, then moved to
                        meta/dropped_image_columns/.
  4. action hand ch2  - the Inspire DG2 channel 2 saturates at ~1652 in the measured state,
                        but the commands go up to 1800. Clipped to 1650 so the target is
                        reachable. Pre-clip values -> meta/action_ch2_preclip.npz.
  5. hand state rows  - isolated sensor garbage (e.g. 29702, -19706, -1018) in the 26 hand
                        state dims. Each bad row's hand block is linearly interpolated
                        from the nearest good rows of the same episode. Originals ->
                        meta/state_row_precorrection.npz.

Every step is idempotent: a split that already has the .npz / .bak records is skipped for
that step, so re-running never double-applies. Stale loader caches (stats_gr00t.*.json,
steps_data_index.pkl) are removed whenever the data changes.

Usage:
    python prepare_car_dataset.py playground/Datasets/car-few-tasks-v1 \
        --ref playground/Datasets/car-left-turn-signal/val [--dry-run]
"""

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# 65-dim state: head 2 | legs 12 | waist 1 | L arm 7 | R arm 7 | pad 8 | L hand 13 | R hand 13 | head_cmd 2
STATE_HANDS = [(37, 50), (50, 63)]
# 42-dim action: head 2 | L arm 7 | L hand 13 | R arm 7 | R hand 13
ACTION_CH2 = {"left_ch2": 9 + 2, "right_ch2": 29 + 2}
CH2_MAX = 1650.0
HAND_STATE_VALID = (-100.0, 2100.0)  # raw Inspire units are ~0..2050; small negatives are noise


def data_files(split: Path) -> list[str]:
    return sorted(glob.glob(str(split / "data" / "*" / "*.parquet")))


def write_like(path: str, table: pa.Table, cols: dict[str, np.ndarray]) -> None:
    """Replace list<float> columns of `table` and write it back with the original schema."""
    for name, arr in cols.items():
        i = table.schema.get_field_index(name)
        field = table.schema.field(i)
        new = pa.array(list(arr.astype(np.float32)), type=field.type)
        table = table.set_column(i, field, new)
    tmp = path + ".tmp"
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def fix_info(split: Path, ref: Path, dry: bool) -> None:
    info_p, bak_p = split / "meta" / "info.json", split / "meta" / "info.json.bak"
    info = json.load(open(info_p))
    if "{chunk_index" in info["data_path"]:
        print("  info.json: already in loader format")
        return
    r = json.load(open(ref / "meta" / "info.json"))
    for k in ["data_path", "video_path", "chunks_size", "data_files_size_in_mb", "video_files_size_in_mb"]:
        info[k] = r[k]
    for feat in ["observation.state", "action"]:
        if info["features"][feat]["shape"] != r["features"][feat]["shape"]:
            raise SystemExit(f"{feat} shape {info['features'][feat]['shape']} != ref {r['features'][feat]['shape']}")
        info["features"][feat]["names"] = r["features"][feat]["names"]
    print(f"  info.json: path templates -> {info['data_path']}, joint names from ref")
    if not dry:
        shutil.copy2(info_p, bak_p)
        json.dump(info, open(info_p, "w"), indent=4)


def fix_tasks(split: Path, dry: bool) -> None:
    p = split / "meta" / "tasks.parquet"
    t = pd.read_parquet(p)
    if "task" not in t.columns:
        print("  tasks.parquet: already indexed by task")
        return
    t = t.set_index("task")[["task_index"]]
    print(f"  tasks.parquet: {len(t)} tasks -> indexed by task string")
    if not dry:
        t.to_parquet(p)


def fix_modality(split: Path, ref: Path, dry: bool) -> None:
    p = split / "meta" / "modality.json"
    if p.exists():
        print("  modality.json: present")
        return
    # state/action/annotation slices from the ref; the video section from THIS split's cameras
    # (exports differ: front_view/left_wrist_view/... vs head_left_2/head_right_2/wrist_left_2/...).
    mod = json.load(open(ref / "meta" / "modality.json"))
    feats = json.load(open(split / "meta" / "info.json"))["features"]
    cams = [k.replace("observation.images.", "") for k, v in feats.items()
            if k.startswith("observation.images.") and v.get("dtype") == "video"]
    mod["video"] = {c: {"original_key": f"observation.images.{c}"} for c in cams}
    print(f"  modality.json: state/action from ref, video = {cams}")
    if not dry:
        json.dump(mod, open(p, "w"), indent=4)


def fix_episodes(split: Path, dry: bool) -> None:
    """Add data/chunk_index + data/file_index (parsed from data_path) to meta/episodes/*."""
    for p in sorted(glob.glob(str(split / "meta" / "episodes" / "*" / "*.parquet"))):
        e = pd.read_parquet(p)
        if "data/chunk_index" in e.columns:
            print(f"  episodes {Path(p).name}: data/chunk_index present")
            continue
        idx = e["data_path"].str.extract(r"chunk-(\d+)/file-(\d+)")
        if idx.isna().any().any():
            raise SystemExit(f"{p}: data_path without chunk/file index")
        e["data/chunk_index"] = idx[0].astype("int64")
        e["data/file_index"] = idx[1].astype("int64")
        print(f"  episodes {Path(p).name}: added data/chunk_index, data/file_index ({len(e)} episodes)")
        if not dry:
            shutil.copy2(p, p + ".bak")
            e.to_parquet(p)


def fix_image_columns(split: Path, dry: bool) -> None:
    """Drop the per-row `observation.images.*` {path, timestamp} pointer columns.

    Their presence makes the loader treat the set as image-only (it PIL-opens the mp4).
    They duplicate meta/episodes (file index + from_timestamp), which is checked first;
    the dropped columns are kept under meta/dropped_image_columns/.
    """
    e = pd.concat(pd.read_parquet(p) for p in sorted(glob.glob(str(split / "meta" / "episodes" / "*" / "*.parquet"))))
    e = e.set_index("episode_index")
    fps = json.load(open(split / "meta" / "info.json"))["fps"]
    out_dir = split / "meta" / "dropped_image_columns"
    for f in data_files(split):
        table = pq.read_table(f)
        cols = [c for c in table.column_names if c.startswith("observation.images.")]
        if not cols:
            continue
        ep = table.column("episode_index").to_numpy()
        fr = table.column("frame_index").to_numpy()
        for c in cols:
            v = c.replace("observation.images.", "")
            s = table.column(c).combine_chunks()
            paths = s.field("path").to_numpy(zero_copy_only=False)
            ts = s.field("timestamp").to_numpy(zero_copy_only=False)
            vkey = f"videos/observation.images.{v}/"
            want_file = e.loc[ep, vkey + "file_index"].to_numpy()
            want_ts = e.loc[ep, vkey + "from_timestamp"].to_numpy() + fr / fps
            got_file = pd.Series(paths).str.extract(r"file-(\d+)\.mp4")[0].astype(int).to_numpy()
            if not (got_file == want_file).all() or np.abs(ts - want_ts).max() > 0.5 / fps:
                raise SystemExit(f"{f}:{c} does not match meta/episodes; not dropping")
        print(f"  {Path(f).name}: dropping {len(cols)} image pointer columns (consistent with meta/episodes)")
        if not dry:
            out_dir.mkdir(exist_ok=True)
            pq.write_table(table.select(["index"] + cols), out_dir / Path(f).name)
            tmp = f + ".tmp"
            pq.write_table(table.drop(cols), tmp)
            os.replace(tmp, f)


def fix_ch2(split: Path, dry: bool) -> bool:
    rec = split / "meta" / "action_ch2_preclip.npz"
    if rec.exists():
        print("  action ch2: already clipped (record exists)")
        return False
    pre = {k: [] for k in ACTION_CH2}
    n_changed = 0
    for f in data_files(split):
        table = pq.read_table(f)
        a = np.stack(table.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
        for k, c in ACTION_CH2.items():
            pre[k].append(a[:, c].copy())
            n_changed += int((a[:, c] > CH2_MAX).sum())
            a[:, c] = np.minimum(a[:, c], CH2_MAX)
        if not dry:
            write_like(f, table, {"action": a})
    pre = {k: np.concatenate(v) for k, v in pre.items()}
    print(f"  action ch2: {n_changed} values clipped to {CH2_MAX:.0f} "
          f"(pre max L {pre['left_ch2'].max():.0f} R {pre['right_ch2'].max():.0f})")
    if not dry:
        np.savez(rec, **pre)
    return n_changed > 0


def fix_hand_state(split: Path, dry: bool) -> bool:
    rec = split / "meta" / "state_row_precorrection.npz"
    if rec.exists():
        print("  hand state: already corrected (record exists)")
        return False
    lo, hi = HAND_STATE_VALID
    rows, originals = [], []
    for f in data_files(split):
        table = pq.read_table(f)
        s = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
        ep = table.column("episode_index").to_numpy()
        gidx = table.column("index").to_numpy()
        hands = np.concatenate([s[:, a:b] for a, b in STATE_HANDS], axis=1)
        bad = ((hands < lo) | (hands > hi)).any(axis=1)
        if not bad.any():
            continue
        for e in np.unique(ep[bad]):
            sel = np.where(ep == e)[0]
            b = bad[sel]
            good = sel[~b]
            if len(good) == 0:
                print(f"    WARNING episode {e}: every row has bad hand state, left untouched")
                continue
            for a_, b_ in STATE_HANDS:
                for d in range(a_, b_):
                    s[sel[b], d] = np.interp(sel[b], good, s[good, d])
        rows.append(gidx[bad])
        originals.append(np.stack(table.column("observation.state").to_numpy(zero_copy_only=False))[bad])
        if not dry:
            write_like(f, table, {"observation.state": s})
    if not rows:
        print("  hand state: no out-of-range rows")
        if not dry:
            np.savez(rec, row=np.zeros(0, np.int64), original=np.zeros((0, 65), np.float32))
        return False
    rows, originals = np.concatenate(rows), np.concatenate(originals)
    print(f"  hand state: {len(rows)} rows interpolated (global index {rows[:8].tolist()}"
          f"{' ...' if len(rows) > 8 else ''})")
    if not dry:
        np.savez(rec, row=rows, original=originals.squeeze())
    return True


def report_short(split: Path, fps: float) -> None:
    e = pd.concat(pd.read_parquet(p) for p in glob.glob(str(split / "meta" / "episodes" / "*" / "*.parquet")))
    for sec in (5, 10):
        n = int((e["length"] < sec * fps).sum())
        print(f"  episodes shorter than {sec:2d} s: {n} / {len(e)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="dataset root holding train/ and val/")
    ap.add_argument("--ref", type=Path, required=True, help="an already-prepared split; car-left-turn-signal/val has the real joint names (its train/ info.json still says position_*)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for split in sorted(p for p in args.root.iterdir() if (p / "meta" / "info.json").exists()):
        print(f"[{split}]")
        fix_info(split, args.ref, args.dry_run)
        fix_tasks(split, args.dry_run)
        fix_modality(split, args.ref, args.dry_run)
        fix_episodes(split, args.dry_run)
        fix_image_columns(split, args.dry_run)
        changed = fix_ch2(split, args.dry_run)
        changed |= fix_hand_state(split, args.dry_run)
        if changed and not args.dry_run:
            for stale in list((split / "meta").glob("stats_gr00t*.json")) + [split / "meta" / "steps_data_index.pkl"]:
                if stale.exists():
                    stale.unlink()
                    print(f"  removed stale cache {stale.name}")
        report_short(split, json.load(open(split / "meta" / "info.json"))["fps"])


if __name__ == "__main__":
    main()
