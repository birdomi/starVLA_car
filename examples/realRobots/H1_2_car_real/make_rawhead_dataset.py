"""Copy of car-turn-signal-ep whose head ACTION is the operator's raw head pose one frame ahead.

    action[t, 0:2] := observation.state[t + 1, 63:65]   (raw_head_0 / raw_head_1, same episode;
                                                         the last frame of an episode keeps its own raw head)

Everything else is unchanged: the other 40 action dims, the 65-D state (incl. 63:65), timestamps, indices.
Why: the recorded head action is a deadband staircase of the raw head pose (it changes in only ~8% of the
frames, ~2.7 deg steps, ~4 frames behind raw_head), while raw_head is continuous; the measured head (state 0:2)
lags the action by ~0.43 s and is sample-and-hold, so it is not used.

Layout of the copy (<dst>/<split>):
    data/      parquet rewritten (same schema, same row order)
    meta/      copied verbatim, INCLUDING stats_gr00t.abs.json: the head is normalized with the ORIGINAL action
               statistics (raw_head q01 / q99 differ from the action's by < 0.3 deg), so System 2 plans of models
               trained on the original data and System 1 targets on this copy share one normalization
    videos     symlink to the original split's videos
    meta/rawhead_note.json  provenance

Usage: python make_rawhead_dataset.py [--src playground/Datasets/car-turn-signal-ep]
                                      [--dst playground/Datasets/car-turn-signal-ep-rawhead] [--splits train,val]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RAW_HEAD = slice(63, 65)  # observation.state raw_head_0 / raw_head_1
HEAD = slice(0, 2)  # action head_0 / head_1


def rewrite(src: Path, dst: Path) -> dict:
    t = pq.read_table(src)
    ep = t.column("episode_index").to_numpy()
    fr = t.column("frame_index").to_numpy()
    if not np.all((np.diff(ep) > 0) | ((np.diff(ep) == 0) & (np.diff(fr) == 1))):
        raise ValueError(f"{src}: rows are not sorted by (episode, frame) with consecutive frames")
    state = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False))
    action = np.stack(t.column("action").to_numpy(zero_copy_only=False))
    nxt = np.arange(len(ep)) + 1
    last = np.r_[ep[1:] != ep[:-1], True]  # last row of each episode -> hold its own raw head
    nxt[last] -= 1
    new = action.copy()
    new[:, HEAD] = state[nxt, RAW_HEAD]
    offsets = pa.array(np.arange(0, new.size + 1, new.shape[1], dtype=np.int32))
    col = pa.ListArray.from_arrays(offsets, pa.array(new.ravel())).cast(t.schema.field("action").type)
    out = t.set_column(t.schema.get_field_index("action"), t.schema.field("action"), col)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dst)
    return {"rows": len(ep), "episodes": int(len(np.unique(ep))),
            "head_change_deg": float(np.degrees(np.abs(new[:, HEAD] - action[:, HEAD]).mean()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="playground/Datasets/car-turn-signal-ep")
    ap.add_argument("--dst", default="playground/Datasets/car-turn-signal-ep-rawhead")
    ap.add_argument("--splits", default="train,val")
    args = ap.parse_args()
    for split in args.splits.split(","):
        src, dst = Path(args.src).resolve() / split, Path(args.dst).resolve() / split
        if dst.exists():
            raise FileExistsError(f"{dst} exists; remove it first")
        dst.mkdir(parents=True)
        shutil.copytree(src / "meta", dst / "meta")
        (dst / "videos").symlink_to(src / "videos")
        files = {}
        for f in sorted((src / "data").rglob("*.parquet")):
            files[str(f.relative_to(src))] = rewrite(f, dst / f.relative_to(src))
            print(split, f.relative_to(src), files[str(f.relative_to(src))], flush=True)
        json.dump({"source": str(src), "action_head": "observation.state[t+1, 63:65] (last frame: [t, 63:65])",
                   "stats": "copied from the source (original action head normalization)", "files": files},
                  open(dst / "meta" / "rawhead_note.json", "w"), indent=2)


if __name__ == "__main__":
    main()
