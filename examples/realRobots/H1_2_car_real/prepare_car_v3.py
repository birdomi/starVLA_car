"""Merge car-1015-v3-ep, car-0901-0923 and car_data_3rd (1019-1027) WITHOUT rewriting any parquet / mp4 (2026-10-29).

car-0901-0923 is (mostly) the same recordings as car-1015-v3 (identical arm trajectories, byte-identical head_left
mp4s), and the two re-split them: ~60-70 % of each one's val is the other's train. car_data_3rd shares nothing.
This script decides, per episode, whether it is trained on, validated on, or dropped, and writes:

  1. <out_root>/car-0901-ov/{train,val} and <out_root>/car-3rd-ov/{train,val}: overlays with their own meta/
     (modality.json in the car-1015-v3 key layout, tasks.jsonl with the car1015_task_map.json instructions) and
     data/ + videos/ symlinked to the raw exports.
       car-0901 state 63-D = the car-1015-v1 layout (head 0:2 | legs/waist 2:15 | arms 15:29 | pad 29:37 | hands 37:63)
       car_data_3rd state 93-D: body 0:35 (= legs/waist 0:13 | arms 13:27 | pad 27:35) | head 35:37 | L arm pos/vel
       37:51 | L hand 51:64 | raw_head 64:66 | R arm pos/vel 66:80 | R hand 80:93 -> arms from the body slice, as in v3
       action 42-D is the same layout in all three. Video keys head_left_2 ... -> observation.images.head_left ...
  2. --lists json {"<dataset>/<split>": {"include": [ids]}} for data_cfg.episode_lists of the train mixture, and
     <lists stem>_val.json for the val mixture (a val dir contributes different episodes to each).

Rules
  * dedup by sha1 of the float32 arm action trajectory (a[:, 2:9] + a[:, 22:29]); priority car-1015-v3 > 3rd > 0901.
  * a val episode whose trajectory appears in ANY train split is trained on (user, 10-29).
  * episodes new to v3 (all of 3rd, the 0901 ones not in v3) must pass rule_v1 (car1015_v3_exclude_episodes.json):
    shorter than 0.5 x train median of its task; turn signal / center hazard without a head look (|yaw| > 20 deg for
    >= 0.5 s); head state at the sensor error value; hand state outside [-100, 2200]; teleop reset (arm cmd all
    zero); 1-frame head / arm command spike (> 15 deg jump, back within 8 deg 4 frames later); dark head_left video.
  * car-0901 episodes with a corrupt mp4 (BAD_ENTRIES.txt) are dropped.
  * --extra-exclude (car_v3_extra_exclude.json, 10-29 scan of the kept episodes, any source incl. v3): episodes with a
    hand / head state sensor dropout (all-zero for >= 30 frames) or a non-returning > 15 deg arm / head command jump,
    written as "exclude" ids next to the include lists (the loader applies both).
  The ch2 action clip (1650) of v3 is applied at load time (data config h1_2_car_3cam_headstate_ch2clip).

Usage:
  python examples/realRobots/H1_2_car_real/prepare_car_v3.py \
      --v3 /appsdata/t509a04/data/car-1015-v3-ep --car0901 /scratch/t509a04/data/car-0901-0923_<uuid> \
      --car3rd /scratch/t509a04/data/car_data_3rd_1019_1027_<uuid> --out-root /appsdata/t509a04/data \
      --lists examples/realRobots/H1_2_car_real/car_v3_episode_lists.json
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ARMS = np.r_[2:9, 22:29]
# per source: state head slice, hand-state columns (for the garbage check), and the overlay modality.json state map
LAYOUT = {
    "0901": dict(head=(0, 2), hands=np.r_[37:63], state={"head": (0, 2), "legs_waist": (2, 15), "left_arm": (15, 22),
                 "right_arm": (22, 29), "body_pad": (29, 37), "left_ee": (37, 50), "right_ee": (50, 63)}),
    "3rd": dict(head=(35, 37), hands=np.r_[51:64, 80:93], state={"head": (35, 37), "legs_waist": (0, 13), "left_arm": (13, 20),
                "right_arm": (20, 27), "body_pad": (27, 35), "left_ee": (51, 64), "right_ee": (80, 93)}),
    "v3": dict(head=(0, 2), hands=np.r_[37:63]),
}
ACTION = {"head": (0, 2), "left_arm": (2, 9), "left_ee": (9, 22), "right_arm": (22, 29), "right_ee": (29, 42)}
CAMS = ["head_left", "head_right", "wrist_left", "wrist_right"]
NEW_TASKS = {  # car_data_3rd instructions with no car1015_task_map.json entry
    "activate the automated windshield washer and inspect its operation": "Activate the automated windshield washer and inspect its operation.",
    "activate the windshield washer and inspect its operation": "Activate the windshield washer and inspect its operation.",
}
LOOK_TASKS = {0, 1, 4, 5}  # turn signal L / R, center hazard on / off (car1015_task_map.json)
D15, D8, LOOK = np.radians(15), np.radians(8), np.radians(20)


def norm(t: str) -> str:
    s = t.lower().replace("(fixed seat)", "").replace("inspact", "inspect").replace("inpact", "inspect").replace("it's", "its")
    s = s.replace("3and", "3 and").replace("speed setting 1", "speed settings 1").replace("speed settingss", "speed settings")
    s = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()
    return s.replace("whole interior lights", "the whole interior light").replace("the the", "the")


def spike(x: np.ndarray) -> bool:
    if len(x) < 6:
        return False
    return bool(((np.abs(x[1:-4] - x[:-5]) > D15) & (np.abs(x[5:] - x[:-5]) < D8)).any())


def scan_file(args):
    src, f = args
    lay = LAYOUT[src]
    t = pq.read_table(f, columns=["observation.state", "action", "episode_index", "task_index"])
    S = np.stack(t.column(0).to_numpy(zero_copy_only=False))
    A = np.stack(t.column(1).to_numpy(zero_copy_only=False))
    E, T = t.column(2).to_numpy(), t.column(3).to_numpy()
    out = []
    for ep in np.unique(E):
        m = E == ep
        s, a = S[m], A[m]
        look, run = 0, 0
        for v in np.abs(a[:, 0]) > LOOK:
            run = run + 1 if v else 0
            look = max(look, run)
        hs = s[:, slice(*lay["head"])]
        out.append(dict(
            ep=int(ep), raw_task_index=int(T[m][0]), len=int(m.sum()), look_frames=look,
            hash=hashlib.sha1(np.ascontiguousarray(a[:, ARMS], dtype=np.float32).tobytes()).hexdigest(),
            head_sensor=bool((np.abs(np.degrees(hs) - np.array([-281.9, 309.5])) < 1).all(1).any() or (np.abs(hs) > np.pi).any()),
            hand_garbage=bool(((s[:, lay["hands"]] < -100) | (s[:, lay["hands"]] > 2200)).any()),
            teleop_reset=bool((np.abs(a[:, 2:9]).max(1) < 1e-6).any() or (np.abs(a[:, 22:29]).max(1) < 1e-6).any()),
            head_spike=spike(a[:, 0:2]), arm_spike=spike(a[:, ARMS]),
        ))
    return out


def brightness(args):
    path, n = args
    import av

    vals = []
    try:
        with av.open(path) as c:
            for i, fr in enumerate(c.decode(video=0)):
                if i in (n // 4, n // 2):
                    vals.append(fr.to_ndarray(format="gray").mean())
                if i > n // 2:
                    break
    except Exception:
        return -1.0
    return min(vals) if vals else -1.0


def task_texts(root: Path, split: str) -> dict:
    m = root / split / "meta"
    if (m / "tasks.jsonl").exists():
        return {d["task_index"]: d["task"] for d in map(json.loads, open(m / "tasks.jsonl"))}
    return {int(r.task_index): idx for idx, r in pd.read_parquet(m / "tasks.parquet").iterrows()}


def write_overlay(name: str, src_root: Path, split: str, out_root: Path, canon: dict):
    d = out_root / name / split
    (d / "meta").mkdir(parents=True, exist_ok=True)
    for sub in ("data", "videos"):
        if not (d / sub).exists():
            os.symlink(src_root / split / sub, d / sub)
    for f in ("info.json", "episodes.jsonl", "stats.json"):
        shutil.copy(src_root / split / "meta" / f, d / "meta" / f)
    raw = task_texts(src_root, split)
    with open(d / "meta" / "tasks.jsonl", "w") as f:
        for i in sorted(raw):
            f.write(json.dumps({"task_index": i, "task": canon[norm(raw[i])]}) + "\n")
    lay = LAYOUT["0901" if name == "car-0901-ov" else "3rd"]["state"]
    mod = {
        "state": {k: {"original_key": "observation.state", "start": a, "end": b} for k, (a, b) in lay.items()},
        "action": {k: {"original_key": "action", "start": a, "end": b} for k, (a, b) in ACTION.items()},
        "video": {f"{c}_2": {"original_key": f"observation.images.{c}"} for c in CAMS},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    json.dump(mod, open(d / "meta" / "modality.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", type=Path, required=True)
    ap.add_argument("--car0901", type=Path, required=True)
    ap.add_argument("--car3rd", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--lists", type=Path, required=True)
    ap.add_argument("--task-map", type=Path, default=HERE / "car1015_task_map.json")
    ap.add_argument("--rule", type=Path, default=HERE / "car1015_v3_exclude_episodes.json")
    ap.add_argument("--dark", type=float, default=10.0, help="head_left mean gray level below this = dark video")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--extra-exclude", type=Path, default=HERE / "car_v3_extra_exclude.json")
    a = ap.parse_args()

    tm = json.load(open(a.task_map))
    canon, n2i = {}, {}
    for t in tm["tasks"]:
        for r in [t["task"]] + t["raw"]:
            canon[norm(r)], n2i[norm(r)] = t["task"], t["task_index"]
    canon.update(NEW_TASKS)
    med = {int(k): v for k, v in json.load(open(a.rule))["train_task_median_s"].items()}

    srcs = {"v3": a.v3, "0901": a.car0901, "3rd": a.car3rd}
    jobs = [(k, f) for k, r in srcs.items() for s in ("train", "val") for f in sorted(glob.glob(f"{r}/{s}/data/*/*.parquet"))]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for (k, f), res in zip(jobs, ex.map(scan_file, jobs, chunksize=4)):
            for r in res:
                rows.append(dict(src=k, split=Path(f).parts[-4], **r))
    df = pd.DataFrame(rows)
    txt = {(k, s): task_texts(r, s) for k, r in srcs.items() for s in ("train", "val")}
    df["task"] = [norm(txt[(k, s)][i]) for k, s, i in zip(df.src, df.split, df.raw_task_index)]
    unknown = set(df.task) - set(canon)
    assert not unknown, f"instructions with no canonical text: {unknown}"
    df["task_index"] = df.task.map(n2i)
    print("episodes:", df.groupby(["src", "split"]).size().to_dict())

    # rule_v1 on the episodes new to v3
    v3_hashes = set(df[df.src == "v3"].hash)
    new = (df.src != "v3") & ~df.hash.isin(v3_hashes)
    for t in df[new & df.task_index.isna()].task.unique():  # new tasks: median from their own train episodes
        med_s = df[(df.task == t) & (df.split == "train") & new].len.median() / 30
        df.loc[df.task == t, "med_s"] = med_s
    known = df.task_index.notna()
    df.loc[known, "med_s"] = df.loc[known, "task_index"].astype(int).map(med)
    df["short"] = df.len / 30 < 0.5 * df.med_s
    df["no_look"] = df.task_index.isin(LOOK_TASKS) & (df.look_frames < 15)
    bad_ids = set()
    for line in open(a.car0901 / "BAD_ENTRIES.txt"):
        m = re.search(r"/(train|val)/videos/.*episode_(\d+)\.mp4", line)
        if m:
            bad_ids.add((m.group(1), int(m.group(2))))
    df["corrupt_mp4"] = [(k == "0901") and ((s, e) in bad_ids) for k, s, e in zip(df.src, df.split, df.ep)]
    vid = df[new]
    paths = [f"{srcs[k]}/{s}/videos/chunk-{e // 1000:03d}/observation.images.head_left/episode_{e:06d}.mp4"
             for k, s, e in zip(vid.src, vid.split, vid.ep)]
    with ProcessPoolExecutor(a.workers) as ex:
        df.loc[new, "bright"] = list(ex.map(brightness, zip(paths, vid.len), chunksize=4))
    df["dark"] = new & (df.bright < a.dark)
    flags = ["short", "no_look", "head_sensor", "hand_garbage", "teleop_reset", "head_spike", "arm_spike", "corrupt_mp4", "dark"]
    df["rejected"] = new & df[flags].any(axis=1)
    print("rule_v1 on new episodes:", {f: int((new & df[f]).sum()) for f in flags}, "rejected", int(df.rejected.sum()))

    # dedup (priority v3 > 3rd > 0901) and final split: a val episode seen in any train split is trained on
    train_hashes = set(df[(df.split == "train") & ~df.rejected].hash)
    df["final"] = np.where((df.split == "val") & df.hash.isin(train_hashes), "train", df.split)
    df["prio"] = df.src.map({"v3": 0, "3rd": 1, "0901": 2})
    keep = df[~df.rejected].sort_values(["prio", "split"]).drop_duplicates("hash")
    df["kept"] = df.index.isin(keep.index)

    # The val dirs feed both mixtures with different episodes, so the train and val mixtures get separate list files
    # (<lists> for car_v3_3cam_headstate, <lists stem>_val.json for car_v3_val_3cam_headstate).
    names = {"v3": a.v3.name, "0901": "car-0901-ov", "3rd": "car-3rd-ov"}
    lists_tr, lists_va, summary = {}, {}, {}
    for k in srcs:
        for s in ("train", "val"):
            d = df[(df.src == k) & (df.split == s)]
            tr = sorted(d[d.kept & (d.final == "train")].ep.tolist())
            va = sorted(d[d.kept & (d.final == "val")].ep.tolist())
            summary[f"{names[k]}/{s}"] = dict(total=len(d), train=len(tr), val=len(va), dropped=len(d) - len(tr) - len(va))
            if s == "train":
                assert not va
                if k == "v3":
                    assert len(tr) == len(d), "every v3 train episode should be kept"
                else:
                    lists_tr[f"{names[k]}/train"] = {"include": tr}
            else:
                lists_va[f"{names[k]}/val"] = {"include": va}
                if k == "v3":  # the only val dir in the train mixture (see car_v3_3cam_headstate)
                    lists_tr[f"{names[k]}/val"] = {"include": tr}
                else:
                    assert not tr, f"{names[k]}/val has train episodes but is not in the train mixture"
    apply_extra_exclude(df, lists_tr, lists_va, summary, names, a.extra_exclude)
    lists_tr["_summary"] = lists_va["_summary"] = summary
    a.lists.write_text(json.dumps(lists_tr, indent=1))
    a.lists.with_name(a.lists.stem + "_val.json").write_text(json.dumps(lists_va, indent=1))
    for k in ("0901", "3rd"):
        for s in ("train", "val"):
            write_overlay(names[k], srcs[k], s, a.out_root, canon)
    df.to_parquet(a.lists.with_suffix(".episodes.parquet"))
    for key, v in summary.items():
        print(f"  {key:22s} {v}")
    tot = df[df.kept]
    print("merged: train", int((tot.final == "train").sum()), "eps", f"{tot[tot.final == 'train'].len.sum() / 30 / 3600:.1f} h",
          "| val", int((tot.final == "val").sum()), "| tasks", tot.task.nunique())


def apply_extra_exclude(df, lists_tr, lists_va, summary, names, path):
    """'exclude' ids from --extra-exclude ({"<dataset>/<split>": {"<ep>": {reason, final}}}) into the train / val lists
    (by the episode's final split); df gets the reason (extra_exclude), summary the per-split count."""
    extra = {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")} if path and path.exists() else {}
    df["extra_exclude"] = [extra.get(f"{names[k]}/{s}", {}).get(str(e), {}).get("reason", "")
                           for k, s, e in zip(df.src, df.split, df.ep)]
    for key, eps in extra.items():
        for e, info in eps.items():
            lists = lists_tr if info["final"] == "train" else lists_va
            lists.setdefault(key, {}).setdefault("exclude", []).append(int(e))
        summary[key]["extra_excluded"] = len(eps)
    for lists in (lists_tr, lists_va):
        for v in lists.values():
            if "exclude" in v:
                v["exclude"] = sorted(set(v["exclude"]))
    print("extra exclusions:", {k: len(v) for k, v in extra.items()})


if __name__ == "__main__":
    main()
