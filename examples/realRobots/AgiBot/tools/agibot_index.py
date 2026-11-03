#!/usr/bin/env python3
"""Episode index of the downloaded AgiBot World 2026 (fetch_agibot.py: raw/<HF path>/[data/]{meta,data,videos}) for
examples/realRobots/EgoDex/tools/flora_subtasks.py, one row per episode:
  {"path": "agibot/<dataset dir>/<episode>.npz" (unique id), "split", "theme", "task", "episode", "n", "fps",
   "description" (the instruction), "task_text" (tasks.jsonl as is), "src" (the LeRobot dataset dir),
   "video" / "left_wrist" / "right_wrist" (top_head / hand_left / hand_right mp4), "frame_bytes" (RGB bytes of one
   frame of the three cameras, sizes the caption stage's memory budget)}
The instruction: ImitationLearning tasks.jsonl holds only the collection name ("Homestay 1 - Imitation Learning."), so
its episodes take their (Sub)Task Frame comments (near-duplicates dropped, joined with ". Then "); the other themes
take the episode's tasks.jsonl text, underscores turned to spaces.

  agibot_index.py /scratch/t509a04/data/AgiBotWorld2026/raw <out dir> [--split agibot] [--themes ImitationLearning,...]
"""
import argparse
import glob
import json
import re
from pathlib import Path

CAMS = {"video": "top_head", "left_wrist": "hand_left", "right_wrist": "hand_right"}
FRAME_TYPES = {"taskframe", "subtaskframe"}  # spelled "Task Frame", "SubTask Frame", "Sub-task frame", ... by task


def is_task_frame(e):
    return re.sub(r"[^a-z]", "", e.get("frame_type_name", "").lower()) in FRAME_TYPES


def words(s):
    return set(re.findall(r"[a-z]+", s.lower())) - {"the", "a", "an", "arm", "left", "right", "with"}


def frame_instruction(kf):
    """Distinct (Sub)Task Frame comments of one episode, in time order."""
    es = sorted([e for e in (kf.get("dual") or []) + (kf.get("single") or []) if is_task_frame(e)],
                key=lambda e: e["start"])
    out = []
    for e in es:
        c = (e.get("frame_detail") or {}).get("comment", "").strip().rstrip(".")
        if c and not any(len(words(c) & words(o)) / max(len(words(c) | words(o)), 1) > 0.6 for o in out):
            out.append(c)
    return ". Then ".join(out) + "." if out else ""


def clean_task(t):
    return re.sub(r"\s+", " ", t.replace("_", " ")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--split", default="agibot")
    ap.add_argument("--themes", default="ImitationLearning,ReinforcementLearning,PhysicalReasoning")
    a = ap.parse_args()
    rows, missing = [], 0
    for theme in a.themes.split(","):
        for info_f in sorted(glob.glob(str(a.raw / theme / "**/meta/info.json"), recursive=True)):
            d = Path(info_f).parent.parent  # the LeRobot dataset dir (…/data for an extracted archive)
            info = json.loads(Path(info_f).read_text())
            fps = float(info["fps"])
            feats = info.get("features", {})
            fb = sum(int(feats[f"observation.images.{c}"]["shape"][0]) * int(feats[f"observation.images.{c}"]["shape"][1]) * 3
                     for c in CAMS.values() if f"observation.images.{c}" in feats)
            rel = str(d.relative_to(a.raw)).removesuffix("/data")
            task = next(p for p in rel.split("/") if p.startswith("task_"))
            kfs = info.get("key_frame") or {}
            for l in open(d / "meta/episodes.jsonl"):
                e = json.loads(l)
                ep = int(e["episode_index"])
                vids = {k: d / f"videos/chunk-{ep // int(info.get('chunks_size', 1000)):03d}/observation.images.{c}/episode_{ep:06d}.mp4"
                        for k, c in CAMS.items()}
                if not all(v.exists() for v in vids.values()):
                    missing += 1
                    continue
                text = (e.get("tasks") or [""])[0]
                desc = frame_instruction(kfs.get(str(ep), {})) if theme == "ImitationLearning" else ""
                rows.append({"path": f"{a.split}/{rel}/{ep}.npz", "split": a.split, "theme": theme, "task": task,
                             "episode": ep, "n": int(e["length"]), "fps": fps,
                             "description": desc or clean_task(text), "task_text": text, "src": str(d),
                             **{k: str(v) for k, v in vids.items()}, "frame_bytes": fb})
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"index_{a.split}.jsonl"
    with open(out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    hours = sum((r["n"] - 1) / r["fps"] for r in rows) / 3600
    print(f"{len(rows)} episodes ({hours:.0f} h), {missing} without all three videos -> {out}")


if __name__ == "__main__":
    main()
