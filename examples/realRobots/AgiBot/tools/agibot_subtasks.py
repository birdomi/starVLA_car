#!/usr/bin/env python3
"""AgiBot World 2026 subtask labels in annotate_subtasks.py's format (read by starVLA/dataloader/egodex_stream.py:
<out_dir>/subtasks_*.jsonl, {"path", "description", "duration", "subtasks": [{"text", "start", "end"}], "ok"}), with an
ordinal on every repeated subtask text so that repeated steps stay distinct candidates: a text occurring k > 1 times in
an episode becomes "<text> (first)", "<text> (second)", ... in time order ("The left arm places the coin into the
piggy bank (second)"). Texts count as the same after lowercasing, dropping punctuation and articles.

  native:   the episode's labelled steps (instruction_segments, track "default", non-empty) of the index rows
            (agibot_index.py); boundaries as labelled (no stretch to [0, duration]: before the first step the
            dataloader reads step 1 not started, after the last step everything done); "task_frames": the episode's
            (Sub)Task Frame comments with their spans, ordinals likewise.
  ordinals: flora_subtasks.py outputs (subtasks_*.jsonl in a dir) -> the same rows with ordinals on the subtasks.

  agibot_subtasks.py native <index_agibot.jsonl> <out_dir>
  agibot_subtasks.py ordinals <flora out dir> <out_dir>
"""
import argparse
import collections
import glob
import json
import re
from pathlib import Path

ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
FRAME_TYPES = {"taskframe", "subtaskframe"}


def ordinal(k: int) -> str:
    return ORDINALS[k] if k < len(ORDINALS) else f"{k + 1}th"


def norm(text: str) -> str:
    return " ".join(w for w in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split() if w not in ("a", "an", "the"))


def with_ordinals(subs):
    """[{"text", ...}] -> copies, a repeated text suffixed with its occurrence in time order."""
    total = collections.Counter(norm(s["text"]) for s in subs)
    seen = collections.Counter()
    out = []
    for s in sorted(subs, key=lambda s: s["start"]):
        k = norm(s["text"])
        r = dict(s)
        if total[k] > 1:
            r["text"] = f"{s['text'].strip().rstrip('.')} ({ordinal(seen[k])})"
        seen[k] += 1
        out.append(r)
    return out


def native(index: Path, out_dir: Path):
    infos, rows, n_rep = {}, [], 0
    for l in open(index):
        e = json.loads(l)
        if e["src"] not in infos:
            infos.clear()  # the index runs dataset by dataset
            infos[e["src"]] = json.loads((Path(e["src"]) / "meta/info.json").read_text())
        info, fps, ep = infos[e["src"]], float(e["fps"]), str(e["episode"])
        steps = [{"text": s["instruction"].strip(), "start": round(s["start_frame_index"] / fps, 3),
                  "end": round(s["end_frame_index"] / fps, 3)}
                 for s in (info.get("instruction_segments") or {}).get(ep, [])
                 if s["track"] == "default" and s["instruction"].strip() and s["end_frame_index"] > s["start_frame_index"]]
        kf = (info.get("key_frame") or {}).get(ep, {})
        frames = [{"text": (x.get("frame_detail") or {}).get("comment", "").strip(), "start": round(x["start"] / fps, 3),
                   "end": round(x["end"] / fps, 3)}
                  for x in (kf.get("dual") or []) + (kf.get("single") or [])
                  if re.sub(r"[^a-z]", "", x.get("frame_type_name", "").lower()) in FRAME_TYPES
                  and (x.get("frame_detail") or {}).get("comment", "").strip()]
        subs = with_ordinals(steps)
        n_rep += any(s["text"] != t["text"] for s, t in zip(subs, sorted(steps, key=lambda s: s["start"])))
        rows.append({"path": e["path"], "description": e["description"], "duration": round((e["n"] - 1) / fps, 3),
                     "subtasks": subs, "ok": bool(subs), "task_frames": with_ordinals(frames)})
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "subtasks_agibot.jsonl"
    with open(out, "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{len(rows)} episodes, {sum(r['ok'] for r in rows)} with steps, {n_rep} with repeated steps -> {out}")


def ordinals(src: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = n_rep = 0
    for f in sorted(glob.glob(str(src / "subtasks_*.jsonl"))):
        rows = {}
        for l in open(f):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            rows[r["path"]] = r  # a later row of a path (a redo) wins
        with open(out_dir / Path(f).name, "w") as fo:
            for r in rows.values():
                r = {k: v for k, v in r.items() if k != "flora"}
                if r.get("ok", True) and r.get("subtasks"):
                    subs = with_ordinals(r["subtasks"])
                    n_rep += any(a["text"] != b["text"] for a, b in zip(subs, r["subtasks"]))
                    r["subtasks"] = subs
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} episodes, {n_rep} with repeated subtasks -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["native", "ordinals"])
    ap.add_argument("src", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args()
    native(a.src, a.out_dir) if a.mode == "native" else ordinals(a.src, a.out_dir)
