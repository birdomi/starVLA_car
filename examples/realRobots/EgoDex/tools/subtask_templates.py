#!/usr/bin/env python3
"""Task-level subtask labels: one step template per task, then every episode aligned to it (vLLM offline, e.g.
Qwen3.8-Flash-Next-FP8), with the boundaries snapped to Flora's (flora_subtasks.py) where they are close.

  1. template: per task, up to --template-episodes episodes as timestamped frames (1 fps) with their Flora captions and
     the task text -> the 2..6 steps every episode of the task goes through (short imperative phrases).
  2. align: per episode, timestamped frames (--fps) and the numbered template -> each step's start / end (steps in
     order, a step not performed may be skipped). Post-processing: contiguous cover of [0, duration]
     (annotate_subtasks.clean), each inner boundary moved to the nearest Flora boundary within --snap s, consecutive
     equal steps merged.
Outputs in <out_dir>: templates_<split>.json {task: [step texts]}, subtasks_<split>.jsonl in annotate_subtasks.py's
format ({"path", "description", "duration", "subtasks": [{"text", "start", "end"}], "ok", "steps": [template index per
subtask]}), read by starVLA/dataloader/egodex_stream.py like the Flora / Flash-Next ones.

  subtask_templates.py <frames_root> <flora_subtasks_dir> <out_dir> --split carval --model <path> [--tp 4]
"""
import argparse
import base64
import collections
import glob
import json
import random
from pathlib import Path

import numpy as np

from annotate_subtasks import clean

TEMPLATE_PROMPT = """Above are {k} recordings of the same task, filmed from the robot's / person's head camera: "{task}".
Each frame is preceded by its timestamp; each recording is followed by an automatic, noisy segmentation of it.

Write the sequence of {lo} to {hi} steps that EVERY recording of this task goes through, in order: short imperative
phrases naming the action and the object (e.g. "reach for the turn-signal lever", "push the lever down", "watch the
indicator blink", "return the hand"). Steps must be visible in the video, at a level where each lasts at least a second;
merge tiny motions, and include waiting / inspecting periods when the task asks for them. Answer with JSON only."""

ALIGN_PROMPT = """These {n} frames are one recording ({duration:.1f} s) of the task "{task}"; the text before each frame
is its timestamp. The task's steps are:
{steps}

Give the start and end time in seconds of each step as performed in this recording, in step order. A step that is not
performed may be left out; every other step appears once. The first starts at 0, each starts where the previous ends,
the last ends at {duration:.1f}. Answer with JSON only."""

T_SCHEMA = {"type": "object", "properties": {"steps": {"type": "array", "minItems": 1, "maxItems": 8,
                                                         "items": {"type": "string"}}}, "required": ["steps"]}
A_SCHEMA = {"type": "object", "properties": {"segments": {"type": "array", "minItems": 1, "items": {
    "type": "object", "properties": {"step": {"type": "integer"}, "start": {"type": "number"}, "end": {"type": "number"}},
    "required": ["step", "start", "end"]}}}, "required": ["segments"]}


def frames(root: Path, ep: dict, fps_in: float, max_images: int):
    """-> [(t, base64 JPEG)] every 1 / fps_in s (fewer when over max_images)."""
    step = max(1, round(float(ep["fps"]) / fps_in))
    n = int(ep["n"])
    if n / step > max_images:
        step = int(np.ceil(n / max_images))
    with np.load(root / ep["path"]) as z:
        o, j = z["offsets"], z["jpeg"]
        return [(i / float(ep["fps"]), base64.b64encode(j[o[i]:o[i + 1]].tobytes()).decode()) for i in range(0, n, step)]


def image_block(fr):
    out = []
    for t, b in fr:
        out.append({"type": "text", "text": f"t={t:.1f}s"})
        out.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    return out


def snap(subs, flora_bounds, tol):
    """Inner boundaries -> the nearest Flora boundary within tol s (kept in order, no empty subtask)."""
    s = [dict(x) for x in subs]
    for i in range(1, len(s)):
        b = s[i]["start"]
        near = [f for f in flora_bounds if abs(f - b) <= tol and s[i - 1]["start"] < f < s[i]["end"]]
        if near:
            b = min(near, key=lambda f: abs(f - b))
            s[i - 1]["end"] = s[i]["start"] = round(b, 3)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_root", type=Path)
    ap.add_argument("flora_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--template-episodes", type=int, default=3)
    ap.add_argument("--min-steps", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-images", type=int, default=64)
    ap.add_argument("--snap", type=float, default=0.75)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    eps = [json.loads(l) for f in sorted(glob.glob(str(a.frames_root / f"index_{a.split}*.jsonl"))) for l in open(f)]
    flora = {}
    for f in glob.glob(str(a.flora_dir / "subtasks_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            flora[r["path"]] = r
    by_task = collections.defaultdict(list)
    for e in eps:
        by_task[e["task"]].append(e)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, enable_expert_parallel=a.tp > 1, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_memory_utilization, kv_cache_dtype="fp8", enable_prefix_caching=False,
              limit_mm_per_prompt={"image": a.template_episodes * 32}, trust_remote_code=True, moe_backend="triton")
    chat = dict(use_tqdm=False, chat_template_kwargs={"enable_thinking": False})

    # 1. templates
    rng = random.Random(a.seed)
    msgs, tasks = [], sorted(by_task)
    for t in tasks:
        sample = rng.sample(by_task[t], min(a.template_episodes, len(by_task[t])))
        content = []
        for k, e in enumerate(sample):
            content.append({"type": "text", "text": f"Recording {k + 1} ({(int(e['n']) - 1) / float(e['fps']):.1f} s):"})
            content += image_block(frames(a.frames_root, e, 1.0, 32))
            segs = flora.get(e["path"], {}).get("flora") or []
            seg_txt = "; ".join(f"[{s['start_s']:.1f}-{s['end_s']:.1f}s] {s['caption']}" for s in segs) or "(none)"
            content.append({"type": "text", "text": f"Automatic segmentation of recording {k + 1}: {seg_txt}"})
        content.append({"type": "text", "text": TEMPLATE_PROMPT.format(k=len(sample), task=by_task[t][0]["description"],
                                                                        lo=a.min_steps, hi=a.max_steps)})
        msgs.append([{"role": "user", "content": content}])
    sp = SamplingParams(temperature=0.0, max_tokens=512, structured_outputs=StructuredOutputsParams(json=T_SCHEMA))
    templates = {}
    for t, r in zip(tasks, llm.chat(msgs, sp, **chat)):
        try:
            steps = [s.strip() for s in json.loads(r.outputs[0].text)["steps"] if s.strip()][: a.max_steps]
        except (json.JSONDecodeError, KeyError):
            steps = []
        templates[t] = steps or [by_task[t][0]["description"]]
        print(f"[template] {t}: {templates[t]}", flush=True)
    (a.out_dir / f"templates_{a.split}.json").write_text(json.dumps(templates, indent=1))

    # 2. alignment
    msgs, meta = [], []
    for e in eps:
        steps = templates[e["task"]]
        dur = (int(e["n"]) - 1) / float(e["fps"])
        content = image_block(frames(a.frames_root, e, a.fps, a.max_images))
        content.append({"type": "text", "text": ALIGN_PROMPT.format(
            n=len(content) // 2, duration=dur, task=e["description"],
            steps="\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps)))})
        msgs.append([{"role": "user", "content": content}])
        meta.append((e, steps, dur))
    sp = SamplingParams(temperature=0.0, max_tokens=1024, structured_outputs=StructuredOutputsParams(json=A_SCHEMA))
    n_ok = 0
    with open(a.out_dir / f"subtasks_{a.split}.jsonl", "w") as f:
        for (e, steps, dur), r in zip(meta, llm.chat(msgs, sp, **chat)):
            subs, ids = None, []
            try:
                segs = json.loads(r.outputs[0].text)["segments"]
                segs = sorted((s for s in segs if 1 <= int(s["step"]) <= len(steps)), key=lambda s: float(s["start"]))
                raw = clean([{"text": str(int(s["step"]) - 1), "start": s["start"], "end": s["end"]} for s in segs], dur)
                if raw:
                    bounds = [b for x in (flora.get(e["path"], {}).get("flora") or []) for b in (x["start_s"], x["end_s"])]
                    raw = snap(raw, bounds, a.snap)
                    merged = []
                    for x in raw:  # consecutive equal steps -> one subtask
                        if merged and merged[-1]["text"] == x["text"]:
                            merged[-1]["end"] = x["end"]
                        else:
                            merged.append(dict(x))
                    ids = [int(x["text"]) for x in merged]
                    subs = [{"text": steps[k], "start": x["start"], "end": x["end"]} for k, x in zip(ids, merged)]
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                subs = None
            ok = bool(subs)
            n_ok += ok
            if not ok:
                subs, ids = [{"text": e["description"], "start": 0.0, "end": round(dur, 3)}], []
            f.write(json.dumps({"path": e["path"], "description": e["description"], "duration": round(dur, 3),
                                "subtasks": subs, "ok": ok, "steps": ids}) + "\n")
    print(f"[align] {n_ok}/{len(eps)} aligned -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
