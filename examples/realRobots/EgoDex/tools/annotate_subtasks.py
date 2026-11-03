#!/usr/bin/env python3
"""Subtask annotation of EgoDex episodes with a large VLM (vLLM offline, e.g. Qwen3.8-Flash-Next-FP8).

Input: the frame files of egodex_frames.py (<frames_root>/index_<split>.jsonl, 10 fps JPEGs). Every episode goes to the
VLM as timestamped frames (--fps, default 2) + its task description; the VLM splits the task into the 2..max subtasks
performed in this video with start / end times (JSON-constrained output). Written per shard:
  <out_dir>/subtasks_<split>_<shard>of<n>.jsonl:
  {"path", "description", "duration": s, "subtasks": [{"text", "start", "end"}, ...]}  (s, contiguous, cover [0, duration])
Resumable: paths already in the shard's output are skipped.

Run with the vLLM env (/home01/t509a04/envs/vllm), one process per node (tensor parallel over its GPUs):
  annotate_subtasks.py <frames_root> <out_dir> --split test --model <path> --shard 0 --num-shards 1 [--tp 4]
"""

import argparse
import base64
import io
import json
import random
from pathlib import Path

import numpy as np

PROMPT = """These {n} images are frames of one egocentric video ({duration:.1f} s), in order; the text before each image \
is its timestamp. The person performs the task: "{task}".

Split what the person actually does in this video into {lo} to {hi} consecutive subtasks (short imperative phrases, \
e.g. "pick up the lid", "place the lid on the cup"). Give each subtask's start and end time in seconds: the first starts \
at 0, each one starts where the previous ends, the last ends at {duration:.1f}. Answer with JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {"subtasks": {"type": "array", "minItems": 1, "items": {
        "type": "object",
        "properties": {"text": {"type": "string"}, "start": {"type": "number"}, "end": {"type": "number"}},
        "required": ["text", "start", "end"]}}},
    "required": ["subtasks"],
}


def frames_b64(npz: Path, step: int):
    with np.load(npz) as z:
        o, jpeg = z["offsets"], z["jpeg"]
        idx = list(range(0, len(o) - 1, step))
        return idx, [base64.b64encode(jpeg[o[i]:o[i + 1]].tobytes()).decode() for i in idx]


def build_messages(ep, root: Path, fps_in: float, max_images: int, lo: int, hi: int):
    step = max(1, round(float(ep["fps"]) / fps_in))
    n_all = int(ep["n"])
    if n_all / step > max_images:  # long episode: spread max_images frames
        step = int(np.ceil(n_all / max_images))
    idx, imgs = frames_b64(root / ep["path"], step)
    duration = (n_all - 1) / float(ep["fps"])
    content = []
    for i, b in zip(idx, imgs):
        content.append({"type": "text", "text": f"t={i / float(ep['fps']):.1f}s"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    content.append({"type": "text", "text": PROMPT.format(n=len(idx), duration=duration, task=ep["description"],
                                                          lo=lo, hi=hi)})
    return [{"role": "user", "content": content}], duration


def clean(subtasks, duration: float):
    """-> contiguous [{"text", "start", "end"}] covering [0, duration], or None."""
    s = []
    for x in subtasks or []:
        try:
            t, a, b = str(x["text"]).strip(), float(x["start"]), float(x["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if t:
            s.append([t, min(max(a, 0.0), duration), min(max(b, 0.0), duration)])
    if not s:
        return None
    s.sort(key=lambda x: x[1])
    for i in range(1, len(s)):  # snap the boundaries: start_i = end_{i-1} (their midpoint when they disagree)
        m = 0.5 * (s[i - 1][2] + s[i][1]) if s[i][1] != s[i - 1][2] else s[i][1]
        s[i - 1][2] = s[i][1] = m
    s[0][1], s[-1][2] = 0.0, duration
    s = [x for x in s if x[2] - x[1] > 1e-3]
    return [{"text": t, "start": round(a, 3), "end": round(b, 3)} for t, a, b in s] or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-images", type=int, default=96)
    ap.add_argument("--min-subtasks", type=int, default=2)
    ap.add_argument("--max-subtasks", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=256, help="episodes per vLLM call (output flushed after each)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="random subset of the split (before sharding), 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--moe-backend", default="triton",
                    help="Hopper: triton (flashinfer's fused MoE JIT does not build with the system gcc 7)")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    with open(a.frames_root / f"index_{a.split}.jsonl") as f:
        eps = [json.loads(l) for l in f]
    if a.sample:  # same subset as flora_subtasks.py --sample / --seed
        eps = random.Random(a.seed).sample(eps, min(a.sample, len(eps)))
    eps = eps[a.shard::a.num_shards]
    if a.limit:
        eps = eps[: a.limit]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"subtasks_{a.split}_{a.shard}of{a.num_shards}.jsonl"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {json.loads(l)["path"] for l in f if l.strip()}
    eps = [e for e in eps if e["path"] not in done]
    print(f"[annotate] {a.split} shard {a.shard}/{a.num_shards}: {len(eps)} to do ({len(done)} done)", flush=True)
    if not eps:
        return

    llm = LLM(model=a.model, tensor_parallel_size=a.tp, enable_expert_parallel=a.tp > 1, max_model_len=a.max_model_len,
              gpu_memory_utilization=0.88, kv_cache_dtype="fp8", enable_prefix_caching=False,
              limit_mm_per_prompt={"image": a.max_images}, trust_remote_code=True, moe_backend=a.moe_backend)
    try:  # vLLM >= 0.11
        from vllm.sampling_params import StructuredOutputsParams
        sp = SamplingParams(temperature=0.0, max_tokens=1024, structured_outputs=StructuredOutputsParams(json=SCHEMA))
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        sp = SamplingParams(temperature=0.0, max_tokens=1024, guided_decoding=GuidedDecodingParams(json=SCHEMA))

    n_ok = n_bad = 0
    for c0 in range(0, len(eps), a.chunk):
        chunk = eps[c0:c0 + a.chunk]
        built = [build_messages(e, a.frames_root, a.fps, a.max_images, a.min_subtasks, a.max_subtasks) for e in chunk]
        res = llm.chat([m for m, _ in built], sp, use_tqdm=False,
                       chat_template_kwargs={"enable_thinking": False})
        with open(out, "a") as f:
            for e, (_, dur), r in zip(chunk, built, res):
                try:
                    sub = clean(json.loads(r.outputs[0].text).get("subtasks"), dur)
                except (json.JSONDecodeError, AttributeError):
                    sub = None
                ok = sub is not None
                if ok:
                    n_ok += 1
                else:
                    n_bad += 1
                    sub = [{"text": e["description"], "start": 0.0, "end": round(dur, 3)}]  # fallback: one subtask
                f.write(json.dumps({"path": e["path"], "description": e["description"], "duration": round(dur, 3),
                                    "subtasks": sub, "ok": ok}) + "\n")
        print(f"[annotate] {c0 + len(chunk)}/{len(eps)} (ok {n_ok}, fallback {n_bad})", flush=True)


if __name__ == "__main__":
    main()
