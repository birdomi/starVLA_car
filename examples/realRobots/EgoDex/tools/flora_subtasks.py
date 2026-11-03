#!/usr/bin/env python3
"""Subtask annotation of EgoDex episodes with NVIDIA Flora (Flora-TempSeg -> Flora-Captioner), through the models' own
reference pipeline (batch_utils/ in the Flora-TempSeg repo: native 16 s / 4 Hz windows, stitching, BEFORE / TARGET /
AFTER caption blocks) on local weights, human profile.

Two passes per shard so each model loads once: TempSeg over all episodes (intervals -> flora_segments_*.jsonl), then the
Captioner over all intervals. Written per shard, in annotate_subtasks.py's format (read by
starVLA/dataloader/egodex_stream.py):
  <out_dir>/subtasks_<split>_<shard>of<n>.jsonl:
  {"path", "description", "duration": s, "subtasks": [{"text", "start", "end"}, ...] (contiguous, cover [0, duration]),
   "ok", "flora": [{"start_s", "end_s", "caption", "status"}, ...] (Flora's raw intervals: may overlap / leave gaps)}
`ok` is false (one subtask = the task description) when Flora fails on the episode or returns no complete caption.
Resumable: paths already in a shard's outputs are skipped.

Videos: the row's "video" path when the index has one (with "left_wrist" / "right_wrist" for the three-camera profiles,
e.g. examples/realRobots/AgiBot/tools/agibot_index.py), else <raw_root>/<split>/<task>/<N>.mp4 or members of
<raw_root>/<split>.zip (extracted one at a time to --tmp).
Profiles (--profile): human (one camera, the default), or dex / tianji (Flora's robot profiles: the main camera above
the two wrist cameras in one mosaic for TempSeg, three-view caption blocks for the Captioner).
Episodes: <frames_root>/<--index> (default index_<split>*.jsonl, the converter's index or its shards), then every
--num-shards-th from --shard.

Backends, one process per GPU:
  hf (reference): the Flora env (/home01/t509a04/envs/flora312: transformers 5.8.1, peft 0.18.1), both passes:
    CUDA_VISIBLE_DEVICES=0 flora_subtasks.py <frames_root> <raw_root> <out_dir> --split test --shard 0 --num-shards 4
  vllm (flora_vllm.py; ~20x faster, LoRA merged by flora_merge_tempseg.py, captioner as flora_captioner_bf16.py): the
    vLLM env (/home01/t509a04/envs/vllm129), one stage per process, episodes in --threads threads:
    flora_subtasks.py ... --backend vllm --stage tempseg && flora_subtasks.py ... --backend vllm --stage caption
"""

import argparse
import glob
import json
import threading
import random
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotate_subtasks import clean  # noqa: E402

PM = Path("/scratch/t509a04/workspace/starVLA/playground/Pretrained_models")
QUIET = {"generation_start", "generation_end", "manifest_verified_reuse"}


def setup(tempseg: Path, base: Path, captioner: Path):
    """Import batch_utils with its downloads pointed at the local copies; -> (pipeline, media, CachedSession)."""
    sys.path.insert(0, str(tempseg))
    import huggingface_hub
    from batch_utils import media, models, pipeline

    local = {"tempseg": tempseg, "captioner": captioner}
    models.snapshot = lambda kind: local[kind]
    hub_download = huggingface_hub.snapshot_download

    def snapshot_download(repo_id, *a, **k):  # the TempSeg base (Qwen3.5-9B) is resolved inside Session.load
        return str(base) if repo_id == models.MODELS["tempseg_base"]["repo"] else hub_download(repo_id, *a, **k)

    huggingface_hub.snapshot_download = snapshot_download
    event = models.runtime_event

    def runtime_event(name, **values):
        if name not in QUIET:
            event(name, **values)

    models.runtime_event = pipeline.runtime_event = runtime_event

    class CachedSession(models.Session):
        """Keeps the loaded model across episodes (the reference Session reloads it on every load())."""

        kind = None

        def load(self, kind):
            if self.kind != kind:
                super().load(kind)
                self.kind = kind
            return self.model, self.proc

        def close(self):
            self.kind = None
            super().close()

    return pipeline, media, CachedSession


def seeking_decode(media):
    """media.decode_selected that seeks to the keyframe before the first wanted frame instead of decoding from frame 0.
    Flora decodes per TempSeg window and, with three cameras, per caption segment and camera, each time from the first
    frame: O(segments x length) frames per episode, the bottleneck of a three-camera run (one 80 s AgiBot episode,
    ~40k frames decoded under the GIL). Frames are matched to their index by the probed timestamps and converted as
    the original does; anything unexpected (a seek past the first wanted frame, a timestamp not in the probe, a frame
    left over) falls back to the original."""
    import bisect
    from fractions import Fraction

    import av

    orig = media.decode_selected

    def decode_selected(video, wanted, canonical=False):
        wanted = set(wanted)
        if not wanted or min(wanted) < 0 or max(wanted) >= len(video.pts_us) or min(wanted) < 120:
            return orig(video, wanted, canonical)  # invalid (orig raises), or near the start: nothing to skip
        lo, hi, result = min(wanted), max(wanted), {}
        with av.open(str(video.path), options={"err_detect": "explode"}) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            zero = None
            for frame in container.decode(stream):  # the first frame's pts: the origin of video.pts_us
                zero = frame.pts
                break
            container.seek(zero + int(Fraction(video.pts_us[lo], 1000000) / stream.time_base), stream=stream,
                           backward=True, any_frame=False)
            first = True
            for frame in container.decode(stream):
                t = Fraction(frame.pts - zero) * stream.time_base * 1000000
                i = bisect.bisect_left(video.pts_us, t)
                if i >= len(video.pts_us) or video.pts_us[i] != t or (first and i > lo):
                    return orig(video, wanted, canonical)
                first = False
                if i in wanted:
                    use_canonical = (canonical and frame.format.name == "yuv420p" and frame.width % 16 == 0
                                     and frame.height % 2 == 0 and int(frame.colorspace) in (2, 5, 6)
                                     and int(frame.color_range) in (0, 1))
                    if use_canonical:
                        from canonical_rgb import canonical_rgb
                        result[i] = canonical_rgb(frame)
                    else:
                        result[i] = frame.to_image().convert("RGB")
                if i >= hi:
                    break
        return result if set(result) == wanted else orig(video, wanted, canonical)

    media.decode_selected = decode_selected  # frames_for() looks it up in media's globals


class Videos:
    """Episode mp4 by its frames-index path, from <raw_root>/<split>/ or <raw_root>/<split>.zip."""

    def __init__(self, raw_root: Path, split: str, tmp: str):
        self.dir = raw_root / split
        self.zip = None if self.dir.is_dir() else zipfile.ZipFile(raw_root / f"{split}.zip")
        Path(tmp).mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="flora_", dir=tmp))

    def get(self, path: str) -> Path:
        rel = Path(path).with_suffix(".mp4")  # <split>/<task>/<N>.mp4
        if self.zip is None:
            return self.dir.parent / rel
        out = self.tmp / f"{rel.parent.name}__{rel.name}"  # episodes of different tasks share numbers
        with self.zip.open(str(rel)) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return out

    def done(self, p: Path):
        if self.zip is not None:
            p.unlink(missing_ok=True)

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def to_subtasks(flora, duration: float):
    """Flora's complete intervals -> contiguous subtasks (annotate_subtasks.clean), or None. Flora keeps parallel /
    nested intervals; one that overlaps an already kept (longer-first) interval by more than half of its own length
    is dropped, smaller overlaps / gaps are split at their midpoint by clean()."""
    iv = sorted(((c["start_s"], c["end_s"], c["caption"].strip()) for c in flora
                 if c["status"] == "complete" and c["caption"].strip() and c["end_s"] > c["start_s"]),
                key=lambda x: x[0] - x[1])  # longest first
    kept = []
    for a, b, t in iv:
        if all(min(b, kb) - max(a, ka) <= 0.5 * (b - a) for ka, kb, _ in kept):
            kept.append((a, b, t))
    return clean([{"text": t, "start": a, "end": b} for a, b, t in sorted(kept)], duration)


# a dead inference engine is no property of the episode: such rows are redone on a rerun, and the process stops
ENGINE_ERRORS = ("EngineDeadError", "OutOfMemoryError", "EngineCore")


def engine_error(err: BaseException) -> bool:
    return any(k in f"{type(err).__name__}: {err}" for k in ENGINE_ERRORS)


class MemoryBudget:
    """Bounds the decoded frames in flight: the captioner decodes every frame of an episode's caption blocks at full
    resolution at once (a long episode: thousands of 1080p frames), so episodes take a share by their estimated frames;
    one that alone exceeds the budget runs by itself."""

    def __init__(self, gb: float):
        self.cap, self.used, self.cv = gb * 2 ** 30, 0.0, threading.Condition()

    def __enter__(self):
        return self

    def take(self, nbytes: float):
        with self.cv:
            while self.used > 0 and self.used + nbytes > self.cap:
                self.cv.wait()
            self.used += nbytes
        return nbytes

    def give(self, nbytes: float):
        with self.cv:
            self.used -= nbytes
            self.cv.notify_all()
        try:  # hand the freed frames back to the OS (glibc keeps them in its arenas otherwise)
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass


def read_paths(f: Path):
    """Paths already written to f, but those that failed on the engine (to redo)."""
    if not f.exists():
        return set()
    out = set()
    for l in open(f):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:  # a line cut by a killed process
            continue
        err = r.get("error") or ""
        if err != "missing" and not any(k in err for k in ENGINE_ERRORS):  # missing: its TempSeg row had not been made
            out.add(r["path"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_root", type=Path)
    ap.add_argument("raw_root", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--split", required=True, help="one split or several, comma-separated (one engine for all)")
    ap.add_argument("--index", default=None,
                    help="index file glob under frames_root, {split} filled in (default index_{split}*.jsonl)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--tag", default=None, help="output name part (default <split>_<shard>of<num_shards>)")
    ap.add_argument("--segments-tag", default=None,
                    help="read / write the TempSeg rows under this tag (a finished shard split over more processes)")
    ap.add_argument("--reverse", action="store_true", help="last episodes first (a helper meeting another process)")
    ap.add_argument("--skip-done-glob", default=None,
                    help="also skip the paths done in these subtasks files (glob under out_dir), e.g. another process's")
    ap.add_argument("--sample", type=int, default=0, help="random subset of the split (before sharding), 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instruction", choices=["none", "description"], default="none",
                    help="pass the EgoDex task description as Flora's optional episode instruction")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--stage", choices=["all", "tempseg", "caption"], default="all")
    ap.add_argument("--threads", type=int, default=64, help="vllm: episodes in flight")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--mem-budget-gb", type=float, default=0.0,
                    help="caption stage: host RAM for decoded frames in flight (0 = unbounded)")
    ap.add_argument("--tempseg", type=Path, default=PM / "Flora-TempSeg")
    ap.add_argument("--tempseg-merged", type=Path, default=PM / "Flora-TempSeg-merged")
    ap.add_argument("--base", type=Path, default=PM / "Qwen3.5-9B")
    ap.add_argument("--captioner", type=Path, default=PM / "Flora-Captioner")
    ap.add_argument("--captioner-bf16", type=Path, default=PM / "Flora-Captioner-bf16")
    ap.add_argument("--tmp", default="/dev/shm")
    ap.add_argument("--decode-from-start", action="store_true",
                    help="Flora's own frame decoding (from frame 0 on every call) instead of seeking to a keyframe")
    ap.add_argument("--profile", choices=["human", "dex", "tianji"], default="human",
                    help="Flora action profile; dex / tianji need the index's left_wrist / right_wrist videos")
    ap.add_argument("--max-duration-s", type=float, default=0.0,
                    help="raise Flora's input limit (batch_utils media.probe: 120 s / 7200 frames, a demo-API guard; "
                         "segmentation runs on 16 s windows either way), 0 = keep it. AgiBot episodes reach ~430 s")
    a = ap.parse_args()
    if a.backend == "vllm" and a.stage == "all":
        ap.error("--backend vllm runs one --stage per process (tempseg, then caption)")

    splits = a.split.split(",")
    eps = []
    for sp in splits:
        for fn in sorted(glob.glob(str(a.frames_root / (a.index or "index_{split}*.jsonl").format(split=sp)))):
            with open(fn) as f:
                eps += [dict(json.loads(l), split=sp) for l in f if l.strip()]
    if a.sample:
        eps = random.Random(a.seed).sample(eps, min(a.sample, len(eps)))
    eps = eps[a.shard::a.num_shards]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{'+'.join(splits)}_{a.shard}of{a.num_shards}"
    seg_f = a.out_dir / f"flora_segments_{a.segments_tag or tag}.jsonl"
    out_f = a.out_dir / f"subtasks_{tag}.jsonl"
    done = read_paths(out_f)
    for f in sorted(a.out_dir.glob(a.skip_done_glob)) if a.skip_done_glob else []:
        done |= read_paths(f)
    eps = [e for e in eps if e["path"] not in done]
    if a.reverse:
        eps = eps[::-1]
    segmented = read_paths(seg_f)
    print(f"[flora] {tag}: {len(eps)} to do ({len(done)} done, {len(segmented)} segmented)", flush=True)
    if not eps or (a.stage == "tempseg" and all(e["path"] in segmented for e in eps)):
        return

    pipeline, media, CachedSession = setup(a.tempseg.resolve(), a.base.resolve(), a.captioner.resolve())
    if not a.decode_from_start:
        seeking_decode(media)
    if a.max_duration_s:  # media.py imported these names from config: patch them where probe() reads them
        media.MAX_DURATION_S, media.MAX_FRAME_COUNT = a.max_duration_s, int(a.max_duration_s * 60)
    videos = {}  # split -> Videos, for rows without their own video paths
    three = a.profile != "human"
    canon = {"human": "Human", "dex": "Dex", "tianji": "Tianji"}[a.profile]

    def open_videos(e):
        """-> ({role: mp4 path}, release)"""
        if e.get("video"):
            roles = {"main": e["video"]}
            if three:
                roles.update(left_wrist=e["left_wrist"], right_wrist=e["right_wrist"])
            return {r: Path(x) for r, x in roles.items()}, (lambda: None)
        if three:
            raise ValueError("the dex / tianji profiles need the index's video / left_wrist / right_wrist paths")
        if e["split"] not in videos:
            videos[e["split"]] = Videos(a.raw_root, e["split"], a.tmp)
        vs = videos[e["split"]]
        p = vs.get(e["path"])
        return {"main": p}, (lambda: vs.done(p))

    def probe_all(paths):
        v = {r: media.probe(x) for r, x in paths.items()}
        media.check_synchronized(v)  # three views: same clock (the reference pipeline checks this in annotate())
        return v
    instr = (lambda e: e["description"]) if a.instruction == "description" else (lambda e: "")
    quiet = lambda *x, **k: None  # noqa: E731
    if a.backend == "hf":
        session = CachedSession()
    else:
        from batch_utils import models
        from flora_vllm import Engine, RecordingProcessor

        kind = "tempseg" if a.stage == "tempseg" else "captioner"
        proc = RecordingProcessor(models.processor(kind, (a.tempseg if kind == "tempseg" else a.captioner).resolve()))

        class VllmSession:  # the processor only; generation goes to the engine
            def load(self, _kind):
                return None, proc

            def close(self):
                pass

        session = VllmSession()
        engine = Engine(str(a.tempseg_merged if kind == "tempseg" else a.captioner_bf16), max_batch=a.threads,
                        tensor_parallel_size=1, dtype="bfloat16", gpu_memory_utilization=a.gpu_memory_utilization,
                        max_model_len=32768, limit_mm_per_prompt={"video": 1 if kind == "tempseg" else 3},
                        enable_prefix_caching=False, max_num_seqs=2 * a.threads)  # one request per thread in flight
        pipeline.generate = engine.generate
    lock = threading.Lock()

    def run(fn, items, f, label):
        t0, k = time.time(), 0
        for row in (map(fn, items) if a.backend == "hf" else ThreadPoolExecutor(a.threads).map(fn, items)):
            with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
                k += 1
                if k % 200 == 0 or k == len(items):
                    print(f"[flora] {label} {k}/{len(items)} ({(time.time() - t0) / k:.2f} s/episode)", flush=True)

    def segment_one(e):
        row, release = {"path": e["path"]}, (lambda: None)
        try:
            paths, release = open_videos(e)
            v = probe_all(paths)
            segs, _ = pipeline.segment(v, a.profile, session, instr(e), quiet)
            row.update(duration=v["main"].duration_s, segments=[
                {"start_s": s["start_s"], "end_s": s["end_s"],
                 "provenance": s["metadata"].get("segmentation_provenance", "fine")} for s in segs])
        except Exception as err:  # noqa: BLE001 - recorded, the episode falls back to one subtask
            if engine_error(err):
                raise
            row["error"] = f"{type(err).__name__}: {err}"
            if not isinstance(err, (ValueError, RuntimeError)):
                traceback.print_exc()
        finally:
            release()
        return row

    # pass 1: TempSeg intervals
    if a.stage in ("all", "tempseg"):
        with open(seg_f, "a") as f:
            run(segment_one, [x for x in eps if x["path"] not in segmented], f, "tempseg")
        session.close()
        if a.stage == "tempseg":
            for vs in videos.values():
                vs.close()
            print(f"[flora] {tag} tempseg done", flush=True)
            return
    seg_rows = {}
    for l in open(seg_f):  # the last row of a path wins (a redo after an engine error)
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get("path"):
            seg_rows[r["path"]] = r

    # pass 2: captions of the intervals
    counts = {"ok": 0, "fallback": 0}

    budget = MemoryBudget(a.mem_budget_gb) if a.mem_budget_gb > 0 else None

    def caption_one(e):
        r = dict(seg_rows.get(e["path"], {"error": "missing"}))
        dur = float(r.get("duration", (int(e["n"]) - 1) / float(e["fps"])))
        raw = []
        # decoded RGB per frame of every camera (index "frame_bytes"; else 1080p ~6.3 MB), twice for a three-view
        # mosaic; the single-view captioner decodes the frames of all its caption blocks (up to 64 per caption) at once,
        # the three-view one one caption at a time (its BEFORE / TARGET / AFTER blocks, bounded here by 192 frames)
        fb = float(e.get("frame_bytes") or 6.3e6) * (2 if three else 1)
        frames = 192 if three else 64 * len(r.get("segments") or [])
        need = min(frames, int(dur * float(e.get("fps", 30))) + 1) * fb
        held = budget.take(need) if budget is not None and r.get("segments") else 0.0
        try:
            return _caption(e, r, dur, raw)
        finally:
            if held:
                budget.give(held)

    def _caption(e, r, dur, raw):
        if r.get("segments"):
            release = lambda: None  # noqa: E731
            try:
                paths, release = open_videos(e)
                v = probe_all(paths)
                segs = [{"start_s": s["start_s"], "end_s": s["end_s"], "metadata": {}} for s in r["segments"]]
                if three:
                    caps, _ = pipeline.caption_three(v, segs, session, instr(e), quiet, profile=canon)
                else:
                    caps, _ = pipeline.caption_single(v, segs, "Human", session, instr(e), quiet)
                raw = [{"start_s": c["start_s"], "end_s": c["end_s"], "caption": c["caption"], "status": c["status"]}
                       for c in caps]
            except Exception as err:  # noqa: BLE001
                if engine_error(err):
                    raise
                r["error"] = f"{type(err).__name__}: {err}"
                if not isinstance(err, (ValueError, RuntimeError)):
                    traceback.print_exc()
            finally:
                release()
        sub = to_subtasks(raw, dur)
        ok = sub is not None
        with lock:
            counts["ok" if ok else "fallback"] += 1
        if not ok:
            sub = [{"text": e["description"], "start": 0.0, "end": round(dur, 3)}]
        row = {"path": e["path"], "description": e["description"], "duration": round(dur, 3), "subtasks": sub,
               "ok": ok, "flora": raw}
        if r.get("error"):
            row["error"] = r["error"]
        return row

    with open(out_f, "a") as f:
        run(caption_one, eps, f, "captions")
    session.close()
    for vs in videos.values():
        vs.close()
    print(f"[flora] {tag} done: ok {counts['ok']}, fallback {counts['fallback']}", flush=True)


if __name__ == "__main__":
    main()
