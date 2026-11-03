"""vLLM backend for Flora's reference pipeline (flora_subtasks.py --backend vllm).

Flora's own code keeps the windows, frame plans, prompts, stitching and caption blocks: its processor is wrapped so a
call records what it was handed (the chat text, the videos, their metadata) instead of building tensors, and the
pipeline's `generate` sends that record to one vLLM engine, which batches the requests of many episodes run in
threads. TempSeg is served from its LoRA merged into the base (flora_merge_tempseg.py).
"""
import queue
import threading
import time
from concurrent.futures import Future

import numpy as np

_local = threading.local()


class RecordingProcessor:
    """Flora's processor, with __call__ recording its inputs (and, with `count_tokens`, the HF prompt length)."""

    def __init__(self, proc, count_tokens: bool = False):
        self._proc, self._count = proc, count_tokens

    def __getattr__(self, name):
        return getattr(self._proc, name)

    def __call__(self, text=None, videos=None, video_metadata=None, **kw):
        _local.call = {"text": text, "videos": videos, "video_metadata": video_metadata, "kw": kw}
        if self._count:
            _local.call["hf_len"] = int(self._proc(text=text, videos=videos, video_metadata=video_metadata,
                                                   **kw)["input_ids"].shape[1])
        return {}


def _meta(m) -> dict:
    get = (lambda k: m.get(k)) if isinstance(m, dict) else (lambda k: getattr(m, k, None))
    out = {"fps": float(get("fps")), "frames_indices": [int(i) for i in get("frames_indices")],
           "total_num_frames": int(get("total_num_frames")), "do_sample_frames": False}
    out["duration"] = out["total_num_frames"] / out["fps"]
    return out


def _frames(v) -> np.ndarray:
    return np.stack([np.asarray(f.convert("RGB") if hasattr(f, "convert") else f) for f in v]).astype(np.uint8)


class Engine:
    """One vLLM engine; `generate` may be called from many threads, a batcher thread groups the requests."""

    def __init__(self, model: str, max_batch: int = 64, wait_s: float = 0.05, **llm_kw):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, **llm_kw)
        self.sp = {n: SamplingParams(temperature=0.0, max_tokens=n) for n in (512, 1024)}
        self.q: "queue.Queue[tuple]" = queue.Queue()
        self.max_batch, self.wait_s = max_batch, wait_s
        self.stats = {"requests": 0, "batches": 0, "gen_s": 0.0, "len_mismatch": 0, "len_checked": 0}
        threading.Thread(target=self._loop, daemon=True).start()

    def generate(self, _model, _proc, _inputs, budget):
        call = _local.call
        req = {"prompt": call["text"][0],
               "multi_modal_data": {"video": [(_frames(v), _meta(m)) for v, m in
                                              zip(call["videos"], call["video_metadata"])]},
               "mm_processor_kwargs": {"do_sample_frames": False}}
        fut: Future = Future()
        self.q.put((req, int(budget), fut, call.get("hf_len")))
        text = fut.result()
        if not text:
            raise RuntimeError("The model did not return a complete output for this input.")
        return text

    def _loop(self):
        while True:
            items = [self.q.get()]
            end = time.monotonic() + self.wait_s
            while len(items) < self.max_batch:
                try:
                    items.append(self.q.get(timeout=max(0.0, end - time.monotonic())))
                except queue.Empty:
                    break
            for budget in {b for _, b, _, _ in items}:
                group = [it for it in items if it[1] == budget]
                t0 = time.monotonic()
                try:
                    outs = self.llm.generate([g[0] for g in group], self.sp[budget], use_tqdm=False)
                except Exception as err:  # noqa: BLE001 - every waiting episode gets the error
                    for g in group:
                        g[2].set_exception(err)
                    continue
                self.stats["gen_s"] += time.monotonic() - t0
                self.stats["batches"] += 1
                self.stats["requests"] += len(group)
                for (req, _, fut, hf_len), o in zip(group, outs):
                    if hf_len is not None:
                        self.stats["len_checked"] += 1
                        self.stats["len_mismatch"] += int(len(o.prompt_token_ids) != hf_len)
                    c = o.outputs[0]
                    fut.set_result(c.text.strip() if c.finish_reason == "stop" else "")
