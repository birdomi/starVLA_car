"""Streamed EgoDex frames for the episode-memory pretraining (starVLA/training/train_memory_pretrain.py).

Reads the per-episode JPEG files of examples/realRobots/EgoDex/tools/egodex_frames.py (<root>/index_<split>.jsonl, or
its shards index_<split>_<k>of<n>.jsonl).
Every batch position ("slot") walks whole episodes from frame 0, one step every `stride` +- `jitter` frames of the
stored fps; at an episode's end the slot moves on to a new random episode. The walk lives in the batch sampler (main
process), so batch k + 1 of a slot always continues batch k, also with prefetching workers.

A batch is a list of dicts: image (PIL), lang, progress t / (T - 1), elapsed (s), stream [slot, step in the episode,
episode counter of the slot], and with subtask annotations (annotate_subtasks.py output, `subtasks_dir`): subtasks
(texts of the episode's subtasks, [] when not annotated), sub_cur (index of the subtask in progress), sub_progress
((t - start) / (end - start) of that subtask, in [0, 1]), sub_done (bool per subtask: ended at or before the frame)."""
import glob
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler


def load_index(root: str, splits: Sequence[str], min_frames: int = 1) -> List[dict]:
    eps = []
    for s in splits:
        files = sorted(glob.glob(os.path.join(root, f"index_{s}.jsonl")) +
                       glob.glob(os.path.join(root, f"index_{s}_*of*.jsonl")))
        if not files:
            raise FileNotFoundError(f"egodex_stream: no index_{s}*.jsonl in {root}")
        for fn in files:
            with open(fn) as f:
                eps += [r for r in map(json.loads, f) if int(r["n"]) >= min_frames]
    if not eps:
        raise ValueError(f"egodex_stream: no episode with >= {min_frames} frames in {root} {list(splits)}")
    return eps


def load_subtasks(subtasks_dir: Optional[str]) -> Dict[str, list]:
    """path -> [{"text", "start", "end"}] (only episodes the VLM annotated, not the one-subtask fallbacks)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(subtasks_dir, "subtasks_*.jsonl"))) if subtasks_dir else []:
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # a line still being written by the annotator
                    continue
                if r.get("ok", True) and r["subtasks"]:  # a later row of a path (a redo) wins
                    out[r["path"]] = r["subtasks"]
    return out


class EgoDexFrames(Dataset):
    def __init__(self, root: str, episodes: List[dict], subtasks: Optional[Dict[str, list]] = None):
        self.root, self.episodes, self.subtasks = Path(root), episodes, subtasks or {}

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, item):
        slot, e, t, step, count = item
        ep = self.episodes[e]
        with np.load(self.root / ep["path"]) as z:
            o = z["offsets"]
            img = Image.open(io.BytesIO(z["jpeg"][o[t]:o[t + 1]].tobytes())).convert("RGB")
        n, sec = int(ep["n"]), t / float(ep["fps"])
        sub = self.subtasks.get(ep["path"], [])
        cur = max(next((j for j, s in enumerate(sub) if sec < s["end"]), len(sub) - 1), 0)
        sp = 0.0
        if sub:
            a, b = float(sub[cur]["start"]), float(sub[cur]["end"])
            sp = min(max((sec - a) / max(b - a, 1e-6), 0.0), 1.0)
        return {"image": img, "lang": ep["description"], "progress": t / max(n - 1, 1), "elapsed": sec,
                "stream": np.array([slot, step, count], dtype=np.int64), "subtasks": [s["text"] for s in sub],
                "sub_cur": cur, "sub_progress": sp, "sub_done": [s["end"] <= sec for s in sub]}


class StreamWalkSampler(Sampler):
    """Yields batches [(slot, episode, frame, step in the episode, episode counter)] forever."""

    def __init__(self, episodes: List[dict], batch_size: int, stride: int, jitter: int, seed: int):
        self.n = np.array([int(e["n"]) for e in episodes])
        self.b, self.stride, self.jitter = int(batch_size), int(stride), int(jitter)
        if not 0 <= self.jitter < self.stride:
            raise ValueError(f"StreamWalkSampler: need 0 <= jitter < stride, got {jitter}, {stride}")
        self.rng = np.random.default_rng(seed)

    def _new(self):
        return [int(self.rng.integers(len(self.n))), 0, 0]  # episode, frame, step

    def __iter__(self):
        walks = [self._new() for _ in range(self.b)]
        counts = [0] * self.b
        while True:
            yield [(s, e, t, k, counts[s]) for s, (e, t, k) in enumerate(walks)]
            for s, w in enumerate(walks):
                nxt = w[1] + self.stride + int(self.rng.integers(-self.jitter, self.jitter + 1))
                if nxt >= self.n[w[0]]:
                    walks[s], counts[s] = self._new(), counts[s] + 1
                else:
                    w[1], w[2] = nxt, w[2] + 1


class _Collate:
    """The batch as a list; with `prepare`, (batch, prepare(batch)) made in the worker (e.g. the VLMs' processor
    inputs, train_memory_pretrain.CpuInputs)."""

    def __init__(self, prepare=None):
        self.prepare = prepare

    def __call__(self, batch):
        return batch if self.prepare is None else (batch, self.prepare(batch))


def build_stream_loader(root: str, splits: Sequence[str], batch_size: int, stride: int = 10, jitter: int = 3,
                        min_frames: int = 1, seed: int = 0, num_workers: int = 8, prefetch_factor: int = 4,
                        subtasks_dir: Optional[str] = None, prepare=None):
    eps = load_index(root, splits, min_frames)
    ds = EgoDexFrames(root, eps, load_subtasks(subtasks_dir))
    loader = DataLoader(ds, batch_sampler=StreamWalkSampler(eps, batch_size, stride, jitter, seed),
                        num_workers=num_workers, collate_fn=_Collate(prepare),
                        prefetch_factor=prefetch_factor if num_workers else None,
                        persistent_workers=num_workers > 0, pin_memory=False)
    return loader, eps
