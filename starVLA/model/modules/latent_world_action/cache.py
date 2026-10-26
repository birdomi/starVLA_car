"""TensorCache: a small on-disk cache of per-key tensors (frozen C-RADIO features, predicted System 1.5 hidden states).

Layout: <root>/meta.json (what the cached tensors depend on) + one torch.save file per key (<root>/<key>.pt, the key's
"/" become sub-directories). Writes are atomic (tmp file + os.replace), so concurrent ranks / dataloader workers never
read a partial file. A cache opened with different meta (e.g. another C-RADIO resolution or other System 1.5
weights) raises instead of silently returning stale tensors.
"""

import json
import os
import re
import uuid
from typing import Dict, Optional

import torch

_SAFE = re.compile(r"[^A-Za-z0-9_.\-/]")


class TensorCache:
    def __init__(self, root: str, meta: Dict, readonly: bool = False):
        self.root, self.readonly = os.path.abspath(root), readonly
        self.meta = json.loads(json.dumps(meta, sort_keys=True, default=str))
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, "meta.json")
        if os.path.exists(path):
            with open(path) as f:
                stored = json.load(f)
            if stored != self.meta:
                diff = {k: (stored.get(k), self.meta.get(k)) for k in set(stored) | set(self.meta) if stored.get(k) != self.meta.get(k)}
                raise ValueError(f"TensorCache {self.root}: cached tensors were made with different settings "
                                 f"{{key: (cached, current)}} = {diff}; use another cache_dir or delete it")
        elif not readonly:
            self._atomic_write(path, lambda f: f.write(json.dumps(self.meta, indent=1, sort_keys=True).encode()))
        self.hits = self.misses = 0

    def _path(self, key: str) -> str:
        key = _SAFE.sub("_", str(key)).strip("/")
        if not key or any(part in ("", ".", "..") for part in key.split("/")):
            raise ValueError(f"invalid cache key {key!r}")
        return os.path.join(self.root, key + ".pt")

    @staticmethod
    def _atomic_write(path: str, write_fn) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{uuid.uuid4().hex}.tmp"
        with open(tmp, "wb") as f:
            write_fn(f)
        os.replace(tmp, path)

    def get(self, key: str) -> Optional[Dict[str, torch.Tensor]]:
        path = self._path(key)
        if os.path.exists(path):
            try:
                out = torch.load(path, map_location="cpu", weights_only=True)
                self.hits += 1
                return out
            except Exception:  # truncated by a crash before os.replace cannot happen; anything else = recompute
                pass
        self.misses += 1
        return None

    def put(self, key: str, tensors: Dict[str, torch.Tensor]) -> None:
        if self.readonly:
            return
        cpu = {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}
        self._atomic_write(self._path(key), lambda f: torch.save(cpu, f))
