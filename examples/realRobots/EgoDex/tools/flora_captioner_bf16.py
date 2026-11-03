#!/usr/bin/env python3
"""Flora-Captioner (fp32 safetensors with Transformer Engine `_extra_state` entries) -> a bf16 copy vLLM loads: the
`_extra_state` tensors are dropped, the GDN A_log / dt_bias stay fp32, everything else is cast to bf16. The tokenizer,
processor and chat template files are copied as they are.

  flora_captioner_bf16.py [--src .../Flora-Captioner] [--out .../Flora-Captioner-bf16]
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PM = Path("/scratch/t509a04/workspace/starVLA/playground/Pretrained_models")
KEEP_FP32 = ("A_log", "dt_bias")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=PM / "Flora-Captioner")
    ap.add_argument("--out", type=Path, default=PM / "Flora-Captioner-bf16")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((a.src / "model.safetensors.index.json").read_text())
    weight_map = {}
    for shard in sorted(set(index["weight_map"].values())):
        sd = load_file(a.src / shard)
        out = {k: (v if k.endswith(KEEP_FP32) or not v.is_floating_point() else v.to(torch.bfloat16)).contiguous()
               for k, v in sd.items() if "_extra_state" not in k}
        save_file(out, a.out / shard, metadata={"format": "pt"})
        weight_map.update({k: shard for k in out})
        print(shard, len(sd), "->", len(out), flush=True)
    (a.out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2))
    for f in a.src.iterdir():
        if f.is_file() and not f.name.startswith("model") and f.suffix in (".json", ".jinja", ".txt"):
            shutil.copy(f, a.out / f.name)
    cfg = json.loads((a.out / "config.json").read_text())
    for c in (cfg, cfg.get("text_config", {})):
        for k in ("torch_dtype", "dtype"):
            if k in c:
                c[k] = "bfloat16"
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2))
    print("->", a.out, flush=True)


if __name__ == "__main__":
    main()
