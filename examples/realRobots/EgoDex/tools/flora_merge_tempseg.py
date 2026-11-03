#!/usr/bin/env python3
"""Merge the Flora-TempSeg LoRA (fp32, rank 32, also on the GDN projections) into its Qwen3.5-9B base, for serving
with vLLM (whose LoRA path does not cover the linear-attention projections). Merged in fp32 on the CPU, saved bf16; the
trained processor / tokenizer / chat template of the TempSeg repo are copied next to the weights.

  flora_merge_tempseg.py [--base .../Qwen3.5-9B] [--adapter .../Flora-TempSeg] [--out .../Flora-TempSeg-merged]
Run with the Flora env (/home01/t509a04/envs/flora312) on a node with ~80 GB of free RAM.
"""
import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

PM = Path("/scratch/t509a04/workspace/starVLA/playground/Pretrained_models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=PM / "Qwen3.5-9B")
    ap.add_argument("--adapter", type=Path, default=PM / "Flora-TempSeg")
    ap.add_argument("--out", type=Path, default=PM / "Flora-TempSeg-merged")
    a = ap.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.float32, local_files_only=True)
    model = PeftModel.from_pretrained(model, a.adapter, local_files_only=True).merge_and_unload()
    model.to(torch.bfloat16).save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
    for f in ("processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(a.adapter / f, a.out / f)
    for f in a.base.glob("*.json"):  # e.g. preprocessor / video_preprocessor configs the TempSeg repo does not ship
        if not (a.out / f.name).exists():
            shutil.copy(f, a.out / f.name)
    print("merged ->", a.out, flush=True)


if __name__ == "__main__":
    main()
