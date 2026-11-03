#!/usr/bin/env python3
"""Per-step predictions of a pretrained episode memory on held-out EgoDex streams, for inspection.

Loads a train_memory_pretrain.py checkpoint (steps_<k>.pt / latest.pt) and one of its VLMs, walks test episodes from
frame 0 every data.stride frames (no jitter, the memory carried over the whole walk, as evaluate()), and writes per
episode and step: the ground truth (episode / subtask progress, current subtask, finished subtasks), the predictions
(expected progress, the subtask distribution over the candidates, finished-subtask probabilities, expected subtask
progress) and the frame (JPEG, base64) -> one JSON file.
  memory_eval_dump.py <run dir> <out.json> [--ckpt steps_10000.pt] [--vlm qwen35_2b] [--episodes 24] [--gpu-gb 8]
"""
import argparse
import base64
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from starVLA.dataloader.egodex_stream import EgoDexFrames, load_index, load_subtasks  # noqa: E402
from starVLA.model.modules.dual_system.memory_pretrain import (FrozenT5Encoder, MultiVLMMemory,  # noqa: E402
                                                               expected_progress, vlm_key)
from starVLA.training.train_memory_pretrain import load_vlm, vlm_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ckpt", default="steps_10000.pt")
    ap.add_argument("--vlm", default="qwen35_2b")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--gpu-gb", type=float, default=0.0, help="cap this process's GPU memory (sharing a GPU)")
    ap.add_argument("--thumb", type=int, default=224, help="frame width in the dump")
    ap.add_argument("--root", default=None, help="frames root (default: the run's data.root)")
    ap.add_argument("--split", default=None, help="split to walk (default: the run's eval.splits)")
    ap.add_argument("--subtasks-dir", default=None, help="subtask annotations (default: the run's data.subtasks_dir)")
    ap.add_argument("--per-task", type=int, default=0,
                    help="pick up to this many episodes per task (all episodes of the split) instead of the eval sample")
    a = ap.parse_args()

    dev = torch.device("cuda", 0)
    if a.gpu_gb:
        torch.cuda.set_per_process_memory_fraction(a.gpu_gb * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
    cfg = OmegaConf.load(a.run_dir / "config.yaml")
    sd = torch.load(a.run_dir / "checkpoints" / a.ckpt, map_location="cpu", weights_only=False)
    step = int(sd["step"])
    # every VLM's width from its token_resize, so the checkpoint loads whole
    dims = {}
    for v in cfg.vlms:
        w = sd["model"][f"token_resize.{vlm_key(v.name)}.0.weight"]
        dims[v.name] = int(w.shape[0])
    mc, dc = cfg.memory, cfg.decision
    t5 = FrozenT5Encoder.from_pretrained(str(REPO / cfg.t5.path), dev)
    model = MultiVLMMemory(dims, token_dim=mc.token_dim, num_tokens=mc.num_tokens, d_model=mc.d_model, depth=mc.depth,
                           num_heads=mc.num_heads, mlp_ratio=mc.mlp_ratio, progress_bins=mc.progress_bins,
                           attention=mc.attention, t5_dim=t5.dim, decision_layers=dc.num_layers,
                           decision_heads=dc.num_heads, decision_mlp_ratio=dc.mlp_ratio,
                           decision_progress_bins=dc.progress_bins, subtask_temperature=dc.subtask_temperature)
    model.load_state_dict(sd["model"])
    model.to(dev).eval()
    spec = next(v for v in cfg.vlms if v.name == a.vlm)
    iface, _ = load_vlm(OmegaConf.to_container(spec), dev)

    # the evaluate() sample (same seed), then a spread over the lengths, annotated episodes only
    root = a.root or cfg.data.root
    eps = load_index(root, [a.split] if a.split else list(cfg.eval.splits), int(cfg.data.get("min_frames", 1)))
    if not a.per_task:
        eps = random.Random(0).sample(eps, int(cfg.eval.max_episodes))
    subs = load_subtasks(a.subtasks_dir or cfg.data.subtasks_dir)
    ds = EgoDexFrames(root, eps, subs)
    stride, cap = int(cfg.data.stride), int(cfg.eval.max_updates)
    cand = [i for i, e in enumerate(eps) if len(subs.get(e["path"], [])) >= 2]
    if a.per_task:
        seen = {}
        pick = []
        for i in sorted(cand, key=lambda i: (eps[i]["task"], eps[i]["episode"])):
            if seen.get(eps[i]["task"], 0) < a.per_task:
                seen[eps[i]["task"]] = seen.get(eps[i]["task"], 0) + 1
                pick.append(i)
    else:
        cand.sort(key=lambda i: eps[i]["n"])
        pick = [cand[int(j)] for j in np.linspace(0, len(cand) - 1, a.episodes)]

    out = {"step": step, "vlm": a.vlm, "ckpt": a.ckpt, "episodes": []}
    for i in pick:
        e = eps[i]
        n = int(e["n"])
        k_max = min((n - 1) // stride + 1, cap)
        m = model.init(1).to(dev)
        texts = subs[e["path"]]
        feats, cvalid = t5.candidates([[s["text"] for s in texts]])
        rows = []
        for k in range(k_max):
            ex = ds[(0, i, k * stride, k, 0)]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h, mask = vlm_frames(iface, [ex["image"]], [ex["lang"]])
                m = model.update(a.vlm, m, h, mask)
                hd = model.decision_heads(model.decide(m), feats, cvalid)
            im = ex["image"].copy()
            im.thumbnail((a.thumb, a.thumb))
            b = io.BytesIO()
            im.save(b, "JPEG", quality=80)
            rows.append({
                "t": round(k * stride / float(e["fps"]), 2), "img": base64.b64encode(b.getvalue()).decode(),
                "progress": round(float(ex["progress"]), 4), "progress_pred": round(float(hd["progress"][0]), 4),
                "progress_probs": [round(float(x), 4) for x in hd["progress_probs"][0]],
                "sub_cur": int(ex["sub_cur"]), "sub_probs": [round(float(x), 4) for x in hd["subtask_logits"][0].float().softmax(-1)],
                "sub_done": [bool(x) for x in ex["sub_done"]],
                "done_probs": [round(float(x), 4) for x in torch.sigmoid(hd["done_logits"][0].float())],
                "sub_progress": round(float(ex["sub_progress"]), 4),
                "sub_progress_pred": round(float(hd["subtask_progress"][0]), 4),
            })
        out["episodes"].append({"path": e["path"], "task": e["task"], "description": e["description"],
                                "duration": round((n - 1) / float(e["fps"]), 2), "subtasks": texts, "steps": rows})
        print(f"{e['path']}: {k_max} steps", flush=True)
    a.out.write_text(json.dumps(out))
    print(f"-> {a.out} (checkpoint step {step}, {a.vlm})", flush=True)


if __name__ == "__main__":
    main()
