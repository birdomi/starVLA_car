"""Inference sampling for QwenGR00T_TriSystem on validation episodes.

For every sampled step t of the chosen episodes: current frame + instruction -> [System 2 S_t] + [C-RADIO Z_t]
-> System 1.5 ODE -> dZ_hat -> System 1 Action DiT ODE -> action chunk. Writes per episode
    episode<k>.mp4   current | GT future (t+H) | |dZ| GT | |dZ_hat|   (per-patch change magnitude)
    episode<k>.npz   predicted / GT normalized action chunks per step
and metrics.json (System 1.5 rel_mse / cos; System 1 action RMSE with dZ_hat, with the GT dZ and without dZ).

  CUDA_VISIBLE_DEVICES=5 python starVLA/training/trisystem_infer.py \
      --checkpoint playground/Checkpoints/<run>/checkpoints/steps_50000_pytorch_model.pt \
      --data_mix car_few_tasks_v1_val_front_future30 --episodes 0 1 2 --stride 15 --out playground/eval_results/<run>
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework  # noqa: F401  (registers frameworks)
from starVLA.model.framework.VLM4A.QwenGR00T_TriSystem import Qwen_GR00T_TriSystem


def write_mp4(path, frames, fps):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0, "ffmpeg failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_mix", default=None, help="default: datasets.val_data.data_mix of the run")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--stride", type=int, default=15, help="frames between sampled steps t")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--system15_steps", type=int, default=None)
    ap.add_argument("--system1_steps", type=int, default=None, help="action ODE steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = Qwen_GR00T_TriSystem.from_pretrained(args.checkpoint).cuda().eval()
    cfg = model.config
    if args.system15_steps is not None:
        cfg.framework.tri_system.eval.system15_steps = args.system15_steps
    if args.system1_steps is not None:
        cfg.framework.tri_system.eval.system1_steps = args.system1_steps
    # dataset settings from the run's complete config (config.yaml only keeps the keys the training accessed)
    full_yaml = Path(args.checkpoint).parents[1] / "config.full.yaml"
    dsets = OmegaConf.load(full_yaml).datasets if full_yaml.exists() else cfg.datasets
    data_cfg = OmegaConf.create(OmegaConf.to_container(dsets.get("val_data", None) or dsets.vla_data, resolve=True))
    if args.data_mix:
        data_cfg.data_mix = args.data_mix
    data_cfg.data_root_dir = Path(data_cfg.data_root_dir)
    ds = get_vla_dataset(data_cfg=data_cfg).datasets[0]  # LeRobotSingleDataset
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ep in args.episodes:
        traj = ds.trajectory_ids[ep]
        length = int(ds.trajectory_lengths[ep])
        steps = list(range(0, length, args.stride))
        rows, metrics, a_pred, a_gt = [], [], [], []
        for b in range(0, len(steps), args.batch_size):
            exs = [ds._pack_sample(ds.transforms(ds.get_step_data(traj, t)), trajectory_id=traj, base_index=t)
                   for t in steps[b:b + args.batch_size]]
            res = model.evaluate_batch(exs, seed=args.seed + b)
            h = res["grid"].shape[0] // len(exs)
            rows += [res["grid"][i * h:(i + 1) * h] for i in range(len(exs))]
            metrics.append({k: v * len(exs) for k, v in res["metrics"].items()})
            a_pred.append(res["actions_pred"])
            a_gt.append(np.stack([np.asarray(ex["action"], dtype=np.float32)[-model.action_horizon:] for ex in exs]))
        n = len(steps)
        agg = {k: sum(m[k] for m in metrics) / n for k in metrics[0]}
        write_mp4(out_dir / f"episode{ep}.mp4", rows, args.fps)
        np.savez(out_dir / f"episode{ep}.npz", steps=np.asarray(steps), actions_pred=np.concatenate(a_pred),
                 actions_gt=np.concatenate(a_gt))
        summary[f"episode{ep}"] = {"trajectory_id": int(traj), "steps": steps, "metrics": agg}
        print(f"episode {ep} (traj {traj}, {n} steps): " + ", ".join(f"{k} {v:.4f}" for k, v in agg.items()))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"checkpoint": args.checkpoint, "data_mix": str(data_cfg.data_mix), "episodes": summary}, f, indent=1)
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
