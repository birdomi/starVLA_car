#!/usr/bin/env python3
"""Direct point fitting on normalized demo chunks; no VLM/policy training."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from starVLA.model.modules.action_model.temporal_action_splat import TemporalActionSplat


def fit(target, count, steps, lr, delta_weight):
    batch, horizon, dim = target.shape
    splat = TemporalActionSplat(dim, horizon, count)
    points = torch.nn.Parameter(torch.zeros(batch, count, splat.parameter_dim))
    indices = (splat.anchors * (horizon - 1)).round().long()
    with torch.no_grad():
        points[..., :dim] = target[:, indices]
    optimizer = torch.optim.Adam([points], lr=lr)
    initial_rmse = (splat(points) - target).square().mean().sqrt().item()
    for _ in range(steps):
        optimizer.zero_grad()
        error = splat(points) - target
        loss = error.abs().mean()
        if horizon > 1:
            loss = loss + delta_weight * (error[:, 1:] - error[:, :-1]).abs().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = splat(points)
        error = prediction - target
        metrics = {"num_points": count, "steps": steps, "lr": lr, "delta_weight": delta_weight,
                   "initial_rmse": initial_rmse, "final_rmse": error.square().mean().sqrt().item(),
                   "final_mae": error.abs().mean().item()}
    return metrics, points.detach().numpy(), prediction.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True, help="[B,H,D] .npy, or .npz with target key")
    parser.add_argument("--max-chunks", type=int, default=16)
    parser.add_argument("--num-points", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--delta-weight", type=float, default=0.1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.max_chunks < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("steps, max-chunks and lr must be positive and finite")
    if not np.isfinite(args.delta_weight) or args.delta_weight < 0:
        parser.error("delta-weight must be finite and nonnegative")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    loaded = np.load(args.actions, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            array = loaded["target"]
    else:
        array = loaded
    if array.ndim != 3 or min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError("Expected finite, nonempty normalized [B,H,D] actions")
    target = torch.from_numpy(array[:args.max_chunks]).float()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"experiment": "per-chunk oracle point fit; no observation policy or task-success evaluation",
              "source": str(args.actions), "shape": list(target.shape), "fits": []}
    for count in args.num_points:
        metrics, points, prediction = fit(target, count, args.steps, args.lr, args.delta_weight)
        report["fits"].append(metrics)
        np.savez_compressed(args.out / f"fit_k{count}.npz", points=points, prediction=prediction, target=target.numpy())
        print(json.dumps(metrics), flush=True)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
