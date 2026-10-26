#!/usr/bin/env python3
"""Fit action-time primitives directly to demo chunks, without a VLM.

This measures representation fit only, NOT policy generalization/success.
Each chunk gets its own optimizable bank, initialized from that chunk.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from starVLA.model.modules.action_model.gaussian_action_field import GaussianActionField


def read_egodex(root, horizon, max_chunks):
    """Read complete future chunks; same rel conversion and per-key norm as training."""
    import pyarrow.parquet as pq
    from starVLA.dataloader.gr00t_lerobot.transform.state_action import Normalizer

    root = Path(root)
    modalities = json.loads((root / "meta/modality.json").read_text())["action"]
    stats = json.loads((root / "meta/stats_gr00t.rel.json").read_text())["statistics"]["action"]
    keys = sorted(modalities.items(), key=lambda item: item[1]["start"])
    normalizers = []
    for name, spec in keys:
        lo, hi = spec["start"], spec["end"]
        selected = {k: v[lo:hi] for k, v in stats.items() if k in ("q01", "q99", "min", "max")}
        normalizers.append(Normalizer("min_max" if "rot6d" in name else "q99", selected))
    chunks, sources = [], []
    for directory in sorted((root / "data").glob("chunk-*")):
        for path in sorted(directory.glob("*.parquet")):
            data = pq.read_table(path, columns=["action", "observation.state", "episode_index", "frame_index"]).to_pydict()
            episode = np.asarray(data["episode_index"])
            for ep in np.unique(episode):
                idx = np.flatnonzero(episode == ep)
                frames = np.asarray(data["frame_index"])[idx]
                order = np.argsort(frames)
                idx, frames = idx[order], frames[order]
                if len(idx) < horizon:
                    continue
                # Up to two disjoint chunks per episode, no endpoint padding.
                for start in range(0, len(idx) - horizon + 1, max(horizon, len(idx) // 2)):
                    if not np.all(np.diff(frames[start:start + horizon]) == 1):
                        continue
                    a = np.asarray([data["action"][i] for i in idx[start:start + horizon]], dtype=np.float32)
                    state = np.asarray(data["observation.state"][idx[start]], dtype=np.float32)
                    pieces = []
                    for (name, spec), normalizer in zip(keys, normalizers):
                        lo, hi = spec["start"], spec["end"]
                        values = a[:, lo:hi].copy()
                        if "rot6d" not in name:
                            values -= state[lo:hi]
                        pieces.append(normalizer.forward(torch.from_numpy(values)))
                    chunks.append(torch.cat(pieces, -1))
                    sources.append({"file": str(path), "episode": int(ep), "frame": int(frames[start])})
                    if len(chunks) == max_chunks:
                        return torch.stack(chunks), sources
    if not chunks:
        raise ValueError("No complete action chunks found")
    return torch.stack(chunks), sources


def fit(target, count, steps, lr):
    batch, horizon, dim = target.shape
    field = GaussianActionField(dim, horizon, count)
    bank = torch.nn.Parameter(torch.zeros(batch, count, field.parameter_dim))
    # Oracle initialization: sample demonstrated action centers at anchor times.
    indices = (field.center_logits.sigmoid() * (horizon - 1)).round().long()
    with torch.no_grad():
        bank[..., 3:3 + dim] = target[:, indices]
    optimizer = torch.optim.Adam([bank], lr=lr)
    initial_nll = -field.log_prob(bank, target).mean().item() / dim
    initial_pred = field.select_actions(bank)
    initial_rmse = (initial_pred - target).square().mean().sqrt().item()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = -field.log_prob(bank, target).mean() / dim
        loss.backward()
        torch.nn.utils.clip_grad_norm_([bank], 10.0)
        optimizer.step()
    with torch.no_grad():
        prediction = field.select_actions(bank)
        residual = prediction - target
        report = {
            "num_gaussians": count, "steps": steps, "initial_nll_per_dim": initial_nll,
            "final_nll_per_dim": -field.log_prob(bank, target).mean().item() / dim,
            "initial_rmse": initial_rmse, "final_rmse": residual.square().mean().sqrt().item(),
            "final_mae": residual.abs().mean().item(),
        }
    return report, bank.detach().numpy(), prediction.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--actions", type=Path, help="Normalized finite [B,H,D] .npy actions")
    source.add_argument("--egodex-root", type=Path, help="LeRobot EgoDex root with stats_gr00t.rel.json")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--max-chunks", type=int, default=16)
    parser.add_argument("--num-gaussians", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.max_chunks < 1 or args.horizon < 1 or args.lr <= 0:
        parser.error("steps, max-chunks, horizon and lr must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if args.actions:
        target = torch.from_numpy(np.load(args.actions, allow_pickle=False)).float()[:args.max_chunks]
        sources = [{"file": str(args.actions)}]
    else:
        target, sources = read_egodex(args.egodex_root, args.horizon, args.max_chunks)
    if target.ndim != 3 or min(target.shape) < 1 or not torch.isfinite(target).all():
        raise ValueError("Expected nonempty, finite [B,H,D] targets")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"experiment": "per-chunk oracle fitting; no observation policy or success evaluation",
              "shape": list(target.shape), "sources": sources, "fits": []}
    for count in args.num_gaussians:
        result, bank, prediction = fit(target, count, args.steps, args.lr)
        report["fits"].append(result)
        np.savez_compressed(args.out / f"fit_k{count}.npz", parameters=bank,
                            prediction=prediction, target=target.numpy())
        print(json.dumps(result), flush=True)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
