"""Data preprocessing for QwenGR00T_TriSystem: C-RADIOv4 feature statistics.

Samples [frame t, ..., frame t+H] examples from the yaml's training mixture (same loader, same
obs_image_size), encodes frame t and the last future frame with the frozen C-RADIOv4 encoder (same
resolution / feature / outlier mask as the model) and writes

    z_mean, z_std [D_c]  per-channel mean / std of the valid patch tokens (both frames)
    dz_rms        [D_c]  per-channel RMS of dZ = z_n(t+H) - z_n(t)   (System 1.5 data = dZ / dz_rms)

to framework.radio.stats_path (torch.save, loaded by PatchFeatureStats) plus a .json summary.

  CUDA_VISIBLE_DEVICES=5 python starVLA/training/trisystem_prepare_stats.py \
      --config_yaml examples/realRobots/H1_2_car_real/train_files/trisystem/stage1_system15.yaml --num_samples 4096
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.modules.tri_system.radio_encoder import RadioPatchEncoder
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def _collate(batch):
    return batch


def _frames(examples, frame_size, device):
    h, w = frame_size
    arr = np.stack([np.stack([np.asarray(im.convert("RGB").resize((w, h)) if im.size != (w, h) else im.convert("RGB"))
                              for im in (ex["image"][0], ex["image"][-1])]) for ex in examples])
    return torch.from_numpy(arr).to(device).permute(0, 1, 4, 2, 3).float().div_(255.0)  # [B, 2, 3, H, W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_yaml", required=True)
    ap.add_argument("--num_samples", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="default: framework.radio.stats_path")
    ap.add_argument("--seed", type=int, default=0)
    args, rest = ap.parse_known_args()
    # config overrides in the trainer's form: --key=value or --key value
    cfg = OmegaConf.merge(OmegaConf.load(args.config_yaml), OmegaConf.from_dotlist(normalize_dotlist_args(rest)))
    rc, tri = cfg.framework.radio, cfg.framework.tri_system
    out = args.out or rc.get("stats_path", None)
    if not out:
        raise ValueError("no output path: set framework.radio.stats_path or --out")
    frame_size = tuple(int(v) for v in tri.frame_size)  # loader frames; C-RADIO resizes them to radio.input_size
    radio_size = tuple(int(v) for v in rc.get("input_size", frame_size))
    device = torch.device("cuda")

    data_cfg = cfg.datasets.vla_data
    data_cfg.data_root_dir = Path(data_cfg.data_root_dir)
    dataset = get_vla_dataset(data_cfg=data_cfg)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)
    kw = dict(batch_size=args.batch_size, collate_fn=_collate, num_workers=args.num_workers)
    if args.num_workers > 0 and data_cfg.get("multiprocessing_context", None):
        kw["multiprocessing_context"] = str(data_cfg.multiprocessing_context)
    loader = DataLoader(Subset(dataset, idx.tolist()), **kw)

    enc = RadioPatchEncoder(rc.path, feature=rc.get("feature", "backbone"), input_size=radio_size,
                            outlier_norm=float(rc.get("outlier_norm", 100.0)), mask_outliers=bool(rc.get("mask_outliers", True)),
                            dtype=rc.get("dtype", "bfloat16"), teacher_ckpt=rc.get("teacher_ckpt", None)).to(device).eval()
    d = enc.feat_dim
    s1 = torch.zeros(d, dtype=torch.float64, device=device)
    s2 = torch.zeros(d, dtype=torch.float64, device=device)
    sd2 = torch.zeros(d, dtype=torch.float64, device=device)
    n_tok = n_pair = 0
    n_invalid = n_all = 0
    per_sample_change = []
    t0 = time.time()
    with torch.no_grad():
        for b, examples in enumerate(loader):
            fr = _frames(examples, frame_size, device)
            z, valid = enc(fr.flatten(0, 1))
            z = z.double().unflatten(0, (fr.shape[0], 2))  # [B, 2, N, D]
            valid = valid.unflatten(0, (fr.shape[0], 2))
            both = valid[:, 0] & valid[:, 1]  # [B, N]
            zv = z[valid]  # [M, D]
            s1 += zv.sum(0)
            s2 += (zv * zv).sum(0)
            n_tok += zv.shape[0]
            dz = (z[:, 1] - z[:, 0])[both]
            sd2 += (dz * dz).sum(0)
            n_pair += dz.shape[0]
            n_invalid += int((~valid).sum())
            n_all += valid.numel()
            raw = (z[:, 1] - z[:, 0]).pow(2).sum(-1).sqrt() / z[:, 0].pow(2).sum(-1).sqrt().clamp_min(1e-6)
            per_sample_change += [(raw[i][both[i]]).mean().item() for i in range(raw.shape[0])]
            if b % 10 == 0:
                print(f"[{b * args.batch_size}/{len(idx)}] {time.time() - t0:.0f}s", flush=True)
    mean = s1 / n_tok
    std = (s2 / n_tok - mean * mean).clamp_min(1e-12).sqrt()
    dz_rms = (sd2 / n_pair).sqrt() / std  # RMS of the change in normalized units
    pc = np.asarray(per_sample_change)
    meta = {
        "config_yaml": args.config_yaml, "data_mix": str(data_cfg.data_mix), "num_samples": int(len(idx)),
        "frame_size": list(frame_size), "radio_input_size": list(radio_size), "feature": str(rc.get("feature", "backbone")),
        "future_offsets": [int(v) for v in tri.future_offsets], "feat_dim": d, "tokens_per_frame": int(z.shape[2]),
        "invalid_token_frac": n_invalid / max(n_all, 1),
        "dz_rms_normalized": {"mean": float(dz_rms.mean()), "min": float(dz_rms.min()), "max": float(dz_rms.max())},
        "raw_relative_change_per_sample": {q: float(np.quantile(pc, p)) for q, p in
                                           [("p10", 0.1), ("p50", 0.5), ("p90", 0.9)]},
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save({"z_mean": mean.float().cpu(), "z_std": std.float().cpu(), "dz_rms": dz_rms.float().cpu(), "meta": meta}, out)
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
