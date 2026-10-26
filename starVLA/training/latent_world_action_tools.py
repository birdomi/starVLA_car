"""QwenLatentWorldAction utilities (run from the repo root with PYTHONPATH=.).

  stats              C-RADIO feature statistics (z_mean, z_std, dz_rms) -> framework.radio.stats_path
  convert-trisystem  QwenGR00T_TriSystem stage-1 checkpoint -> System 1.5 weights + stats of this framework
  cache-hidden       precompute the predicted System 1.5 hidden states H_f of every training step
                     (stage B with system1.hidden_source: cache)
  infer              offline inference + metrics of a trained checkpoint on validation examples

Examples (RoboCasa-100):
  Y=examples/simBenchmarks/Robocasa_tabletop/train_files/latent_world_action
  python starVLA/training/latent_world_action_tools.py stats --config_yaml $Y/joint_rc100.yaml
  python starVLA/training/latent_world_action_tools.py convert-trisystem \
      --src playground/Checkpoints/starvla_trisystem_s1p5_rc100_robocasa_gr1/checkpoints/steps_30000_pytorch_model.pt \
      --out playground/Checkpoints/lwa_from_trisystem_s1p5_rc100/system15_steps_30000.pt
  python starVLA/training/latent_world_action_tools.py cache-hidden --config_yaml $Y/stageB_system1_rc100.yaml
  python starVLA/training/latent_world_action_tools.py infer \
      --checkpoint playground/Checkpoints/<run>/checkpoints/steps_N_pytorch_model.pt --num_samples 64
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset


def _collate(batch):
    return batch


def _load_cfg(path, rest):
    from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

    return OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(normalize_dotlist_args(rest)))


def _loader(dataset, idx, batch_size, num_workers, data_cfg):
    kw = dict(batch_size=batch_size, collate_fn=_collate, num_workers=num_workers)
    if num_workers > 0 and data_cfg.get("multiprocessing_context", None):
        kw["multiprocessing_context"] = str(data_cfg.multiprocessing_context)
    return DataLoader(Subset(dataset, [int(i) for i in idx]), **kw)


# ----------------------------------------------------------------------------------------------- stats
def cmd_stats(args, rest):
    """Per-channel mean / std of the valid C-RADIO patch tokens (current + future frames) and RMS of the change
    dZ = z_n(future) - z_n(current) in normalized units, over examples whose future frame was not clamped."""
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from starVLA.model.framework.share_tools import merge_framework_config
    from starVLA.model.framework.VLM4A.QwenLatentWorldAction import LatentWorldActionConfig
    from starVLA.model.modules.latent_world_action import CRADIOVisualEncoder

    cfg = merge_framework_config(LatentWorldActionConfig, _load_cfg(args.config_yaml, rest))
    rc, lw = cfg.framework.radio, cfg.framework.latent_world
    out = args.out or rc.get("stats_path", None)
    if not out:
        raise ValueError("no output path: set framework.radio.stats_path or --out")
    layout = list(lw.image_layout)
    t_i = [i for i, v in enumerate(layout) if v == "t"][int(lw.radio_view)]
    f_i = [i for i, v in enumerate(layout) if v == "f"][-1]
    data_cfg = cfg.datasets.vla_data
    data_cfg.data_root_dir = Path(data_cfg.data_root_dir)
    data_cfg.emit_frame_meta = True
    dataset = get_vla_dataset(data_cfg=data_cfg)
    idx = np.random.default_rng(args.seed).choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)
    loader = _loader(dataset, idx, args.batch_size, args.num_workers, data_cfg)
    device = torch.device("cuda")
    enc = CRADIOVisualEncoder.from_config(OmegaConf.merge(rc, {"cache_dir": None, "stats_path": None})).to(device).eval()
    size = tuple(int(v) for v in data_cfg.obs_image_size)  # (W, H)
    d = enc.feat_dim
    s1, s2, sd2 = (torch.zeros(d, dtype=torch.float64, device=device) for _ in range(3))
    n_tok = n_pair = n_ex = n_clamped = 0
    t0 = time.time()
    with torch.no_grad():
        for b, examples in enumerate(loader):
            keep = [ex for ex in examples if ex.get("future_valid", True)]
            n_clamped += len(examples) - len(keep)
            if not keep:
                continue
            frames = np.stack([np.stack([np.asarray(ex["image"][i].convert("RGB").resize(size)) for i in (t_i, f_i)])
                               for ex in keep])
            x = torch.from_numpy(frames).to(device).flatten(0, 1).permute(0, 3, 1, 2).float().div_(255.0)
            z, valid = enc.encode_raw(x)
            z, valid = z.double().unflatten(0, (len(keep), 2)), valid.unflatten(0, (len(keep), 2))
            zv = z[valid]
            s1 += zv.sum(0)
            s2 += (zv * zv).sum(0)
            n_tok += zv.shape[0]
            dz = (z[:, 1] - z[:, 0])[valid[:, 0] & valid[:, 1]]
            sd2 += (dz * dz).sum(0)
            n_pair += dz.shape[0]
            n_ex += len(keep)
            if b % 10 == 0:
                print(f"[{(b + 1) * args.batch_size}/{len(idx)}] {time.time() - t0:.0f}s", flush=True)
    mean = s1 / n_tok
    std = (s2 / n_tok - mean * mean).clamp_min(1e-12).sqrt()
    dz_rms = (sd2 / n_pair).sqrt() / std
    meta = {"config_yaml": args.config_yaml, "data_mix": str(data_cfg.data_mix), "num_examples": n_ex,
            "num_clamped_skipped": n_clamped, "radio_path": str(rc.path), "radio_input_size": list(rc.input_size),
            "feature": str(rc.feature), "image_layout": layout, "future_offset": int(lw.future_offset), "feat_dim": d,
            "tokens_per_frame": int(z.shape[2]),
            "dz_rms_normalized": {"mean": float(dz_rms.mean()), "min": float(dz_rms.min()), "max": float(dz_rms.max())}}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save({"z_mean": mean.float().cpu(), "z_std": std.float().cpu(), "dz_rms": dz_rms.float().cpu(), "meta": meta}, out)
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1), "\nsaved", out)


# ----------------------------------------------------------------------------------------------- convert
def cmd_convert_trisystem(args, rest):
    from starVLA.model.modules.latent_world_action.convert import convert_trisystem_system15

    sd = torch.load(args.src, map_location="cpu", weights_only=True)
    out = convert_trisystem_system15(sd)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    run_cfg = Path(args.src).parent.parent / "config.yaml"
    hint = {}
    if run_cfg.exists():
        fw = OmegaConf.to_container(OmegaConf.load(run_cfg).framework, resolve=True)
        s15, rc = fw.get("system15", {}), fw.get("radio", {})
        hint = {"framework.system15": {"adaln_cond": "tau+s2", "d_model": s15.get("d_model"), "depth": s15.get("depth"),
                                       "num_heads": s15.get("num_heads"), "mlp_ratio": s15.get("mlp_ratio", 4.0)},
                "framework.radio": {k: rc.get(k) for k in ("path", "input_size", "feature", "outlier_norm") if k in rc}}
    print(f"wrote {len(out)} tensors -> {args.out}\nuse it with trainer.pretrained_checkpoint: {args.out}\n"
          f"            trainer.reload_modules: system15,radio_encoder.stats\nand these framework settings:\n"
          f"{json.dumps(hint, indent=1)}")


# ----------------------------------------------------------------------------------------------- cache-hidden
def _build_for_training(cfg, checkpoint=None):
    """Build the framework from a training yaml and load weights exactly like the trainer
    (trainer.pretrained_checkpoint + trainer.reload_modules)."""
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

    model = build_framework(cfg)
    ckpt = checkpoint or cfg.trainer.get("pretrained_checkpoint", None)
    if ckpt:
        model = TrainerUtils.load_pretrained_backbones(model, ckpt, reload_modules=cfg.trainer.get("reload_modules", None))
    return model


def cmd_cache_hidden(args, rest):
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    cfg = _load_cfg(args.config_yaml, rest)
    cfg.datasets.vla_data.emit_frame_meta = True
    model = _build_for_training(cfg, args.checkpoint).cuda().eval()
    if model.hidden_source != "cache":
        raise ValueError("the yaml must set framework.system1.hidden_source: cache (+ hidden_cache_dir)")
    data_cfg = cfg.datasets.vla_data
    data_cfg.data_root_dir = Path(data_cfg.data_root_dir)
    steps = ConcatDataset(get_vla_dataset(data_cfg=data_cfg).datasets)  # every (episode, step) once
    idx = np.arange(args.shard, len(steps), args.num_shards)[: args.max_samples or None]
    loader = _loader(steps, idx, args.batch_size, args.num_workers, data_cfg)
    t0, n, miss = time.time(), 0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b, examples in enumerate(loader):
            model.shapes.reset()
            miss += model.cached_hidden(examples)[2]
            n += len(examples)
            if b % 20 == 0:
                rate = n / max(time.time() - t0, 1e-6)
                print(f"[{n}/{len(idx)}] computed {miss}, {rate:.1f} samples/s, eta {(len(idx) - n) / rate / 60:.0f} min",
                      flush=True)
    print(f"done: {n} samples, {miss} computed, cache {model._hidden_cache_store().root}")


# ----------------------------------------------------------------------------------------------- infer
def cmd_infer(args, rest):
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from starVLA.model.framework.base_framework import baseframework
    from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

    overrides = normalize_dotlist_args(rest) or None  # e.g. --framework.system15.sample_steps=8
    model = baseframework.from_pretrained(args.checkpoint, config_overrides=overrides).cuda().eval()
    if args.use_bf16:
        model = model.to(torch.bfloat16)
    ds_cfg = model.config.datasets
    data_cfg = ds_cfg.get("val_data", None) if args.split == "val" else None
    data_cfg = data_cfg if data_cfg is not None else ds_cfg.vla_data
    data_cfg = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True)) if not isinstance(data_cfg, dict) \
        else OmegaConf.create(data_cfg)
    data_cfg.data_root_dir = Path(data_cfg.data_root_dir)
    data_cfg.emit_frame_meta = True
    ds = get_vla_dataset(data_cfg=data_cfg)
    rng = np.random.default_rng(args.seed)
    examples = [ds[int(i)] for i in rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)]
    t0 = time.time()
    metrics = model.evaluate_batch(examples, seed=args.seed)
    # the deployment path: current view(s) + instruction + state only
    current = [{"image": [ex["image"][i] for i in model.t_idx], "lang": ex["lang"],
                **({"state": ex["state"]} if "state" in ex else {})} for ex in examples[: args.num_predict]]
    t1 = time.time()
    pred = model.predict_action(current, seed=args.seed, return_latent=True)
    t_pred = (time.time() - t1) / max(len(current), 1)
    gt = np.stack([np.asarray(ex["action"], dtype=np.float32)[-model.action_horizon:] for ex in examples[: args.num_predict]])
    metrics["predict_action/rmse"] = float(np.sqrt(((pred["normalized_actions"] - gt) ** 2).mean()))
    metrics["predict_action/sec_per_example_batched"] = t_pred
    out = Path(args.out or Path(args.checkpoint).parent.parent / "infer")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions.npz", actions_pred=pred["normalized_actions"], actions_gt=gt,
                        latent_change=pred["latent_change"], valid=pred["valid"], lang=np.array([c["lang"] for c in current]))
    with open(out / "metrics.json", "w") as f:
        json.dump({"checkpoint": args.checkpoint, "split": args.split, "num_samples": len(examples),
                   "metrics": metrics, "eval_sec": time.time() - t0}, f, indent=1)
    print(json.dumps(metrics, indent=1), f"\nsaved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats")
    p.add_argument("--config_yaml", required=True)
    p.add_argument("--num_samples", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("convert-trisystem")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("cache-hidden")
    p.add_argument("--config_yaml", required=True)
    p.add_argument("--checkpoint", default=None, help="default: trainer.pretrained_checkpoint of the yaml")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_samples", type=int, default=0)
    p = sub.add_parser("infer")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=("val", "train"), default="val")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--num_predict", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--out", default=None)
    args, rest = ap.parse_known_args()
    {"stats": cmd_stats, "convert-trisystem": cmd_convert_trisystem, "cache-hidden": cmd_cache_hidden,
     "infer": cmd_infer}[args.cmd](args, rest)


if __name__ == "__main__":
    main()
