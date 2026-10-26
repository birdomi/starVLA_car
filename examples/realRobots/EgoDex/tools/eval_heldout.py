#!/usr/bin/env python3
"""Held-out evaluation for EgoDex action heads (regressor / WTA / flow-matching).

Normalisation is done with the MODEL's own training statistics
(run_dir/dataset_statistics.json), never the test set's, so different runs
are comparable. Errors are reported in raw delta space (metres per step) and
as absolute keypoint position error (metres) after reconstructing the
trajectory from the raw state at t.

Usage:
  eval_heldout.py --run_dir <run> [--ckpt steps_40000] [--stride 30] [--batch_size 64] [--max_samples N] [--out res.json]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image

sys.path.insert(0, "/mnt/DATA2/yyg/starVLA")
ROT_DIMS = list(range(3, 9)) + list(range(27, 33))          # wrist rot6d (min_max normalised, absolute)
POS_DIMS = [d for d in range(48) if d not in ROT_DIMS]      # 12 keypoints x 3 (q99 normalised, delta)
KEYPOINTS = {"L_wrist": (0, 3), "L_thumb": (9, 12), "L_index": (12, 15), "L_middle": (15, 18), "L_ring": (18, 21), "L_pinky": (21, 24),
             "R_wrist": (24, 27), "R_thumb": (33, 36), "R_index": (36, 39), "R_middle": (39, 42), "R_ring": (42, 45), "R_pinky": (45, 48)}


def make_norm(stats):
    q01, q99 = np.array(stats["q01"], np.float64), np.array(stats["q99"], np.float64)
    mn, mx = np.array(stats["min"], np.float64), np.array(stats["max"], np.float64)
    lo, hi = q01.copy(), q99.copy(); lo[ROT_DIMS], hi[ROT_DIMS] = mn[ROT_DIMS], mx[ROT_DIMS]
    ok = hi != lo
    def norm(x):  # raw -> [-1,1] (clamped like the training transform)
        y = x.astype(np.float64).copy(); y[..., ok] = np.clip(2 * (x[..., ok] - lo[ok]) / (hi[ok] - lo[ok]) - 1, -2.2, 2.2); return y
    def denorm(y):
        x = y.astype(np.float64).copy(); x[..., ok] = (y[..., ok] + 1) / 2 * (hi[ok] - lo[ok]) + lo[ok]; return x
    return norm, denorm


def reconstruct_abs(state_pos, deltas, p0, mode="delta"):
    """delta (bidirectional, anchored at p0) or rel (a[k]-s[t]) -> absolute keypoint positions. state_pos (B,36) at t, deltas (B,W,36)."""
    if mode == "rel":
        return state_pos[:, None, :] + deltas
    B, W, D = deltas.shape; out = np.zeros_like(deltas)
    out[:, p0] = state_pos + deltas[:, p0]
    for i in range(p0 + 1, W): out[:, i] = out[:, i - 1] + deltas[:, i]
    for i in range(p0 - 1, -1, -1): out[:, i] = out[:, i + 1] + deltas[:, i]
    return out


def find_single_dataset(ds):
    seen, found = set(), []
    def walk(o, d=0):
        if id(o) in seen or d > 4: return
        seen.add(id(o))
        if hasattr(o, "get_step_data") and hasattr(o, "all_steps"): found.append(o); return
        for v in (vars(o).values() if hasattr(o, "__dict__") else []):
            if isinstance(v, (list, tuple)):
                for x in v: walk(x, d + 1)
            elif hasattr(v, "__dict__"): walk(v, d + 1)
    walk(ds); return found[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True); ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--stride", type=int, default=30); ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_samples", type=int, default=0); ap.add_argument("--out", default="")
    ap.add_argument("--test_mix", default=""); ap.add_argument("--past_frame", default="data", choices=["data", "same"], help="two-frame runs: 'same' replaces the past frame(s) with the current one (ablation)"); ap.add_argument("--img_size", default="auto", help="N or WxH resize before the VLM processor; 'raw' = native frame. Training packs 224x224 (LeRobotSingleDataset._pack_sample), so keep the default unless the dataset resize is changed"); ap.add_argument("--dump", default="", help="save per-sample errors (.npz)"); a = ap.parse_args()

    _seen = []; _img_size_cfg = [None]
    def _prep_img(arr):
        """Training-time image path: the dataset resizes every frame to 224x224 before the Qwen processor (datasets.py _pack_sample)."""
        im = Image.fromarray(arr)
        if not _seen:
            _seen.append(1); print(f"[eval] raw frame {im.size[0]}x{im.size[1]} -> img_size={a.img_size}", flush=True)
        if str(a.img_size).lower() == "raw":
            return im
        if str(a.img_size).lower() == "auto":  # the run's own training size (datasets.vla_data.obs_image_size), default 224
            wh = [int(x) for x in (_img_size_cfg[0] or [224, 224])]
        else:
            wh = [int(x) for x in str(a.img_size).lower().split("x")]
        return im.resize((wh[0], wh[-1]))
    from omegaconf import OmegaConf
    run = Path(a.run_dir); cfg = OmegaConf.load(run / "config.full.yaml")
    am = cfg.framework.action_model
    windows = list(am.get("multi_windows", []) or []); H = int(am.action_horizon)
    action_mode = str(cfg.datasets.vla_data.get("action_mode", "delta"))
    p0 = H // 2 if windows else 0                                   # index of t inside the dataset window
    train_mix = str(cfg.datasets.vla_data.get("data_mix", ""))
    _img_size_cfg[0] = list(cfg.datasets.vla_data.get("obs_image_size", None) or [])
    default_mix = "egodex_test_pf64_epsplit" if windows else ("egodex_test_hf15" if train_mix.endswith("_hf15") else "egodex_test_epsplit")
    cfg.datasets.vla_data.data_mix = a.test_mix or default_mix
    cfg.output_dir = str(run / "eval_tmp"); os.makedirs(cfg.output_dir, exist_ok=True)
    for k in ("num_workers", "multiprocessing_context", "preprocess_in_workers"):
        cfg.datasets.vla_data.pop(k, None)
    cfg.datasets.vla_data.video_backend = "decord"            # eval decodes in-process; old runs used torchcodec_cuda+spawn
    cfg.datasets.vla_data.video_backend_kwargs = {"num_threads": 4}

    # ---- model ----
    from starVLA.model.framework.base_framework import build_framework
    model = build_framework(cfg)
    ck = run / "checkpoints" / (sorted((run / "checkpoints").glob("steps_*_pytorch_model.pt"), key=lambda p: int(p.name.split("_")[1]))[-1].name if a.ckpt == "latest" else (a.ckpt if a.ckpt.endswith(".pt") else f"{a.ckpt}_pytorch_model.pt"))
    sd = torch.load(ck, map_location="cpu", mmap=True); missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if not k.startswith("qwen_vl_interface")], f"missing keys: {missing[:5]}"
    model = model.cuda().eval(); head = model.action_model
    print(f"[eval] {run.name} @ {ck.name} | windows {windows or [H]} | p0 {p0} | action_mode {action_mode}", flush=True)
    print(f"[eval] test_mix {cfg.datasets.vla_data.data_mix} | img_size {a.img_size} (cfg {_img_size_cfg[0]}) | past_frame {a.past_frame}", flush=True)

    # ---- stats (model's own) ----
    st = json.load(open(run / "dataset_statistics.json")); tag = list(st.keys())[0]
    a_norm, a_denorm = make_norm(st[tag]["action"]); s_norm, _ = make_norm(st[tag]["state"])

    # ---- data (raw path) ----
    from starVLA.dataloader import build_dataloader
    d = find_single_dataset(build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py).dataset)
    akeys, skeys = d.modality_keys["action"], d.modality_keys["state"]; vkey, lkey = d.modality_keys["video"][0], d.modality_keys["language"][0]
    idxs = list(range(0, len(d.all_steps), a.stride)); idxs = idxs[: a.max_samples] if a.max_samples else idxs
    print(f"[eval] test steps {len(d.all_steps)} -> {len(idxs)} samples (stride {a.stride})", flush=True)

    wins = windows or [H]
    # slot layout of the prediction: pf-experts/separate -> concat of windows; single/nested -> H slots
    slot_slices = {}
    if windows and str(am.get("multi_window_mode", "experts")) in ("experts", "separate"):
        s0 = 0
        for w in wins: slot_slices[w] = (s0, s0 + w); s0 += w
    elif windows:  # nested: window = centre slice of the H slots
        for w in wins: slot_slices[w] = (p0 - w // 2, p0 + w // 2)
    else:          # single future-only window
        slot_slices[H] = (0, H)
    tgt_slices = {w: (p0 - w // 2, p0 + w // 2) if windows else (0, H) for w in wins}

    acc = {w: {"l1_norm": [], "pos_delta_l1_m": [], "rot_l1": [], "abs_err_m_per_step": [], "abs_err_m_kp": []} for w in wins}
    dump = {w: {"err": [], "gt_disp": [], "pred_disp": [], "rot_err": []} for w in wins} if a.dump else None
    t0 = time.time()
    for b in range(0, len(idxs), a.batch_size):
        raws, examples = [], []
        for i in idxs[b : b + a.batch_size]:
            tid, base = d.all_steps[i]; r = d.get_step_data(int(tid), int(base))
            frames = list(r[vkey])                                                      # (T, H, W, 3): oldest first, last = t
            if a.past_frame == "same":
                frames = [frames[-1]] * len(frames)
            imgs = [_prep_img(f) for f in frames]
            state_raw = np.concatenate([r[k] for k in skeys], axis=1)[0]                # (48,)
            act_raw = np.concatenate([r[k] for k in akeys], axis=1)                     # (H, 48) bidirectional delta / delta
            raws.append((state_raw, act_raw))
            examples.append({"image": imgs, "lang": r[lkey][0], "state": s_norm(state_raw[None]).astype(np.float32)})
        with torch.no_grad():
            pred_norm = model.predict_action(examples=examples)["normalized_actions"].astype(np.float64)  # (B, S, 48)
        pred_raw = a_denorm(pred_norm)
        state_pos = np.stack([s for s, _ in raws])[:, POS_DIMS]; gt_raw = np.stack([g for _, g in raws])
        for w in wins:
            (lo, hi), (tlo, thi) = slot_slices[w], tgt_slices[w]
            pw, gw, pn, gn = pred_raw[:, lo:hi], gt_raw[:, tlo:thi], pred_norm[:, lo:hi], a_norm(gt_raw[:, tlo:thi])
            acc[w]["l1_norm"].append(np.abs(pn - gn).mean())
            acc[w]["pos_delta_l1_m"].append(np.abs(pw[..., POS_DIMS] - gw[..., POS_DIMS]).mean())
            acc[w]["rot_l1"].append(np.abs(pw[..., ROT_DIMS] - gw[..., ROT_DIMS]).mean())
            pp = w // 2 if windows else 0
            ap_, ag_ = reconstruct_abs(state_pos, pw[..., POS_DIMS], pp, action_mode), reconstruct_abs(state_pos, gw[..., POS_DIMS], pp, action_mode)
            err = np.linalg.norm((ap_ - ag_).reshape(len(pw), w, 12, 3), axis=-1)       # (B, w, 12 keypoints)
            acc[w]["abs_err_m_per_step"].append(err.mean(axis=(0, 2))); acc[w]["abs_err_m_kp"].append(err.mean(axis=(0, 1)))
            if dump is not None:  # per-sample: keypoint error, GT displacement from t (hold-still baseline), rot6d error
                disp = np.linalg.norm((ag_ - state_pos[:, None, :]).reshape(len(pw), w, 12, 3), axis=-1)
                dump[w]["err"].append(err.astype(np.float32)); dump[w]["gt_disp"].append(disp.astype(np.float32))
                dump[w]["pred_disp"].append(np.linalg.norm((ap_ - state_pos[:, None, :]).reshape(len(pw), w, 12, 3), axis=-1).astype(np.float32))
                dump[w]["rot_err"].append(np.abs(pw[..., ROT_DIMS] - gw[..., ROT_DIMS]).reshape(len(pw), w, 2, 6).mean(-1).astype(np.float32))
        if (b // a.batch_size) % 20 == 0: print(f"  {b + len(examples)}/{len(idxs)}  {time.time() - t0:.0f}s", flush=True)

    res = {"run": run.name, "ckpt": ck.name, "samples": len(idxs), "windows": {}}
    for w in wins:
        A = acc[w]; per_step = np.mean(A["abs_err_m_per_step"], axis=0); kp = np.mean(A["abs_err_m_kp"], axis=0); pp = w // 2 if windows else 0
        fut = per_step[pp:]; past = per_step[:pp]
        res["windows"][str(w)] = {
            "l1_norm": float(np.mean(A["l1_norm"])), "pos_delta_l1_cm": float(np.mean(A["pos_delta_l1_m"]) * 100), "rot6d_l1": float(np.mean(A["rot_l1"])),
            "abs_err_cm_past": float(past.mean() * 100) if len(past) else None, "abs_err_cm_future": float(fut.mean() * 100),
            "abs_err_cm_future30": float(fut[:30].mean() * 100), "abs_err_cm_at_future_step": {str(k): float(fut[k - 1] * 100) for k in (1, 4, 8, 16, 30, 32) if k <= len(fut)},
            "abs_err_cm_per_step": [float(x * 100) for x in per_step], "abs_err_cm_per_keypoint": {n: float(kp[i] * 100) for i, n in enumerate(KEYPOINTS)},
        }
        r = res["windows"][str(w)]
        print(f"[w{w:>2}] L1norm {r['l1_norm']:.4f} | delta {r['pos_delta_l1_cm']:.3f} cm/step | abs err past {r['abs_err_cm_past'] if r['abs_err_cm_past'] is None else round(r['abs_err_cm_past'],2)} cm, future {r['abs_err_cm_future']:.2f} cm, future30 {r['abs_err_cm_future30']:.2f} cm | @+1 {r['abs_err_cm_at_future_step']['1']:.2f} @+8 {r['abs_err_cm_at_future_step'].get('8', float('nan')):.2f} @+30 {r['abs_err_cm_at_future_step'].get('30', float('nan')):.2f} cm")
    out = Path(a.out) if a.out else run / f"eval_heldout_{ck.name.replace('_pytorch_model.pt','')}.json"
    json.dump(res, open(out, "w"), indent=1); print("[eval] saved", out)
    if dump is not None:
        np.savez_compressed(a.dump, **{f"w{w}_{k}": np.concatenate(v, axis=0) for w in wins for k, v in dump[w].items()}, windows=np.array(wins), p0=p0)
        print("[eval] dumped per-sample errors ->", a.dump)


if __name__ == "__main__":
    main()
