#!/usr/bin/env python3
"""Overlay predicted vs GT future wrist trajectories on the input RGB frame (EgoDex test set).

chest-frame keypoints -> world (via leftArm/rightArm transforms in the raw HDF5) -> camera at
frame t (transforms/camera) -> pixels (camera/intrinsic, scaled to the 640x360 video).
Usage: viz_trajectory.py --run_dir <run> --ckpt steps_50000 --out <dir> [--n 8]
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, h5py
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "/mnt/DATA2/yyg/starVLA"); sys.path.insert(0, "/mnt/DATA2/yyg/starVLA/examples/realRobots/EgoDex/tools")
from eval_heldout import make_norm, find_single_dataset, POS_DIMS, ROT_DIMS

R_FIX = np.array([[0, 0, -1.], [-1, 0, 0], [0, 1, 0]])
TIPS = ["ThumbTip", "IndexFingerTip", "MiddleFingerTip", "RingFingerTip", "LittleFingerTip"]
RAW = {"test": Path("/mnt/DATA2/yyg/egodex/data/test"), "train": Path("/mnt/DATA2/yyg/egodex/data/part1")}


def chest_frames(f):
    T = f["transforms"]
    la = (R_FIX @ T["leftArm"][:, :3, 3].T).T; ra = (R_FIX @ T["rightArm"][:, :3, 3].T).T
    chest = (la + ra) / 2; y = la - ra; y[:, 2] = 0; y /= np.linalg.norm(y, axis=1, keepdims=True)
    z = np.zeros_like(y); z[:, 2] = 1; x = np.cross(y, z)
    return chest, np.stack([x, y, z], axis=2)  # (n,3), (n,3,3) chest axes as columns


def project(p_world, T_cw, K):
    """EgoDex camera (verified numerically): transforms/camera is camera->world and the camera
    frame is OpenCV-style (x right, y down, z forward). p_world (...,3) -> pixel (...,2), depth."""
    ph = np.concatenate([p_world, np.ones_like(p_world[..., :1])], -1)
    pc = ph @ T_cw.T; x, y, z = pc[..., 0], pc[..., 1], pc[..., 2]; d = z
    u = K[0, 0] * x / d + K[0, 2]; v = K[1, 1] * y / d + K[1, 2]
    return np.stack([u, v], -1), d


def rot6d_to_R(r6):  # [col0(3), col1(3)] -> 3x3 (columns)
    c0 = r6[:3] / (np.linalg.norm(r6[:3]) + 1e-8); c1 = r6[3:] - c0 * (c0 @ r6[3:]); c1 /= (np.linalg.norm(c1) + 1e-8)
    return np.stack([c0, c1, np.cross(c0, c1)], axis=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run_dir", required=True); ap.add_argument("--ckpt", default="latest")
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--split", default="test", choices=["test", "train"]); ap.add_argument("--select", default="motion", choices=["motion", "lowloss", "random"])
    ap.add_argument("--pool", type=int, default=2000, help="samples scored for --select lowloss/random"); a = ap.parse_args()
    from omegaconf import OmegaConf
    run = Path(a.run_dir); cfg = OmegaConf.load(run / "config.full.yaml"); am = cfg.framework.action_model
    _IMG_SIZE = tuple(int(x) for x in (cfg.datasets.vla_data.get("obs_image_size", None) or (224, 224)))  # (W, H) the run trained with
    def _imgs(r):  # every frame of the video window (e.g. [t-15, t]) resized like training
        return [Image.fromarray(f).resize(_IMG_SIZE) for f in r[vkey]]
    _hf = str(cfg.datasets.vla_data.get("data_mix", "")).endswith("_hf15")  # two-frame runs use the hf15 test/train mixes
    MIX = {"test": "egodex_test_hf15" if _hf else "egodex_test_epsplit", "train": "egodex_part1_hf15" if _hf else "egodex_part1_epsplit"}
    H = int(am.action_horizon); mode = str(cfg.datasets.vla_data.get("action_mode", "delta")); assert not am.get("multi_windows")
    cfg.datasets.vla_data.data_mix = MIX[a.split]; cfg.output_dir = str(run / "eval_tmp"); os.makedirs(cfg.output_dir, exist_ok=True)
    for k in ("num_workers", "multiprocessing_context", "preprocess_in_workers"): cfg.datasets.vla_data.pop(k, None)
    cfg.datasets.vla_data.video_backend = "decord"; cfg.datasets.vla_data.video_backend_kwargs = {"num_threads": 4}
    from starVLA.model.framework.base_framework import build_framework
    model = build_framework(cfg)
    ck = run / "checkpoints" / (sorted((run / "checkpoints").glob("steps_*_pytorch_model.pt"), key=lambda p: int(p.name.split("_")[1]))[-1].name if a.ckpt == "latest" else f"{a.ckpt}_pytorch_model.pt")
    model.load_state_dict(torch.load(ck, map_location="cpu", mmap=True), strict=False); model = model.cuda().eval()
    st = json.load(open(run / "dataset_statistics.json")); tag = list(st)[0]
    a_norm, a_denorm = make_norm(st[tag]["action"]); s_norm, _ = make_norm(st[tag]["state"])
    from starVLA.dataloader import build_dataloader
    d = find_single_dataset(build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py).dataset)
    akeys, skeys, vkey, lkey = d.modality_keys["action"], d.modality_keys["state"], d.modality_keys["video"][0], d.modality_keys["language"][0]
    h5s = sorted(RAW[a.split].rglob("*.hdf5"), key=lambda p: (p.parent.name, int(p.stem)))

    if a.select in ("lowloss", "random"):
        # score a pool of samples with the model; keep the lowest future-30 abs keypoint error (moving hands only)
        stride = max(1, len(d.all_steps) // a.pool); pool = list(range(0, len(d.all_steps), stride))[: a.pool]
        scores = []
        for b in range(0, len(pool), 64):
            raws, ex = [], []
            for i in pool[b:b+64]:
                tid, base = d.all_steps[i]; r = d.get_step_data(int(tid), int(base))
                s0 = np.concatenate([r[k] for k in skeys], 1)[0]; g = np.concatenate([r[k] for k in akeys], 1)
                raws.append((s0, g)); ex.append({"image": _imgs(r), "lang": r[lkey][0], "state": s_norm(s0[None]).astype(np.float32)})
            with torch.no_grad(): pr = a_denorm(model.predict_action(examples=ex)["normalized_actions"].astype(np.float64))
            for j, (s0, g) in enumerate(raws):
                sp = s0[POS_DIMS]
                pa_ = sp[None] + pr[j][:, POS_DIMS] if mode == "rel" else sp[None] + np.cumsum(pr[j][:, POS_DIMS], 0)
                ga_ = sp[None] + g[:, POS_DIMS] if mode == "rel" else sp[None] + np.cumsum(g[:, POS_DIMS], 0)
                err = np.linalg.norm((pa_ - ga_).reshape(H, 12, 3), axis=-1).mean() * 100
                motion = np.linalg.norm((ga_[-1] - sp).reshape(12, 3), axis=-1)[[0, 6]].mean() * 100
                scores.append((err, motion, pool[b + j]))
        if a.select == "random": np.random.default_rng(0).shuffle(scores)   # unbiased: random moving samples
        else: scores.sort()
        picks, used = [], set()
        for err, motion, i in scores:
            if motion < 6.0: continue                      # visibly moving samples only
            tid, base = d.all_steps[i]; tid, base = int(tid), int(base)
            if tid in used: continue
            with h5py.File(h5s[tid], "r") as f:            # both wrists must be inside the frame at t
                K = f["camera"]["intrinsic"][:] * np.array([[1/3, 1, 1/3], [1, 1/3, 1/3], [1, 1, 1]]); K[2, 2] = 1
                T_cw = np.linalg.inv(f["transforms"]["camera"][base])
                uv, dep = project(np.stack([f["transforms"][f"{sd}Hand"][base][:3, 3] for sd in ("left", "right")]), T_cw, K)
            if not (np.all(dep > 0.05) and np.all((uv[:, 0] > 20) & (uv[:, 0] < 620) & (uv[:, 1] > 20) & (uv[:, 1] < 340))): continue
            picks.append((i, tid, base, f"{a.select} err {err:.2f}cm", float(motion))); used.add(tid)
            if len(picks) >= a.n: break
        print(f"[viz] pool {len(scores)} samples: err median {np.median([s[0] for s in scores]):.2f} cm | picked {len(picks)} {a.select} moving samples", flush=True)
    else:
      # pick samples from the eval dump by wrist motion (fast / medium / static), distinct episodes
      dump = np.load(sorted(run.glob("eval_heldout_steps_*_dump.npz"))[-1]); disp = dump[f"w{H}_gt_disp"][:, -1, [0, 6]].mean(1) * 100
      order = np.argsort(-disp); picks, used = [], set()
      for want, sel in (("fast", order[: 400]), ("medium", order[len(order)//2 - 200: len(order)//2 + 200]), ("static", order[-400:])):
          rng = np.random.default_rng(0); cand = rng.permutation(sel)
          for i in cand:
              tid, base = d.all_steps[int(i) * 30]; tid = int(tid)
              if tid in used: continue
              picks.append((int(i), tid, int(base), want, float(disp[i]))); used.add(tid)
              if sum(1 for p in picks if p[3] == want) >= {"fast": 4, "medium": 2, "static": 2}[want]: break
      picks = picks[: a.n]

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); font = ImageFont.load_default(); tiles = []
    for i, tid, base, want, dcm in picks:
        r = d.get_step_data(tid, base); imgs = _imgs(r)
        state_raw = np.concatenate([r[k] for k in skeys], 1)[0]; gt_raw = np.concatenate([r[k] for k in akeys], 1)
        with torch.no_grad():
            pred = a_denorm(model.predict_action(examples=[{"image": imgs, "lang": r[lkey][0], "state": s_norm(state_raw[None]).astype(np.float32)}])["normalized_actions"].astype(np.float64))[0]
        # absolute chest-frame positions for every future step (rel: s + delta)
        def absolute(win):
            w = win.copy()
            if mode == "rel": w[:, POS_DIMS] = state_raw[POS_DIMS] + win[:, POS_DIMS]
            else:
                cur = state_raw[POS_DIMS].copy()
                for k in range(len(w)): cur = cur + win[k, POS_DIMS]; w[k, POS_DIMS] = cur
            return w
        P, G = absolute(pred), absolute(gt_raw)
        with h5py.File(h5s[tid], "r") as f:
            chest, Rc = chest_frames(f); K = f["camera"]["intrinsic"][:] * np.array([[1/3, 1, 1/3], [1, 1/3, 1/3], [1, 1, 1]]); K[2, 2] = 1
            T_wc = f["transforms"]["camera"][base]; T_cw = np.linalg.inv(T_wc); n = chest.shape[0]
            wrist_gt_h5 = {s: (R_FIX.T @ R_FIX @ f["transforms"][f"{s}Hand"][:, :3, 3].T).T for s in ("left", "right")}  # world ARKit
        frame = Image.fromarray(r[vkey][-1]).resize((640, 360)); dr = ImageDraw.Draw(frame)
        for side, off, col_gt, col_pr in (("left", 0, (0, 230, 0), (255, 60, 60)), ("right", 24, (0, 200, 255), (255, 160, 0))):
            def world_traj(W):
                pts = []
                for k in range(H):
                    fidx = min(base + k + 1, n - 1); p_chest = W[k, off:off + 3]
                    pts.append(R_FIX.T @ (Rc[fidx] @ p_chest + chest[fidx]))
                return np.array(pts)
            for W, col, width in ((G, col_gt, 3), (P, col_pr, 2)):
                uv, dep = project(world_traj(W), T_cw, K); ok = dep > 0.05
                pts = [tuple(uv[k]) for k in range(H) if ok[k]]
                if len(pts) > 1: dr.line(pts, fill=col, width=width)
                for k in (0, H - 1):
                    if ok[k]: dr.ellipse([uv[k, 0] - 4, uv[k, 1] - 4, uv[k, 0] + 4, uv[k, 1] + 4], outline=col, width=2)
            # current wrist (from HDF5, frame t) as a white dot: sanity check of the projection
            uv0, d0 = project(wrist_gt_h5[side][base][None], T_cw, K)
            if d0[0] > 0.05: dr.ellipse([uv0[0, 0] - 5, uv0[0, 1] - 5, uv0[0, 0] + 5, uv0[0, 1] + 5], fill=(255, 255, 255))
        dr.rectangle([0, 0, 640, 26], fill=(0, 0, 0)); dr.text((4, 2), f"ep{tid} t={base} [{want}, wrist moves {dcm:.1f}cm/+30] {r[lkey][0][:70]}", fill=(255, 255, 255), font=font)
        dr.text((4, 14), "GT: green=L cyan=R | pred: red=L orange=R | white=wrist@t", fill=(200, 200, 200), font=font)
        frame.save(out / f"ep{tid}_t{base}_{want}.png"); tiles.append(frame); print("saved", out / f"ep{tid}_t{base}_{want}.png", flush=True)
    cols = 2; rows = (len(tiles) + 1) // 2; grid = Image.new("RGB", (640 * cols, 360 * rows), (30, 30, 30))
    for j, t in enumerate(tiles): grid.paste(t, ((j % cols) * 640, (j // cols) * 360))
    grid.save(out / "grid.png"); print("grid ->", out / "grid.png")


if __name__ == "__main__":
    main()
