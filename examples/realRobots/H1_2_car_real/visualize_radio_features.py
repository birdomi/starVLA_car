"""C-RADIOv4 dense-feature visualization for one episode of a LeRobot v3 car dataset.

For each camera, PCA(3) of the patch features is fit over the whole episode, so colors are
comparable across frames. Outputs (in --out):
  pca_ep<N>.mp4        original | PCA-RGB, front row + wrist row
  keyframes_ep<N>.png  evenly spaced frames, original over PCA
  temporal_ep<N>.png   frame x frame feature similarity, frame-to-frame change, joint trajectories
  features_ep<N>.pt    summary tokens, PCA bases, explained variance

Example:
  CUDA_VISIBLE_DEVICES=5 python examples/realRobots/H1_2_car_real/visualize_radio_features.py \
      --root playground/Datasets/car-left-turn-signal/val --episode 2
"""
import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchcodec.decoders import VideoDecoder

CAMS = ["observation.images.front_view", "observation.images.left_wrist_view", "observation.images.right_wrist_view"]


def load_radio(path, pkg="cradio_hf"):
    """Import the HF remote-code dir as a package and load it.

    transformers 5.5 AutoModel(trust_remote_code) copies only part of the nested relative
    imports (utils.py is missing) and fails, so we register `path` as a package ourselves.
    """
    if pkg not in sys.modules:
        mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(pkg, loader=None, is_package=True))
        mod.__path__ = [path]
        sys.modules[pkg] = mod
    hf = importlib.import_module(pkg + ".hf_model")
    return hf.RADIOModel.from_pretrained(path)


def decode_episode(root, cam, ep, timestamps):
    c, f = int(ep[f"videos/{cam}/chunk_index"]), int(ep[f"videos/{cam}/file_index"])
    path = os.path.join(root, "videos", cam, f"chunk-{c:03d}", f"file-{f:03d}.mp4")
    # CPU decoder: the CUDA one returns wrong frames after backward seeks on a reused decoder.
    dec = VideoDecoder(path, device="cpu", num_ffmpeg_threads=8)
    secs = (float(ep[f"videos/{cam}/from_timestamp"]) + timestamps).tolist()
    return dec.get_frames_played_at(seconds=secs).data  # (N, 3, H, W) uint8


@torch.no_grad()
def extract(model, frames, bs, dtype, patch):
    N, _, H, W = frames.shape
    h, w = H // patch, W // patch
    summaries, feats = [], []
    for i in range(0, N, bs):
        x = frames[i:i + bs].cuda(non_blocking=True).float() / 255.0
        if (H, W) != (h * patch, w * patch):
            x = F.interpolate(x, (h * patch, w * patch), mode="bilinear", antialias=True, align_corners=False)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            s, f = model(x)
        summaries.append(s.float().cpu())
        feats.append(f.half().cpu())
    return torch.cat(summaries), torch.cat(feats), (h, w)


@torch.no_grad()
def fit_pca(feats, k=3, frame_stride=4, q=0.01):
    D = feats.shape[-1]
    X = feats[::frame_stride].reshape(-1, D).cuda().float()
    mu = X.mean(0)
    X -= mu
    evals, evecs = torch.linalg.eigh(X.T @ X / (len(X) - 1))
    evals, evecs = evals.flip(0), evecs.flip(1)
    V = evecs[:, :k].contiguous()
    P = X @ V
    sign = torch.sign((P ** 3).mean(0))  # deterministic orientation: positive skew
    V, P = V * sign, P * sign
    lo, hi = torch.quantile(P, q, dim=0), torch.quantile(P, 1 - q, dim=0)
    return dict(mean=mu, V=V, lo=lo, hi=hi, explained=(evals[:k] / evals.sum()).cpu())


@torch.no_grad()
def project(feats, pca, hw, bs=64):
    out = []
    for i in range(0, len(feats), bs):
        P = (feats[i:i + bs].cuda().float() - pca["mean"]) @ pca["V"]
        P = ((P - pca["lo"]) / (pca["hi"] - pca["lo"])).clamp(0, 1)
        out.append((P * 255).round().byte().reshape(-1, *hw, 3).cpu())
    return torch.cat(out).numpy()  # (N, h, w, 3) uint8 RGB


def label(img, text, scale=0.9):
    cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def tile(frame_chw, pca_hw3, size, name, t=None):
    W, H = size
    rgb = cv2.resize(np.ascontiguousarray(frame_chw.permute(1, 2, 0).numpy()), (W, H), interpolation=cv2.INTER_AREA)
    pc = cv2.resize(pca_hw3, (W, H), interpolation=cv2.INTER_LINEAR)
    label(rgb, name if t is None else f"{name}  t={t:.2f}s")
    label(pc, "PCA(3)")
    return np.concatenate([rgb, pc], axis=1)


def compose(i, frames, pcas, names, ts):
    top = tile(frames[0][i], pcas[0][i], (960, 540), names[0], ts[i])
    rows = [top]
    if len(frames) > 1:
        rows.append(np.concatenate([tile(frames[c][i], pcas[c][i], (480, 360), names[c]) for c in range(1, len(frames))], axis=1))
    return np.concatenate(rows, axis=0)


def write_mp4(path, frames_iter, size, fps):
    W, H = size
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
           "-movflags", "+faststart", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames_iter:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    assert p.wait() == 0, "ffmpeg failed"


@torch.no_grad()
def cos_matrix(x):
    x = F.normalize(x.cuda().float().flatten(1), dim=1)
    return (x @ x.T).cpu().numpy()


def temporal_plot(path, ts, summaries, pooled, feats_change, names, action, action_names, title, n_teacher_summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = summaries[0].shape[1] // n_teacher_summ
    mats = [("front patch features (pooled 5x5)", cos_matrix(pooled[0]))]
    teacher_names = ["SigLIP2 summary", "DINOv3 summary"][:n_teacher_summ]
    mats += [(f"front {tn}", cos_matrix(summaries[0][:, k * D:(k + 1) * D])) for k, tn in enumerate(teacher_names)]

    fig = plt.figure(figsize=(18, 11), layout="constrained")
    gs = fig.add_gridspec(3, len(mats), height_ratios=[1.6, 0.6, 0.8])
    ext = [ts[0], ts[-1], ts[-1], ts[0]]
    for j, (nm, m) in enumerate(mats):
        ax = fig.add_subplot(gs[0, j])
        off = m[~np.eye(len(m), dtype=bool)]
        im = ax.imshow(m, extent=ext, cmap="viridis", vmin=np.quantile(off, 0.01), vmax=np.quantile(off, 0.99))
        ax.set_title(f"cos sim: {nm}", fontsize=10)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("t [s]")
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = fig.add_subplot(gs[1, :])
    for c, nm in enumerate(names):
        ax.plot(ts[1:], feats_change[c], lw=1, label=nm)
    ax.set_ylabel("1 - cos(f_t, f_t-1)\n(mean over patches)")
    ax.set_xlim(ts[0], ts[-1])
    ax.legend(fontsize=8, loc="upper right")
    ax = fig.add_subplot(gs[2, :])
    ax2 = ax.twinx()
    for k, nm in enumerate(action_names):
        tgt = ax2 if "hand" in nm else ax
        tgt.plot(ts, action[:, k], lw=1, ls="--" if "hand" in nm else "-", label=nm)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("joint cmd [rad]")
    ax2.set_ylabel("hand cmd (raw)")
    ax.set_xlim(ts[0], ts[-1])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, ncol=6, loc="upper left")
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="playground/Datasets/car-left-turn-signal/val")
    ap.add_argument("--episode", type=int, default=2)
    ap.add_argument("--model", default="playground/Pretrained_models/C-RADIOv4-SO400M")
    ap.add_argument("--cams", nargs="+", default=CAMS)
    ap.add_argument("--front-res", type=int, nargs=2, default=None, metavar=("H", "W"),
                    help="front input resolution (multiple of 16); default native 720x1280")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-keyframes", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(8)
    dtype = getattr(torch, args.dtype)
    tag = os.path.basename(os.path.normpath(args.model)).lower()
    out = args.out or os.path.join("playground/eval_results/radio_features", f"{tag}_{os.path.basename(os.path.dirname(os.path.normpath(args.root)))}_{os.path.basename(os.path.normpath(args.root))}_ep{args.episode}")
    os.makedirs(out, exist_ok=True)

    eps = pd.read_parquet(os.path.join(args.root, "meta/episodes/chunk-000/file-000.parquet"))
    ep = eps[eps.episode_index == args.episode].iloc[0]
    tasks = pd.read_parquet(os.path.join(args.root, "meta/tasks.parquet"))
    task = ep["tasks"][0]
    if np.issubdtype(np.asarray(ep["tasks"]).dtype, np.integer):  # prepared sets store task_index here
        task = dict(zip(tasks.task_index, tasks.index))[int(task)]
    data = pd.read_parquet(os.path.join(args.root, "data", f"chunk-{int(ep['data/chunk_index']):03d}", f"file-{int(ep['data/file_index']):03d}.parquet"))
    data = data[data.episode_index == args.episode].sort_values("frame_index")
    ts = data.timestamp.to_numpy()
    action = np.stack(data.action.to_numpy())
    info = json.load(open(os.path.join(args.root, "meta/info.json")))
    anames = info["features"]["action"]["names"]
    print(f"episode {args.episode}: {len(ts)} frames, task={task!r}")

    t0 = time.time()
    model = load_radio(args.model).eval().cuda()
    patch = model.patch_size
    print(f"loaded {tag} in {time.time() - t0:.1f}s, patch {patch}, {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    frames, summaries, feats, hws, pcas, names = [], [], [], [], [], []
    stats = {}
    for cam in args.cams:
        t0 = time.time()
        fr = decode_episode(args.root, cam, ep, ts)
        t_dec = time.time() - t0
        if cam.endswith("front_view") and args.front_res is not None:
            fr_in = F.interpolate(fr.float(), tuple(args.front_res), mode="bilinear", antialias=True).round().clamp(0, 255).byte()
        else:
            fr_in = fr
        t0 = time.time()
        s, f, hw = extract(model, fr_in, args.bs, dtype, patch)
        torch.cuda.synchronize()
        t_fwd = time.time() - t0
        pca = fit_pca(f)
        pcas.append(project(f, pca, hw))
        frames.append(fr)
        summaries.append(s)
        feats.append(f)
        hws.append(hw)
        nm = cam.split(".")[-1]
        names.append(nm)
        stats[nm] = dict(input_hw=list(fr_in.shape[-2:]), grid_hw=list(hw), summary_dim=s.shape[1], feat_dim=f.shape[-1],
                         decode_s=round(t_dec, 1), forward_s=round(t_fwd, 1), ms_per_frame=round(1000 * t_fwd / len(fr), 1),
                         pca_explained=[round(v, 4) for v in pca["explained"].tolist()],
                         pca=dict(V=pca["V"].cpu(), mean=pca["mean"].cpu(), lo=pca["lo"].cpu(), hi=pca["hi"].cpu()))
        print(f"{nm}: in {tuple(fr_in.shape[-2:])} -> grid {hw}, feat {tuple(f.shape)}, decode {t_dec:.1f}s, "
              f"fwd {t_fwd:.1f}s ({1000 * t_fwd / len(fr):.0f} ms/frame), PCA expl {stats[nm]['pca_explained']}")
    print(f"peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    vid = os.path.join(out, f"pca_ep{args.episode}.mp4")
    first = compose(0, frames, pcas, names, ts)
    write_mp4(vid, (compose(i, frames, pcas, names, ts) for i in range(len(ts))), (first.shape[1], first.shape[0]), info["fps"])
    print("wrote", vid)

    idx = np.linspace(0, len(ts) - 1, args.n_keyframes).round().astype(int)
    cols = []
    for i in idx:
        col = [tile(frames[c][i], pcas[c][i], (480, 270) if c == 0 else (480, 360), names[c], ts[i]) for c in range(len(frames))]
        # stack original over PCA instead of side by side
        col = [np.concatenate([x[:, :480], x[:, 480:]], axis=0) for x in col]
        cols.append(np.concatenate(col, axis=0))
    grid = np.concatenate(cols, axis=1)
    kf = os.path.join(out, f"keyframes_ep{args.episode}.png")
    cv2.imwrite(kf, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print("wrote", kf)

    pooled, change = [], []
    for f, hw in zip(feats, hws):
        g = f.reshape(len(f), *hw, -1)
        pooled.append(torch.cat([F.adaptive_avg_pool2d(g[i:i + 64].cuda().float().permute(0, 3, 1, 2), 5).cpu()
                                 for i in range(0, len(g), 64)]))
        cs = torch.cat([F.cosine_similarity(f[i:j].cuda().float(), f[i + 1:j + 1].cuda().float(), dim=-1).mean(-1).cpu()
                        for i in range(0, len(f) - 1, 64) for j in [min(i + 64, len(f) - 1)]])
        change.append((1 - cs).numpy())
    sel = [k for k, n in enumerate(anames) if n.startswith("head") or (n.startswith("left_") and "hand" not in n)]
    sel += [anames.index("left_hand_00"), anames.index("left_hand_04")]
    # summary = concat of per-teacher CLS tokens (SigLIP2, DINOv3 for C-RADIOv4)
    n_summ = summaries[0].shape[1] // feats[0].shape[-1] if summaries[0].shape[1] % feats[0].shape[-1] == 0 else 1
    tp = os.path.join(out, f"temporal_ep{args.episode}.png")
    temporal_plot(tp, ts, summaries, pooled, change, names, action[:, sel], [anames[k] for k in sel],
                  f"{tag}  |  {args.root}  ep {args.episode}  |  {task}", n_summ)
    print("wrote", tp)

    torch.save(dict(summaries=dict(zip(names, summaries)), stats=stats, timestamps=ts, task=task), os.path.join(out, f"features_ep{args.episode}.pt"))
    json.dump(dict(model=args.model, root=args.root, episode=args.episode, task=task, n_frames=len(ts), dtype=args.dtype,
                   cams={k: {kk: vv for kk, vv in v.items() if kk != "pca"} for k, v in stats.items()}),
              open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("done ->", out)


if __name__ == "__main__":
    main()
