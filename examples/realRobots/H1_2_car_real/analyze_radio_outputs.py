"""What does C-RADIOv4 output? Structure / statistics of backbone + teacher-adaptor outputs on one car episode.

Needs the full training checkpoint (`*_half.pth.tar`, has the teacher heads) next to the HF dir.
Writes stats.json + figures to --out.

  CUDA_VISIBLE_DEVICES=6 python examples/realRobots/H1_2_car_real/analyze_radio_outputs.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_radio_features import CAMS, decode_episode, load_radio  # noqa: E402

FRONT, LWRIST, RWRIST = CAMS
SPACES = ["backbone", "siglip2-g", "dino_v3_7b", "sam3"]


def attach_adaptors(model, pth, pkg="cradio_hf"):
    """Build the per-teacher heads from the training checkpoint (HF config ships none)."""
    import argparse as _ap
    torch.serialization.add_safe_globals([_ap.Namespace])
    ck = torch.load(pth, map_location="cpu", weights_only=True)
    args, sd = ck["args"], ck["state_dict"]
    GenericAdaptor = sys.modules[pkg + ".adaptor_generic"].GenericAdaptor
    for idx, t in enumerate(args.teachers):
        name = t["name"]
        st = {}
        for src, dst in [(f"_heads.{name}.", "summary."), (f"_feature_projections.{name}.", "feature.")]:
            st.update({dst + k[len(src):]: v.float() for k, v in sd.items() if k.startswith(src)})
        # GenericAdaptor instead of the registry's SigLIP2Adaptor: same heads, no text-tower download
        ad = GenericAdaptor(args, t, st)
        ad.head_idx = idx
        model.radio_model.adaptors[name] = ad
    return args.teachers


@torch.no_grad()
def run(model, x, dtype=torch.bfloat16):
    """Returns (raw token sequence incl. prefix, {space: (summary, features)})."""
    rm = model.radio_model
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        xc = rm.input_conditioner(x)
        y = rm.model.forward_features(xc)
        out = rm._extract_final(xc, y)
    res = {"backbone": (out["backbone"].summary.float(), out["backbone"].features.float())}
    for k in SPACES[1:]:
        res[k] = (out[k].summary.float(), out[k].features.float())
    return y.float(), res


def to_input(frames_u8, hw=None):
    x = frames_u8.cuda().float() / 255.0
    if hw is not None and tuple(x.shape[-2:]) != tuple(hw):
        x = F.interpolate(x, hw, mode="bilinear", antialias=True, align_corners=False)
    return x


def linear_cka(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    hsic = (a.T @ b).norm() ** 2
    return (hsic / ((a.T @ a).norm() * (b.T @ b).norm())).item()


def pca_basis(x, k=3):
    mu = x.mean(0)
    xc = x - mu
    ev, V = torch.linalg.eigh(xc.T @ xc / (len(x) - 1))
    ev, V = ev.flip(0), V.flip(1)
    P = xc @ V[:, :k]
    s = torch.sign((P ** 3).mean(0))
    V = V[:, :k] * s
    P = P * s
    return dict(mu=mu, V=V, lo=torch.quantile(P, 0.01, dim=0), hi=torch.quantile(P, 0.99, dim=0), ev=ev)


def pca_rgb(feat, basis, hw):
    P = ((feat - basis["mu"]) @ basis["V"] - basis["lo"]) / (basis["hi"] - basis["lo"])
    return P.clamp(0, 1).reshape(*hw, 3).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="playground/Datasets/car-left-turn-signal/val")
    ap.add_argument("--episode", type=int, default=2)
    ap.add_argument("--model", default="playground/Pretrained_models/C-RADIOv4-SO400M")
    ap.add_argument("--pth", default="playground/Pretrained_models/C-RADIOv4-SO400M_pth/c-radio_v4-so400m_half.pth.tar")
    ap.add_argument("--stride", type=int, default=8, help="frame stride for episode statistics")
    ap.add_argument("--out", default="playground/eval_results/radio_features/output_analysis_c-radiov4-so400m_ep2")
    ap.add_argument("--gpu-mem-gb", type=float, default=20.0, help="hard cap for this process (GPU is shared with training)")
    ap.add_argument("--outlier-norm", type=float, default=100.0, help="patch tokens above this L2 norm count as high-norm outliers")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    torch.manual_seed(0)
    torch.set_num_threads(8)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = pd.read_parquet(os.path.join(args.root, "meta/episodes/chunk-000/file-000.parquet"))
    ep = eps[eps.episode_index == args.episode].iloc[0]
    data = pd.read_parquet(os.path.join(args.root, "data", f"chunk-{int(ep['data/chunk_index']):03d}",
                                        f"file-{int(ep['data/file_index']):03d}.parquet"))
    ts = data[data.episode_index == args.episode].sort_values("frame_index").timestamp.to_numpy()
    frames = {c: decode_episode(args.root, c, ep, ts) for c in CAMS}

    model = load_radio(args.model).eval().cuda()
    teachers = attach_adaptors(model, args.pth)
    model.cuda()
    n_prefix = model.radio_model.model.patch_generator.num_skip
    n_cls = model.radio_model.model.patch_generator.num_cls_tokens
    S = {"teachers": teachers, "num_prefix_tokens": n_prefix, "num_cls_tokens": n_cls,
         "num_registers": n_prefix - n_cls, "depth": len(model.radio_model.model.blocks),
         "final_norm": type(model.radio_model.model.norm).__name__}

    # ---------------- 1. output structure on one front frame ----------------
    x = to_input(frames[FRONT][:1])
    y, res = run(model, x)
    S["structure"] = {"input": list(x.shape), "raw_tokens": list(y.shape)}
    for k, (s, f) in res.items():
        S["structure"][k] = {"summary": list(s.shape), "features": list(f.shape),
                             "summary_norm": round(s.norm(dim=-1).mean().item(), 2),
                             "feature_norm_median": round(f.norm(dim=-1).median().item(), 2)}
    print(json.dumps(S["structure"], indent=1))

    # ---------------- 2. episode pass (front, every `stride` frames) ----------------
    idx = np.arange(0, len(ts), args.stride)
    Hp, Wp = 45, 80
    lum_grid = F.avg_pool2d(frames[FRONT][idx].float().mean(1, keepdim=True), 16).flatten(1)  # (N, 3600) patch luminance
    n_samp = 128
    samp = {k: [] for k in SPACES}
    samp_lum, samp_pos, prefix_all, summ = [], [], [], {k: [] for k in SPACES}
    norms_all, out_vec, out_pos = [], [], []
    for b in range(0, len(idx), 8):
        bi = idx[b:b + 8]
        y, res = run(model, to_input(frames[FRONT][bi]))
        prefix_all.append(y[:, :n_prefix].cpu())
        fb = res["backbone"][1]
        nb = fb.norm(dim=-1)
        norms_all.append(nb.cpu())
        oi = (nb > args.outlier_norm).nonzero()
        out_vec.append(fb[oi[:, 0], oi[:, 1]].cpu())
        out_pos.append(torch.stack([torch.as_tensor(bi, device=oi.device)[oi[:, 0]], oi[:, 1]], 1).cpu())
        tok = torch.stack([torch.randperm(Hp * Wp)[:n_samp] for _ in bi]).cuda()
        for k in SPACES:
            f = res[k][1]
            samp[k].append(torch.gather(f, 1, tok[..., None].expand(-1, -1, f.shape[-1])).flatten(0, 1).cpu())
            summ[k].append(res[k][0].cpu())
        samp_lum.append(torch.gather(lum_grid[b:b + 8].cuda(), 1, tok).flatten().cpu())
        samp_pos.append(tok.flatten().cpu())
    samp = {k: torch.cat(v) for k, v in samp.items()}
    summ = {k: torch.cat(v) for k, v in summ.items()}
    prefix_all = torch.cat(prefix_all)  # (N, 10, 1152)
    norms_all = torch.cat(norms_all)  # (N, 3600)
    samp_lum, samp_pos = torch.cat(samp_lum), torch.cat(samp_pos)
    out_vec, out_pos = torch.cat(out_vec), torch.cat(out_pos)
    N = len(idx)

    # high-norm outlier tokens: how many, where, which channels
    per_frame = (norms_all > args.outlier_norm).sum(1)
    S["outliers"] = {"threshold_norm": args.outlier_norm, "frames": N, "tokens_per_frame": Hp * Wp,
                     "frac_tokens": round((norms_all > args.outlier_norm).float().mean().item(), 5),
                     "per_frame_count": {"min": int(per_frame.min()), "median": int(per_frame.median()), "max": int(per_frame.max())},
                     "frac_frames_with_any": round((per_frame > 0).float().mean().item(), 3)}
    if len(out_vec):
        e = out_vec ** 2
        chan_share = e.sum(0) / e.sum()
        topc = chan_share.sort(descending=True)
        pos = out_pos[:, 1]
        uniq, cnt = pos.unique(return_counts=True)
        o = cnt.argsort(descending=True)[:8]
        S["outliers"].update({
            "norm_median_of_outliers": round(out_vec.norm(dim=-1).median().item(), 1),
            "energy_share_top_channels": [(int(c), round(v, 3)) for c, v in zip(topc.indices[:5].tolist(), topc.values[:5].tolist())],
            "mean_pairwise_cos": round((F.normalize(out_vec, dim=-1) @ F.normalize(out_vec, dim=-1).T).mean().item(), 3),
            "n_distinct_positions": int(len(uniq)),
            "top_positions_rc_count": [(int(uniq[j]) // Wp, int(uniq[j]) % Wp, int(cnt[j])) for j in o],
            "lum_at_outliers_mean": round(F.avg_pool2d(frames[FRONT][out_pos[:, 0]].float().mean(1, keepdim=True), 16).flatten(1)
                                          .gather(1, out_pos[:, 1:2]).mean().item(), 1),
            "lum_all_mean": round(lum_grid.mean().item(), 1),
        })
    print(json.dumps(S["outliers"], indent=1))

    # prefix tokens
    pn = prefix_all.norm(dim=-1)
    pf = F.normalize(prefix_all, dim=-1)
    cross_frame = [(pf[:, j] @ pf[:, j].T).mean().item() for j in range(n_prefix)]
    S["prefix_tokens"] = {
        "names": [t["name"] + "_cls" for t in teachers] + [f"reg{j}" for j in range(n_prefix - n_cls)],
        "norm_mean": [round(v, 1) for v in pn.mean(0).tolist()],
        "norm_std": [round(v, 2) for v in pn.std(0).tolist()],
        "cross_frame_cos_mean": [round(v, 4) for v in cross_frame],
        "pairwise_cos_frame0": (pf[0] @ pf[0].T).numpy().round(3).tolist(),
    }

    # backbone patch-token statistics
    X = samp["backbone"].cuda()
    mu = X.mean(0)
    var = X.var(0)
    ev = torch.linalg.eigvalsh(torch.cov(X.T)).flip(0)
    Xn = F.normalize(X, dim=-1)
    Xc = F.normalize(X - mu, dim=-1)
    pr = torch.randperm(len(X), device=X.device)[:4000]
    cos_raw = (Xn[pr[:2000]] * Xn[pr[2000:]]).sum(-1)
    cos_c = (Xc[pr[:2000]] * Xc[pr[2000:]]).sum(-1)
    top_var = var.sort(descending=True)
    top_mean = mu.abs().sort(descending=True)
    bb_all = pca_basis(X, k=3)
    load1 = bb_all["V"][:, 0] ** 2
    inl = X.norm(dim=-1) <= args.outlier_norm
    Xi = X[inl]
    bb = pca_basis(Xi, k=3)  # inlier PCA: what the dominant *content* direction is
    inl_c = inl.cpu()
    pc1 = ((Xi - bb["mu"]) @ bb["V"][:, 0]).cpu()
    row = (samp_pos // Wp).float()[inl_c]
    nm = norms_all.flatten()
    evi = torch.linalg.eigvalsh(torch.cov(Xi.T)).flip(0)
    vari = Xi.var(0)
    Xin = F.normalize(Xi, dim=-1)
    pri = torch.randperm(len(Xi), device=X.device)[:4000]
    S["patch_tokens_inliers"] = {
        "n": int(inl.sum()), "mean_vector_norm": round(Xi.mean(0).norm().item(), 1),
        "max_abs_value": round(Xi.abs().max().item(), 1),
        "var_share_top1_channel": round((vari.max() / vari.sum()).item(), 4),
        "pca_explained_cum": {k: round((evi[:k].sum() / evi.sum()).item(), 3) for k in [1, 3, 10, 32, 100, 256]},
        "effective_rank_participation": round((evi.sum() ** 2 / (evi ** 2).sum()).item(), 1),
        "cos_random_pairs_raw": round((Xin[pri[:2000]] * Xin[pri[2000:]]).sum(-1).mean().item(), 3),
    }
    S["patch_tokens"] = {
        "n_tokens_sampled": len(X),
        "token_norm": {"median": round(nm.median().item(), 1), "p1": round(nm.quantile(0.01).item(), 1),
                       "p99": round(nm.quantile(0.99).item(), 1), "max": round(nm.max().item(), 1),
                       "frac_gt_2x_median": float((nm > 2 * nm.median()).float().mean())},
        "mean_vector_norm": round(mu.norm().item(), 1),
        "mean_token_norm": round(X.norm(dim=-1).mean().item(), 1),
        "max_abs_value": round(X.abs().max().item(), 1),
        "top_channels_by_abs_mean": [(int(i), round(mu[i].item(), 1)) for i in top_mean.indices[:5].tolist()],
        "var_share_top1_channel": round((top_var.values[0] / var.sum()).item(), 4),
        "var_share_top10_channels": round((top_var.values[:10].sum() / var.sum()).item(), 4),
        "cos_random_pairs_raw": round(cos_raw.mean().item(), 3),
        "cos_random_pairs_centered": round(cos_c.mean().item(), 3),
        "pca_explained_cum": {k: round((ev[:k].sum() / ev.sum()).item(), 3) for k in [1, 3, 10, 32, 100, 256]},
        "effective_rank_participation": round((ev.sum() ** 2 / (ev ** 2).sum()).item(), 1),
        "pc1_loading_top10_channel_share": round(load1.sort(descending=True).values[:10].sum().item(), 3),
        "inlier_pc1_corr_luminance": round(np.corrcoef(pc1.numpy(), samp_lum[inl_c].numpy())[0, 1], 3),
        "inlier_pc1_corr_row": round(np.corrcoef(pc1.numpy(), row.numpy())[0, 1], 3),
        "inlier_norm_corr_luminance": round(np.corrcoef(Xi.norm(dim=-1).cpu().numpy(), samp_lum[inl_c].numpy())[0, 1], 3),
    }

    # every output space: dims, anisotropy, spectrum, CKA vs backbone, summary dynamics
    S["spaces"] = {}
    for k in SPACES:
        Z = samp[k].cuda()
        evk = torch.linalg.eigvalsh(torch.cov(Z.T)).flip(0).clamp_min(0)
        Zn = F.normalize(Z, dim=-1)
        sm = F.normalize(summ[k], dim=-1)
        sim = sm @ sm.T
        S["spaces"][k] = {
            "feat_dim": Z.shape[1], "summary_dim": summ[k].shape[1],
            "feat_norm_median": round(Z.norm(dim=-1).median().item(), 2),
            "cos_random_pairs_raw": round((Zn[pr[:2000]] * Zn[pr[2000:]]).sum(-1).mean().item(), 3),
            "pca_top1": round((evk[0] / evk.sum()).item(), 3), "pca_top10": round((evk[:10].sum() / evk.sum()).item(), 3),
            "effective_rank": round((evk.sum() ** 2 / (evk ** 2).sum()).item(), 1),
            "cka_vs_backbone": round(linear_cka(samp["backbone"][:8000].cuda().double(), Z[:8000].double()), 3),
            "cka_vs_backbone_inliers": round(linear_cka(X[inl][:8000].double(), Z[inl][:8000].double()), 3),
            "summary_norm": round(summ[k].norm(dim=-1).mean().item(), 2),
            "summary_cos_min_over_episode": round(sim.min().item(), 3),
            "summary_cos_mean_over_episode": round(sim.mean().item(), 3),
        }
    print(json.dumps({k: S[k] for k in ["patch_tokens", "patch_tokens_inliers", "spaces"]}, indent=1))
    torch.cuda.empty_cache()

    # ---------------- 3. PCA maps of every output space ----------------
    show = [0, 60, 297, 450, 640]
    bases = {k: pca_basis(samp[k].cuda()[inl]) for k in SPACES}
    outs = {}
    torch.cuda.empty_cache()
    for i in show:
        _, r = run(model, to_input(frames[FRONT][i:i + 1]))
        outs[i] = {k: r[k][1][0] for k in SPACES}
    fig, axs = plt.subplots(len(show), 1 + len(SPACES), figsize=(4.2 * (1 + len(SPACES)), 2.5 * len(show)), layout="constrained")
    for a, i in enumerate(show):
        axs[a, 0].imshow(frames[FRONT][i].permute(1, 2, 0).numpy())
        axs[a, 0].set_title(f"frame {i} (t={ts[i]:.1f}s)", fontsize=9)
        for b, k in enumerate(SPACES):
            axs[a, b + 1].imshow(pca_rgb(outs[i][k], bases[k], (Hp, Wp)), interpolation="nearest")
            if a == 0:
                d = S["spaces"][k]["feat_dim"]
                axs[a, b + 1].set_title(f"{k} ({d}-d) PCA, top3={(bases[k]['ev'][:3].sum() / bases[k]['ev'].sum()).item():.2f}", fontsize=9)
    for ax in axs.flat:
        ax.axis("off")
    fig.savefig(os.path.join(args.out, "spaces_pca.png"), dpi=100)
    plt.close(fig)

    # token norm map + PC1 map (what is the dominant direction?)
    fig, axs = plt.subplots(2, 3, figsize=(15, 6), layout="constrained")
    for a, i in enumerate([0, 297, 640][:3]):
        f = outs[i]["backbone"] if i in outs else None
        axs[0, a].imshow(f.norm(dim=-1).reshape(Hp, Wp).log10().cpu(), cmap="magma")
        axs[0, a].set_title(f"log10 token norm, frame {i} (max {f.norm(dim=-1).max().item():.0f})", fontsize=9)
        axs[1, a].imshow(((f - bb["mu"]) @ bb["V"][:, 0]).reshape(Hp, Wp).cpu(), cmap="coolwarm")
        axs[1, a].set_title(f"inlier PC1 score, frame {i}", fontsize=9)
    for ax in axs.flat:
        ax.axis("off")
    fig.savefig(os.path.join(args.out, "norm_pc1_maps.png"), dpi=100)
    plt.close(fig)

    # ---------------- 4. query-patch similarity (centered cosine), incl. cross-camera ----------------
    queries = [("wheel logo", 0, 31, 36), ("L robot hand", 60, 35, 12), ("R robot hand", 0, 28, 62), ("mirror blinker", 297, 24, 20)]
    tgt = [(FRONT, 0), (FRONT, 297), (FRONT, 640), (LWRIST, 0), (LWRIST, 297), (RWRIST, 0)]
    tgt_feats = []
    for cam, i in tgt:
        _, r = run(model, to_input(frames[cam][i:i + 1]))
        tgt_feats.append(r["backbone"][1][0])
    center = Xi.mean(0)
    fig, axs = plt.subplots(len(queries), len(tgt) + 1, figsize=(3.2 * (len(tgt) + 1), 2.3 * len(queries)), layout="constrained")
    S["query_similarity"] = {}
    for a, (qn, qi, qr, qc) in enumerate(queries):
        q = F.normalize(outs[qi]["backbone"][qr * Wp + qc] - center, dim=0)
        img = frames[FRONT][qi].permute(1, 2, 0).numpy()
        axs[a, 0].imshow(img)
        axs[a, 0].scatter([qc * 16 + 8], [qr * 16 + 8], s=80, c="cyan", marker="x", linewidths=2)
        axs[a, 0].set_title(f"query: {qn} (f{qi})", fontsize=9)
        S["query_similarity"][qn] = {}
        for b, ((cam, i), tf) in enumerate(zip(tgt, tgt_feats)):
            hw = (Hp, Wp) if cam == FRONT else (30, 40)
            s = (F.normalize(tf - center, dim=-1) @ q).reshape(hw).cpu().numpy()
            axs[a, b + 1].imshow(frames[cam][i].permute(1, 2, 0).numpy())
            axs[a, b + 1].imshow(np.kron(s, np.ones((16, 16))), cmap="jet", alpha=0.55, vmin=0, vmax=0.8)
            axs[a, b + 1].set_title(f"{cam.split('.')[-1].replace('_view', '')} f{i}  max={s.max():.2f}", fontsize=8)
            S["query_similarity"][qn][f"{cam.split('.')[-1]}_f{i}_max"] = round(float(s.max()), 3)
    for ax in axs.flat:
        ax.axis("off")
    fig.savefig(os.path.join(args.out, "query_similarity.png"), dpi=90)
    plt.close(fig)

    # ---------------- 5. resolution sweep ----------------
    res_list = [(256, 448), (368, 656), (512, 912), (720, 1280), (1024, 1824), (1152, 2048)]
    rf = [0, 297, 640]
    S["resolution"] = {}
    ref = {}
    maps = {}
    for hw in res_list:
        x = to_input(frames[FRONT][rf], hw)
        torch.cuda.synchronize()
        t0 = time.time()
        _, r = run(model, x)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / len(rf)
        g = (hw[0] // 16, hw[1] // 16)
        pooled = {k: F.adaptive_avg_pool2d(r[k][1].reshape(len(rf), *g, -1).permute(0, 3, 1, 2), (16, 28)) for k in SPACES}
        maps[hw] = r["backbone"][1][0].reshape(*g, -1).cpu()
        if hw == (720, 1280):
            ref = {k: (r[k][0].cpu(), pooled[k].cpu()) for k in SPACES}
        S["resolution"][f"{hw[0]}x{hw[1]}"] = {"tokens": g[0] * g[1], "ms_per_frame": round(1000 * dt, 1),
                                                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                                                "_summ": {k: r[k][0].cpu() for k in SPACES},
                                                "_pool": {k: pooled[k].cpu() for k in SPACES},
                                                "feat_norm_median": round(r["backbone"][1].norm(dim=-1).median().item(), 1),
                                                "outlier_frac": round((r["backbone"][1].norm(dim=-1) > args.outlier_norm).float().mean().item(), 5)}
        del r, pooled
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    for key, v in S["resolution"].items():
        for k in SPACES:
            s, p = v["_summ"][k], v["_pool"][k]
            v[f"{k}_summary_cos_vs_720p"] = round(F.cosine_similarity(s, ref[k][0].cpu(), dim=-1).mean().item(), 4)
            v[f"{k}_spatial_cos_vs_720p"] = round(F.cosine_similarity(p, ref[k][1].cpu(), dim=1).mean().item(), 4)
        v.pop("_summ")
        v.pop("_pool")
    allf = torch.cat([m.flatten(0, 1) for m in maps.values()])
    allf = allf[allf.norm(dim=-1) <= args.outlier_norm]
    rb = pca_basis(allf[torch.randperm(len(allf))[:60000]].cuda())
    fig, axs = plt.subplots(1, len(res_list), figsize=(4 * len(res_list), 2.6), layout="constrained")
    for ax, (hw, m) in zip(axs, maps.items()):
        ax.imshow(pca_rgb(m.flatten(0, 1).cuda(), rb, m.shape[:2]), interpolation="nearest")
        ax.set_title(f"{hw[0]}x{hw[1]} -> {m.shape[0]}x{m.shape[1]} tokens", fontsize=9)
        ax.axis("off")
    fig.savefig(os.path.join(args.out, "resolution_pca.png"), dpi=100)
    plt.close(fig)

    # ---------------- 6. numerics: dtype + batch-size dependence ----------------
    xb = to_input(frames[FRONT][idx[:8]])
    _, r32 = run(model, xb, torch.float32)
    _, r16 = run(model, xb, torch.bfloat16)
    _, r16b1 = run(model, xb[:1], torch.bfloat16)
    _, r32b1 = run(model, xb[:1], torch.float32)
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    S["numerics"] = {
        "bf16_vs_fp32_feat_rel": round(rel(r16["backbone"][1], r32["backbone"][1]), 5),
        "bf16_vs_fp32_feat_cos": round(F.cosine_similarity(r16["backbone"][1], r32["backbone"][1], dim=-1).mean().item(), 5),
        "bf16_vs_fp32_siglip_summary_cos": round(F.cosine_similarity(r16["siglip2-g"][0], r32["siglip2-g"][0], dim=-1).mean().item(), 5),
        "bf16_B1_vs_B8_feat_rel": round(rel(r16b1["backbone"][1][0], r16["backbone"][1][0]), 5),
        "fp32_B1_vs_B8_feat_rel": round(rel(r32b1["backbone"][1][0], r32["backbone"][1][0]), 7),
    }

    # ---------------- 7. layer-wise ----------------
    D = S["depth"]
    torch.cuda.empty_cache()
    chunks = []
    for b in range(0, len(idx[::6]), 4):
        xi = to_input(frames[FRONT][idx[::6][b:b + 4]])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            it = model.radio_model.forward_intermediates(xi, indices=list(range(D)), output_fmt="NLC", intermediates_only=True)
        chunks.append([t.half().cpu() for t in it])
        del it
    inter = [torch.cat([c[l] for c in chunks]).float() for l in range(D)]
    del chunks
    fin = inter[-1].flatten(0, 1)
    sel = torch.randperm(len(fin))[:6000]
    S["layers"] = {"patch_norm_median": [], "outlier_frac_norm_gt_3x_median": [], "max_norm": [], "cka_vs_last": [], "pca_top1": []}
    fin_in = fin[sel].norm(dim=-1) <= args.outlier_norm
    for t in inter:
        z = t.flatten(0, 1)[sel].double()
        tn = t.norm(dim=-1)
        S["layers"]["patch_norm_median"].append(round(tn.median().item(), 1))
        S["layers"]["outlier_frac_norm_gt_3x_median"].append(round((tn > 3 * tn.median()).float().mean().item(), 5))
        S["layers"]["max_norm"].append(round(tn.max().item(), 1))
        S["layers"]["cka_vs_last"].append(round(linear_cka(z[fin_in].cuda(), fin[sel][fin_in].double().cuda()), 3))
        evl = torch.linalg.eigvalsh(torch.cov(z[fin_in].T.float().cuda())).flip(0).clamp_min(0)
        S["layers"]["pca_top1"].append(round((evl[0] / evl.sum()).item(), 3))
    show_layers = [3, 8, 13, 18, 22, D - 1]
    fig, axs = plt.subplots(1, len(show_layers), figsize=(4 * len(show_layers), 2.6), layout="constrained")
    for ax, l in zip(axs, show_layers):
        z = inter[l]
        zz = z.flatten(0, 1)[sel]
        lb = pca_basis(zz[zz.norm(dim=-1) <= 3 * zz.norm(dim=-1).median()].cuda())
        ax.imshow(pca_rgb(z[0].cuda(), lb, (Hp, Wp)), interpolation="nearest")
        ax.set_title(f"block {l + 1}/{D}  CKA→last {S['layers']['cka_vs_last'][l]:.2f}", fontsize=9)
        ax.axis("off")
    fig.savefig(os.path.join(args.out, "layers_pca.png"), dpi=100)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 3.4), layout="constrained")
    L = np.arange(1, D + 1)
    for ax, key in zip(axs, ["patch_norm_median", "max_norm", "cka_vs_last"]):
        if key == "max_norm":
            ax.set_yscale("log")
        ax.plot(L, S["layers"][key], "o-", ms=3)
        ax.set_xlabel("block")
        ax.set_title(key)
        ax.grid(alpha=0.3)
    fig.savefig(os.path.join(args.out, "layers_curves.png"), dpi=100)
    plt.close(fig)

    json.dump(S, open(os.path.join(args.out, "stats.json"), "w"), indent=1)
    print(json.dumps({k: S[k] for k in ["resolution", "numerics", "layers", "query_similarity"]}, indent=1))
    print("done ->", args.out)


if __name__ == "__main__":
    main()
