"""Multi-VLM episode-memory pretraining on streamed EgoDex (no action head).

Every step, every frozen VLM of `vlms` encodes the same batch of frames (one per stream slot, each slot walking its
episode from frame 0, starVLA/dataloader/egodex_stream.py); each VLM drives its own memory stream (per-VLM token_resize,
shared memory transformer: starVLA/model/modules/dual_system/memory_pretrain.py, MultiVLMMemory) and contributes,
through the Joint Decision Transformer over the memory tokens M_t, the episode-progress loss (20 bins) and, on annotated episodes,
the current-subtask / finished-subtasks (frozen-T5 candidates of the episode's subtasks) / subtask-progress losses. Truncated BPTT over the last `memory.tbptt`
updates per stream (cached VLM tokens, the older memory detached), as QwenGR00T_EpisodeMemory's stream mode.

No DDP wrapper: the trainable module is small, so after every backward its gradients are averaged over the ranks in one
flat all-reduce (the frozen VLMs are never wrapped or broadcast).

Launch (one process per GPU; torchrun / accelerate launch env vars):
  accelerate launch --num_processes 4 starVLA/training/train_memory_pretrain.py --config_yaml <yaml> [key=value ...]
Outputs in <run_root_dir>/<run_id>/: checkpoints/steps_<k>.pt (trainable state + optimizer), checkpoints/steps_<k>_
memory_<vlm>.pt (episode_memory.* for QwenGR00T_EpisodeMemory, episode_memory.init_from), log.jsonl, eval.jsonl; with
wandb_project set, the same rows go to Weights & Biases from rank 0 (one run per run_id, continued on resume; best
effort: training goes on without it).
"""
import argparse
import json
import math
import os
import random
import secrets
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf

from starVLA.dataloader.egodex_stream import EgoDexFrames, build_stream_loader, load_index, load_sources, load_subtasks
from starVLA.model.modules.dual_system.memory_pretrain import (FrozenT5Encoder, MultiVLMMemory, done_loss,
                                                               progress_bin_targets, progress_loss, subtask_loss)
from starVLA.model.modules.vlm import get_vlm_model


# ----------------------------------------------------------------------------------------------------------------------
def load_vlm(spec, device):
    cfg = OmegaConf.create({"framework": {"qwenvl": {k: v for k, v in spec.items() if k != "name"}},
                            "datasets": {"vla_data": {}}})
    cfg.framework.qwenvl.base_vlm = spec["path"]
    iface = get_vlm_model(cfg).to(device).eval()
    for p in iface.parameters():
        p.requires_grad_(False)
    return iface, int(iface.model.config.hidden_size)


def masked_mean(h, mask):
    m = mask.to(h.dtype)[..., None]
    return (h * m).sum(-2) / m.sum(-2).clamp_min(1)


def image_groups(images) -> list:
    """Sample indices grouped by their image count (one VLM call per group: a batch mixes one-camera and three-camera
    sources, and not every interface takes a ragged image count)."""
    by = {}
    for i, ims in enumerate(images):
        by.setdefault(len(ims), []).append(i)
    return [by[k] for k in sorted(by)]


def as_image_lists(images) -> list:
    return [list(x) if isinstance(x, (list, tuple)) else [x] for x in images]


def ex_images(ex) -> list:
    """A sample's camera images (egodex_stream "images"; older batches: the one "image")."""
    return ex["images"] if "images" in ex else [ex["image"]]


@torch.no_grad()
def vlm_frames(iface, images, langs, inputs=None):
    """Frozen VLM on each sample's frame(s) (one image, or a list: the source's cameras) -> (last hidden state
    [B, L, H], mask [B, L]); samples with different image counts run as separate calls, right-padded to one L.
    `inputs`: [(sample indices, processor output)] made beforehand (on the CPU, by a dataloader worker: CpuInputs),
    else built here."""
    images = as_image_lists(images)
    if inputs is None:
        inputs = [(idx, iface.build_qwenvl_inputs(images=[images[i] for i in idx], instructions=[langs[i] for i in idx]))
                  for idx in image_groups(images)]
    dev = next(iface.parameters()).device
    parts = []
    for idx, inp in inputs:
        inp = inp.to(dev, non_blocking=True)
        out = iface(**inp, output_hidden_states=True, return_dict=True, logits_to_keep=1)
        h = out.hidden_states[-1]
        m = inp.get("attention_mask", None)
        m = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device) if m is None else m.to(h.device).bool()
        parts.append((idx, h, m))
    if len(parts) == 1 and parts[0][0] == list(range(len(images))):
        return parts[0][1], parts[0][2]
    L = max(h.shape[1] for _, h, _ in parts)
    h0 = parts[0][1]
    hs = h0.new_zeros(len(images), L, h0.shape[2])
    ms = torch.zeros(len(images), L, dtype=torch.bool, device=h0.device)
    for idx, h, m in parts:
        ix = torch.tensor(idx, device=h0.device)
        hs[ix, :h.shape[1]], ms[ix, :h.shape[1]] = h, m
    return hs, ms


class CpuInputs:
    """Builds every VLM's processor inputs of a batch on the CPU, in the dataloader workers (forked: the processors are
    shared, the models never touched there): each interface's own build_qwenvl_inputs on a stand-in holding its
    processor and config, with the model's device the CPU."""

    def __init__(self, vlms: dict):
        import types

        cpu = types.SimpleNamespace(device=torch.device("cpu"))
        self.parts = {n: (type(i).build_qwenvl_inputs, types.SimpleNamespace(processor=i.processor, config=i.config,
                                                                              model=cpu)) for n, i in vlms.items()}

    def __call__(self, batch):
        torch.set_num_threads(2)
        images, langs = [ex_images(ex) for ex in batch], [ex["lang"] for ex in batch]
        groups = image_groups(images)
        return {n: [(idx, build(stand_in, images=[images[i] for i in idx], instructions=[langs[i] for i in idx]))
                    for idx in groups] for n, (build, stand_in) in self.parts.items()}


class Streams:
    """Per-(VLM, slot) memory stream state: detached base memory, cached VLM tokens of the last tbptt - 1 updates."""

    def __init__(self, tbptt: int):
        self.tbptt, self.s = tbptt, {}

    def get(self, slot, step, count):
        st = self.s.get(slot)
        if step == 0 or st is None or st["key"] != (count, step - 1):
            st = {"base": None, "frames": [], "valid": step == 0}
        st["key"] = (count, step)
        self.s[slot] = st
        return st


def _acc(pred: torch.Tensor, tgt: torch.Tensor, w: torch.Tensor) -> float:
    return float(((pred == tgt).float() * w).sum() / w.sum().clamp_min(1))


def step_vlm(model, name, iface, streams: Streams, batch, t5: Optional[FrozenT5Encoder], w, dev):
    """One VLM on one batch -> (weighted loss, logs)."""
    h = vlm_frames(iface, [ex_images(ex) for ex in batch], [ex["lang"] for ex in batch])
    return step_all(model, {name: h}, {name: streams}, batch, t5, w, dev)


def length_groups(lengths: dict, ratio: float = 1.6) -> list:
    """VLM names in groups of similar token count (longest / shortest <= ratio), so stacking them pads little."""
    groups = []
    for n in sorted(lengths, key=lengths.get):
        if groups and lengths[n] <= ratio * lengths[groups[-1][0]]:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def step_all(model, hidden: dict, streams: dict, batch, t5: Optional[FrozenT5Encoder], w, dev, group_ratio: float = 1.6):
    """Every VLM's stream on one batch at once -> (loss = the mean over the VLMs of each one's weighted loss, logs
    "<vlm>/<key>"). hidden: VLM -> (last hidden state [B, L_v, H_v], mask) of the batch's frames. Each VLM's cached
    tokens go through its token_resize at their own length; VLMs of similar token count (length_groups) are padded to
    their group's longest and stacked on the batch axis, so the TBPTT unroll of the shared memory runs once per group
    (padding every VLM to the longest one costs ~2x the activation memory), and the decision heads once over all
    V x B streams."""
    names, b = list(hidden), len(batch)
    keys = [tuple(int(x) for x in ex["stream"]) for ex in batch]
    states = {n: [streams[n].get(*k) for k in keys] for n in names}
    tbptt = streams[names[0]].tbptt
    seqs = {n: [(st["frames"] + [hidden[n][0][i][hidden[n][1][i]]])[-tbptt:] for i, st in enumerate(states[n])]
            for n in names}
    lens = [len(q) for q in seqs[names[0]]]  # the same walk for every VLM
    n_up = max(lens)
    vk = torch.zeros(n_up, b, dtype=torch.bool, device=dev)
    for i, l in enumerate(lens):  # right-aligned: the current frame at n_up - 1 for every sample
        vk[n_up - l:, i] = True
    valid = torch.tensor([st["valid"] for st in states[names[0]]], device=dev)
    wv = valid.float()
    progress = torch.tensor([ex["progress"] for ex in batch], dtype=torch.float32, device=dev)
    # subtask terms: the episode's annotated subtasks are the candidates; skipped (no made-up target) without them
    cands = [list(ex["subtasks"]) for ex in batch]
    cur = torch.tensor([int(ex["sub_cur"]) for ex in batch], device=dev)
    sub_p = torch.tensor([float(ex.get("sub_progress", 0.0)) for ex in batch], dtype=torch.float32, device=dev)
    has = valid & torch.tensor([0 <= int(ex["sub_cur"]) < len(c) for ex, c in zip(batch, cands)], device=dev)
    use_sub = t5 is not None and bool(has.any())
    V = len(names)
    rep = lambda t: t.repeat(V, *([1] * (t.dim() - 1)))  # noqa: E731 - per-sample tensors for the V stacked streams
    logs = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        init = model.init(1).to(dev)[0]
        lengths = {n: max(t.shape[0] for q in seqs[n] for t in q) for n in names}
        last, first = {}, {}
        for group in length_groups(lengths, group_ratio):
            xs, ms = [], []
            width = max(lengths[n] for n in group)
            for n in group:
                h = hidden[n][0]
                tk = h.new_zeros(n_up, b, lengths[n], h.shape[-1])
                mk = torch.zeros(n_up, b, width, dtype=torch.bool, device=dev)
                for i, q in enumerate(seqs[n]):
                    for s_, t in enumerate(q, start=n_up - len(q)):
                        tk[s_, i, : t.shape[0]], mk[s_, i, : t.shape[0]] = t, True
                xs.append(F.pad(model.resize(n, tk), (0, 0, 0, width - lengths[n])))  # resized at its own length
                ms.append(mk)
            x, mk = torch.cat(xs, dim=1), torch.cat(ms, dim=1)  # [n_up, G * B, L_group, token_dim]
            m = torch.stack([init if st["base"] is None else st["base"] for n in group for st in states[n]])
            vk_g = vk.repeat(1, len(group))
            f_ = m
            for s_ in range(n_up):
                m = model.memory.update(m, x[s_], mk[s_], vk_g[s_])
                if s_ == 0:
                    f_ = m
            for gi, n in enumerate(group):
                last[n], first[n] = m[gi * b:(gi + 1) * b], f_[gi * b:(gi + 1) * b]
        m = torch.cat([last[n] for n in names])  # [V * B, N, d] in the order of `names`
        # decisions from the memory tokens M_t only
        z = model.decide(m)
        feats, cvalid = t5.candidates([c if ok else [] for c, ok in zip(cands, has.tolist())]) if use_sub else (None, None)
        out = model.decision_heads(z, rep(feats) if use_sub else None, rep(cvalid) if use_sub else None)
        bins = progress_bin_targets(progress, model.progress_bins)
        # every VLM has the same samples and masks: the mean over the V * B rows is the mean of the per-VLM means
        loss = w["progress"] * progress_loss(out["progress_logits"], rep(bins), rep(wv))
        if use_sub:
            done = torch.zeros_like(cvalid)
            for i, ex in enumerate(batch):
                if has[i]:
                    done[i, :len(ex["sub_done"])] = torch.tensor(ex["sub_done"], dtype=torch.bool, device=dev)
            sbins = progress_bin_targets(sub_p, model.progress_bins)
            loss = (loss + w["subtask"] * subtask_loss(out["subtask_logits"], rep(cur), rep(has))
                    + w["subtask_done"] * done_loss(out["done_logits"], rep(done), rep(cvalid), rep(has))
                    + w["subtask_progress"] * progress_loss(out["subtask_progress_logits"], rep(sbins), rep(has.float())))
        with torch.no_grad():  # per-VLM logs
            wh = has.float()
            for vi, n in enumerate(names):
                sl = slice(vi * b, (vi + 1) * b)
                pl = out["progress_logits"][sl]
                lg = {"progress_ce": float(progress_loss(pl, bins, wv)), "progress_acc": _acc(pl.argmax(-1), bins, wv),
                      "progress_mae": float(((out["progress"][sl] - progress).abs() * wv).sum() / wv.sum().clamp_min(1)),
                      "subtask_frac": float(has.float().mean()), "valid_frac": float(valid.float().mean())}
                if use_sub:
                    sl_l, dl = out["subtask_logits"][sl], out["done_logits"][sl]
                    dw = (cvalid & has[:, None]).float()
                    lg.update({
                        "subtask_ce": float(subtask_loss(sl_l, cur, has)),
                        "subtask_acc": _acc(sl_l.argmax(-1), cur, wh),
                        "subtask_candidates": float(cvalid[has].sum(-1).float().mean()),
                        "subtask_done_bce": float(done_loss(dl, done, cvalid, has)),
                        "subtask_done_acc": float((((dl > 0) == done).float() * dw).sum() / dw.sum().clamp_min(1)),
                        "subtask_progress_ce": float(progress_loss(out["subtask_progress_logits"][sl], sbins, wh)),
                        "subtask_progress_mae": float(((out["subtask_progress"][sl] - sub_p).abs() * wh).sum()
                                                      / wh.sum().clamp_min(1))})
                logs.update({f"{n}/{k}": v for k, v in lg.items()})
    for vi, n in enumerate(names):
        for i, (st, q) in enumerate(zip(states[n], seqs[n])):
            if len(q) == tbptt:  # the oldest cached frame leaves the window -> folded into the detached base
                st["base"], st["frames"] = first[n][i].detach(), q[1:]
            else:
                st["frames"] = q
    return loss, logs


@torch.no_grad()
def evaluate(model, vlms, t5, cfg, rank: int, world: int, dev) -> dict:
    """Held-out streams: cfg.eval.splits of data.root (keys "<vlm>/<metric>", "mean/<metric>"), or with cfg.eval.sources
    ([{name, root, splits, subtasks_dir, cameras, max_episodes, max_updates}]) each source on its own (the same keys
    under "<source>/"). -> {key: value, ...}."""
    ec, dc = cfg.eval, cfg.data
    if not ec.get("sources", None):
        eps = load_index(dc.root, list(ec.splits), int(dc.get("min_frames", 1)))
        return evaluate_episodes(model, vlms, t5, eps, EgoDexFrames(dc.root, eps, None), cfg, rank, world, dev,
                                 load_subtasks(dc.get("subtasks_dir", None)), int(ec.get("max_episodes", 0)),
                                 int(ec.max_updates))
    res = {}
    for src in ec.sources:
        src = OmegaConf.to_container(src, resolve=True)
        eps, subs = load_sources([src], int(dc.get("min_frames", 1)))
        r = evaluate_episodes(model, vlms, t5, eps, EgoDexFrames(None, eps, None), cfg, rank, world, dev, subs,
                              int(src.get("max_episodes", ec.get("max_episodes", 0))),
                              int(src.get("max_updates", ec.max_updates)))
        res.update({f"{src['name']}/{k}": v for k, v in r.items()})
    return res


@torch.no_grad()
def evaluate_episodes(model, vlms, t5, eps, ds, cfg, rank, world, dev, subtasks, max_episodes: int, cap: int) -> dict:
    """Every episode (a seeded sample of max_episodes) walked from frame 0 every data.stride frames (no jitter), up to
    `cap` updates, the memory carried over the whole walk (no truncation). Per VLM: episode / subtask progress MAE,
    progress bin accuracy, current-subtask and finished-subtask accuracy (annotated episodes). The episodes are split
    over the ranks; the sums are all-reduced."""
    ec, dc = cfg.eval, cfg.data
    if max_episodes and len(eps) > max_episodes:
        eps = random.Random(0).sample(eps, max_episodes)
    ds.episodes, ds.subtasks = eps, subtasks
    stride, bs = int(dc.stride), int(ec.batch_size)
    names = list(vlms)
    keys = ["progress_ae", "progress_hit", "n", "subtask_hit", "subtask_progress_ae", "n_sub", "done_hit", "n_done"]
    acc = torch.zeros(len(names), len(keys), dtype=torch.float64, device=dev)
    mine = list(range(rank, len(eps), world))
    was_training = model.training
    model.eval()
    for g0 in range(0, len(mine), bs):
        group = mine[g0:g0 + bs]
        steps = [min((int(eps[e]["n"]) - 1) // stride + 1, cap) for e in group]
        mem = {n: model.init(len(group)).to(dev) for n in names}
        for k in range(max(steps)):
            live = [i for i, s_ in enumerate(steps) if k < s_]
            items = [ds[(i, group[i], k * stride, k, 0)] for i in live]
            alive = torch.zeros(len(group), dtype=torch.bool, device=dev)
            alive[live] = True
            prog = torch.tensor([ex["progress"] for ex in items], device=dev)
            cands = [ex["subtasks"] for ex in items]
            has = torch.tensor([bool(c) for c in cands], device=dev)
            feats, cvalid = t5.candidates(cands) if (t5 is not None and bool(has.any())) else (None, None)
            for vi, name in enumerate(names):
                h, mask = vlm_frames(vlms[name], [ex_images(ex) for ex in items], [ex["lang"] for ex in items])
                full_h = h.new_zeros(len(group), h.shape[1], h.shape[2])
                full_m = torch.zeros(len(group), h.shape[1], dtype=torch.bool, device=dev)
                full_h[live], full_m[live] = h, mask
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    mem[name] = model.update(name, mem[name], full_h, full_m, alive)
                    z = model.decide(mem[name][live])
                    out = model.decision_heads(z, feats, cvalid)
                bins = progress_bin_targets(prog, model.progress_bins)
                a = acc[vi]
                a[0] += (out["progress"] - prog).abs().sum()
                a[1] += (out["progress_logits"].argmax(-1) == bins).sum()
                a[2] += len(live)
                if feats is not None:
                    cur = torch.tensor([ex["sub_cur"] for ex in items], device=dev)
                    sp = torch.tensor([ex["sub_progress"] for ex in items], device=dev)
                    done = torch.zeros_like(cvalid)
                    for i, ex in enumerate(items):
                        done[i, :len(ex["sub_done"])] = torch.tensor(ex["sub_done"], dtype=torch.bool, device=dev)
                    a[3] += ((out["subtask_logits"].argmax(-1) == cur) & has).sum()
                    a[4] += ((out["subtask_progress"] - sp).abs() * has).sum()
                    a[5] += has.sum()
                    dmask = cvalid & has[:, None]
                    a[6] += (((out["done_logits"] > 0) == done) & dmask).sum()
                    a[7] += dmask.sum()
    if world > 1:
        dist.all_reduce(acc)
    model.train(was_training)
    res = {}
    for vi, name in enumerate(names):
        a = dict(zip(keys, acc[vi].tolist()))
        n, ns, nd = max(a["n"], 1), max(a["n_sub"], 1), max(a["n_done"], 1)
        res.update({f"{name}/progress_mae": a["progress_ae"] / n, f"{name}/progress_acc": a["progress_hit"] / n,
                    f"{name}/subtask_acc": a["subtask_hit"] / ns, f"{name}/subtask_progress_mae":
                    a["subtask_progress_ae"] / ns, f"{name}/subtask_done_acc": a["done_hit"] / nd})
    for m in ("progress_mae", "progress_acc", "subtask_acc", "subtask_progress_mae", "subtask_done_acc"):
        res[f"mean/{m}"] = float(np.mean([res[f"{n}/{m}"] for n in names]))
    res["episodes"], res["frames"] = len(eps), float(acc[0, 2])
    return res


def init_wandb(cfg, out_dir: Path):
    """rank 0: the run's W&B run (id kept in <out_dir>/wandb_id.txt so a resubmit continues it), or None."""
    if not cfg.get("wandb_project", None) or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb

        id_f = out_dir / "wandb_id.txt"
        run_id = id_f.read_text().strip() if id_f.exists() else secrets.token_hex(4)  # W&B ids: [a-z0-9]
        id_f.write_text(run_id)
        return wandb.init(project=cfg.wandb_project, entity=cfg.get("wandb_entity", None), name=cfg.run_id, id=run_id,
                          resume="allow", dir=str(out_dir), config=OmegaConf.to_container(cfg, resolve=True))
    except Exception as err:  # noqa: BLE001 - logging must not stop training
        print(f"[wandb] disabled: {type(err).__name__}: {err}", flush=True)
        return None


def wandb_log(run, row: dict, step: int, prefix: str = ""):
    if run is None:
        return
    try:
        run.log({prefix + k: v for k, v in row.items() if k != "step" and isinstance(v, (int, float))}, step=step)
    except Exception as err:  # noqa: BLE001
        print(f"[wandb] log failed: {type(err).__name__}: {err}", flush=True)


def dc_subtasks(cfg) -> bool:
    """Subtask annotations configured (data.subtasks_dir, or any data.sources' subtasks_dir)."""
    return bool(cfg.data.get("subtasks_dir", None)) or any(s.get("subtasks_dir", None) for s in cfg.data.get("sources", []) or [])


def allreduce_grads(params, world):
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1).float() for g in grads])
    dist.all_reduce(flat)
    flat /= world
    o = 0
    for p, g in zip(params, grads):
        p.grad = flat[o:o + g.numel()].view_as(g).to(p.dtype)
        o += g.numel()


# ----------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_yaml", required=True)
    a, rest = ap.parse_known_args()
    cfg = OmegaConf.merge(OmegaConf.load(a.config_yaml), OmegaConf.from_dotlist([r.lstrip("-") for r in rest]))

    # worker -> trainer tensors (CpuInputs) through /dev/shm files: the default file-descriptor sharing goes through
    # /tmp sockets, and a lost one kills the rank (FileNotFoundError in resource_sharer after ~4k steps)
    torch.multiprocessing.set_sharing_strategy("file_system")
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if cfg.get("gpu_memory_fraction", None):  # sharing the GPUs with another job (srun --overlap)
        torch.cuda.set_per_process_memory_fraction(float(cfg.gpu_memory_fraction), dev)
    if world > 1:
        dist.init_process_group("nccl")
    seed = int(cfg.seed) + rank
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)

    out_dir = Path(cfg.run_root_dir) / cfg.run_id
    ckpt_dir = out_dir / "checkpoints"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out_dir / "config.yaml")
    wb = init_wandb(cfg, out_dir) if rank == 0 else None
    if world > 1:
        dist.barrier()  # a slow W&B start on rank 0 must not leave the others alone at the first collective

    vlms, dims = {}, {}
    for spec in cfg.vlms:
        t0 = time.time()
        vlms[spec.name], dims[spec.name] = load_vlm(OmegaConf.to_container(spec), dev)
        if rank == 0:
            print(f"[vlm] {spec.name}: hidden {dims[spec.name]}, {time.time() - t0:.0f}s", flush=True)

    mc = cfg.memory
    dcfg = cfg.decision
    t5 = None
    if dc_subtasks(cfg) and any(float(cfg.loss.get(k, 0.0)) for k in ("subtask", "subtask_done", "subtask_progress")):
        t5 = FrozenT5Encoder.from_pretrained(cfg.t5.path, dev, cache_size=int(cfg.t5.get("cache_size", 50000)))
    model = MultiVLMMemory(dims, token_dim=mc.token_dim, num_tokens=mc.num_tokens, d_model=mc.d_model, depth=mc.depth,
                           num_heads=mc.num_heads, mlp_ratio=mc.mlp_ratio, progress_bins=mc.progress_bins,
                           attention=mc.attention, t5_dim=t5.dim if t5 is not None else int(cfg.t5.get("dim", 768)),
                           decision_layers=dcfg.num_layers, decision_heads=dcfg.num_heads,
                           decision_mlp_ratio=dcfg.mlp_ratio, decision_progress_bins=dcfg.progress_bins,
                           subtask_temperature=dcfg.subtask_temperature).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    if world > 1:
        for p in model.state_dict().values():
            dist.broadcast(p, 0)
    if rank == 0:
        print(f"[model] trainable {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)

    tc = cfg.trainer
    opt = torch.optim.AdamW(params, lr=tc.lr, betas=tuple(tc.betas), weight_decay=tc.weight_decay)

    def lr_at(s):
        if s < tc.warmup_steps:
            return tc.lr * (s + 1) / tc.warmup_steps
        p = (s - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps)
        return tc.lr * (tc.min_lr_rate + (1 - tc.min_lr_rate) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))

    step = 0
    # resume from the newest of the kept checkpoints (steps_<k>.pt) and the rolling one (latest.pt, every resume_every)
    found = []
    if tc.get("resume", True):
        found = [(int(p.stem.split("_")[1]), p) for p in ckpt_dir.glob("steps_*[0-9].pt")]
        if (ckpt_dir / "latest.pt").exists():
            found.append((int(torch.load(ckpt_dir / "latest.pt", map_location="cpu", weights_only=False)["step"]),
                          ckpt_dir / "latest.pt"))
    if found:
        path = max(found, key=lambda x: x[0])[1]
        sd = torch.load(path, map_location=dev, weights_only=False)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["optimizer"])
        step = int(sd["step"])
        if rank == 0:
            print(f"[resume] {path} (step {step}); streams restart at new episodes", flush=True)

    dc = cfg.data
    sources = OmegaConf.to_container(dc.sources, resolve=True) if dc.get("sources", None) else None
    loader, eps = build_stream_loader(dc.get("root", None), list(dc.get("splits", [])), dc.per_device_batch_size,
                                      stride=dc.stride, jitter=dc.jitter, min_frames=int(dc.get("min_frames", 1)),
                                      seed=seed * 1000 + step, num_workers=dc.num_workers,
                                      subtasks_dir=dc.get("subtasks_dir", None), sources=sources,
                                      prepare=CpuInputs(vlms) if dc.get("cpu_inputs", True) and dc.num_workers else None)
    if rank == 0:
        hours = {}
        for e in eps:
            k = e.get("_source", "data")
            hours[k] = hours.get(k, 0.0) + int(e["n"]) / float(e["fps"]) / 3600
        print(f"[data] {len(eps)} episodes, " + ", ".join(f"{k} {v:.0f} h" for k, v in hours.items()) +
              f"; {world} ranks x {dc.per_device_batch_size} streams", flush=True)

    if cfg.get("eval", None) and cfg.eval.get("only", False):  # evaluate the resumed (or initial) model and stop
        ev = evaluate(model, vlms, t5, cfg, rank, world, dev)
        if rank == 0:
            ev["step"] = step
            with open(out_dir / "eval.jsonl", "a") as f:
                f.write(json.dumps(ev) + "\n")
            wandb_log(wb, ev, step, "eval/")
            print("[eval] " + " ".join(f"{k}={v:.4g}" for k, v in ev.items() if "mean/" in k), flush=True)
        if wb is not None:
            wb.finish()
        if world > 1:
            dist.destroy_process_group()
        return
    w = {k: float(v) for k, v in cfg.loss.items()}
    streams = {n: Streams(int(mc.tbptt)) for n in vlms}
    log_f = open(out_dir / "log.jsonl", "a") if rank == 0 else None
    agg, t_last = {}, time.time()
    it = iter(loader)
    model.train()
    while step < tc.max_steps:
        t_data = time.time()
        batch = next(it)
        batch, pre = batch if isinstance(batch, tuple) else (batch, {})
        t_data = time.time() - t_data
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        hidden = {n: vlm_frames(iface, [ex_images(ex) for ex in batch], [ex["lang"] for ex in batch], pre.get(n))
                  for n, iface in vlms.items()}
        total, logs = step_all(model, hidden, streams, batch, t5, w, dev)
        opt.zero_grad(set_to_none=True)
        total.backward()
        if world > 1:
            allreduce_grads(params, world)
        gn = torch.nn.utils.clip_grad_norm_(params, tc.grad_clip)
        opt.step()
        step += 1
        logs.update({"loss": float(total.detach()), "grad_norm": float(gn), "lr": lr_at(step - 1), "time/data": t_data})
        for k, v in logs.items():
            agg.setdefault(k, []).append(v)
        if step % tc.log_every == 0 and rank == 0:
            row = {"step": step, "time/step": (time.time() - t_last) / tc.log_every,
                   **{k: float(np.mean(v)) for k, v in agg.items()}}
            t_last, agg = time.time(), {}
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            wandb_log(wb, row, step, "train/")
            keys = ["loss", "grad_norm", "time/step"] + [k for k in row if k.endswith(
                ("progress_mae", "subtask_acc", "subtask_done_acc", "subtask_progress_mae"))]
            print(" ".join(f"{k}={row[k]:.4g}" for k in ["step"] + keys if k in row), flush=True)
        if step % int(tc.get("resume_every", 1000)) == 0 and step % tc.save_every and rank == 0:
            # rolling resume point (overwritten): a crash between the kept checkpoints loses at most resume_every steps
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                        "config": OmegaConf.to_container(cfg)}, ckpt_dir / "latest.tmp")
            os.replace(ckpt_dir / "latest.tmp", ckpt_dir / "latest.pt")
        if step % tc.save_every == 0 or step == tc.max_steps:
            if rank == 0:
                torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                            "config": OmegaConf.to_container(cfg)}, ckpt_dir / f"steps_{step}.pt")
                for n in vlms:
                    torch.save(model.export_for_vlm(n), ckpt_dir / f"steps_{step}_memory_{n}.pt")
                print(f"[save] step {step}", flush=True)
            if world > 1:
                dist.barrier()
            if cfg.get("eval", None) and cfg.eval.get("enabled", True):
                t_ev = time.time()
                ev = evaluate(model, vlms, t5, cfg, rank, world, dev)
                if rank == 0:
                    ev.update({"step": step, "time/eval": time.time() - t_ev})
                    with open(out_dir / "eval.jsonl", "a") as f:
                        f.write(json.dumps(ev) + "\n")
                    wandb_log(wb, ev, step, "eval/")
                    print("[eval] " + " ".join(f"{k}={v:.4g}" for k, v in ev.items() if "mean/" in k), flush=True)
    if rank == 0 and step >= tc.max_steps:
        (out_dir / "DONE").write_text(f"{step}\n")  # the sbatch chain stops here
    if wb is not None:
        wb.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
