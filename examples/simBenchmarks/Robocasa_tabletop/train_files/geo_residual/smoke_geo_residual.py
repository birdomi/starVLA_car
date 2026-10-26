# GPU smoke of QwenGR00T_GeoResidual: build + load the frozen System 2 (trainer.pretrained_checkpoint / reload_modules, or
# argv[2] to override the checkpoint), B=4 real examples: 3 forward/backward/AdamW steps (loss, residual metrics, grads per
# module, peak memory, timings), predict_action fresh (plan_kf back; GEORES_OFF output == plan) and cached (age 0.2 s).
import os, sys, time, collections, numpy as np, torch
from pathlib import Path
from omegaconf import OmegaConf
torch.cuda.set_per_process_memory_fraction(40 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
from starVLA.model.framework.base_framework import build_framework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

cfg = OmegaConf.load(sys.argv[1])
ck = sys.argv[2] if len(sys.argv) > 2 else cfg.trainer.pretrained_checkpoint
t = time.time(); m = build_framework(cfg); print(f"built {time.time() - t:.0f}s {type(m).__name__}", flush=True)
m = TrainerUtils.load_pretrained_backbones(m, ck, reload_modules=cfg.trainer.reload_modules)
m = m.cuda().train()
tr, fz = collections.Counter(), collections.Counter()
for n, p in m.named_parameters():
    (tr if p.requires_grad else fz)[n.split(".")[0]] += p.numel()
print("trainable M:", {k: round(v / 1e6, 2) for k, v in tr.items()}, "| frozen M:", {k: round(v / 1e6, 1) for k, v in fz.items()})
dc = cfg.datasets.vla_data.copy(); dc.data_root_dir = Path(dc.data_root_dir)
ds = get_vla_dataset(data_cfg=dc)
ex = [ds[int(i)] for i in np.random.default_rng(0).integers(0, len(ds), 4)]
print("example images:", len(ex[0]["image"]), "state", np.asarray(ex[0]["state"]).shape)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
torch.cuda.reset_peak_memory_stats()
for it in range(3):
    torch.cuda.synchronize(); t = time.time()
    out = m(ex); loss = out["action_loss"]; loss.backward()
    g = collections.defaultdict(float)
    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            g[n.split(".")[0]] += p.grad.float().pow(2).sum().item()
    opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
    print(f"step {it}: loss {loss.item():.4f} {time.time() - t:.2f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB | "
          + " ".join(f"{k[4:]} {v:.3g}" for k, v in out.items() if k.startswith("geo/"))
          + f" | grad norm {{{', '.join(f'{k}: {v ** 0.5:.3g}' for k, v in g.items())}}}", flush=True)
m.eval()
cur = [{"image": [e["image"][-1]], "lang": e["lang"], "state": np.asarray(e["state"])[-1:]} for e in ex]
t = time.time(); o = m.predict_action(cur); torch.cuda.synchronize()
print(f"predict fresh: {o['normalized_actions'].shape} plan_kf {None if 'plan_kf' not in o else o['plan_kf'].shape} {time.time() - t:.2f}s "
      f"| corrected first {m.K} vs plan max|d| {np.abs(o['normalized_actions'][:, :m.K] - o['plan_kf'][:, :m.K]).max():.4f}")
os.environ["STARVLA_GEORES_OFF"] = "1"
torch.manual_seed(0); o_off = m.predict_action(cur); os.environ.pop("STARVLA_GEORES_OFF")
print("GEORES_OFF fresh: output == plan:", np.allclose(o_off["normalized_actions"], o_off["plan_kf"]))
cached = [dict(c, plan_kf=o["plan_kf"][i], plan_age_s=0.2) for i, c in enumerate(cur)]
t = time.time(); o2 = m.predict_action(cached); torch.cuda.synchronize()
print(f"predict cached (age 0.2 s): {o2['normalized_actions'].shape} plan_kf back: {'plan_kf' in o2} {time.time() - t:.2f}s "
      f"| tail == plan[8:16]: {np.allclose(o2['normalized_actions'][:, m.K:m.K + 8], o['plan_kf'][:, 8:16])}")
print("SMOKE DONE")
