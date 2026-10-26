# QwenGR00T smoke (Helix baseline chain): build each yaml -> trainable params per module, B=4 forward/backward/AdamW steps on
# real data (grads per module), predict_action on the same examples.
import sys, time, collections, numpy as np, torch
from pathlib import Path
from omegaconf import OmegaConf
torch.cuda.set_per_process_memory_fraction(40 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
from starVLA.model.framework.base_framework import build_framework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset

for yaml_path in sys.argv[1:]:
    cfg = OmegaConf.load(yaml_path)
    print(f"\n===== {Path(yaml_path).name}", flush=True)
    t = time.time(); m = build_framework(cfg); print(f"built {time.time() - t:.0f}s {type(m).__name__}", flush=True)
    m = m.cuda().train()
    tr = collections.Counter(); fz = collections.Counter()
    for n, p in m.named_parameters():
        (tr if p.requires_grad else fz)[n.split(".")[0]] += p.numel()
    print("trainable M:", {k: round(v / 1e6, 1) for k, v in tr.items()}, "| frozen M:", {k: round(v / 1e6, 1) for k, v in fz.items()})
    dc = cfg.datasets.vla_data.copy(); dc.data_root_dir = Path(dc.data_root_dir)
    ds = get_vla_dataset(data_cfg=dc)
    rng = np.random.default_rng(0)
    ex = [ds[int(i)] for i in rng.integers(0, len(ds), 4)]
    print("example images:", len(ex[0]["image"]), ex[0]["image"][0].size)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    for it in range(3):
        torch.cuda.synchronize(); t = time.time()
        out = m(ex); loss = out["action_loss"]; loss.backward()
        g = collections.defaultdict(float)
        for n, p in m.named_parameters():
            if p.requires_grad and p.grad is not None:
                g[n.split(".")[0]] += p.grad.float().pow(2).sum().item()
        nograd = sorted({n.split(".")[0] for n, p in m.named_parameters() if p.requires_grad and p.grad is None})
        opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
        print(f"step {it}: loss {loss.item():.4f} rmse {out['action_rmse']:.3f} {time.time() - t:.2f}s peak "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB | vlm {out['timing/vlm_fwd']:.2f}s"
              f" | grad norm per module {{{', '.join(f'{k}: {v ** 0.5:.3g}' for k, v in g.items())}}} | no-grad modules {nograd}", flush=True)
    m.eval()
    o = m.predict_action([{"image": e["image"], "lang": e["lang"], "state": e["state"]} for e in ex])["normalized_actions"]
    print("predict:", o.shape, f"{np.abs(o).max():.2f}")
    del m, opt, out, loss; torch.cuda.empty_cache()
