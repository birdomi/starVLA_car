# CPU checks of QwenGR00T_GeoResidual without running System 2 (Qwen3.5 needs CUDA): loader deltas / state layout, FK + IK
# on real configurations, MoGe-2 neck parity vs the official forward, forward/backward (System 2 plan replaced by a noisy
# ground truth), predict_action composition (zero head = identity), STARVLA_GEORES_OFF.
#   cd repo && PYTHONPATH=$PWD python examples/.../geo_residual/check_geo_residual_cpu.py <yaml>
import os, sys, collections, numpy as np, torch
from pathlib import Path
from omegaconf import OmegaConf
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.modules.geo_residual.gr1_kinematics import ARM, GR1Kinematics, state_sincos_to_raw
from starVLA.model.modules.geo_residual.se3 import axis_angle_to_matrix, rotation_residual

cfg = OmegaConf.load(sys.argv[1])
torch.manual_seed(0); np.random.seed(0)
dc = cfg.datasets.vla_data.copy(); dc.data_root_dir = Path(dc.data_root_dir)
ds = get_vla_dataset(data_cfg=dc)
ex = [ds[int(i)] for i in np.random.default_rng(1).integers(0, len(ds), 6)]
e = ex[0]
print("example keys", sorted(e.keys()), "| images", len(e["image"]), e["image"][0].size, "| state", np.asarray(e["state"]).shape,
      "| action", np.asarray(e["action"]).shape)

import json
st = json.load(open(cfg.framework.geo_residual.action_stats))["gr1"]["action"]
amin, amax = np.array(st["min"]), np.array(st["max"]); rng = amax - amin
unnorm = lambda x: np.where(rng > 0, (x + 1) / 2 * rng + amin, x)
s_raw = state_sincos_to_raw(torch.tensor(np.stack([np.asarray(x["state"])[-1] for x in ex]))).numpy()
a0 = unnorm(np.stack([np.asarray(x["action"])[0] for x in ex]))
print("|state(t) - action(t)| arms+waist (rad, ~0.05 lag expected): median", np.round(np.median(np.abs(s_raw[:, list(range(14)) + [26, 27, 28]] - a0[:, list(range(14)) + [26, 27, 28]]), 0), 3).tolist())

kin = GR1Kinematics()
w = kin.joints_to_wrists(torch.tensor(s_raw, dtype=torch.float32))
print("current wrist positions (base frame, m) left", np.round(w["left"][0].numpy(), 3).tolist()[:3], "right", np.round(w["right"][0].numpy(), 3).tolist()[:3])
# IK on real configurations: from a perturbed plan back to the ground-truth wrist poses
A = torch.tensor(unnorm(np.concatenate([np.asarray(x["action"])[:4] for x in ex])), dtype=torch.float32)  # [24, 29]
for s in ("left", "right"):
    waist, q_gt = A[:, 26:29], A[:, ARM[s]]
    p_gt, R_gt = kin.fk(s, waist, q_gt)
    q0 = q_gt + torch.randn_like(q_gt) * 0.05
    qi = kin.ik(s, waist, q0, p_gt, R_gt)
    p1, R1 = kin.fk(s, waist, qi); p0, R0 = kin.fk(s, waist, q0)
    print(f"IK {s} real configs (q noise 0.05 rad): pos err {((p0 - p_gt).norm(dim=-1) * 1e3).median():.1f} -> "
          f"{((p1 - p_gt).norm(dim=-1) * 1e3).max():.3f} mm max | rot {rotation_residual(R_gt, R0).norm(dim=-1).median():.3f} -> "
          f"{rotation_residual(R_gt, R1).norm(dim=-1).max():.2e} rad max")

# MoGe-2 neck parity vs the official forward (fp32, CPU)
from starVLA.model.modules.geo_residual.moge2_geometry import MoGe2NeckFeatures
from starVLA.model.framework.VLM4A.QwenGR00T_RadioAdaLN import _images_to_tensor
gc = cfg.framework.geo_residual
neck = MoGe2NeckFeatures(gc.moge_path, gc.moge_num_tokens, gc.moge_crop, gc.neck_levels, dtype="float32")
x = _images_to_tensor([x["image"][-1] for x in ex[:2]], (256, 256), "cpu")
ours = neck(x)
cap = {}
h = neck.model.neck.register_forward_hook(lambda m, i, o: cap.__setitem__("n", o))
with torch.no_grad():
    neck.model.forward(x[..., 42:214, 0:256], num_tokens=gc.moge_num_tokens)
h.remove()
ref = torch.cat([torch.nn.functional.avg_pool2d(cap["n"][l].float(), 2 ** l) for l in gc.neck_levels], 1)
print("MoGe neck feature", tuple(ours.shape), "grid", neck.grid, "out_dim", neck.out_dim, "| max|d| vs official forward",
      (ours - ref).abs().max().item(), "| per-level std", [round(cap["n"][l].float().std().item(), 3) for l in gc.neck_levels])

# framework (System 2 replaced by noisy ground truth)
from starVLA.model.framework.base_framework import build_framework
m = build_framework(cfg)
tr, fz = collections.Counter(), collections.Counter()
for n_, p in m.named_parameters():
    (tr if p.requires_grad else fz)[n_.split(".")[0]] += p.numel()
print("trainable M:", {k: round(v / 1e6, 2) for k, v in tr.items()}, "| frozen M:", {k: round(v / 1e6, 1) for k, v in fz.items()})
m.train(); print("eval-mode kept:", not m.qwen_vl_interface.training, not m.action_model.training, not m.geo_backbone.training,
                 "| residual modules training:", m.residual_decoder.training)
gt_all = torch.tensor(np.stack([np.asarray(x["action"]) for x in ex]), dtype=torch.float32)
def fake_plan(images, langs, states):
    return (gt_all[: len(images)] + torch.randn_like(gt_all[: len(images)]) * 0.03).clamp(-1, 1)
m._system2_plan = fake_plan
out = m(ex[:4])
out["action_loss"].backward()
g = collections.defaultdict(float)
for n_, p in m.named_parameters():
    if p.grad is not None:
        g[n_.split(".")[0]] += p.grad.pow(2).sum().item()
print("loss", round(out["action_loss"].item(), 4), {k: round(v, 3) for k, v in out.items() if k.startswith("geo/")})
print("grad norm per module", {k: f"{v ** 0.5:.3g}" for k, v in g.items()})
# a few steps of the residual parts only: the loss must go down on this fixed batch (plan noise fixed)
torch.manual_seed(1); fixed = fake_plan(ex[:4], None, None); m._system2_plan = lambda *a: fixed
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
np.random.seed(3); ls = []
for it in range(6):
    np.random.seed(3); o = m(ex[:4]); o["action_loss"].backward(); opt.step(); opt.zero_grad(); ls.append(round(o["action_loss"].item(), 4))
print("overfit 6 steps:", ls)

m.eval()
for p_ in m.residual_head.parameters():
    torch.nn.init.zeros_(p_)
plan = gt_all[:2].numpy()
exs = [{"image": [x["image"][-1]], "lang": x["lang"], "state": np.asarray(x["state"])[-1:], "plan_kf": plan[i], "plan_age_s": 0.2}
       for i, x in enumerate(ex[:2])]
o = m.predict_action(exs)
print("predict (zero head, age 0.2 s = offset 4):", o["normalized_actions"].shape, "plan_kf returned:", "plan_kf" in o,
      "| max|out[:4] - plan[4:8]| =", np.abs(o["normalized_actions"][:, :4] - plan[:, 4:8]).max(),
      "| tail ok:", np.allclose(o["normalized_actions"][:, 4:12], plan[:, 8:16]))
torch.nn.init.normal_(m.residual_head.proj.weight, std=0.05)
o2 = m.predict_action(exs)
os.environ["STARVLA_GEORES_OFF"] = "1"; o3 = m.predict_action(exs); os.environ.pop("STARVLA_GEORES_OFF")
print("random head: max|corrected - plan| arms", np.abs(o2["normalized_actions"][:, :4, :14] - plan[:, 4:8, :14]).max().round(4),
      "hands", np.abs(o2["normalized_actions"][:, :4, 14:26] - plan[:, 4:8, 14:26]).max().round(4),
      "waist", np.abs(o2["normalized_actions"][:, :4, 26:] - plan[:, 4:8, 26:]).max(),
      "| GEORES_OFF = plan:", np.allclose(o3["normalized_actions"][:, :4], plan[:, 4:8]))
