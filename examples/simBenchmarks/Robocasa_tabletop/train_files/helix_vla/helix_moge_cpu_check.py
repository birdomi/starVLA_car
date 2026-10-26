# CPU check of Helix_VLA with radio.backbone moge (build, frozen split, S1 eye -> RadioCondition shapes, bad backbone). Run from repo root:
#   PYTHONPATH=$PWD python helix_vla/helix_moge_cpu_check.py <yaml>   (10-22 OK: 86.9M frozen eye, 257 cond tokens x 1024)
import sys, collections, torch
from omegaconf import OmegaConf
from starVLA.model.framework.base_framework import build_framework
cfg = OmegaConf.load(sys.argv[1])
m = build_framework(cfg)
print(type(m).__name__, "| eye", type(m.radio_encoder).__name__, m.radio_encoder.trainable_desc, "| radio_size", m.radio_size)
tr, fz = collections.Counter(), collections.Counter()
for n, p in m.named_parameters():
    (tr if p.requires_grad else fz)[n.split(".")[0]] += p.numel()
print("trainable M:", {k: round(v / 1e6, 1) for k, v in tr.items()}, "| frozen M:", {k: round(v / 1e6, 1) for k, v in fz.items()})
m.train(); print("radio_encoder.training", m.radio_encoder.training, "| radio_cond n_prefix", m.radio_cond.n_prefix)
x = torch.rand(3, 3, 256, 256)
with torch.no_grad():
    f = m.radio_encoder(x); r_tok, r_mask = m.radio_cond(f, 3)
print("eye tokens", tuple(f.tokens.shape), "prefix", tuple(f.prefix.shape), "-> cond tokens", tuple(r_tok.shape), "mask", tuple(r_mask.shape), bool(r_mask.all()))
sd = m.state_dict(); ks = [k for k in sd if k.startswith("radio_encoder.")]
print("state_dict radio_encoder keys", len(ks), ks[:3], "...", "| persistent mean/std?", any(k.endswith((".mean", ".std")) for k in ks))
try:
    bad = OmegaConf.load(sys.argv[1]); bad.framework.radio.backbone = "nope"; build_framework(bad); print("BAD BACKBONE ACCEPTED")
except ValueError as e:
    print("bad backbone -> ValueError OK:", str(e)[:80])
