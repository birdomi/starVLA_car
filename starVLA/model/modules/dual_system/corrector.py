"""Inference-time corrector for a frozen QwenGR00T_DualSystem (System 2 + System 1), trained offline on cached rc100
outputs (scratchpad study train_corrector.py; same layer names, so its state dict loads as is).

input  x = Gaussian-smoothed System 1 chunk a_hat[t .. t+H-1] (normalised absolute), ActionMap tokens (H steps x 5 columns)
memory C-RADIO summary (2) + pooled patch tokens (k x k masked average of the 16 x 16 grid; zeros if trained --noimg)
       [+ System 2 keyframe plan tokens (plancond)] [+ executed past a[t-P .. t-1] tokens (past)]
       [+ one state token in the action sequence (state)]
out    corr: x + head(h);  reg: head(h)

Enabled in QwenGR00T_DualSystem.predict_action by env STARVLA_DUAL_CORRECTOR=<run>.pt (+ <run>.json next to it with the
training args). Fast System 1 (train_s1fast.py: input = the AGED System 2 plan on the S1 window, cfg input "plan", own
horizon cfg H, condition C-RADIO only): env STARVLA_DUAL_FASTS1=<run>.pt, QwenGR00T_DualSystem._predict_fast_s1.
The executed past comes from the client (ROBOCASA_NUM_PAST_STEPS=P, normalised by the policy wrapper into
examples[b]["past_action"]); examples[b]["past_action_count"] = number of real (not padded) steps, else all valid.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

COLS = [(0, 7), (7, 14), (14, 20), (20, 26), (26, 29)]


class _Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.n1, self.n2, self.n3, self.nc = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, nh, batch_first=True)
        self.ca = nn.MultiheadAttention(d, nh, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(0.0), nn.Linear(4 * d, d))

    def forward(self, h, c, cmask):
        q = self.n1(h)
        h = h + self.sa(q, q, q, need_weights=False)[0]
        h = h + self.ca(self.n2(h), self.nc(c), self.nc(c), key_padding_mask=cmask, need_weights=False)[0]
        return h + self.mlp(self.n3(h))


class S1Corrector(nn.Module):
    def __init__(self, cfg: dict, H: int, d_state: int, n_kf: int, d_c: int = 1152, d_a: int = 29):
        super().__init__()
        self.cfg, self.H = cfg, H
        d, L = int(cfg["d"]), int(cfg["layers"])
        self.pool = int(cfg.get("pool", 16))
        self.n_patch = 0 if self.pool == 0 else self.pool * self.pool
        self.past_n = int(cfg.get("past_n", 40))
        self.inp = nn.ModuleList([nn.Linear(b - a, d) for a, b in COLS])
        self.out = nn.ModuleList([nn.Linear(d, b - a) for a, b in COLS])
        self.t_emb = nn.Parameter(torch.zeros(H, d))
        self.c_emb = nn.Parameter(torch.zeros(len(COLS), d))
        self.cproj = nn.Linear(d_c, d)
        self.pos = nn.Parameter(torch.zeros(2 + self.n_patch, d))
        self.state = nn.Linear(d_state, d) if cfg.get("state") else None
        self.kproj = nn.Linear(d_a, d) if cfg.get("plancond") else None
        self.kpos = nn.Parameter(torch.zeros(n_kf, d)) if cfg.get("plancond") else None
        self.pproj = nn.Linear(d_a, d) if cfg.get("past") else None
        self.ppos = nn.Parameter(torch.zeros(self.past_n, d)) if cfg.get("past") else None
        self.blocks = nn.ModuleList([_Block(d, 8) for _ in range(L)])
        self.norm = nn.LayerNorm(d)
        sigma = float(cfg.get("sigma", 2.0))
        if sigma > 0:
            r = int(math.ceil(3 * sigma))
            k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / sigma) ** 2)
            self.register_buffer("ker", k / k.sum(), persistent=False)
        else:
            self.ker = None

    @classmethod
    def load(cls, pt_path: str, H: int, d_state: int, n_kf: int):
        cfg = json.load(open(Path(pt_path).with_suffix(".json")))["args"]
        net = cls(cfg, int(cfg.get("H", H)), d_state, n_kf)
        net.load_state_dict(torch.load(pt_path, map_location="cpu"), strict=True)
        return net.eval()

    def smooth(self, x):  # [B, H, D] along time, edge-replicate
        if self.ker is None:
            return x
        r = (len(self.ker) - 1) // 2
        y = F.pad(x.transpose(1, 2), (r, r), mode="replicate")
        return F.conv1d(y, self.ker.view(1, 1, -1).expand(x.shape[-1], 1, -1), groups=x.shape[-1]).transpose(1, 2)

    def image_tokens(self, radio):
        """C-RADIO summary + k x k masked-average-pooled patches (as pool_tok.py), [B, 2 + k^2, D_c], mask True = ignore."""
        pre = radio.prefix.float()
        b = pre.shape[0]
        if self.n_patch == 0:
            tok = pre
        else:
            x, v = radio.tokens.float(), radio.valid.float()
            g = int(round(x.shape[1] ** 0.5)); k = self.pool; f = g // k
            xs = (x * v[..., None]).view(b, k, f, k, f, -1).sum((2, 4)); vs = v.view(b, k, f, k, f).sum((2, 4))
            tok = torch.cat([pre, (xs / vs.clamp_min(1)[..., None]).view(b, k * k, -1)], 1)
        if self.cfg.get("noimg"):
            tok = torch.zeros_like(tok)
        return tok, torch.zeros(b, tok.shape[1], dtype=torch.bool, device=tok.device)

    def forward(self, s1, radio, state=None, kf=None, past=None, past_valid=None):
        """s1 [B, >= H, D] System 1 output (cfg input plan: the System 2 plan on the S1 window), normalised absolute ->
        corrected [B, H, D]."""
        src = torch.zeros_like(s1[:, :self.H]) if self.cfg.get("input") == "zero" else s1[:, :self.H].float()
        x = self.smooth(src)
        b = x.shape[0]
        h = torch.stack([f(x[..., a:bb]) for f, (a, bb) in zip(self.inp, COLS)], 2)
        h = (h + self.t_emb[None, :, None] + self.c_emb[None, None]).flatten(1, 2)
        if self.state is not None:
            h = torch.cat([h, self.state(state.reshape(b, -1).float())[:, None]], 1)
        tok, tmask = self.image_tokens(radio)
        c = self.cproj(tok) + self.pos[None]
        if self.kproj is not None:
            c = torch.cat([c, self.kproj(kf.float()) + self.kpos[None]], 1)
            tmask = torch.cat([tmask, torch.zeros(b, kf.shape[1], dtype=torch.bool, device=x.device)], 1)
        if self.pproj is not None:
            pa, pv = past[:, -self.past_n:].float(), past_valid[:, -self.past_n:]
            c = torch.cat([c, self.pproj(pa) + self.ppos[None]], 1)
            tmask = torch.cat([tmask, ~pv], 1)
        for blk in self.blocks:
            h = blk(h, c, tmask)
        h = self.norm(h[:, :self.H * len(COLS)]).view(b, self.H, len(COLS), -1)
        y = torch.cat([f(h[:, :, i]) for i, f in enumerate(self.out)], -1)
        return x + y if self.cfg.get("mode", "corr") == "corr" else y


def past_from_examples(examples, P: int, d_a: int, device):
    """examples[b]["past_action"] [P', D] normalised (policy wrapper), chronological; past_action_count = real steps."""
    pa = torch.zeros(len(examples), P, d_a, device=device)
    pv = torch.zeros(len(examples), P, dtype=torch.bool, device=device)
    for b, ex in enumerate(examples):
        if "past_action" not in ex:
            continue
        a = torch.as_tensor(np.asarray(ex["past_action"], dtype=np.float32), device=device)[-P:]
        pa[b, P - len(a):] = a
        n = int(ex.get("past_action_count", len(a)))
        if n > 0:
            pv[b, P - min(n, len(a)):] = True
    return pa, pv
