"""Robot-side inference wrapper for the H1-2 car tasks (starVLA QwenGR00T).

Wraps normalization, prediction, un-normalization and rel->abs reconstruction
into one call, using the TRAIN statistics bundled under stats/.

    from deploy_infer import CarPolicy
    policy = CarPolicy(task="left-turn-signal",
                       ckpt="checkpoints/steps_20000_pytorch_model.pt")
    chunk = policy.predict(front, left_wrist, right_wrist, state63)  # (16, 42)

Inputs:
    front / left_wrist / right_wrist: RGB uint8 arrays (any size; resized to
        224x224 internally), camera order MUST be front, left, right.
    state63: raw robot state, same 63-dim layout as observation.state in the
        dataset (head 2, legs 12, waist 1, L arm 7, R arm 7, pad 8, L hand 13,
        R hand 13).
Output:
    (16, 42) absolute action chunk @ 20 Hz:
        head(2, rad) + L arm(7, rad) + L hand(13, raw Inspire units)
        + R arm(7, rad) + R hand(13, raw units)
    Arms/head are clipped to the train action range +-0.15 rad margin; hands
    to the train raw range. Send with the sol_tauff gravity feedforward
    (sim/gravity_comp.py) as in the previous deployments.

Requires the starVLA repo on PYTHONPATH (bundled: starvla_src.tar.gz).
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BUNDLE = Path(__file__).resolve().parent

TASK_PROMPTS = {  # task -> default instruction (train tasks.jsonl, task_index 0)
    "left-turn-signal": "Activate the left turn signal and inspect its operation.",
    "right-turn-signal": "Activate the right turn signal and inspect its operation.",
    # interiorlight has TWO instructions — pass prompt= to pick "off":
    #   "Turn off both front interior lights and inspect its operation"
    "car-interiorlight-onoff": "Turn on both front interior lights and inspect its operation",
}

# action(42) slices and their normalization / reference-state slices
ACT = {
    "head": (slice(0, 2), "rel", slice(0, 2)),
    "larm": (slice(2, 9), "rel", slice(15, 22)),
    "lhand": (slice(9, 22), "abs", None),
    "rarm": (slice(22, 29), "rel", slice(22, 29)),
    "rhand": (slice(29, 42), "abs", None),
}
ARM_CLIP_MARGIN = 0.15  # rad beyond the train action range


class CarPolicy:
    def __init__(self, task, ckpt, config_yaml=None, device="cuda", prompt=None):
        from omegaconf import OmegaConf

        from starVLA.model.framework.base_framework import build_framework

        assert task in TASK_PROMPTS, f"unknown task {task}"
        self.task = task
        self.prompt = prompt or TASK_PROMPTS[task]
        self.device = device

        cfg = OmegaConf.load(config_yaml or BUNDLE / "config/starvla_qwengr00t_car3task.yaml")
        self.model = build_framework(cfg)
        sd = torch.load(BUNDLE / ckpt if not Path(ckpt).is_absolute() else ckpt, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not missing, f"missing keys: {missing[:5]}"
        self.model = self.model.to(device).eval()

        stats = json.load(open(BUNDLE / f"stats/{task}.stats_gr00t.rel.json"))["statistics"]
        a = stats["action"]
        self.a_q01, self.a_q99 = np.array(a["q01"]), np.array(a["q99"])
        self.a_min, self.a_max = np.array(a["min"]), np.array(a["max"])
        s = stats["observation.state"]
        self.s_min, self.s_max = np.array(s["min"]), np.array(s["max"])
        self.s_span = self.s_max - self.s_min

    def _norm_state(self, state63):
        x = np.asarray(state63, dtype=np.float64)
        out = np.zeros(63)
        ok = self.s_span > 1e-9
        out[ok] = 2.0 * (x[ok] - self.s_min[ok]) / self.s_span[ok] - 1.0
        return out  # constant dims (pads) -> 0, matching training

    def _unnorm_action(self, pred_n, state63):
        out = np.empty_like(pred_n, dtype=np.float64)
        for _, (sl, mode, ref) in ACT.items():
            x = pred_n[:, sl]
            if mode == "rel":
                phys = (x + 1.0) / 2.0 * (self.a_q99[sl] - self.a_q01[sl]) + self.a_q01[sl]
                # clip in rel space to the train delta range +- margin, then anchor
                phys = np.clip(phys, self.a_min[sl] - ARM_CLIP_MARGIN, self.a_max[sl] + ARM_CLIP_MARGIN)
                out[:, sl] = phys + np.asarray(state63)[ref]
            else:
                phys = (x + 1.0) / 2.0 * (self.a_max[sl] - self.a_min[sl]) + self.a_min[sl]
                out[:, sl] = np.clip(phys, self.a_min[sl], self.a_max[sl])
        return out

    @torch.no_grad()
    def predict(self, front, left_wrist, right_wrist, state63):
        imgs = [Image.fromarray(np.asarray(im)).convert("RGB").resize((224, 224)) for im in (front, left_wrist, right_wrist)]
        example = {
            "image": imgs,
            "lang": self.prompt,
            "state": self._norm_state(state63)[None, :].astype(np.float16),
        }
        pred_n = self.model.predict_action([example])["normalized_actions"][0]  # (16, 42)
        return self._unnorm_action(np.asarray(pred_n, dtype=np.float64), state63)


if __name__ == "__main__":
    # offline smoke test against a bundled sample episode
    import pandas as pd
    import sys

    task = sys.argv[1] if len(sys.argv) > 1 else "left-turn-signal"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/steps_20000_pytorch_model.pt"
    pq = sorted((BUNDLE / "sample" / task).glob("data/chunk-*/*.parquet"))[0]
    df = pd.read_parquet(pq)
    state = np.stack(df["observation.state"])[0]
    gt = np.stack(df["action"])[:16]

    import decord

    frames = {}
    for view in ("front_view", "left_wrist_view", "right_wrist_view"):
        vp = sorted((BUNDLE / "sample" / task).glob(f"videos/chunk-*/observation.images.{view}/*.mp4"))[0]
        frames[view] = decord.VideoReader(str(vp))[0].asnumpy()

    policy = CarPolicy(task, ckpt)
    chunk = policy.predict(frames["front_view"], frames["left_wrist_view"], frames["right_wrist_view"], state)
    arm = np.r_[0:9, 22:29]
    print("chunk:", chunk.shape, "| arm/head MAE vs GT first chunk:", float(np.abs(chunk[: len(gt), arm] - gt[:, arm]).mean()))

    import time

    for _ in range(2):  # warmup
        policy.predict(frames["front_view"], frames["left_wrist_view"], frames["right_wrist_view"], state)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        policy.predict(frames["front_view"], frames["left_wrist_view"], frames["right_wrist_view"], state)
        times.append(time.perf_counter() - t0)
    print(f"predict latency: mean {np.mean(times)*1000:.0f} ms, max {np.max(times)*1000:.0f} ms "
          f"(chunk budget @20Hz: 16 steps = 800 ms)")
