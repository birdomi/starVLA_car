"""Robot-side policy for the H1-2 car models (QwenGR00T: Qwen3.5 VLM + GR00T flow-matching DiT-B, optionally with the
streamed episode memory of QwenGR00T_EpisodeMemory), run exactly as trained and evaluated (eval_val_stream.py /
eval_val_car_v3.py): VLM in bfloat16, memory + DiT in fp32, the training-time normalization
(deployment/model_server/policy_norm_processor.py on the bundled dataset_statistics.json).

    from deploy_infer import CarPolicy
    policy = CarPolicy()                                          # everything from this bundle
    policy.reset("Activate the left turn signal and inspect its operation.")
    out = policy.predict({"head_left_2": rgb, "wrist_left_2": rgb, "wrist_right_2": rgb}, state63, t=0.0)
    out["actions"]    # (50, 42) absolute targets at 30 Hz, the chunk starting at this observation

Inputs
  images   the 3 cameras, RGB uint8 (H, W, 3), as a dict by name or a sequence in CAMERAS order (head, left wrist, right
           wrist). Any size: each is resized to 224 x 224 with PIL's default filter, aspect ratio NOT kept, as in training
           (recorded at 1280 x 720 head, 640 x 480 wrists).
  state63  the robot state in the recorded 63-D observation.state layout; the model's 42-D state is read from it with
           stats/modality.json (head 0:2, left arm 15:22, right arm 22:29, left hand 37:50, right hand 50:63). Or pass
           state42= (that 42-D vector, in this order) instead.
  t        seconds since the episode start (the robot's clock; drives the memory schedule).
Output (dict)
  actions         (50, 42) float64 absolute: head(2) L arm(7) L hand(13) R arm(7) R hand(13), the dataset's action
                  units (head / arms rad, hands raw Inspire units), 30 Hz (50 steps = 1.67 s)
  progress        expected episode progress in [0, 1] read from the memory (None for a model without memory)
  memory_updated  whether this call advanced the memory (False for a model without memory)
  latency_ms      wall time of the call

Memory schedule (memory models only; a model without memory is stateless and ignores t): the memory advances once per `update_interval_s` (default 1.0 s) from the episode start, on the frame
of that call. Training drew every interval from 30 +- 9 frames (0.7 - 1.3 s at 30 Hz), so anything in that range is
in distribution; much faster or slower updates are not. Calls in between (a control loop faster than 1 Hz) are side
queries: the chunk uses the last committed memory updated with the current frame, and that update is dropped
(eval_val_stream.py --exec). Call reset() at every episode start: the memory carries the whole episode.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BUNDLE = Path(__file__).resolve().parent
_SRC = BUNDLE / "starVLA_src"
if (_SRC / "starVLA").is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

CAMERAS = ("head_left_2", "wrist_left_2", "wrist_right_2")  # training order: head, left wrist, right wrist
IMAGE_SIZE = (224, 224)


class CarPolicy:
    def __init__(self, ckpt=None, base_vlm=None, device="cuda", dtype="bfloat16", update_interval_s=1.0, seed=0):
        from omegaconf import OmegaConf

        from deployment.model_server.policy_norm_processor import PolicyNormProcessor
        from starVLA.model.framework.base_framework import baseframework

        ckpt = Path(ckpt) if ckpt else sorted((BUNDLE / "model/checkpoints").glob("*.pt"))[-1]
        cfg = OmegaConf.load(ckpt.parents[1] / "config.yaml")
        base_vlm = Path(base_vlm) if base_vlm else BUNDLE / "base_vlm" / Path(str(cfg.framework.qwenvl.base_vlm)).name
        overrides = [f"framework.qwenvl.base_vlm={base_vlm}", f"framework.qwenvl.autocast_dtype={dtype}"]
        # The checkpoint holds every memory / decision weight: drop the training-time init_from paths (pretraining files
        # that are not in the bundle).
        mem = cfg.framework.get("episode_memory", None)
        if mem is not None and mem.get("init_from", None):
            overrides.append("framework.episode_memory.init_from=null")
        if mem is not None and mem.get("decision", None) is not None and mem.decision.get("init_from", None):
            overrides.append("framework.episode_memory.decision.init_from=null")
        model = baseframework.from_pretrained(str(ckpt), config_overrides=overrides)
        self.has_memory = bool(getattr(model, "mem_tbptt", 0))  # streamed QwenGR00T_EpisodeMemory
        if getattr(model, "episode_memory", None) is not None and not self.has_memory:
            raise ValueError(f"{ckpt}: a window-mode episode memory (not streamed) is not supported here")
        model.qwen_vl_interface.to(getattr(torch, dtype))
        self.model = model.to(device).eval()

        # The car DataConfig has keys of unequal width (head 2, arms 7, hands 13), which PolicyNormProcessor cannot infer
        # from the combined statistics: hand it the per-key dims from the dataset layout.
        from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP

        modality = json.load(open(BUNDLE / "stats/modality.json"))
        # A mixture of the plain car data config and its _ch2clip twin (car-v3) has one set of merged statistics, but
        # PolicyNormProcessor wants one robot type: use the plain one. The twins share every state / action transform;
        # the ch2 clip acts on training targets only (skipped on unapply).
        mix_name = str(cfg.datasets.vla_data.data_mix)
        robots = sorted({r for _, _, r in DATASET_NAMED_MIXTURES[mix_name]})
        if len({r.removesuffix("_ch2clip") for r in robots}) != 1:
            raise ValueError(f"data_mix {mix_name}: robot types {robots} do not share one transform set")
        if len(robots) > 1:
            DATASET_NAMED_MIXTURES[mix_name] = [(DATASET_NAMED_MIXTURES[mix_name][0][0], 1.0, robots[0])]
        dc = ROBOT_TYPE_CONFIG_MAP[robots[0]]
        width = lambda mod, key: modality[mod][key.split(".", 1)[1]]["end"] - modality[mod][key.split(".", 1)[1]]["start"]
        dc.state_key_dims = {k: width("state", k) for k in dc.state_keys}
        dc.action_key_dims = {k: width("action", k) for k in dc.action_keys}
        self.proc = PolicyNormProcessor(str(ckpt))
        self.horizon = int(cfg.framework.action_model.action_horizon)
        self.update_interval_s = float(update_interval_s)
        self.seed = seed

        st = modality["state"]
        self.state_index = np.concatenate([np.arange(st[k.split(".", 1)[1]]["start"], st[k.split(".", 1)[1]]["end"])
                                           for k in self.proc.state_keys])
        self.tasks = json.load(open(BUNDLE / "stats/tasks.json"))
        self.reset(None)

    def reset(self, instruction):
        """New episode: the instruction (one of self.tasks, as trained) and an empty memory."""
        if instruction is not None and instruction not in self.tasks:
            print(f"[CarMemoryPolicy] WARNING instruction not among the {len(self.tasks)} trained tasks: {instruction!r}")
        self.instruction = instruction
        self.memory = None   # committed memory [1, N, d] (None: the next call is the episode's first update)
        self.t_mem = None    # t of the last memory update
        self.calls = 0

    def state42(self, state63):
        return np.asarray(state63, np.float64)[self.state_index]

    def _example(self, images, state42):
        if isinstance(images, dict):
            images = [images[c] for c in CAMERAS]
        assert len(images) == len(CAMERAS), f"need {len(CAMERAS)} images ({CAMERAS}), got {len(images)}"
        imgs = [Image.fromarray(np.asarray(im, np.uint8)).resize(IMAGE_SIZE) for im in images]
        state = self.proc.apply_state(np.asarray(state42, np.float32))[None].astype(np.float16)  # (1, 42) as the loader
        return {"image": imgs, "lang": self.instruction, "state": state}

    @torch.no_grad()
    def predict(self, images, state63=None, t=0.0, state42=None):
        assert self.instruction is not None, "call reset(instruction) at the episode start"
        t0 = time.perf_counter()
        s42 = np.asarray(state42, np.float64) if state42 is not None else self.state42(state63)
        if self.seed is not None:
            torch.manual_seed(self.seed * 1000003 + self.calls)
        example = self._example(images, s42)
        update = False
        if self.has_memory:
            update = self.memory is None or t - self.t_mem >= self.update_interval_s - 1e-6
            out = self.model.predict_action([example], memory=self.memory)
            if update:
                self.memory, self.t_mem = out["memory"], float(t)
        else:
            out = self.model.predict_action([example])
        self.calls += 1
        actions = self.proc.unapply_actions(np.asarray(out["normalized_actions"][0], np.float32)).astype(np.float64)
        progress = float(out["progress"][0]) if "progress" in out else None
        return {"actions": actions, "progress": progress, "memory_updated": bool(update),
                "latency_ms": (time.perf_counter() - t0) * 1e3}


CarMemoryPolicy = CarPolicy  # the name of the first bundles
