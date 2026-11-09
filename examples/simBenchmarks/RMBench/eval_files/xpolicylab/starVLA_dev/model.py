"""XPolicyLab adapter: RMBench (aloha-agilex) -> a starVLA policy server of THIS repo.

Unlike XPolicyLab's bundled `starVLA` adapter (RoboDojo / ARX-X5, vendored upstream runtime with a q99 state contract),
this one speaks the protocol of `deployment/model_server/server_policy.py` here and replays the RoboTwin client
(`examples/simBenchmarks/Robotwin/eval_files/model2robotwin_interface.py`) conventions, since RMBench data converted by
`../../../train_files/convert_rmbench_to_lerobot.py` uses the RoboTwin 14-D layout:

  * cameras [cam_head, cam_left_wrist, cam_right_wrist] resized to image_size (the training order: cam_high, left, right)
  * XPolicyLab packs joints as [left 6, left grip, right 6, right grip] (= RoboTwin env order); the model (AgilexDataConfig
    keys) uses [left 6, right 6, left grip, right grip] -> reordered both ways
  * state is sent RAW as `state_raw` (model order) and normalized by the server with the training transform
  * the server returns unnormalized absolute joint actions; each get_action() is one inference and returns the first
    execute_horizon steps of the chunk (the RMBench loop executes them and calls update_obs between steps)
  * stream episode-memory checkpoints (QwenGR00T_EpisodeMemory, episode_memory.stream; detected from the run config):
    every request carries mem_slot / mem_t (episode seconds = executed steps / control_hz) / mem_reset (first call of an
    episode); the server (deployment/model_server/policy_wrapper.py) keeps the memory per slot and advances it at most
    once per STARVLA_MEM_INTERVAL_S (default 1.0 s). execute_horizon defaults to the training stream stride (e.g. 16
    frames at 15 Hz = 1.07 s), so the calls land on the trained update grid (frames 0, 16, 32, ...) and each advances
    the memory

Process layout: RMBench sim (env client) -> XPolicyLab policy server running this Model (CPU only, RMBench env)
-> starVLA websocket server on a GPU (starVLA env). Every key below can be overridden by the env var in brackets.
"""
from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state, unpack_robot_state

ENV_TO_MODEL = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
MODEL_TO_ENV = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
CAMERAS = ["cam_head", "cam_left_wrist", "cam_right_wrist"]


def _cfg(cfg: dict, key: str, env: str, default=None):
    value = os.environ.get(env)
    if value is None:
        value = cfg.get(key, default)
    return None if value in ("", "null", "None") else value


def _run_configs(ckpt_path: str | None) -> list[dict]:
    """The run dir's configs (searched upward from the checkpoint), config.full.yaml (resolved defaults) first."""
    if not ckpt_path:
        return []
    for parent in [Path(ckpt_path)] + list(Path(ckpt_path).parents)[:3]:
        found = [parent / n for n in ("config.full.yaml", "config.yaml") if (parent / n).is_file()]
        if found:
            out = []
            for cfg_file in found:
                with open(cfg_file) as f:
                    out.append(yaml.safe_load(f) or {})
            return out
    return []


def _checkpoint_include_state(configs: list[dict]) -> bool:
    for cfg in configs:
        vla_data = (cfg.get("datasets") or {}).get("vla_data") or {}
        if "include_state" in vla_data:
            return bool(vla_data["include_state"])
    return False


def _checkpoint_stream_memory(configs: list[dict]) -> bool:
    """QwenGR00T_EpisodeMemory with episode_memory.stream: the memory is carried across calls of one episode."""
    for cfg in configs:
        fw = cfg.get("framework") or {}
        if fw.get("name"):
            return "EpisodeMemory" in str(fw["name"]) and bool((fw.get("episode_memory") or {}).get("stream"))
    return False


def _checkpoint_stream_stride(configs: list[dict]) -> int | None:
    """datasets.vla_data.stream.stride: frames between two memory updates in training."""
    for cfg in configs:
        stream = ((cfg.get("datasets") or {}).get("vla_data") or {}).get("stream") or {}
        if "stride" in stream:
            return int(stream["stride"])
    return None


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        cfg = dict(model_cfg)
        self.action_type = cfg.get("action_type") or "joint"
        if self.action_type != "joint":
            raise ValueError("starVLA_dev supports action_type='joint' only.")
        self.env_cfg_type = cfg.get("env_cfg_type") or "aloha_agilex"
        self.dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_dim = sum(self.dim_info["arm_dim"]) + sum(self.dim_info["ee_dim"])
        if self.action_dim != 14:
            raise ValueError(f"starVLA_dev expects the 14-D aloha-agilex layout, got {self.env_cfg_type}: {self.dim_info}")

        starvla_root = Path(_cfg(cfg, "starvla_root", "STARVLA_ROOT", "/mnt/DATA2/yyg/starVLA")).expanduser()
        if str(starvla_root) not in sys.path:
            sys.path.append(str(starvla_root))  # appended: never shadow RMBench's own envs/ scripts/ packages
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        host = _cfg(cfg, "starvla_server_host", "STARVLA_SERVER_HOST", "127.0.0.1")
        port = int(_cfg(cfg, "starvla_server_port", "STARVLA_SERVER_PORT", 5694))
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.chunk_size = int(meta["action_chunk_size"])
        self.unnorm_key = _cfg(cfg, "unnorm_key", "STARVLA_UNNORM_KEY")
        self.use_ddim = str(_cfg(cfg, "use_ddim", "STARVLA_USE_DDIM", True)).lower() in ("1", "true", "yes")
        self.num_ddim_steps = int(_cfg(cfg, "num_ddim_steps", "STARVLA_NUM_DDIM_STEPS", 10))
        size = _cfg(cfg, "image_size", "STARVLA_IMAGE_SIZE", [224, 224])
        if isinstance(size, str):
            size = [int(v) for v in size.replace("x", ",").split(",")]
        self.image_size = (int(size[0]), int(size[1]))
        run_cfgs = _run_configs(meta.get("ckpt_path"))
        send_state = str(_cfg(cfg, "send_state", "STARVLA_SEND_STATE", "auto")).lower()
        self.send_state = _checkpoint_include_state(run_cfgs) if send_state == "auto" else \
            send_state in ("1", "true", "yes")
        # stream episode memory (QwenGR00T_EpisodeMemory): the server keeps one memory per mem_slot, reset at the
        # episode start and advanced at most once per STARVLA_MEM_INTERVAL_S (server side) of episode time mem_t
        memory = str(_cfg(cfg, "memory", "STARVLA_MEMORY", "auto")).lower()
        self.memory = _checkpoint_stream_memory(run_cfgs) if memory == "auto" else memory in ("1", "true", "yes")
        self.control_hz = float(_cfg(cfg, "control_hz", "STARVLA_CONTROL_HZ", 15))
        self.mem_slot = f"rmbench-{os.getpid()}-{id(self)}"
        self._steps = 0  # actions handed out since reset() = executed steps (the loop runs every returned action)
        # execute_horizon: null = the full chunk, or for a stream memory the training stream stride, so the calls (one
        # memory update each) fall on the trained update grid: frames 0, stride, 2 stride, ...
        horizon = _cfg(cfg, "execute_horizon", "STARVLA_EXECUTE_HORIZON")
        stride = _checkpoint_stream_stride(run_cfgs) if self.memory else None
        if horizon is not None:
            self.execute_horizon = int(horizon)
            if stride is not None and self.execute_horizon != stride:
                print(f"[starVLA_dev][WARN] execute_horizon={self.execute_horizon} != training stream stride {stride}: "
                      f"policy calls leave the trained memory update grid", flush=True)
        else:
            self.execute_horizon = self.chunk_size if stride is None else stride
        if not 0 < self.execute_horizon <= self.chunk_size:
            raise ValueError(f"execute_horizon={self.execute_horizon} must be in [1, chunk {self.chunk_size}]")
        # history inputs, as ROBOTWIN_STATE_HIST / ROBOTWIN_NUM_PAST_STEPS in the RoboTwin client
        self.state_hist = int(_cfg(cfg, "state_hist", "STARVLA_STATE_HIST", 0))
        self.num_past_steps = int(_cfg(cfg, "num_past_steps", "STARVLA_NUM_PAST_STEPS", 0))
        self._states = deque(maxlen=max(self.state_hist, 1))
        self._past = deque(maxlen=max(self.num_past_steps, 1))
        self._obs = None
        print(f"[starVLA_dev] server {host}:{port} chunk={self.chunk_size} execute_horizon={self.execute_horizon} "
              f"send_state={self.send_state} memory={self.memory} state_hist={self.state_hist} "
              f"num_past_steps={self.num_past_steps} "
              f"unnorm_key={self.unnorm_key} image_size={self.image_size} ckpt={meta.get('ckpt_path')}", flush=True)

    # ------------------------------------------------------------------ obs
    def _image(self, obs: dict, cam: str) -> np.ndarray:
        image = np.asarray(obs["vision"][cam]["color"])  # RGB uint8 HWC, decoded by the XPolicyLab server
        return cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)

    def update_obs(self, obs):
        instruction = obs.get("instruction") or (obs.get("instructions") or [""])[0]
        qpos = pack_robot_state(obs, self.action_type, self.dim_info, source_type="obs").astype(np.float32)
        qpos = qpos.reshape(-1)[ENV_TO_MODEL]
        self._states.append(qpos)
        self._obs = {"lang": str(instruction), "image": [self._image(obs, cam) for cam in CAMERAS], "qpos": qpos}

    def update_obs_batch(self, obs_list):
        raise NotImplementedError("starVLA_dev runs one env per policy server (eval_batch: false)")

    # --------------------------------------------------------------- action
    def get_action(self):
        if self._obs is None:
            raise AssertionError("update_obs must be called before get_action")
        qpos = self._obs["qpos"]
        example = {"lang": self._obs["lang"], "image": self._obs["image"]}
        if self.state_hist > 0:
            hist = list(self._states)
            example["state_raw"] = np.stack([hist[0]] * (self.state_hist - len(hist)) + hist).astype(np.float32)
        elif self.send_state:
            example["state_raw"] = qpos
        if self.num_past_steps > 0:
            past = list(self._past)
            example["past_action_abs"] = np.stack([qpos] * (self.num_past_steps - len(past)) + past).astype(np.float32)
            example["current_qpos"] = qpos
            example["past_mode"] = "abs"
        if self.memory:
            example.update(mem_slot=self.mem_slot, mem_t=self._steps / self.control_hz, mem_reset=self._steps == 0)
        request = {"examples": [example], "do_sample": False, "use_ddim": self.use_ddim,
                   "num_ddim_steps": self.num_ddim_steps, "unnorm_key": self.unnorm_key}
        response = self.client.predict_action(request)
        chunk = np.asarray(response["data"]["actions"][0], dtype=np.float32)[: self.execute_horizon]  # model order
        if chunk.shape[-1] != self.action_dim:
            raise ValueError(f"server returned action dim {chunk.shape[-1]}, expected {self.action_dim}")
        self._past.extend(chunk)  # assumes the whole returned chunk is executed (true until the episode ends)
        self._steps += len(chunk)
        return [unpack_robot_state(a[MODEL_TO_ENV], self.action_type, self.dim_info, source_type="obs") for a in chunk]

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("starVLA_dev runs one env per policy server (eval_batch: false)")

    def reset(self):
        self._obs = None
        self._steps = 0
        self._states.clear()
        self._past.clear()
