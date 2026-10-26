from collections import deque
from typing import Dict, Optional, Sequence

import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.simBenchmarks.Robocasa_tabletop.eval_files.adaptive_ensemble import AdaptiveEnsembler


class PolicyWarper:
    # 29-D layout in loader action-key order
    ACTION_KEY_DIMS = (("left_arm", 7), ("right_arm", 7), ("left_hand", 6), ("right_hand", 6), ("waist", 3))

    @classmethod
    def _build_anchor_mask(cls, keys) -> "np.ndarray":
        """(1, 1, 29) bool: True on the dims whose predictions are anchored to the current qpos."""
        total = sum(d for _, d in cls.ACTION_KEY_DIMS)
        mask = np.zeros(total, dtype=bool)
        if not keys:
            mask[:] = True   # loader default: the mode applies to every action key
            return mask[None, None, :]
        known = {k for k, _ in cls.ACTION_KEY_DIMS}
        unknown = [k for k in keys if k.replace("action.", "") not in known]
        assert not unknown, f"ROBOCASA_ACTION_MODE_KEYS has unknown keys {unknown}; expected from {sorted(known)}"
        want = {k.replace("action.", "") for k in keys}
        off = 0
        for name, dim in cls.ACTION_KEY_DIMS:
            if name in want:
                mask[off : off + dim] = True
            off += dim
        return mask[None, None, :]

    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble=False,  # @Jinhui
        action_ensemble_horizon: Optional[int] = 3,  # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="0.0.0.0",
        port=10095,
        n_action_steps=2,
        send_state: bool = True,
    ) -> None:

        # build client to connect server policy
        # --- history-conditioned / delta heads (QwenGR00T_PatchRegressor) ---
        #   action_mode: "abs" (default) or "delta" (server returns per-step displacements anchored at the
        #   current qpos; we integrate them). num_past_steps > 0 sends the last P executed absolute actions.
        self.action_mode = os.environ.get("ROBOCASA_ACTION_MODE", "abs")  # abs | delta | rel
        # ROBOCASA_ACTION_MODE_KEYS: comma-separated loader action keys the rel/delta mode was TRAINED on
        # (must mirror the yaml's datasets.vla_data.action_mode_apply_keys). Unset -> every key, which is the
        # loader's own default. Keys left out are returned as ABSOLUTE targets and are not anchored to qpos.
        # GR1 note: the 6-finger hand ACTION is a [-3, 3] command while the hand STATE is a [0, 1.86] finger
        # angle, so "action - state" is meaningless for the hands -> train/eval them as abs.
        self.action_mode_keys = [k.strip() for k in os.environ.get("ROBOCASA_ACTION_MODE_KEYS", "").split(",") if k.strip()]
        self._anchor_mask = self._build_anchor_mask(self.action_mode_keys)
        self.num_past_steps = int(os.environ.get("ROBOCASA_NUM_PAST_STEPS", "0"))
        # ROBOCASA_S2_EVERY=N > 0 (QwenGR00T_DualSystem fast System 1, server env STARVLA_DUAL_FASTS1): the client keeps
        # the System 2 plan the server returned ("plan_kf") and sends it back with its age (plan_age_s, at
        # ROBOCASA_ACTION_RATE Hz) on every call; the server re-runs System 2 only when the plan is N or more env steps
        # old (or at an episode start), System 1 on every call (n_action_steps env steps apart).
        self.s2_every = int(os.environ.get("ROBOCASA_S2_EVERY", "0"))
        self.action_rate = float(os.environ.get("ROBOCASA_ACTION_RATE", "20"))
        self._plan, self._plan_age = [], []
        self._hist: list[deque] = []
        # ROBOCASA_STATE_HIST=H (>0): send the last H OBSERVED raw states (29-D, loader key order) as ``state_raw``
        # (H, 29); the server normalizes them with the training-time state transform (raw min_max for the
        # state-history image2image PatchRegressor). Replaces the sin/cos ``state`` that other checkpoints expect.
        self.state_hist = int(os.environ.get("ROBOCASA_STATE_HIST", "0"))
        # ROBOCASA_VLM_PREV_FRAME=1 (QwenGR00T_DualSystem, dual_system.stale_input vlm with video_offsets
        # [-n_action_steps, 0]): every camera's [previous call's frame, frame t]; the VLM sees the first one, C-RADIO /
        # state / System 1 frame t. An episode's first call sends frame t twice (the training fallback where t - lag
        # lies before the episode). reset_episode() (simulation_env, every episode start) clears the previous frame.
        self.vlm_prev_frame = os.environ.get("ROBOCASA_VLM_PREV_FRAME", "0") == "1"
        self._prev_images: list = []
        # ROBOCASA_VLM_PREV_FRAMES=K > 1 (dual_system.memory, video_offsets [-K * n_action_steps, ..., -n_action_steps,
        # 0]): every camera's frames of the last K calls + frame t, oldest first, and per example ``frame_valid`` per
        # (camera, frame), camera-major: False = no such call yet in this episode (frame t is sent in its place; the
        # model skips that memory update, the training fallback). Takes precedence over ROBOCASA_VLM_PREV_FRAME.
        self.vlm_prev_frames = int(os.environ.get("ROBOCASA_VLM_PREV_FRAMES", "0") or 0)
        self._prev_hist: list = []
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        # Checkpoints trained without proprioceptive state (e.g. Qwen3-VL-OFT-Robocasa)
        # must not receive a `state` key: QwenOFT appends discretized state tokens to
        # the instruction whenever `state` is present, corrupting the prompt format.
        self.send_state = send_state

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon  # 0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.n_action_steps = n_action_steps

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        server_meta = self.client.get_server_metadata()
        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, server_meta: {server_meta} ***")

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    @staticmethod
    def _select_instruction(instructions, batch_index: int):
        if isinstance(instructions, np.ndarray):
            instructions = instructions.tolist()
        if isinstance(instructions, tuple):
            instructions = list(instructions)
        if not isinstance(instructions, list):
            return instructions
        if len(instructions) == 0:
            raise ValueError("instructions must not be an empty list")
        if batch_index < len(instructions):
            return instructions[batch_index]
        if len(instructions) == 1:
            return instructions[0]
        raise IndexError(
            f"instructions has length {len(instructions)}, but batch index {batch_index} was requested"
        )

    @staticmethod
    def _normalize_task_descriptions(task_descriptions):
        if isinstance(task_descriptions, np.ndarray):
            task_descriptions = task_descriptions.tolist()
        if isinstance(task_descriptions, tuple):
            task_descriptions = list(task_descriptions)
        if not isinstance(task_descriptions, list):
            return task_descriptions

        normalized = []
        for item in task_descriptions:
            if isinstance(item, np.ndarray):
                item = item.tolist()
            if isinstance(item, tuple):
                item = list(item)
            if isinstance(item, list) and len(item) == 1:
                item = item[0]
            normalized.append(item)
        return normalized

    def reset_episode(self, index: Optional[int] = None) -> None:
        """Per-episode client state (ROBOCASA_VLM_PREV_FRAME / _FRAMES): forget the previous frames of env `index`
        (None = all)."""
        if index is None:
            self._prev_images = []
            self._prev_hist = []
            self._plan, self._plan_age = [], []
            return
        if index < len(self._plan):
            self._plan[index] = None
        if index < len(self._prev_images):
            self._prev_images[index] = None
        if index < len(self._prev_hist):
            self._prev_hist[index] = None

    @staticmethod
    def _with_prev_frames(prev: list, images: list) -> list:
        """images [B][V] (frame t per camera) + prev [B] ([V] or None) -> [B][V * 2], camera-major [prev_v, t_v] per camera
        (the loader order of video_offsets [-lag, 0]); a missing previous frame = frame t."""
        out = []
        for b, cur in enumerate(images):
            pv = prev[b] if b < len(prev) and prev[b] is not None else cur
            out.append([im for v in range(len(cur)) for im in (pv[v], cur[v])])
        return out

    @staticmethod
    def _with_prev_window(hist: list, images: list, k: int):
        """images [B][V] (frame t per camera) + hist [B] (up to k earlier calls' [V], oldest first, or None) ->
        (images [B][V * (k + 1)], frame_valid [B][V * (k + 1)]), camera-major [call -k .. call -1, t] per camera (the
        loader order of video_offsets [-k * lag, .., -lag, 0]); a missing call = frame t, frame_valid False."""
        out, valid = [], []
        for b, cur in enumerate(images):
            h = list(hist[b] or [])[-k:] if b < len(hist) else []
            past = [None] * (k - len(h)) + h
            out.append([im for v in range(len(cur)) for im in [cur[v] if p is None else p[v] for p in past] + [cur[v]]])
            valid.append([ok for _ in range(len(cur)) for ok in [p is not None for p in past] + [True]])
        return out, valid

    def reset(self, task_description: str or tuple) -> None:
        self._hist = []  # executed-action history (PatchRegressor)
        self._plan, self._plan_age = [], []  # ROBOCASA_S2_EVERY: client-held System 2 plan per env
        self._prev_images = []
        self._prev_hist = []
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(self, observations, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Execute one inference step.
        :param image: Input image (H, W, 3) in uint8 format
        :param task_description: Task description text
        :return: (raw_actions, processed_actions)
        """

        task_description = self._normalize_task_descriptions(
            observations["annotation.human.coarse_action"]
        )
        ego_view = observations["video.ego_view"]  # (N, 1, H, W, 3)
        images = ego_view

        input_state = None
        if self.send_state and self.state_hist <= 0:
            state = {}
            state["left_arm"] = observations["state.left_arm"]
            state["right_arm"] = observations["state.right_arm"]  # (N, 1, 7)
            state["left_hand"] = observations["state.left_hand"]  # (N, 1, 6)
            state["right_hand"] = observations["state.right_hand"]  # (N, 1, 6)
            state["waist"] = observations["state.waist"]  # (N, 1, 3)

            state = self.normalize_state(state)
            input_state = []
            for key in state.keys():
                input_state.append(state[key])
            input_state = np.concatenate(input_state, axis=-1)

        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        # image: Image.Image = Image.fromarray(image)

        images = [[self._resize_image(img) for img in sample] for sample in images]  # (B, N_view, H, W, 3)
        frame_valid = None
        if self.vlm_prev_frames > 1:
            cur_images = images
            images, frame_valid = self._with_prev_window(self._prev_hist, cur_images, self.vlm_prev_frames)
            self._prev_hist = [(list(self._prev_hist[b] or []) if b < len(self._prev_hist) else []) + [cur_images[b]]
                               for b in range(len(cur_images))]
            self._prev_hist = [h[-self.vlm_prev_frames:] for h in self._prev_hist]
            if not getattr(self, "_vlm_prev_logged", False) and all(all(v) for v in frame_valid):
                print(f"[vlm_prev_frames] {len(images[0])} images per example (last {self.vlm_prev_frames} calls + t per camera)", flush=True)
                self._vlm_prev_logged = True
        elif self.vlm_prev_frame:
            cur_images = images
            fresh = [b >= len(self._prev_images) or self._prev_images[b] is None for b in range(len(cur_images))]
            images = self._with_prev_frames(self._prev_images, images)  # (B, N_view * 2, H, W, 3)
            self._prev_images = list(cur_images)
            if not getattr(self, "_vlm_prev_logged", False) and not any(fresh):
                print(f"[vlm_prev_frame] {len(images[0])} images per example ([previous call, t] per camera)", flush=True)
                self._vlm_prev_logged = True
        if input_state is not None:
            input_state = [input_s for input_s in input_state]  # B, state_dim*(sin, cos)

        # prepare vla input
        examples = []
        batch_size = len(images)
        instructions = self.task_description
        for b in range(batch_size):
            example = {
                "image": images[b],  # A list of multi-view images for a single sample
                "lang": self._select_instruction(instructions, b),
            }
            if input_state is not None:
                example["state"] = input_state[b]  # N_history, 58 #Hack BUG
            if frame_valid is not None:
                example["frame_valid"] = frame_valid[b]
            examples.append(example)

        # raw 29-D qpos (loader action-key order) for delta integration / history conditioning
        raw_qpos = np.concatenate(
            [np.asarray(observations[k])[:, -1] for k in ("state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand", "state.waist")],
            axis=-1,
        )  # (B, 29)
        if self.state_hist > 0:
            # Per-env-step state window from the MultiStepWrapper (simulation_env sets state_delta_indices =
            # range(-H+1, 1) when ROBOCASA_STATE_HIST=H): observations["state.*"] are (N, H, d). The policy is called
            # once per action chunk, so a per-call deque would span H*n_action_steps env steps -> wrong history.
            win = np.concatenate(
                [np.asarray(observations[k], dtype=np.float32) for k in ("state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand", "state.waist")],
                axis=-1,
            )  # (N, T, 29)
            assert win.ndim == 3 and win.shape[1] == self.state_hist, (
                f"ROBOCASA_STATE_HIST={self.state_hist} needs per-step state windows (N, {self.state_hist}, 29) from the env "
                f"wrapper; got {win.shape}. Run through simulation_env.py (it sets state_delta_indices from ROBOCASA_STATE_HIST)."
            )
            for b in range(batch_size):
                examples[b]["state_raw"] = win[b]  # (H, 29) chronological, ends at s[t]
        if self.num_past_steps > 0:
            while len(self._hist) < batch_size:
                self._hist.append(deque(maxlen=self.num_past_steps))
            for b in range(batch_size):
                h = list(self._hist[b])
                pad = [raw_qpos[b]] * (self.num_past_steps - len(h))  # before the first step: "no motion"
                examples[b]["past_action_abs"] = np.stack(pad + h, axis=0).astype(np.float32)  # (P, 29) chronological
                examples[b]["past_action_count"] = len(h)  # real executed steps (the rest is qpos padding; DualSystem corrector masks it)
                examples[b]["current_qpos"] = raw_qpos[b].astype(np.float32)
                examples[b]["past_mode"] = self.action_mode if self.action_mode in ("rel", "abs") else "delta"
        if self.s2_every > 0:
            while len(self._plan) < batch_size:
                self._plan.append(None)
                self._plan_age.append(0)
            for b in range(batch_size):
                if self._plan[b] is not None and self._plan_age[b] < self.s2_every:
                    examples[b]["plan_kf"] = self._plan[b]
                    examples[b]["plan_age_s"] = self._plan_age[b] / self.action_rate
        vla_input = {
            "examples": examples,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        vla_input["unnorm_key"] = self.unnorm_key

        # === TRAIN/TEST CONSISTENCY: keep the observation below aligned with training ===
        # Embodied policies degrade SILENTLY (no error) when the eval-time observation
        # differs from what the model saw during TRAINING. Verify these match the
        # training config used for this checkpoint:
        #   - state       : whether proprioceptive state is included (and its dim/order/normalization))
        #   - image size  : resize / crop resolution (e.g. 224x224)
        #   - image count : how many camera views are fed
        #   - image order : the ordering of those camera views
        #   - action normalization: unnorm_key must match the training dataset stats
        # ==============================================================================
        response = self.client.predict_action(vla_input)
        # server already un-normalized via training-time transform
        raw_actions = np.array(response["data"]["actions"])  # (B, chunk, D)
        if self.s2_every > 0:
            plan_kf = response["data"].get("plan_kf")
            for b in range(batch_size):
                if plan_kf is not None:  # System 2 ran on this frame: plan age 0
                    self._plan[b], self._plan_age[b] = np.asarray(plan_kf[b], dtype=np.float32), 0
                self._plan_age[b] += self.n_action_steps
        if self.action_mode == "delta":
            # server un-normalized per-step displacements d[0] = a[t]-q[t], d[i] = a[t+i]-a[t+i-1]
            integrated = raw_qpos[:, None, :] + np.cumsum(raw_actions, axis=1)
            raw_actions = np.where(self._anchor_mask, integrated, raw_actions)
        elif self.action_mode == "rel":
            # server un-normalized displacements from the current qpos: a[t+i] = q[t] + r[i]
            absolute = raw_qpos[:, None, :] + raw_actions
            raw_actions = np.where(self._anchor_mask, absolute, raw_actions)
        if self.num_past_steps > 0:
            for b in range(raw_actions.shape[0]):
                for a in raw_actions[b, : self.n_action_steps]:
                    self._hist[b].append(np.asarray(a, dtype=np.float32))

        # raw_actions shape: (B, chunk, D)
        if self.action_ensemble:
            # Ensemble each sample in the batch
            batch_size = raw_actions.shape[0]
            ensembled_actions = []
            for b in range(batch_size):
                ensembled = self.action_ensembler.ensemble_action(raw_actions[b])[None]  # (1, D)
                ensembled_actions.append(ensembled)
            raw_actions = np.stack(ensembled_actions, axis=0)  # (B, 1, D)

        raw_action = {
            "action.left_arm": raw_actions[:, : self.n_action_steps, :7],  # (B, n_action_steps, 7)
            "action.right_arm": raw_actions[:, : self.n_action_steps, 7:14],  # (B, n_action_steps, 7)
            "action.left_hand": raw_actions[:, : self.n_action_steps, 14:20],  # (B, n_action_steps, 6)
            "action.right_hand": raw_actions[:, : self.n_action_steps, 20:26],  # (B, n_action_steps, 6)
            "action.waist": raw_actions[:, : self.n_action_steps, 26:29],  # (B, n_action_steps, 3)
        }

        return {"actions": raw_action}

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)

    def normalize_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Normalize the state
        """
        for key in state.keys():
            sin_state = np.sin(state[key])
            cos_state = np.cos(state[key])
            state[key] = np.concatenate([sin_state, cos_state], axis=-1)
        return state
