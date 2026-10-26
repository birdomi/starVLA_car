from collections import deque
from typing import Dict, Optional

import cv2 as cv
import os
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

try:
    from examples.simBenchmarks.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
    ) -> None:

        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode} ***"
        )
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode

        # Action mode: "abs", "delta", or "rel"
        self.action_mode = action_mode
        # State tracking for delta/rel modes
        self.initial_state = None  # s_0 for rel mode
        self.prev_action = None  # last absolute action for delta mode

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_chunk_size = None
        self.state_norm_stats = None
        self.raw_actions = None
        # Env layout of joint vectors is [left 6, left gripper, right 6, right gripper]; the model
        # (AgilexDataConfig action/state keys) uses [left 6, right 6, left gripper, right gripper].
        self.env_to_model = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
        self.model_to_env = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
        # ROBOTWIN_SEND_STATE=1: send the raw state (model order) as ``state_raw``; the server normalizes it
        # with the training-time state transform. Required for models trained with include_state: true.
        self.send_state = os.environ.get("ROBOTWIN_SEND_STATE", "0") == "1"
        # ROBOTWIN_NUM_PAST_STEPS=P (>0): keep the last P executed actions (model order, env units) and send
        # them as ``past_action_abs`` + ``current_qpos`` for history-conditioned heads (QwenGR00T_PatchRegressor).
        self.num_past_steps = int(os.environ.get("ROBOTWIN_NUM_PAST_STEPS", "0"))
        # ROBOTWIN_STATE_HIST=H (>0): send the last H OBSERVED states (model order, env units) as ``state_raw``
        # (H, 14); the server normalizes each step. For heads with past_source: state (image2image over states).
        self.state_hist = int(os.environ.get("ROBOTWIN_STATE_HIST", "0"))
        self._state_hist = deque(maxlen=max(self.state_hist, 1))
        self._hist = deque(maxlen=max(self.num_past_steps, 1))
        print(f"*** robotwin client: send_state={self.send_state} num_past_steps={self.num_past_steps} ***")

        server_meta = self.client.get_server_metadata()
        self.action_chunk_size = server_meta["action_chunk_size"]
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode}, "
            f"server_meta: {server_meta} ***"
        )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        # Reset state tracking for delta/rel modes
        self.initial_state = None
        self.prev_action = None
        self._hist.clear()
        self._state_hist.clear()

    def step(
        self,
        example: dict,
        step: int = 0,
    ) -> np.ndarray:
        state = example.get("state", None)
        # if state is not None:
        #     state = self.normalize_state(state, self.state_norm_stats)
        #     state = state[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]]
        #     example["state"] = state.reshape(1, -1)

        # Store initial state for delta/rel modes
        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state to be provided in example")
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)
        images = example["image"]

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)
                # Re-store initial state after reset if in delta/rel mode
                if self.action_mode in ["delta", "rel"] and state is not None:
                    self.initial_state = np.array(state).copy()

        images = [self._resize_image(image) for image in images]
        example["image"] = images
        example_copy = example.copy()
        example_copy.pop("state")
        qpos_model = np.asarray(state, dtype=np.float32)[self.env_to_model] if state is not None else None
        if qpos_model is not None and self.state_hist > 0:
            self._state_hist.append(qpos_model.copy())
            h = list(self._state_hist)
            pad = [qpos_model] * (self.state_hist - len(h))
            example_copy["state_raw"] = np.stack(pad + h, axis=0).astype(np.float32)  # (H, 14) chronological, ends at s[t]
        elif self.send_state and qpos_model is not None:
            example_copy["state_raw"] = qpos_model  # (14,) model order, env units -> server normalizes
        if self.num_past_steps > 0 and qpos_model is not None:
            h = list(self._hist)
            pad = [qpos_model] * (self.num_past_steps - len(h))  # before the first step: "no motion"
            example_copy["past_action_abs"] = np.stack(pad + h, axis=0).astype(np.float32)  # (P, 14) chronological
            example_copy["current_qpos"] = qpos_model
            example_copy["past_mode"] = self.action_mode if self.action_mode in ("rel", "abs") else "delta"
        vla_input = {
            "examples": [example_copy],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        vla_input["unnorm_key"] = self.unnorm_key

        action_chunk_size = self.action_chunk_size

        if step % action_chunk_size == 0 or self.raw_actions is None:
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
            raw_actions = np.array(response["data"]["actions"][0])  # (chunk, D)

            # Convert delta/rel to absolute actions
            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_absolute(raw_actions, state)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_absolute(raw_actions)
            else:
                self.raw_actions = raw_actions

        action_idx = step % action_chunk_size
        if action_idx >= len(self.raw_actions):
            pass

        current_action = self.raw_actions[action_idx]

        # Update prev_action for delta mode (for cross-chunk continuity)
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        if self.num_past_steps > 0:
            self._hist.append(np.asarray(current_action, dtype=np.float32).copy())  # model order, env units
        current_action = current_action[self.model_to_env]
        return current_action

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Convert delta actions to absolute actions."""
        abs_actions = np.zeros_like(delta_actions)
        base = self.prev_action if self.prev_action is not None else self.initial_state
        for i in range(len(delta_actions)):
            abs_actions[i] = delta_actions[i] + base
            base = abs_actions[i]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        """Convert relative actions to absolute actions."""
        return rel_actions + self.initial_state

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    host = usr_args.get("host", "127.0.0.1")
    port = usr_args.get("port", 5694)
    unnorm_key = usr_args.get("unnorm_key", None)
    action_mode = usr_args.get("action_mode", "abs")
    normalization_mode = usr_args.get(
        "action_normalization_mode",
        usr_args.get("normalization_mode", "min_max"),
    )

    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")

    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=host,
        port=port,
        unnorm_key=unnorm_key,
        action_mode=action_mode,
        normalization_mode=normalization_mode,
    )


def reset_model(model):
    model.reset(task_description="")


def eval(TASK_ENV, model, observation):
    # Get instruction
    instruction = TASK_ENV.get_instruction()

    # Prepare images
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]

    # Order: [head, left, right] to match training order
    images = [head_img, left_img, right_img]

    state = observation["joint_action"]["vector"]
    example = {
        "lang": str(instruction),
        "image": images,
        "state": state,  # Required for delta/rel action modes
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # Execute action
    TASK_ENV.take_action(action)
