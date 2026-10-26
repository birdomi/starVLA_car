"""H1-2 car-real (2026-09-18 collection) data registry — LeRobot v3.0, 30 Hz.

Same task as 0820_car_data/left-turn-signal ("Activate the left turn signal and
inspect its operation.") but a newer recording pipeline:

  * state is 65-dim: the familiar 63 (head 2, legs 12, waist 1, L arm 7,
    R arm 7, zero-pad 8, L hand 13, R hand 13) plus `head_cmd` (63:65) — the
    COMMANDED head angle. Verified by correlation: head_cmd tracks the head
    action with r=0.994 at zero lag, while the measured head lags ~0.7 s.
  * action is the usual 42: head(2) L arm(7) L hand(13) R arm(7) R hand(13).
  * 30 Hz. (01a0b21c was logged on a 50 Hz grid with 40.7% duplicated rows;
    both the rows and the videos were resampled to the true 30 Hz control rate.)

Design choices, each backed by a measurement on this data:
  * head IS predicted here, anchored to `head_cmd` rather than the measured
    head: rel std drops 12.83deg -> 7.97deg, i.e. to the same difficulty as the
    arms (9.12deg). On the old data head had to be dropped because no such
    channel existed and `action - measured_head` encoded tracking lag.
  * `legs_waist` and `body_pad` are left out of the state: the pad is always
    exactly 0 and the legs/waist are a fixed seated pose.
  * horizon 30 = 1.0 s at 30 Hz (the deployed 0820 policy used 16 steps @
    20 Hz = 0.8 s; diffusion inference measured 505 ms, so 1.0 s still leaves
    room for synchronous chunk execution).
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform

_ABS_ACTION_KEYS = [  # grip targets are absolute (not rel-anchored)
    "action.left_ee",
    "action.right_ee",
]


class H12CarRealDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.front_view",
        "video.left_wrist_view",
        "video.right_wrist_view",
    ]
    # 44 = head(2) + head_cmd(2) + L arm(7) + R arm(7) + L hand(13) + R hand(13)
    state_keys = [
        "state.head",
        "state.head_cmd",
        "state.left_arm",
        "state.right_arm",
        "state.left_ee",
        "state.right_ee",
    ]
    # 42 = head(2) + L arm(7) + L hand(13) + R arm(7) + R hand(13)
    action_keys = [
        "action.head",
        "action.left_arm",
        "action.left_ee",
        "action.right_arm",
        "action.right_ee",
    ]
    language_keys = ["annotation.human.task_description"]

    observation_indices = [0]
    state_indices = [0]
    # 30 steps @ 30 Hz = 1.0 s; must match framework.action_model.action_horizon.
    action_indices = list(range(30))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # Everything is q99 (q01..q99 rescaled to [-1,1], clamped at +-2.2) rather
        # than min_max: a single corrupted sensor frame is enough to blow out a
        # min/max range and squash a whole channel group to near-constant, which
        # is exactly what one bad row in car-left-turn-signal did to the right
        # hand state (-19706..29702 instead of 0..2050).
        action_modes = {k: "q99" for k in self.action_keys}
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={k: "q99" for k in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_modes,
                ),
            ]
        )


class H12CarLeftTurnDataConfig(H12CarRealDataConfig):
    """car-left-turn-signal (532 ep). Same robot, but the head is carried in
    exactly one place.

    The measured head (s0:2) is dropped from the state, leaving 42 dims. It is
    the camera pose, so the front_view image already shows it and the model can
    read it off the pixels. `head_cmd` (s63:65) is the opposite — a target the
    head has not reached yet, visible nowhere in the observation — and it is the
    anchor the rel target is measured from (`action.head - state.head_cmd`), so
    the loader requires it in `state_keys` anyway (datasets.py raises if the
    anchor key is missing). Feeding both would carry one joint in two different
    roles, and they are far apart: yaw corr 0.78, mean gap 7.45deg, p95 30.6deg.

    `H12CarRealDataConfig` keeps its 44-dim state so the already-trained
    01a0b21c 100k checkpoint stays loadable.
    """

    # 42 = head_cmd(2) + L arm(7) + R arm(7) + L hand(13) + R hand(13)
    state_keys = [
        "state.head_cmd",
        "state.left_arm",
        "state.right_arm",
        "state.left_ee",
        "state.right_ee",
    ]


class H12CarLeftTurnHistoryDataConfig(H12CarLeftTurnDataConfig):
    """Same as h1_2_car_left, but the state carries a 15-frame history.

    `state_indices` becomes [-14, ..., 0], so the dataloader stacks 15 frames
    (0.5 s at 30 Hz) oldest-first and the action head turns them into 15 tokens
    in `sa_embs` instead of one. Used with framework QwenGR00TProgress, whose
    head adds a position embedding over those tokens and reads normalized
    progress off the DiT's future tokens.
    """

    state_indices = list(range(-14, 1))


class H12CarLeftTurnFrontOnlyDataConfig(H12CarLeftTurnDataConfig):
    """Same as h1_2_car_left, but only the head (front_view) camera reaches the VLM.

    Wrist-camera ablation. Caveat measured on car-few-tasks-v1 val: turn signals are
    left-arm tasks, while hazard and wiper are right-arm tasks where the right wrist view is
    the only close-up of the button / stalk.
    """

    video_keys = ["video.front_view"]


class H12CarLeftTurnFrontHist30DataConfig(H12CarLeftTurnFrontOnlyDataConfig):
    """Front camera at t-90, t-60, t-30, t (oldest first) for the HAMLET memory framework.

    History length 4 with stride k = 30 = one action chunk. Each frame gets its own VLM pass
    (QwenGR00T_Memory splits sample["image"]); the last one is the current observation. Frames
    before the episode start are clamped to frame 0 by the loader, so early samples repeat the
    first frame — the deployment cache must pad the same way.
    """

    video_indices = [-90, -60, -30, 0]

    def modality_config(self):
        config = super().modality_config()
        config["video"] = ModalityConfig(delta_indices=self.video_indices, modality_keys=self.video_keys)
        return config


class H12CarLeftTurnFrontFuture30DataConfig(H12CarLeftTurnFrontOnlyDataConfig):
    """Front camera at t and t+30 (1.0 s) for QwenGR00T_TriSystem: [current, future] image pairs.

    The future frame is the System 1.5 target (dZ = Z_f - Z_t) and the System 1 generation target; System 2 and
    the action head only see frame t. Indices past the episode end are clamped to its last frame by the loader.
    Must match framework.tri_system.future_offsets = [30].
    """

    video_indices = [0, 30]

    def modality_config(self):
        config = super().modality_config()
        config["video"] = ModalityConfig(delta_indices=self.video_indices, modality_keys=self.video_keys)
        return config


class H12CarLeftTurnFrontFuture10x3DataConfig(H12CarLeftTurnFrontFuture30DataConfig):
    """Front camera at t, t+10, t+20, t+30: System 1 generates the 3-frame future clip, System 1.5 still
    predicts the change to t+30 (the last frame). framework.tri_system.future_offsets = [10, 20, 30]."""

    video_indices = [0, 10, 20, 30]


class H12CarTurnSignal3CamDataConfig(H12CarLeftTurnDataConfig):
    """car-turn-signal (left + right turn signal, 1143 train / 60 val episodes, 2026-10-07 export).

    Same robot, 65-dim state and 42-dim action layout as car-few-tasks-v1, but the export names
    the cameras head_left_2 / head_right_2 (a stereo pair) / wrist_left_2 / wrist_right_2.
    head_left_2 IS the old front_view (pixel-identical on overlapping episodes), so the camera
    order here (head, left wrist, right wrist) matches the earlier 3-camera car models.
    """

    video_keys = ["video.head_left_2", "video.wrist_left_2", "video.wrist_right_2"]
    # 32-step chunk (1.07 s at 30 Hz) so a keyframe stride of 4 divides it (anchors 3, 7, ..., 31);
    # must match framework.action_model.action_horizon.
    action_indices = list(range(32))



class H12CarTurnSignal3CamPrev30DataConfig(H12CarTurnSignal3CamDataConfig):
    """3 cameras, each at t-30 (1.0 s) and t: [prev, curr] pairs per camera, camera-major.

    For QwenGR00T_GridDiT_ProgressShift (temporal progress-conditioned AdaLN shift): the previous
    frames only feed the progress branch; the DiT cross-attends to frame t as before. Frames before
    the episode start are clamped to frame 0 -> prev == curr at early steps (framework zeroes the
    residual when the two are identical). Must match framework.progress_shift.prev_offset = 30.
    """

    video_indices = [-30, 0]

    def modality_config(self):
        config = super().modality_config()
        config["video"] = ModalityConfig(delta_indices=self.video_indices, modality_keys=self.video_keys)
        return config


class H12CarTurnSignalDualSystemDataConfig(H12CarTurnSignal3CamDataConfig):
    """QwenGR00T_DualSystem window around the System 2 observation t on car-turn-signal-ep (30 Hz), after
    FourierGr1ArmsWaistDualSystemDataConfig (RoboCasa, 20 Hz) / AgilexDualSystemDataConfig (RoboTwin, 3 views):
    dense actions a[t + k] for k = floor(t_min * rate) .. ceil(t_max * rate) (default [-2.0, +4.0] s -> k = -60..120,
    181 steps, the System 2 keyframe span), the 3 cameras (head_left_2, wrist_left_2, wrist_right_2; camera-major) and
    the state at t + video_offsets (frame t = System 2 input; t + o = a System 1 input whose plan age is o / 30 s).
    Must match framework.dual_system {action_rate_hz: 30, keyframes, video_offsets, num_views: 3}. Slots outside the
    episode are padded by the loader and flagged with datasets.*.emit_action_valid: true. State / action are the 42-D
    car layout (head_cmd + arms + hands, q99), no sin/cos -> action_dim = state_dim = 42.
    """

    def __init__(self, video_offsets=(0,), rate: float = 30.0, t_min: float = -2.0, t_max: float = 4.0):
        import math

        self.video_indices = [int(o) for o in video_offsets]
        self.state_indices = list(self.video_indices)
        self.action_indices = list(range(math.floor(t_min * rate + 1e-6), math.ceil(t_max * rate - 1e-6) + 1))

    def modality_config(self):
        cfg = super().modality_config()
        cfg["video"] = ModalityConfig(delta_indices=self.video_indices, modality_keys=self.video_keys)
        cfg["state"] = ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys)
        return cfg


class H12CarTurnSignalDualSystemNoHeadCmdDataConfig(H12CarTurnSignalDualSystemDataConfig):
    """H12CarTurnSignalDualSystemDataConfig without head_cmd (s63:65) in the state: 40-D state = L arm + R arm +
    L hand + R hand -> framework.action_model.state_dim 40. Used with car-turn-signal-ep-rawhead, whose head ACTION
    is head_cmd one frame ahead, so a state carrying head_cmd would hand the model its own target."""

    # 40 = L arm(7) + R arm(7) + L hand(13) + R hand(13)
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_ee",
        "state.right_ee",
    ]


class H12Car3CamNoHeadCmdDataConfig(H12CarTurnSignal3CamDataConfig):
    """Plain chunk policy on the 63-D-state car exports (car-turn-signal-v2-ep, car-1015-v1-ep): 3 cameras (head_left_2,
    wrist_left_2, wrist_right_2) at video_offsets (oldest first, camera-major), 40-D state at t (arms + hands, no head:
    the measured head is a lagging sample-and-hold copy, and these exports have no head_cmd), a[t .. t + horizon - 1]
    (q99; default 50 = 1.67 s, the DualSystem System 1 horizon; must match framework.action_model.action_horizon).
    video_offsets (-15, 0) = Helix_VLA helix.s2_lag 15: System 2 on the frame 0.5 s old, System 1 on frame t.
    prev_action: action rows a[t - 1 .. t + horizon - 1] (Helix_VLA_PrevCmd: row 0 = the previous command -> state).
    head_state: 42-D state = the recorded (measured) head state.head (s0:2) + the 40 above -> action_model.state_dim 42."""

    state_keys = H12CarTurnSignalDualSystemNoHeadCmdDataConfig.state_keys

    def __init__(self, video_offsets=(0,), horizon: int = 50, prev_action: bool = False, head_state: bool = False):
        self.video_indices = [int(o) for o in video_offsets]
        self.action_indices = list(range(-1 if prev_action else 0, int(horizon)))
        if head_state:
            self.state_keys = ["state.head"] + list(type(self).state_keys)

    def modality_config(self):
        cfg = super().modality_config()
        cfg["video"] = ModalityConfig(delta_indices=self.video_indices, modality_keys=self.video_keys)
        return cfg


ROBOT_TYPE_CONFIG_MAP = {
    # 3 cameras, 40-D state at t, a[t .. t+49]; frames t (sync) or [t - 15, t] (Helix_VLA s2_lag 15)
    "h1_2_car_3cam_nohc": H12Car3CamNoHeadCmdDataConfig(video_offsets=(0,)),
    "h1_2_car_3cam_vlm15_nohc": H12Car3CamNoHeadCmdDataConfig(video_offsets=(-15, 0)),
    # + a[t - 1] in front of the chunk (Helix_VLA_PrevCmd: previous head command -> state)
    "h1_2_car_3cam_vlm15_prevcmd_nohc": H12Car3CamNoHeadCmdDataConfig(video_offsets=(-15, 0), prev_action=True),
    # 42-D state = recorded head state (s0:2) + arms + hands at t (Helix_VLA)
    "h1_2_car_3cam_vlm15_headstate": H12Car3CamNoHeadCmdDataConfig(video_offsets=(-15, 0), head_state=True),
    "h1_2_car_3cam_headstate": H12Car3CamNoHeadCmdDataConfig(video_offsets=(0,), head_state=True),  # frames t (QwenGR00T)
    # frames t - 60, -45, -30, -15, t (every 0.5 s over the last 2 s; QwenGR00T_EpisodeMemory update_offsets)
    "h1_2_car_3cam_mem60_headstate": H12Car3CamNoHeadCmdDataConfig(video_offsets=(-60, -45, -30, -15, 0), head_state=True),
    "h1_2_car_turn_3cam": H12CarTurnSignal3CamDataConfig(),
    "h1_2_car_turn_3cam_prev30": H12CarTurnSignal3CamPrev30DataConfig(),
    # QwenGR00T_DualSystem (3 cameras, 30 Hz): a[t-60 .. t+120]; stage 1 (System 2) frame t only; stage 2 (System 1)
    # frames t + {0, 6, 12, 18, 24} = plan ages 0 .. 0.8 s (same ages as the RoboCasa 20 Hz offsets {0, 4, .., 16})
    "h1_2_car_turn_dual_s2": H12CarTurnSignalDualSystemDataConfig(video_offsets=(0,)),
    "h1_2_car_turn_dual": H12CarTurnSignalDualSystemDataConfig(video_offsets=(0, 6, 12, 18, 24)),
    # same without head_cmd in the state (40-D)
    "h1_2_car_turn_dual_s2_nohc": H12CarTurnSignalDualSystemNoHeadCmdDataConfig(video_offsets=(0,)),
    "h1_2_car_turn_dual_nohc": H12CarTurnSignalDualSystemNoHeadCmdDataConfig(video_offsets=(0, 6, 12, 18, 24)),
    # dual_system.stale_input vlm: VLM frame t - 15 (0.5 s), C-RADIO / state / System 1 frame t
    "h1_2_car_turn_dual_vlm15_nohc": H12CarTurnSignalDualSystemNoHeadCmdDataConfig(video_offsets=(-15, 0)),
    # + dual_system.memory: VLM frames t - 60, -45, -30, -15 (every 0.5 s over the last 2 s) for the episode memory updates
    "h1_2_car_turn_dual_vlm15mem_nohc": H12CarTurnSignalDualSystemNoHeadCmdDataConfig(video_offsets=(-60, -45, -30, -15, 0)),
    "h1_2_car_real": H12CarRealDataConfig(),
    "h1_2_car_left": H12CarLeftTurnDataConfig(),
    "h1_2_car_left_hist": H12CarLeftTurnHistoryDataConfig(),
    "h1_2_car_left_front": H12CarLeftTurnFrontOnlyDataConfig(),
    "h1_2_car_left_front_hist30": H12CarLeftTurnFrontHist30DataConfig(),
    "h1_2_car_left_front_future30": H12CarLeftTurnFrontFuture30DataConfig(),
    "h1_2_car_left_front_future10x3": H12CarLeftTurnFrontFuture10x3DataConfig(),
}

_ROOT = "01a0b21c-8a26-7759-a7be-25b7c9c545f6"

DATASET_NAMED_MIXTURES = {
    # car-real 01a0b21c: 36 episodes, resampled 50 Hz -> 30 Hz.
    "car_real_left_turn": [(f"{_ROOT}/train", 1.0, "h1_2_car_real")],
    # car-left-turn-signal: 532 episodes / 412k frames, natively 30 Hz with no
    # duplicated rows. Two task strings (fixed seat / free seat) in one set.
    # Lives in a sibling dataset root, so point data_root_dir at Datasets/.
    "car_left_turn_v2": [("car-left-turn-signal/train", 1.0, "h1_2_car_left")],
    # same data, state history + progress head (framework QwenGR00TProgress)
    "car_left_turn_v2_hist": [("car-left-turn-signal/train", 1.0, "h1_2_car_left_hist")],
    # car-few-tasks-v1: 9 tasks (L/R turn signal, hazard on/off, each +/- "(fixed seat)";
    # wiper speed 1/2), 2608 ep / 1.65M frames, same robot and 65/42 layout as
    # car-left-turn-signal. Its left-turn episodes are a superset of car-left-turn-signal but
    # RE-SPLIT (~52 of its 59 val left-turn episodes were car-left-turn-signal train).
    # Run prepare_car_dataset.py on it first (modality.json + hand ch2 clip).
    "car_few_tasks_v1": [("car-few-tasks-v1/train", 1.0, "h1_2_car_left")],
    # same data, head camera only
    "car_few_tasks_v1_front": [("car-few-tasks-v1/train", 1.0, "h1_2_car_left_front")],
    # same, plus 3 past front frames at stride 30 (QwenGR00T_Memory)
    "car_few_tasks_v1_front_hist30": [("car-few-tasks-v1/train", 1.0, "h1_2_car_left_front_hist30")],
    # smoke-test stand-in until car-few-tasks-v1 lands (same robot / layout)
    "car_left_turn_v2_front": [("car-left-turn-signal/train", 1.0, "h1_2_car_left_front")],
    # QwenGR00T_TriSystem: [frame t, frame t+30] pairs (front camera), train / val
    "car_few_tasks_v1_front_future30": [("car-few-tasks-v1/train", 1.0, "h1_2_car_left_front_future30")],
    "car_few_tasks_v1_val_front_future30": [("car-few-tasks-v1/val", 1.0, "h1_2_car_left_front_future30")],
    # same with the clip t+10, t+20, t+30 for System 1
    "car_few_tasks_v1_front_future10x3": [("car-few-tasks-v1/train", 1.0, "h1_2_car_left_front_future10x3")],
    "car_few_tasks_v1_val_front_future10x3": [("car-few-tasks-v1/val", 1.0, "h1_2_car_left_front_future10x3")],
    # small set for smoke tests (car-left-turn-signal val: 58 episodes)
    "car_left_turn_v2_val_front_future30": [("car-left-turn-signal/val", 1.0, "h1_2_car_left_front_future30")],
    # car-turn-signal: left + right turn (4 prompts, 2 after the fixed-seat merge), head + both wrists.
    # Mostly a re-export / re-split of car-few-tasks-v1's turn-signal episodes (1183 of 1203 overlap).
    # Prepared with prepare_car_dataset.py (2026-10-08).
    "car_turn_signal_3cam": [("car-turn-signal/train", 1.0, "h1_2_car_turn_3cam")],
    # Same episodes re-exported as ONE mp4 PER EPISODE with a 15-frame GOP (make_episode_videos.py,
    # 2026-10-08): the concatenated originals have decode errors in 37 train / 39 val episodes (mismatched
    # stream parameters after concatenation; see meta/corrupted_episodes.json -> exclude them) and a
    # 150-frame GOP that made each random frame decode ~75 frames.
    "car_turn_signal_ep_3cam": [("car-turn-signal-ep/train", 1.0, "h1_2_car_turn_3cam")],
    # same, each camera at t-30 and t (QwenGR00T_GridDiT_ProgressShift)
    "car_turn_signal_ep_3cam_prev30": [("car-turn-signal-ep/train", 1.0, "h1_2_car_turn_3cam_prev30")],
    # QwenGR00T_DualSystem stage 1 / stage 2 (train + held-out val split for the eval hook)
    "car_turn_signal_ep_dual_s2": [("car-turn-signal-ep/train", 1.0, "h1_2_car_turn_dual_s2")],
    "car_turn_signal_ep_dual": [("car-turn-signal-ep/train", 1.0, "h1_2_car_turn_dual")],
    "car_turn_signal_ep_val_dual_s2": [("car-turn-signal-ep/val", 1.0, "h1_2_car_turn_dual_s2")],
    "car_turn_signal_ep_val_dual": [("car-turn-signal-ep/val", 1.0, "h1_2_car_turn_dual")],
    # same episodes, head ACTION = raw head pose one frame ahead (state[t+1, 63:65]; make_rawhead_dataset.py,
    # 2026-10-12); the other 40 action dims and the normalization statistics are the original ones; state WITHOUT
    # head_cmd (40-D). Stage 1 / stage 2, train + val.
    "car_turn_signal_ep_rawhead_dual_s2_nohc": [("car-turn-signal-ep-rawhead/train", 1.0, "h1_2_car_turn_dual_s2_nohc")],
    "car_turn_signal_ep_rawhead_dual_nohc": [("car-turn-signal-ep-rawhead/train", 1.0, "h1_2_car_turn_dual_nohc")],
    "car_turn_signal_ep_rawhead_val_dual_s2_nohc": [("car-turn-signal-ep-rawhead/val", 1.0, "h1_2_car_turn_dual_s2_nohc")],
    "car_turn_signal_ep_rawhead_val_dual_nohc": [("car-turn-signal-ep-rawhead/val", 1.0, "h1_2_car_turn_dual_nohc")],
    # car-turn-signal_v2 re-export (2026-10-13) prepared by prepare_car_v2_dataset.py: same recordings, new split
    # (all 1143 / 60 episodes usable), head ACTION = raw head[t], 63-D state without head_cmd (the nohc configs pick
    # the same 40 dims), 2 instructions (" (fixed seat)" removed). Stage 1 / stage 2, train + val.
    "car_turn_signal_v2_dual_s2_nohc": [("car-turn-signal-v2-ep/train", 1.0, "h1_2_car_turn_dual_s2_nohc")],
    "car_turn_signal_v2_dual_nohc": [("car-turn-signal-v2-ep/train", 1.0, "h1_2_car_turn_dual_nohc")],
    "car_turn_signal_v2_val_dual_s2_nohc": [("car-turn-signal-v2-ep/val", 1.0, "h1_2_car_turn_dual_s2_nohc")],
    "car_turn_signal_v2_val_dual_nohc": [("car-turn-signal-v2-ep/val", 1.0, "h1_2_car_turn_dual_nohc")],
    # VLM 0.5 s behind C-RADIO / state (joint_e2e_vlm15_v2_carturn_3cam.yaml)
    "car_turn_signal_v2_dual_vlm15_nohc": [("car-turn-signal-v2-ep/train", 1.0, "h1_2_car_turn_dual_vlm15_nohc")],
    "car_turn_signal_v2_val_dual_vlm15_nohc": [("car-turn-signal-v2-ep/val", 1.0, "h1_2_car_turn_dual_vlm15_nohc")],
    # v2 train with every look's head hold capped at 0.5 s (rows removed, original timestamps / videos; meta/holdcap_note.json)
    "car_turn_signal_v2_holdcap05_dual_vlm15_nohc": [("car-turn-signal-v2-ep-holdcap05/train", 1.0, "h1_2_car_turn_dual_vlm15_nohc")],
    # episode memory over the VLM frames of the last 2 s (joint_e2e_vlm15_relplan_holdcap05_mem_ft46k_v2_carturn_3cam.yaml)
    "car_turn_signal_v2_holdcap05_dual_vlm15mem_nohc": [("car-turn-signal-v2-ep-holdcap05/train", 1.0, "h1_2_car_turn_dual_vlm15mem_nohc")],
    "car_turn_signal_v2_val_dual_vlm15mem_nohc": [("car-turn-signal-v2-ep/val", 1.0, "h1_2_car_turn_dual_vlm15mem_nohc")],
    # car-1015-v1 (2026-10-15 export, 34 tasks: turn signal, hazard, high beam, wiper, menu, interior light, sunroof,
    # engine) prepared by prepare_car1015_dataset.py (2026-10-19): the ids of car1015_exclude_episodes.json dropped
    # (train 14559 -> 13696, val 766 -> 710 episodes), 34 normalized instructions (car1015_task_map.json, same
    # task_index in both splits), hand ch2 clipped, head ACTION = raw head[t]. Hand state is left 37:50 | right 50:63
    # (meta/modality.json), so the nohc configs pick the same 40 dims as on car-turn-signal-v2-ep. One parquet per
    # episode: leave cache_parquet_tables off. Videos are symlinks to the source mp4s (GOP 250).
    "car1015_v1_dual_s2_nohc": [("car-1015-v1-ep/train", 1.0, "h1_2_car_turn_dual_s2_nohc")],
    "car1015_v1_dual_nohc": [("car-1015-v1-ep/train", 1.0, "h1_2_car_turn_dual_nohc")],
    "car1015_v1_val_dual_s2_nohc": [("car-1015-v1-ep/val", 1.0, "h1_2_car_turn_dual_s2_nohc")],
    "car1015_v1_val_dual_nohc": [("car-1015-v1-ep/val", 1.0, "h1_2_car_turn_dual_nohc")],
    "car1015_v1_dual_vlm15_nohc": [("car-1015-v1-ep/train", 1.0, "h1_2_car_turn_dual_vlm15_nohc")],
    "car1015_v1_val_dual_vlm15_nohc": [("car-1015-v1-ep/val", 1.0, "h1_2_car_turn_dual_vlm15_nohc")],
    "car1015_v1_dual_vlm15mem_nohc": [("car-1015-v1-ep/train", 1.0, "h1_2_car_turn_dual_vlm15mem_nohc")],
    "car1015_v1_val_dual_vlm15mem_nohc": [("car-1015-v1-ep/val", 1.0, "h1_2_car_turn_dual_vlm15mem_nohc")],
    # plain chunk policy (Helix_VLA): 3 cameras at [t - 15, t], 40-D state at t, a[t .. t+49]
    "car1015_v1_3cam_vlm15_nohc": [("car-1015-v1-ep/train", 1.0, "h1_2_car_3cam_vlm15_nohc")],
    "car1015_v1_val_3cam_vlm15_nohc": [("car-1015-v1-ep/val", 1.0, "h1_2_car_3cam_vlm15_nohc")],
    # same with a[t - 1 .. t+49] (Helix_VLA_PrevCmd: the previous head command a[t - 1][0:2] prepended to the state)
    "car1015_v1_3cam_vlm15_prevcmd_nohc": [("car-1015-v1-ep/train", 1.0, "h1_2_car_3cam_vlm15_prevcmd_nohc")],
    "car1015_v1_val_3cam_vlm15_prevcmd_nohc": [("car-1015-v1-ep/val", 1.0, "h1_2_car_3cam_vlm15_prevcmd_nohc")],
    # same with the recorded head state in the state: 42-D = state.head (s0:2) + arms + hands at t, a[t .. t+49]
    "car1015_v1_3cam_vlm15_headstate": [("car-1015-v1-ep/train", 1.0, "h1_2_car_3cam_vlm15_headstate")],
    "car1015_v1_val_3cam_vlm15_headstate": [("car-1015-v1-ep/val", 1.0, "h1_2_car_3cam_vlm15_headstate")],
    # plain chunk policy (QwenGR00T): 3 cameras at t, 42-D state (recorded head + arms + hands) at t, a[t .. t+49]
    "car1015_v1_3cam_headstate": [("car-1015-v1-ep/train", 1.0, "h1_2_car_3cam_headstate")],
    "car1015_v1_val_3cam_headstate": [("car-1015-v1-ep/val", 1.0, "h1_2_car_3cam_headstate")],
    # + frames t - 60, -45, -30, -15 per camera for the episode memory (QwenGR00T_EpisodeMemory)
    "car1015_v1_3cam_mem60_headstate": [("car-1015-v1-ep/train", 1.0, "h1_2_car_3cam_mem60_headstate")],
    "car1015_v1_val_3cam_mem60_headstate": [("car-1015-v1-ep/val", 1.0, "h1_2_car_3cam_mem60_headstate")],
    # car-1015-v2-ep: the car-1015-v1 episodes recorded with the LOW gravity compensation only (10-21,
    # car1015_v2_lowcomp_exclude_episodes.json; 19 tasks: high-beam flash and the 3 tasks with 1-3 low-comp episodes dropped);
    # same layouts as the v1 mixtures above
    "car1015_v2_3cam_vlm15_headstate": [("car-1015-v2-ep/train", 1.0, "h1_2_car_3cam_vlm15_headstate")],
    "car1015_v2_val_3cam_vlm15_headstate": [("car-1015-v2-ep/val", 1.0, "h1_2_car_3cam_vlm15_headstate")],
    "car1015_v2_3cam_headstate": [("car-1015-v2-ep/train", 1.0, "h1_2_car_3cam_headstate")],
    "car1015_v2_val_3cam_headstate": [("car-1015-v2-ep/val", 1.0, "h1_2_car_3cam_headstate")],
    # car-1015-v3-ep: the car-1015-v1 episodes (both comp regimes) of 13 tasks (10-22, car1015_v3_exclude_episodes.json;
    # hazard, high beam, wiper 1->2->3 sequential, steering-wheel menus, interior lights dropped); same layouts as above
    "car1015_v3_3cam_vlm15_headstate": [("car-1015-v3-ep/train", 1.0, "h1_2_car_3cam_vlm15_headstate")],
    "car1015_v3_val_3cam_vlm15_headstate": [("car-1015-v3-ep/val", 1.0, "h1_2_car_3cam_vlm15_headstate")],
    "car1015_v3_3cam_headstate": [("car-1015-v3-ep/train", 1.0, "h1_2_car_3cam_headstate")],
    "car1015_v3_val_3cam_headstate": [("car-1015-v3-ep/val", 1.0, "h1_2_car_3cam_headstate")],
    "car1015_v3_3cam_mem60_headstate": [("car-1015-v3-ep/train", 1.0, "h1_2_car_3cam_mem60_headstate")],
    "car1015_v3_val_3cam_mem60_headstate": [("car-1015-v3-ep/val", 1.0, "h1_2_car_3cam_mem60_headstate")],
}
