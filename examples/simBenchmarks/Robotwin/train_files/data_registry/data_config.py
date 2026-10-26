"""RobotWin benchmark — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig — Agilex (RobotWin, action_indices=16)
# ---------------------------------------------------------------------------
class AgilexDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = ["state.left_joints", "state.right_joints", "state.left_gripper", "state.right_gripper"]
    action_keys = ["action.left_joints", "action.right_joints", "action.left_gripper", "action.right_gripper"]
    # Per-key dims for PolicyNormProcessor (Agilex 6-DOF arms + binary gripper = 14-D total)
    action_key_dims = {"action.left_joints": 6, "action.right_joints": 6, "action.left_gripper": 1, "action.right_gripper": 1}
    state_key_dims  = {"state.left_joints": 6, "state.right_joints": 6, "state.left_gripper": 1, "state.right_gripper": 1}
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max", "state.right_joints": "min_max",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max", "action.right_joints": "min_max",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — Agilex past+future window (QwenGR00T_PatchRegressor)
# ---------------------------------------------------------------------------
class AgilexPastFutureDataConfig(AgilexDataConfig):
    """Bidirectional action window: P past steps a[t-P..t-1] followed by H future steps a[t..t+H-1]
    (used by QwenGR00T_PatchRegressor as executed-action history + regression target). Same min_max /
    binary normalization as AgilexDataConfig, so abs targets match the plain regressor."""

    def __init__(self, past: int, future: int):
        self.action_indices = list(range(-past, future))


# ---------------------------------------------------------------------------
# DataConfig — Agilex 50 (action_indices=50)
# ---------------------------------------------------------------------------
class AgilexData50Config(AgilexDataConfig):
    action_indices = list(range(50))

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "state.left_joints": "min_max", "state.right_joints": "min_max",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                binary_threshold=0.49,
                normalization_modes={
                    "action.left_joints": "min_max", "action.right_joints": "min_max",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


# ---------------------------------------------------------------------------
# DataConfig — ARX X5
# ---------------------------------------------------------------------------
class ArxX5DataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = ["state.left_joints", "state.right_joints", "state.left_gripper", "state.right_gripper"]
    action_keys = ["action.left_joints", "action.right_joints", "action.left_gripper", "action.right_gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.left_joints": "min_max", "state.right_joints": "min_max",
                    "state.left_gripper": "binary", "state.right_gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.left_joints": "min_max", "action.right_joints": "min_max",
                    "action.left_gripper": "binary", "action.right_gripper": "binary",
                },
            ),
        ])


class AgilexStateHistDataConfig(AgilexDataConfig):
    """State HISTORY window s[t-H+1 .. t] (H steps, ends at the current step) + future actions a[t .. t+15].
    Used by QwenGR00T_PatchRegressor with past_source: state (image2image over observed states)."""

    def __init__(self, hist: int):
        self.state_indices = list(range(-hist + 1, 1))

    def modality_config(self):
        cfg = super().modality_config()
        cfg["state"] = ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys)
        return cfg


class AgilexDualSystemDataConfig(AgilexDataConfig):
    """QwenGR00T_DualSystem window around the System 2 observation t (15 Hz): dense actions a[t + k] for
    k = floor(t_min * rate) .. ceil(t_max * rate) (default [-2.0, +4.0] s -> k = -30..60, 91 steps, the System 2
    keyframe span), the 3 cameras (head, left wrist, right wrist) and the state at t + video_offsets (frame t = System 2
    input; t + o = a System 1 input whose plan age is o / 15 s). Must match framework.dual_system {action_rate_hz,
    keyframes, video_offsets, num_views: 3}. Slots outside the episode are padded by the loader and flagged with
    datasets.*.emit_action_valid: true. Same min_max / binary normalization as AgilexDataConfig."""

    def __init__(self, video_offsets=(0,), rate: float = 15.0, t_min: float = -2.0, t_max: float = 4.0):
        import math

        self.video_indices = [int(o) for o in video_offsets]
        self.state_indices = list(self.video_indices)
        self.action_indices = list(range(math.floor(t_min * rate + 1e-6), math.ceil(t_max * rate - 1e-6) + 1))

    def modality_config(self):
        cfg = super().modality_config()
        cfg["video"] = ModalityConfig(delta_indices=self.video_indices, modality_keys=self.video_keys)
        cfg["state"] = ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys)
        return cfg


ROBOT_TYPE_CONFIG_MAP = {
    "robotwin": AgilexDataConfig(),
    "robotwin_sh16": AgilexStateHistDataConfig(16),  # 16-step observed state history
    "robotwin50": AgilexData50Config(),
    "robotwin_pf32": AgilexPastFutureDataConfig(16, 16),  # range(-16, 16): 16 past + 16 future
    "arx_x5": ArxX5DataConfig(),
    # QwenGR00T_DualSystem: a[t-30 .. t+60]; stage 1 (System 2) frame t only; stage 2 (System 1) frames t + {0, 3, 6, 9,
    # 12} = plan ages 0 .. 0.8 s
    "robotwin_dual_s2": AgilexDualSystemDataConfig(video_offsets=(0,)),
    "robotwin_dual": AgilexDualSystemDataConfig(video_offsets=(0, 3, 6, 9, 12)),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}

# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "robotwin_all_sh16": [
        ("Clean/adjust_bottle", 1.0, "robotwin_sh16"), ("Randomized/adjust_bottle", 1.0, "robotwin_sh16"),
        ("Clean/beat_block_hammer", 1.0, "robotwin_sh16"), ("Randomized/beat_block_hammer", 1.0, "robotwin_sh16"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin_sh16"), ("Randomized/blocks_ranking_rgb", 1.0, "robotwin_sh16"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin_sh16"), ("Randomized/blocks_ranking_size", 1.0, "robotwin_sh16"),
        ("Clean/click_alarmclock", 1.0, "robotwin_sh16"), ("Randomized/click_alarmclock", 1.0, "robotwin_sh16"),
        ("Clean/click_bell", 1.0, "robotwin_sh16"), ("Randomized/click_bell", 1.0, "robotwin_sh16"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin_sh16"), ("Randomized/dump_bin_bigbin", 1.0, "robotwin_sh16"),
        ("Clean/grab_roller", 1.0, "robotwin_sh16"), ("Randomized/grab_roller", 1.0, "robotwin_sh16"),
        ("Clean/handover_block", 1.0, "robotwin_sh16"), ("Randomized/handover_block", 1.0, "robotwin_sh16"),
        ("Clean/handover_mic", 1.0, "robotwin_sh16"), ("Randomized/handover_mic", 1.0, "robotwin_sh16"),
        ("Clean/hanging_mug", 1.0, "robotwin_sh16"), ("Randomized/hanging_mug", 1.0, "robotwin_sh16"),
        ("Clean/lift_pot", 1.0, "robotwin_sh16"), ("Randomized/lift_pot", 1.0, "robotwin_sh16"),
        ("Clean/move_can_pot", 1.0, "robotwin_sh16"), ("Randomized/move_can_pot", 1.0, "robotwin_sh16"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin_sh16"), ("Randomized/move_pillbottle_pad", 1.0, "robotwin_sh16"),
        ("Clean/move_playingcard_away", 1.0, "robotwin_sh16"), ("Randomized/move_playingcard_away", 1.0, "robotwin_sh16"),
        ("Clean/move_stapler_pad", 1.0, "robotwin_sh16"), ("Randomized/move_stapler_pad", 1.0, "robotwin_sh16"),
        ("Clean/open_laptop", 1.0, "robotwin_sh16"), ("Randomized/open_laptop", 1.0, "robotwin_sh16"),
        ("Clean/open_microwave", 1.0, "robotwin_sh16"), ("Randomized/open_microwave", 1.0, "robotwin_sh16"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin_sh16"), ("Randomized/pick_diverse_bottles", 1.0, "robotwin_sh16"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin_sh16"), ("Randomized/pick_dual_bottles", 1.0, "robotwin_sh16"),
        ("Clean/place_a2b_left", 1.0, "robotwin_sh16"), ("Randomized/place_a2b_left", 1.0, "robotwin_sh16"),
        ("Clean/place_a2b_right", 1.0, "robotwin_sh16"), ("Randomized/place_a2b_right", 1.0, "robotwin_sh16"),
        ("Clean/place_bread_basket", 1.0, "robotwin_sh16"), ("Randomized/place_bread_basket", 1.0, "robotwin_sh16"),
        ("Clean/place_bread_skillet", 1.0, "robotwin_sh16"), ("Randomized/place_bread_skillet", 1.0, "robotwin_sh16"),
        ("Clean/place_burger_fries", 1.0, "robotwin_sh16"), ("Randomized/place_burger_fries", 1.0, "robotwin_sh16"),
        ("Clean/place_can_basket", 1.0, "robotwin_sh16"), ("Randomized/place_can_basket", 1.0, "robotwin_sh16"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin_sh16"), ("Randomized/place_cans_plasticbox", 1.0, "robotwin_sh16"),
        ("Clean/place_container_plate", 1.0, "robotwin_sh16"), ("Randomized/place_container_plate", 1.0, "robotwin_sh16"),
        ("Clean/place_dual_shoes", 1.0, "robotwin_sh16"), ("Randomized/place_dual_shoes", 1.0, "robotwin_sh16"),
        ("Clean/place_empty_cup", 1.0, "robotwin_sh16"), ("Randomized/place_empty_cup", 1.0, "robotwin_sh16"),
        ("Clean/place_fan", 1.0, "robotwin_sh16"), ("Randomized/place_fan", 1.0, "robotwin_sh16"),
        ("Clean/place_mouse_pad", 1.0, "robotwin_sh16"), ("Randomized/place_mouse_pad", 1.0, "robotwin_sh16"),
        ("Clean/place_object_basket", 1.0, "robotwin_sh16"), ("Randomized/place_object_basket", 1.0, "robotwin_sh16"),
        ("Clean/place_object_scale", 1.0, "robotwin_sh16"), ("Randomized/place_object_scale", 1.0, "robotwin_sh16"),
        ("Clean/place_object_stand", 1.0, "robotwin_sh16"), ("Randomized/place_object_stand", 1.0, "robotwin_sh16"),
        ("Clean/place_phone_stand", 1.0, "robotwin_sh16"), ("Randomized/place_phone_stand", 1.0, "robotwin_sh16"),
        ("Clean/place_shoe", 1.0, "robotwin_sh16"), ("Randomized/place_shoe", 1.0, "robotwin_sh16"),
        ("Clean/press_stapler", 1.0, "robotwin_sh16"), ("Randomized/press_stapler", 1.0, "robotwin_sh16"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin_sh16"), ("Randomized/put_bottles_dustbin", 1.0, "robotwin_sh16"),
        ("Clean/put_object_cabinet", 1.0, "robotwin_sh16"), ("Randomized/put_object_cabinet", 1.0, "robotwin_sh16"),
        ("Clean/rotate_qrcode", 1.0, "robotwin_sh16"), ("Randomized/rotate_qrcode", 1.0, "robotwin_sh16"),
        ("Clean/scan_object", 1.0, "robotwin_sh16"), ("Randomized/scan_object", 1.0, "robotwin_sh16"),
        ("Clean/shake_bottle", 1.0, "robotwin_sh16"), ("Randomized/shake_bottle", 1.0, "robotwin_sh16"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin_sh16"), ("Randomized/shake_bottle_horizontally", 1.0, "robotwin_sh16"),
        ("Clean/stack_blocks_three", 1.0, "robotwin_sh16"), ("Randomized/stack_blocks_three", 1.0, "robotwin_sh16"),
        ("Clean/stack_blocks_two", 1.0, "robotwin_sh16"), ("Randomized/stack_blocks_two", 1.0, "robotwin_sh16"),
        ("Clean/stack_bowls_three", 1.0, "robotwin_sh16"), ("Randomized/stack_bowls_three", 1.0, "robotwin_sh16"),
        ("Clean/stack_bowls_two", 1.0, "robotwin_sh16"), ("Randomized/stack_bowls_two", 1.0, "robotwin_sh16"),
        ("Clean/stamp_seal", 1.0, "robotwin_sh16"), ("Randomized/stamp_seal", 1.0, "robotwin_sh16"),
        ("Clean/turn_switch", 1.0, "robotwin_sh16"), ("Randomized/turn_switch", 1.0, "robotwin_sh16"),
    ],
    "robotwin_clean50_sh16": [
        ("Clean/adjust_bottle", 1.0, "robotwin_sh16"),
        ("Clean/beat_block_hammer", 1.0, "robotwin_sh16"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin_sh16"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin_sh16"),
        ("Clean/click_alarmclock", 1.0, "robotwin_sh16"),
        ("Clean/click_bell", 1.0, "robotwin_sh16"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin_sh16"),
        ("Clean/grab_roller", 1.0, "robotwin_sh16"),
        ("Clean/handover_block", 1.0, "robotwin_sh16"),
        ("Clean/handover_mic", 1.0, "robotwin_sh16"),
        ("Clean/hanging_mug", 1.0, "robotwin_sh16"),
        ("Clean/lift_pot", 1.0, "robotwin_sh16"),
        ("Clean/move_can_pot", 1.0, "robotwin_sh16"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin_sh16"),
        ("Clean/move_playingcard_away", 1.0, "robotwin_sh16"),
        ("Clean/move_stapler_pad", 1.0, "robotwin_sh16"),
        ("Clean/open_laptop", 1.0, "robotwin_sh16"),
        ("Clean/open_microwave", 1.0, "robotwin_sh16"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin_sh16"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin_sh16"),
        ("Clean/place_a2b_left", 1.0, "robotwin_sh16"),
        ("Clean/place_a2b_right", 1.0, "robotwin_sh16"),
        ("Clean/place_bread_basket", 1.0, "robotwin_sh16"),
        ("Clean/place_bread_skillet", 1.0, "robotwin_sh16"),
        ("Clean/place_burger_fries", 1.0, "robotwin_sh16"),
        ("Clean/place_can_basket", 1.0, "robotwin_sh16"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin_sh16"),
        ("Clean/place_container_plate", 1.0, "robotwin_sh16"),
        ("Clean/place_dual_shoes", 1.0, "robotwin_sh16"),
        ("Clean/place_empty_cup", 1.0, "robotwin_sh16"),
        ("Clean/place_fan", 1.0, "robotwin_sh16"),
        ("Clean/place_mouse_pad", 1.0, "robotwin_sh16"),
        ("Clean/place_object_basket", 1.0, "robotwin_sh16"),
        ("Clean/place_object_scale", 1.0, "robotwin_sh16"),
        ("Clean/place_object_stand", 1.0, "robotwin_sh16"),
        ("Clean/place_phone_stand", 1.0, "robotwin_sh16"),
        ("Clean/place_shoe", 1.0, "robotwin_sh16"),
        ("Clean/press_stapler", 1.0, "robotwin_sh16"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin_sh16"),
        ("Clean/put_object_cabinet", 1.0, "robotwin_sh16"),
        ("Clean/rotate_qrcode", 1.0, "robotwin_sh16"),
        ("Clean/scan_object", 1.0, "robotwin_sh16"),
        ("Clean/shake_bottle", 1.0, "robotwin_sh16"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin_sh16"),
        ("Clean/stack_blocks_three", 1.0, "robotwin_sh16"),
        ("Clean/stack_blocks_two", 1.0, "robotwin_sh16"),
        ("Clean/stack_bowls_three", 1.0, "robotwin_sh16"),
        ("Clean/stack_bowls_two", 1.0, "robotwin_sh16"),
        ("Clean/stamp_seal", 1.0, "robotwin_sh16"),
        ("Clean/turn_switch", 1.0, "robotwin_sh16"),
    ],
    "robotwin_clean50_pf32": [  # 50 tasks x 50 clean demos (README 50x50 setting)
        ("Clean/adjust_bottle", 1.0, "robotwin_pf32"),
        ("Clean/beat_block_hammer", 1.0, "robotwin_pf32"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin_pf32"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin_pf32"),
        ("Clean/click_alarmclock", 1.0, "robotwin_pf32"),
        ("Clean/click_bell", 1.0, "robotwin_pf32"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin_pf32"),
        ("Clean/grab_roller", 1.0, "robotwin_pf32"),
        ("Clean/handover_block", 1.0, "robotwin_pf32"),
        ("Clean/handover_mic", 1.0, "robotwin_pf32"),
        ("Clean/hanging_mug", 1.0, "robotwin_pf32"),
        ("Clean/lift_pot", 1.0, "robotwin_pf32"),
        ("Clean/move_can_pot", 1.0, "robotwin_pf32"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin_pf32"),
        ("Clean/move_playingcard_away", 1.0, "robotwin_pf32"),
        ("Clean/move_stapler_pad", 1.0, "robotwin_pf32"),
        ("Clean/open_laptop", 1.0, "robotwin_pf32"),
        ("Clean/open_microwave", 1.0, "robotwin_pf32"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin_pf32"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin_pf32"),
        ("Clean/place_a2b_left", 1.0, "robotwin_pf32"),
        ("Clean/place_a2b_right", 1.0, "robotwin_pf32"),
        ("Clean/place_bread_basket", 1.0, "robotwin_pf32"),
        ("Clean/place_bread_skillet", 1.0, "robotwin_pf32"),
        ("Clean/place_burger_fries", 1.0, "robotwin_pf32"),
        ("Clean/place_can_basket", 1.0, "robotwin_pf32"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin_pf32"),
        ("Clean/place_container_plate", 1.0, "robotwin_pf32"),
        ("Clean/place_dual_shoes", 1.0, "robotwin_pf32"),
        ("Clean/place_empty_cup", 1.0, "robotwin_pf32"),
        ("Clean/place_fan", 1.0, "robotwin_pf32"),
        ("Clean/place_mouse_pad", 1.0, "robotwin_pf32"),
        ("Clean/place_object_basket", 1.0, "robotwin_pf32"),
        ("Clean/place_object_scale", 1.0, "robotwin_pf32"),
        ("Clean/place_object_stand", 1.0, "robotwin_pf32"),
        ("Clean/place_phone_stand", 1.0, "robotwin_pf32"),
        ("Clean/place_shoe", 1.0, "robotwin_pf32"),
        ("Clean/press_stapler", 1.0, "robotwin_pf32"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin_pf32"),
        ("Clean/put_object_cabinet", 1.0, "robotwin_pf32"),
        ("Clean/rotate_qrcode", 1.0, "robotwin_pf32"),
        ("Clean/scan_object", 1.0, "robotwin_pf32"),
        ("Clean/shake_bottle", 1.0, "robotwin_pf32"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin_pf32"),
        ("Clean/stack_blocks_three", 1.0, "robotwin_pf32"),
        ("Clean/stack_blocks_two", 1.0, "robotwin_pf32"),
        ("Clean/stack_bowls_three", 1.0, "robotwin_pf32"),
        ("Clean/stack_bowls_two", 1.0, "robotwin_pf32"),
        ("Clean/stamp_seal", 1.0, "robotwin_pf32"),
        ("Clean/turn_switch", 1.0, "robotwin_pf32"),
    ],
    "robotwin_clean50": [  # 50 tasks x 50 clean demos (README 50x50 setting)
        ("Clean/adjust_bottle", 1.0, "robotwin"),
        ("Clean/beat_block_hammer", 1.0, "robotwin"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin"),
        ("Clean/click_alarmclock", 1.0, "robotwin"),
        ("Clean/click_bell", 1.0, "robotwin"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin"),
        ("Clean/grab_roller", 1.0, "robotwin"),
        ("Clean/handover_block", 1.0, "robotwin"),
        ("Clean/handover_mic", 1.0, "robotwin"),
        ("Clean/hanging_mug", 1.0, "robotwin"),
        ("Clean/lift_pot", 1.0, "robotwin"),
        ("Clean/move_can_pot", 1.0, "robotwin"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin"),
        ("Clean/move_playingcard_away", 1.0, "robotwin"),
        ("Clean/move_stapler_pad", 1.0, "robotwin"),
        ("Clean/open_laptop", 1.0, "robotwin"),
        ("Clean/open_microwave", 1.0, "robotwin"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin"),
        ("Clean/place_a2b_left", 1.0, "robotwin"),
        ("Clean/place_a2b_right", 1.0, "robotwin"),
        ("Clean/place_bread_basket", 1.0, "robotwin"),
        ("Clean/place_bread_skillet", 1.0, "robotwin"),
        ("Clean/place_burger_fries", 1.0, "robotwin"),
        ("Clean/place_can_basket", 1.0, "robotwin"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin"),
        ("Clean/place_container_plate", 1.0, "robotwin"),
        ("Clean/place_dual_shoes", 1.0, "robotwin"),
        ("Clean/place_empty_cup", 1.0, "robotwin"),
        ("Clean/place_fan", 1.0, "robotwin"),
        ("Clean/place_mouse_pad", 1.0, "robotwin"),
        ("Clean/place_object_basket", 1.0, "robotwin"),
        ("Clean/place_object_scale", 1.0, "robotwin"),
        ("Clean/place_object_stand", 1.0, "robotwin"),
        ("Clean/place_phone_stand", 1.0, "robotwin"),
        ("Clean/place_shoe", 1.0, "robotwin"),
        ("Clean/press_stapler", 1.0, "robotwin"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin"),
        ("Clean/put_object_cabinet", 1.0, "robotwin"),
        ("Clean/rotate_qrcode", 1.0, "robotwin"),
        ("Clean/scan_object", 1.0, "robotwin"),
        ("Clean/shake_bottle", 1.0, "robotwin"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin"),
        ("Clean/stack_blocks_three", 1.0, "robotwin"),
        ("Clean/stack_blocks_two", 1.0, "robotwin"),
        ("Clean/stack_bowls_three", 1.0, "robotwin"),
        ("Clean/stack_bowls_two", 1.0, "robotwin"),
        ("Clean/stamp_seal", 1.0, "robotwin"),
        ("Clean/turn_switch", 1.0, "robotwin"),
    ],
    "robotwin_all": [
        ("Clean/adjust_bottle", 1.0, "robotwin"), ("Randomized/adjust_bottle", 1.0, "robotwin"),
        ("Clean/beat_block_hammer", 1.0, "robotwin"), ("Randomized/beat_block_hammer", 1.0, "robotwin"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin"), ("Randomized/blocks_ranking_rgb", 1.0, "robotwin"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin"), ("Randomized/blocks_ranking_size", 1.0, "robotwin"),
        ("Clean/click_alarmclock", 1.0, "robotwin"), ("Randomized/click_alarmclock", 1.0, "robotwin"),
        ("Clean/click_bell", 1.0, "robotwin"), ("Randomized/click_bell", 1.0, "robotwin"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin"), ("Randomized/dump_bin_bigbin", 1.0, "robotwin"),
        ("Clean/grab_roller", 1.0, "robotwin"), ("Randomized/grab_roller", 1.0, "robotwin"),
        ("Clean/handover_block", 1.0, "robotwin"), ("Randomized/handover_block", 1.0, "robotwin"),
        ("Clean/handover_mic", 1.0, "robotwin"), ("Randomized/handover_mic", 1.0, "robotwin"),
        ("Clean/hanging_mug", 1.0, "robotwin"), ("Randomized/hanging_mug", 1.0, "robotwin"),
        ("Clean/lift_pot", 1.0, "robotwin"), ("Randomized/lift_pot", 1.0, "robotwin"),
        ("Clean/move_can_pot", 1.0, "robotwin"), ("Randomized/move_can_pot", 1.0, "robotwin"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin"), ("Randomized/move_pillbottle_pad", 1.0, "robotwin"),
        ("Clean/move_playingcard_away", 1.0, "robotwin"), ("Randomized/move_playingcard_away", 1.0, "robotwin"),
        ("Clean/move_stapler_pad", 1.0, "robotwin"), ("Randomized/move_stapler_pad", 1.0, "robotwin"),
        ("Clean/open_laptop", 1.0, "robotwin"), ("Randomized/open_laptop", 1.0, "robotwin"),
        ("Clean/open_microwave", 1.0, "robotwin"), ("Randomized/open_microwave", 1.0, "robotwin"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin"), ("Randomized/pick_diverse_bottles", 1.0, "robotwin"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin"), ("Randomized/pick_dual_bottles", 1.0, "robotwin"),
        ("Clean/place_a2b_left", 1.0, "robotwin"), ("Randomized/place_a2b_left", 1.0, "robotwin"),
        ("Clean/place_a2b_right", 1.0, "robotwin"), ("Randomized/place_a2b_right", 1.0, "robotwin"),
        ("Clean/place_bread_basket", 1.0, "robotwin"), ("Randomized/place_bread_basket", 1.0, "robotwin"),
        ("Clean/place_bread_skillet", 1.0, "robotwin"), ("Randomized/place_bread_skillet", 1.0, "robotwin"),
        ("Clean/place_burger_fries", 1.0, "robotwin"), ("Randomized/place_burger_fries", 1.0, "robotwin"),
        ("Clean/place_can_basket", 1.0, "robotwin"), ("Randomized/place_can_basket", 1.0, "robotwin"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin"), ("Randomized/place_cans_plasticbox", 1.0, "robotwin"),
        ("Clean/place_container_plate", 1.0, "robotwin"), ("Randomized/place_container_plate", 1.0, "robotwin"),
        ("Clean/place_dual_shoes", 1.0, "robotwin"), ("Randomized/place_dual_shoes", 1.0, "robotwin"),
        ("Clean/place_empty_cup", 1.0, "robotwin"), ("Randomized/place_empty_cup", 1.0, "robotwin"),
        ("Clean/place_fan", 1.0, "robotwin"), ("Randomized/place_fan", 1.0, "robotwin"),
        ("Clean/place_mouse_pad", 1.0, "robotwin"), ("Randomized/place_mouse_pad", 1.0, "robotwin"),
        ("Clean/place_object_basket", 1.0, "robotwin"), ("Randomized/place_object_basket", 1.0, "robotwin"),
        ("Clean/place_object_scale", 1.0, "robotwin"), ("Randomized/place_object_scale", 1.0, "robotwin"),
        ("Clean/place_object_stand", 1.0, "robotwin"), ("Randomized/place_object_stand", 1.0, "robotwin"),
        ("Clean/place_phone_stand", 1.0, "robotwin"), ("Randomized/place_phone_stand", 1.0, "robotwin"),
        ("Clean/place_shoe", 1.0, "robotwin"), ("Randomized/place_shoe", 1.0, "robotwin"),
        ("Clean/press_stapler", 1.0, "robotwin"), ("Randomized/press_stapler", 1.0, "robotwin"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin"), ("Randomized/put_bottles_dustbin", 1.0, "robotwin"),
        ("Clean/put_object_cabinet", 1.0, "robotwin"), ("Randomized/put_object_cabinet", 1.0, "robotwin"),
        ("Clean/rotate_qrcode", 1.0, "robotwin"), ("Randomized/rotate_qrcode", 1.0, "robotwin"),
        ("Clean/scan_object", 1.0, "robotwin"), ("Randomized/scan_object", 1.0, "robotwin"),
        ("Clean/shake_bottle", 1.0, "robotwin"), ("Randomized/shake_bottle", 1.0, "robotwin"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin"), ("Randomized/shake_bottle_horizontally", 1.0, "robotwin"),
        ("Clean/stack_blocks_three", 1.0, "robotwin"), ("Randomized/stack_blocks_three", 1.0, "robotwin"),
        ("Clean/stack_blocks_two", 1.0, "robotwin"), ("Randomized/stack_blocks_two", 1.0, "robotwin"),
        ("Clean/stack_bowls_three", 1.0, "robotwin"), ("Randomized/stack_bowls_three", 1.0, "robotwin"),
        ("Clean/stack_bowls_two", 1.0, "robotwin"), ("Randomized/stack_bowls_two", 1.0, "robotwin"),
        ("Clean/stamp_seal", 1.0, "robotwin"), ("Randomized/stamp_seal", 1.0, "robotwin"),
        ("Clean/turn_switch", 1.0, "robotwin"), ("Randomized/turn_switch", 1.0, "robotwin"),
    ],
    "robotwin_all_pf32": [  # same 50 tasks x {Clean, Randomized}, past+future window for the patch regressor
        ("Clean/adjust_bottle", 1.0, "robotwin_pf32"), ("Randomized/adjust_bottle", 1.0, "robotwin_pf32"),
        ("Clean/beat_block_hammer", 1.0, "robotwin_pf32"), ("Randomized/beat_block_hammer", 1.0, "robotwin_pf32"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin_pf32"), ("Randomized/blocks_ranking_rgb", 1.0, "robotwin_pf32"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin_pf32"), ("Randomized/blocks_ranking_size", 1.0, "robotwin_pf32"),
        ("Clean/click_alarmclock", 1.0, "robotwin_pf32"), ("Randomized/click_alarmclock", 1.0, "robotwin_pf32"),
        ("Clean/click_bell", 1.0, "robotwin_pf32"), ("Randomized/click_bell", 1.0, "robotwin_pf32"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin_pf32"), ("Randomized/dump_bin_bigbin", 1.0, "robotwin_pf32"),
        ("Clean/grab_roller", 1.0, "robotwin_pf32"), ("Randomized/grab_roller", 1.0, "robotwin_pf32"),
        ("Clean/handover_block", 1.0, "robotwin_pf32"), ("Randomized/handover_block", 1.0, "robotwin_pf32"),
        ("Clean/handover_mic", 1.0, "robotwin_pf32"), ("Randomized/handover_mic", 1.0, "robotwin_pf32"),
        ("Clean/hanging_mug", 1.0, "robotwin_pf32"), ("Randomized/hanging_mug", 1.0, "robotwin_pf32"),
        ("Clean/lift_pot", 1.0, "robotwin_pf32"), ("Randomized/lift_pot", 1.0, "robotwin_pf32"),
        ("Clean/move_can_pot", 1.0, "robotwin_pf32"), ("Randomized/move_can_pot", 1.0, "robotwin_pf32"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin_pf32"), ("Randomized/move_pillbottle_pad", 1.0, "robotwin_pf32"),
        ("Clean/move_playingcard_away", 1.0, "robotwin_pf32"), ("Randomized/move_playingcard_away", 1.0, "robotwin_pf32"),
        ("Clean/move_stapler_pad", 1.0, "robotwin_pf32"), ("Randomized/move_stapler_pad", 1.0, "robotwin_pf32"),
        ("Clean/open_laptop", 1.0, "robotwin_pf32"), ("Randomized/open_laptop", 1.0, "robotwin_pf32"),
        ("Clean/open_microwave", 1.0, "robotwin_pf32"), ("Randomized/open_microwave", 1.0, "robotwin_pf32"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin_pf32"), ("Randomized/pick_diverse_bottles", 1.0, "robotwin_pf32"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin_pf32"), ("Randomized/pick_dual_bottles", 1.0, "robotwin_pf32"),
        ("Clean/place_a2b_left", 1.0, "robotwin_pf32"), ("Randomized/place_a2b_left", 1.0, "robotwin_pf32"),
        ("Clean/place_a2b_right", 1.0, "robotwin_pf32"), ("Randomized/place_a2b_right", 1.0, "robotwin_pf32"),
        ("Clean/place_bread_basket", 1.0, "robotwin_pf32"), ("Randomized/place_bread_basket", 1.0, "robotwin_pf32"),
        ("Clean/place_bread_skillet", 1.0, "robotwin_pf32"), ("Randomized/place_bread_skillet", 1.0, "robotwin_pf32"),
        ("Clean/place_burger_fries", 1.0, "robotwin_pf32"), ("Randomized/place_burger_fries", 1.0, "robotwin_pf32"),
        ("Clean/place_can_basket", 1.0, "robotwin_pf32"), ("Randomized/place_can_basket", 1.0, "robotwin_pf32"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin_pf32"), ("Randomized/place_cans_plasticbox", 1.0, "robotwin_pf32"),
        ("Clean/place_container_plate", 1.0, "robotwin_pf32"), ("Randomized/place_container_plate", 1.0, "robotwin_pf32"),
        ("Clean/place_dual_shoes", 1.0, "robotwin_pf32"), ("Randomized/place_dual_shoes", 1.0, "robotwin_pf32"),
        ("Clean/place_empty_cup", 1.0, "robotwin_pf32"), ("Randomized/place_empty_cup", 1.0, "robotwin_pf32"),
        ("Clean/place_fan", 1.0, "robotwin_pf32"), ("Randomized/place_fan", 1.0, "robotwin_pf32"),
        ("Clean/place_mouse_pad", 1.0, "robotwin_pf32"), ("Randomized/place_mouse_pad", 1.0, "robotwin_pf32"),
        ("Clean/place_object_basket", 1.0, "robotwin_pf32"), ("Randomized/place_object_basket", 1.0, "robotwin_pf32"),
        ("Clean/place_object_scale", 1.0, "robotwin_pf32"), ("Randomized/place_object_scale", 1.0, "robotwin_pf32"),
        ("Clean/place_object_stand", 1.0, "robotwin_pf32"), ("Randomized/place_object_stand", 1.0, "robotwin_pf32"),
        ("Clean/place_phone_stand", 1.0, "robotwin_pf32"), ("Randomized/place_phone_stand", 1.0, "robotwin_pf32"),
        ("Clean/place_shoe", 1.0, "robotwin_pf32"), ("Randomized/place_shoe", 1.0, "robotwin_pf32"),
        ("Clean/press_stapler", 1.0, "robotwin_pf32"), ("Randomized/press_stapler", 1.0, "robotwin_pf32"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin_pf32"), ("Randomized/put_bottles_dustbin", 1.0, "robotwin_pf32"),
        ("Clean/put_object_cabinet", 1.0, "robotwin_pf32"), ("Randomized/put_object_cabinet", 1.0, "robotwin_pf32"),
        ("Clean/rotate_qrcode", 1.0, "robotwin_pf32"), ("Randomized/rotate_qrcode", 1.0, "robotwin_pf32"),
        ("Clean/scan_object", 1.0, "robotwin_pf32"), ("Randomized/scan_object", 1.0, "robotwin_pf32"),
        ("Clean/shake_bottle", 1.0, "robotwin_pf32"), ("Randomized/shake_bottle", 1.0, "robotwin_pf32"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin_pf32"), ("Randomized/shake_bottle_horizontally", 1.0, "robotwin_pf32"),
        ("Clean/stack_blocks_three", 1.0, "robotwin_pf32"), ("Randomized/stack_blocks_three", 1.0, "robotwin_pf32"),
        ("Clean/stack_blocks_two", 1.0, "robotwin_pf32"), ("Randomized/stack_blocks_two", 1.0, "robotwin_pf32"),
        ("Clean/stack_bowls_three", 1.0, "robotwin_pf32"), ("Randomized/stack_bowls_three", 1.0, "robotwin_pf32"),
        ("Clean/stack_bowls_two", 1.0, "robotwin_pf32"), ("Randomized/stack_bowls_two", 1.0, "robotwin_pf32"),
        ("Clean/stamp_seal", 1.0, "robotwin_pf32"), ("Randomized/stamp_seal", 1.0, "robotwin_pf32"),
        ("Clean/turn_switch", 1.0, "robotwin_pf32"), ("Randomized/turn_switch", 1.0, "robotwin_pf32"),
    ],
    "robotwin_all_50": [
        ("Clean/adjust_bottle", 1.0, "robotwin50"), ("Randomized/adjust_bottle", 1.0, "robotwin50"),
        ("Clean/beat_block_hammer", 1.0, "robotwin50"), ("Randomized/beat_block_hammer", 1.0, "robotwin50"),
        ("Clean/blocks_ranking_rgb", 1.0, "robotwin50"), ("Randomized/blocks_ranking_rgb", 1.0, "robotwin50"),
        ("Clean/blocks_ranking_size", 1.0, "robotwin50"), ("Randomized/blocks_ranking_size", 1.0, "robotwin50"),
        ("Clean/click_alarmclock", 1.0, "robotwin50"), ("Randomized/click_alarmclock", 1.0, "robotwin50"),
        ("Clean/click_bell", 1.0, "robotwin50"), ("Randomized/click_bell", 1.0, "robotwin50"),
        ("Clean/dump_bin_bigbin", 1.0, "robotwin50"), ("Randomized/dump_bin_bigbin", 1.0, "robotwin50"),
        ("Clean/grab_roller", 1.0, "robotwin50"), ("Randomized/grab_roller", 1.0, "robotwin50"),
        ("Clean/handover_block", 1.0, "robotwin50"), ("Randomized/handover_block", 1.0, "robotwin50"),
        ("Clean/handover_mic", 1.0, "robotwin50"), ("Randomized/handover_mic", 1.0, "robotwin50"),
        ("Clean/hanging_mug", 1.0, "robotwin50"), ("Randomized/hanging_mug", 1.0, "robotwin50"),
        ("Clean/lift_pot", 1.0, "robotwin50"), ("Randomized/lift_pot", 1.0, "robotwin50"),
        ("Clean/move_can_pot", 1.0, "robotwin50"), ("Randomized/move_can_pot", 1.0, "robotwin50"),
        ("Clean/move_pillbottle_pad", 1.0, "robotwin50"), ("Randomized/move_pillbottle_pad", 1.0, "robotwin50"),
        ("Clean/move_playingcard_away", 1.0, "robotwin50"), ("Randomized/move_playingcard_away", 1.0, "robotwin50"),
        ("Clean/move_stapler_pad", 1.0, "robotwin50"), ("Randomized/move_stapler_pad", 1.0, "robotwin50"),
        ("Clean/open_laptop", 1.0, "robotwin50"), ("Randomized/open_laptop", 1.0, "robotwin50"),
        ("Clean/open_microwave", 1.0, "robotwin50"), ("Randomized/open_microwave", 1.0, "robotwin50"),
        ("Clean/pick_diverse_bottles", 1.0, "robotwin50"), ("Randomized/pick_diverse_bottles", 1.0, "robotwin50"),
        ("Clean/pick_dual_bottles", 1.0, "robotwin50"), ("Randomized/pick_dual_bottles", 1.0, "robotwin50"),
        ("Clean/place_a2b_left", 1.0, "robotwin50"), ("Randomized/place_a2b_left", 1.0, "robotwin50"),
        ("Clean/place_a2b_right", 1.0, "robotwin50"), ("Randomized/place_a2b_right", 1.0, "robotwin50"),
        ("Clean/place_bread_basket", 1.0, "robotwin50"), ("Randomized/place_bread_basket", 1.0, "robotwin50"),
        ("Clean/place_bread_skillet", 1.0, "robotwin50"), ("Randomized/place_bread_skillet", 1.0, "robotwin50"),
        ("Clean/place_burger_fries", 1.0, "robotwin50"), ("Randomized/place_burger_fries", 1.0, "robotwin50"),
        ("Clean/place_can_basket", 1.0, "robotwin50"), ("Randomized/place_can_basket", 1.0, "robotwin50"),
        ("Clean/place_cans_plasticbox", 1.0, "robotwin50"), ("Randomized/place_cans_plasticbox", 1.0, "robotwin50"),
        ("Clean/place_container_plate", 1.0, "robotwin50"), ("Randomized/place_container_plate", 1.0, "robotwin50"),
        ("Clean/place_dual_shoes", 1.0, "robotwin50"), ("Randomized/place_dual_shoes", 1.0, "robotwin50"),
        ("Clean/place_empty_cup", 1.0, "robotwin50"), ("Randomized/place_empty_cup", 1.0, "robotwin50"),
        ("Clean/place_fan", 1.0, "robotwin50"), ("Randomized/place_fan", 1.0, "robotwin50"),
        ("Clean/place_mouse_pad", 1.0, "robotwin50"), ("Randomized/place_mouse_pad", 1.0, "robotwin50"),
        ("Clean/place_object_basket", 1.0, "robotwin50"), ("Randomized/place_object_basket", 1.0, "robotwin50"),
        ("Clean/place_object_scale", 1.0, "robotwin50"), ("Randomized/place_object_scale", 1.0, "robotwin50"),
        ("Clean/place_object_stand", 1.0, "robotwin50"), ("Randomized/place_object_stand", 1.0, "robotwin50"),
        ("Clean/place_phone_stand", 1.0, "robotwin50"), ("Randomized/place_phone_stand", 1.0, "robotwin50"),
        ("Clean/place_shoe", 1.0, "robotwin50"), ("Randomized/place_shoe", 1.0, "robotwin50"),
        ("Clean/press_stapler", 1.0, "robotwin50"), ("Randomized/press_stapler", 1.0, "robotwin50"),
        ("Clean/put_bottles_dustbin", 1.0, "robotwin50"), ("Randomized/put_bottles_dustbin", 1.0, "robotwin50"),
        ("Clean/put_object_cabinet", 1.0, "robotwin50"), ("Randomized/put_object_cabinet", 1.0, "robotwin50"),
        ("Clean/rotate_qrcode", 1.0, "robotwin50"), ("Randomized/rotate_qrcode", 1.0, "robotwin50"),
        ("Clean/scan_object", 1.0, "robotwin50"), ("Randomized/scan_object", 1.0, "robotwin50"),
        ("Clean/shake_bottle", 1.0, "robotwin50"), ("Randomized/shake_bottle", 1.0, "robotwin50"),
        ("Clean/shake_bottle_horizontally", 1.0, "robotwin50"), ("Randomized/shake_bottle_horizontally", 1.0, "robotwin50"),
        ("Clean/stack_blocks_three", 1.0, "robotwin50"), ("Randomized/stack_blocks_three", 1.0, "robotwin50"),
        ("Clean/stack_blocks_two", 1.0, "robotwin50"), ("Randomized/stack_blocks_two", 1.0, "robotwin50"),
        ("Clean/stack_bowls_three", 1.0, "robotwin50"), ("Randomized/stack_bowls_three", 1.0, "robotwin50"),
        ("Clean/stack_bowls_two", 1.0, "robotwin50"), ("Randomized/stack_bowls_two", 1.0, "robotwin50"),
        ("Clean/stamp_seal", 1.0, "robotwin50"), ("Randomized/stamp_seal", 1.0, "robotwin50"),
        ("Clean/turn_switch", 1.0, "robotwin50"), ("Randomized/turn_switch", 1.0, "robotwin50"),
    ],
    "robotwin": [
        ("adjust_bottle", 1.0, "robotwin"),
        ("beat_block_hammer", 1.0, "robotwin"),
        ("blocks_ranking_rgb", 1.0, "robotwin"),
        ("blocks_ranking_size", 1.0, "robotwin"),
        ("click_alarmclock", 1.0, "robotwin"),
        ("click_bell", 1.0, "robotwin"),
        ("dump_bin_bigbin", 1.0, "robotwin"),
        ("grab_roller", 1.0, "robotwin"),
        ("handover_block", 1.0, "robotwin"),
        ("handover_mic", 1.0, "robotwin"),
        ("hanging_mug", 1.0, "robotwin"),
        ("lift_pot", 1.0, "robotwin"),
        ("move_can_pot", 1.0, "robotwin"),
        ("move_pillbottle_pad", 1.0, "robotwin"),
        ("move_playingcard_away", 1.0, "robotwin"),
        ("move_stapler_pad", 1.0, "robotwin"),
        ("open_laptop", 1.0, "robotwin"),
        ("open_microwave", 1.0, "robotwin"),
        ("pick_diverse_bottles", 1.0, "robotwin"),
        ("pick_dual_bottles", 1.0, "robotwin"),
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
        ("place_bread_basket", 1.0, "robotwin"),
        ("place_bread_skillet", 1.0, "robotwin"),
        ("place_burger_fries", 1.0, "robotwin"),
        ("place_can_basket", 1.0, "robotwin"),
        ("place_cans_plasticbox", 1.0, "robotwin"),
        ("place_container_plate", 1.0, "robotwin"),
        ("place_dual_shoes", 1.0, "robotwin"),
        ("place_empty_cup", 1.0, "robotwin"),
        ("place_fan", 1.0, "robotwin"),
        ("place_mouse_pad", 1.0, "robotwin"),
        ("place_object_basket", 1.0, "robotwin"),
        ("place_object_scale", 1.0, "robotwin"),
        ("place_object_stand", 1.0, "robotwin"),
        ("place_phone_stand", 1.0, "robotwin"),
        ("place_shoe", 1.0, "robotwin"),
        ("press_stapler", 1.0, "robotwin"),
        ("put_bottles_dustbin", 1.0, "robotwin"),
        ("put_object_cabinet", 1.0, "robotwin"),
        ("rotate_qrcode", 1.0, "robotwin"),
        ("scan_object", 1.0, "robotwin"),
        ("shake_bottle", 1.0, "robotwin"),
        ("shake_bottle_horizontally", 1.0, "robotwin"),
        ("stack_blocks_three", 1.0, "robotwin"),
        ("stack_blocks_two", 1.0, "robotwin"),
        ("stack_bowls_three", 1.0, "robotwin"),
        ("stack_bowls_two", 1.0, "robotwin"),
        ("stamp_seal", 1.0, "robotwin"),
        ("turn_switch", 1.0, "robotwin"),
    ],
    "robotwin_task1": [("adjust_bottle", 1.0, "robotwin")],
    "robotwin_task2": [("place_a2b_left", 1.0, "robotwin"), ("place_a2b_right", 1.0, "robotwin")],
    "arx_x5": [("arx_x5", 1.0, "arx_x5")],
}

# QwenGR00T_DualSystem (Robotwin/train_files/dual_system): the 50x50 clean demos, and the RANDOMIZED demos of the same 50
# tasks as held-out validation (the clean set has no spare demos; domain-randomized scenes, so val RMSE also carries
# that domain gap)
for _suffix in ("dual_s2", "dual"):
    DATASET_NAMED_MIXTURES[f"robotwin_clean50_{_suffix}"] = [
        (name, weight, f"robotwin_{_suffix}") for name, weight, _ in DATASET_NAMED_MIXTURES["robotwin_clean50"]
    ]
    DATASET_NAMED_MIXTURES[f"robotwin_rand50_{_suffix}"] = [
        (name.replace("Clean/", "Randomized/"), weight, f"robotwin_{_suffix}")
        for name, weight, _ in DATASET_NAMED_MIXTURES["robotwin_clean50"]
    ]
