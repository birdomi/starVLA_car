"""H1-2 car datasets (2026-08-20 collection, LeRobot v2.0) data registry.

Mirrors the Isaac-GR00T setup in examples/H1_2_car_turn_signal: three camera
views, 42-dim state/action over the groups shipped in each dataset's
meta/modality.json (head 2 + left_arm 7 + right_arm 7 + left_ee 13 +
right_ee 13; `left_arm`/`right_arm` are the arm slices of the 35-dim body
block, added to modality.json for relative actions). Head and arm actions are
trained RELATIVE to the matching state group (yaml: action_mode: rel), hand
(ee) actions stay absolute.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform

_REL_ACTION_KEYS = [  # relative (action - state[0]) in the train yaml
    "action.h1_2.action.head.joint_position",
    "action.h1_2.action.left_arm.joint_position",
    "action.h1_2.action.right_arm.joint_position",
]
_ABS_ACTION_KEYS = [
    "action.h1_2.action.left_ee.joint_position",
    "action.h1_2.action.right_ee.joint_position",
]


class H12CarDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.front_view",
        "video.left_wrist_view",
        "video.right_wrist_view",
    ]
    # Full raw 63-dim state in original order: head(2) + legs/waist(13) +
    # L arm(7) + R arm(7) + zero pad(8) + L hand(13) + R hand(13).
    # legs_waist / body_pad / left_arm / right_arm are non-overlapping slices
    # of the shipped 35-dim body block, added to meta/modality.json.
    state_keys = [
        "state.h1_2.observation.head.joint_position",
        "state.legs_waist",
        "state.left_arm",
        "state.right_arm",
        "state.body_pad",
        "state.h1_2.observation.left_ee.joint_position",
        "state.h1_2.observation.right_ee.joint_position",
    ]
    action_keys = [
        "action.h1_2.action.head.joint_position",
        "action.h1_2.action.left_arm.joint_position",
        "action.h1_2.action.left_ee.joint_position",
        "action.h1_2.action.right_arm.joint_position",
        "action.h1_2.action.right_ee.joint_position",
    ]
    language_keys = ["annotation.human.task_description"]

    observation_indices = [0]
    # Must match framework.action_model.action_horizon (16 steps @ 20 Hz = 0.8 s).
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # Absolute joint values are bounded -> min_max; relative arm/head deltas
        # get outlier-robust q99 (their stats come from the rel distribution).
        action_modes = {k: "q99" for k in _REL_ACTION_KEYS}
        action_modes.update({k: "min_max" for k in _ABS_ACTION_KEYS})
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={k: "min_max" for k in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=action_modes,
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "h1_2_car": H12CarDataConfig(),
}

DATASET_NAMED_MIXTURES = {
    # Weights ~ frame counts (92k / 99k / 239k) => uniform over frames.
    "car_3task": [
        ("left-turn-signal/train", 0.92, "h1_2_car"),
        ("right-turn-signal/train", 0.99, "h1_2_car"),
        ("car-interiorlight-onoff/train", 2.39, "h1_2_car"),
    ],
    # Equal sampling per task, regardless of dataset size.
    "car_3task_uniform": [
        ("left-turn-signal/train", 1.0, "h1_2_car"),
        ("right-turn-signal/train", 1.0, "h1_2_car"),
        ("car-interiorlight-onoff/train", 1.0, "h1_2_car"),
    ],
}
