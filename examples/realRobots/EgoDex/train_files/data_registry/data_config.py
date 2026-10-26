"""EgoDex (human egocentric hand data, chest frame) data registry.

Dataset: LeRobot v3.0 conversion of Apple EgoDex (see egodex2lerobot.py).
State/action are identical 48-dim vectors: per hand, wrist pos (3) +
wrist rot6d (6) + five fingertip positions (5x3), all absolute in the
chest frame.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform

_HAND_KEYS = [
    "wrist_pos",
    "wrist_rot6d",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
]


class EgoDexHumanDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.ego"]
    state_keys = [f"state.{side}_{k}" for side in ("left", "right") for k in _HAND_KEYS]
    action_keys = [f"action.{side}_{k}" for side in ("left", "right") for k in _HAND_KEYS]
    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    # Must match framework.action_model.action_horizon in the train yaml.
    action_indices = list(range(30))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # rot6d components are already bounded in [-1, 1]; positions get
        # outlier-robust q99 normalization.
        def _norm_modes(keys):
            return {k: ("min_max" if "rot6d" in k else "q99") for k in keys}

        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes=_norm_modes(self.state_keys),
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes=_norm_modes(self.action_keys),
                ),
            ]
        )


class EgoDexHumanPastFutureDataConfig(EgoDexHumanDataConfig):
    """Bidirectional action window: H slots = H/2 PAST steps a[t-H/2..t-1]
    followed by H/2 FUTURE steps a[t..t+H/2-1]. action_mode delta is
    bidirectional (anchored at t: past slots are backward one-step diffs).
    """

    def __init__(self, window: int):
        assert window % 2 == 0, "window must be even (half past, half future)"
        self.window = window
        self.action_indices = list(range(-window // 2, window // 2))


class EgoDexHumanHistFrameDataConfig(EgoDexHumanDataConfig):
    """Current frame plus past reference frame(s): the video window is
    e.g. [-15, 0] (0.5 s ago and now at 30 fps). State / language stay at t
    and the action window is unchanged. Before the episode start the past
    slot is padded with the first frame, so the "past" image equals the
    current one there (no motion cue, which is the truth).
    """

    def __init__(self, past_frames=(-15,)):
        self.video_indices = sorted(set(int(p) for p in past_frames) | {0})

    def modality_config(self):
        cfg = super().modality_config()
        cfg["video"] = ModalityConfig(delta_indices=self.video_indices, modality_keys=self.video_keys)
        return cfg


ROBOT_TYPE_CONFIG_MAP = {
    "egodex_human": EgoDexHumanDataConfig(),
    **{f"egodex_human_pf{h}": EgoDexHumanPastFutureDataConfig(h) for h in (8, 16, 32, 64)},
    # current frame + frame t-15 (0.5 s ago) as a past reference image
    "egodex_human_hf15": EgoDexHumanHistFrameDataConfig((-15,)),
}

DATASET_NAMED_MIXTURES = {
    "egodex_test": [("egodex_lerobot_test", 1.0, "egodex_human")],
    "egodex_full": [("egodex_lerobot", 1.0, "egodex_human")],
    # trex_lerobot: T-Rex converted to the same 48-dim chest-frame layout
    # (trex2lerobot.py); symlinked into the egodex data root.
    "egodex_trex": [
        ("egodex_lerobot_test", 1.0, "egodex_human"),
        ("trex_lerobot", 1.0, "egodex_human"),
    ],
    "trex_only": [("trex_lerobot", 1.0, "egodex_human")],
    # Full EgoDex train split (part1-5 + extra) + TRex, weights proportional to
    # frame counts (in millions) so sampling matches one concatenated dataset.
    "egodex_full_trex": [
        ("egodex_lerobot_part1", 15.35, "egodex_human"),
        ("egodex_lerobot_part2", 15.65, "egodex_human"),
        ("egodex_lerobot_part3", 15.74, "egodex_human"),
        ("egodex_lerobot_part4", 15.94, "egodex_human"),
        ("egodex_lerobot_part5", 15.60, "egodex_human"),
        ("egodex_lerobot_extra", 10.47, "egodex_human"),
        ("trex_lerobot", 5.47, "egodex_human"),
    ],
    # Same mix, but repackaged with ONE mp4 PER EPISODE (convert_epsplit.py):
    # container open drops ~300ms -> ~5ms per sample, removing the dataloader
    # bottleneck of the merged v3.0 videos.
    "egodex_full_trex_epsplit": [
        ("egodex_lerobot_part1_epsplit", 15.35, "egodex_human"),
        ("egodex_lerobot_part2_epsplit", 15.65, "egodex_human"),
        ("egodex_lerobot_part3_epsplit", 15.74, "egodex_human"),
        ("egodex_lerobot_part4_epsplit", 15.94, "egodex_human"),
        ("egodex_lerobot_part5_epsplit", 15.60, "egodex_human"),
        ("egodex_lerobot_extra_epsplit", 10.47, "egodex_human"),
        ("trex_lerobot_epsplit", 5.47, "egodex_human"),
    ],
    # part1 + TRex only (per-episode files), weights proportional to frame counts:
    # 15.35M + 5.47M = 20.8M samples -> 50K steps @ global batch 512 = 1.23 epochs.
    "egodex_part1_trex_epsplit": [
        ("egodex_lerobot_part1_epsplit", 15.35, "egodex_human"),
        ("trex_lerobot_epsplit", 5.47, "egodex_human"),
    ],
    "egodex_part1_epsplit": [("egodex_lerobot_part1_epsplit", 1.0, "egodex_human")],
    # held-out test set (per-episode files) for evaluation, both window layouts
    "egodex_test_epsplit": [("egodex_lerobot_test_epsplit", 1.0, "egodex_human")],
    "egodex_test_pf64_epsplit": [("egodex_lerobot_test_epsplit", 1.0, "egodex_human_pf64")],
    # part1 only with a past+future action window of H steps (half/half).
    **{f"egodex_part1_pf{h}": [("egodex_lerobot_part1_epsplit", 1.0, f"egodex_human_pf{h}")] for h in (8, 16, 32, 64)},
    # two-frame input (t-15, t): train on part1, evaluate on the held-out test set
    "egodex_part1_hf15": [("egodex_lerobot_part1_epsplit", 1.0, "egodex_human_hf15")],
    "egodex_test_hf15": [("egodex_lerobot_test_epsplit", 1.0, "egodex_human_hf15")],
    # two-frame input on the full EgoDex train split + TRex (weights = frame counts in millions,
    # same as egodex_full_trex_epsplit): 4.5x the data of part1 for VLM-unfrozen runs.
    "egodex_full_trex_hf15": [
        ("egodex_lerobot_part1_epsplit", 15.35, "egodex_human_hf15"),
        ("egodex_lerobot_part2_epsplit", 15.65, "egodex_human_hf15"),
        ("egodex_lerobot_part3_epsplit", 15.74, "egodex_human_hf15"),
        ("egodex_lerobot_part4_epsplit", 15.94, "egodex_human_hf15"),
        ("egodex_lerobot_part5_epsplit", 15.60, "egodex_human_hf15"),
        ("egodex_lerobot_extra_epsplit", 10.47, "egodex_human_hf15"),
        ("trex_lerobot_epsplit", 5.47, "egodex_human_hf15"),
    ],
}
