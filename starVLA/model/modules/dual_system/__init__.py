"""Dual-System VLA modules (framework: starVLA/model/framework/VLM4A/QwenGR00T_DualSystem.py).

    backbone.py       SharedRadioBackbone   frozen C-RADIOv4, one instance shared by both systems
    diffusion.py      FlowMatchingScheduler (GR00T-style, default) / X0DiffusionScheduler / RegressionScheduler, build_scheduler
    interpolation.py  KeyframeTimeline, TrajectoryInterpolator, DenseTrajectoryLoss
    layers.py         temporal RoPE, DiT blocks (full bidirectional self-attention + cross-attention + adaLN-Zero)
    system2.py        System2KeyframePlanner (projectors, 61-keyframe DiT, dense-interpolation loss)
    memory.py         EpisodeMemory (recurrent memory over VLM hidden states + episode-progress classifier -> System 2)
    system1.py        TrajectoryTokenizer, System1ActionPolicy (1 s action DiT over [actions, 61 trajectory tokens];
                      optional ActionMap grid: (step, joint group) cells, GridTrajectoryTokenizer)
    runtime.py        AsyncDualSystemRunner (System 2 in a background thread; System 1 never waits for it),
                      AsyncVLMRunner (stale_input vlm: only the VLM in the background; System 2 DiT + System 1 per act)
"""

from .backbone import SharedRadioBackbone, VisualFeatures
from .diffusion import FlowMatchingScheduler, RegressionScheduler, X0DiffusionScheduler, build_scheduler
from .interpolation import DenseTrajectoryLoss, KeyframeTimeline, TrajectoryInterpolator
from .memory import EpisodeMemory
from .runtime import AsyncDualSystemRunner, AsyncVLMRunner, KeyframePlan, VLMCache
from .system1 import GridTrajectoryTokenizer, PlanNoise, System1ActionPolicy, TrajectoryTokenizer, parse_action_groups
from .system2 import System2KeyframePlanner

__all__ = [
    "SharedRadioBackbone", "VisualFeatures", "FlowMatchingScheduler", "X0DiffusionScheduler", "RegressionScheduler", "build_scheduler",
    "DenseTrajectoryLoss", "KeyframeTimeline", "TrajectoryInterpolator", "AsyncDualSystemRunner", "AsyncVLMRunner",
    "KeyframePlan", "VLMCache",
    "PlanNoise", "System1ActionPolicy", "TrajectoryTokenizer", "GridTrajectoryTokenizer", "parse_action_groups",
    "System2KeyframePlanner", "EpisodeMemory",
]
