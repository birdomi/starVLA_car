"""
Helix_VLA_PrevCmd: Helix_VLA with the PREVIOUS COMMAND a[t - 1][dims] (action-normalised) prepended to the state token,
as QwenGR00T_DualSystem dual_system.state_prev_action_dims. For the H1-2 car: helix.state_prev_action_dims [0, 1] = the
head command. The measured head state lags the command by ~0.4 s and is sample-and-hold (it changes on ~1/3 of the
frames during a look), while the previous command is known to the robot at every call.

Training: the data config's action indices start at -1 (a[t - 1 .. t + H - 1], H + 1 rows; the loader clamps t - 1 to
the episode's first frame, so frame 0 sees a[0] as its previous command). Row 0 -> previous command, rows 1.. = the
target chunk. Inference: example["prev_action"] = the last command sent ([P], action-normalised; at an episode start
the first frame's command, as the training fallback). framework.action_model.state_dim = loader state + P.
Everything else (frames, System 2 / System 1, DiT, loss, sampling) is Helix_VLA's.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.Helix_VLA import Helix_VLA, HelixVLADefaultConfig
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class HelixVLAPrevCmdDefaultConfig(HelixVLADefaultConfig):
    name: str = "Helix_VLA_PrevCmd"
    helix: dict = field(default_factory=lambda: {
        "s2_lag": 12,
        "s2_norm": True,
        "s1_vision": True,
        "use_prefix": True,
        "state_prev_action_dims": [0, 1],  # action dims of a[t - 1] prepended to the state (car: head command)
    })


@FRAMEWORK_REGISTRY.register("Helix_VLA_PrevCmd")
class Helix_VLA_PrevCmd(Helix_VLA):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        config = merge_framework_config(HelixVLAPrevCmdDefaultConfig, config)
        super().__init__(config, **kwargs)
        fw = self.config.framework
        dims = [int(d) for d in (fw.helix.get("state_prev_action_dims", None) or [])]
        action_dim, state_dim = int(fw.action_model.action_dim), int(fw.action_model.state_dim)
        if not dims or not all(0 <= d < action_dim for d in dims) or state_dim <= len(dims):
            raise ValueError(f"helix.state_prev_action_dims {dims} needs action dims < {action_dim} and "
                             f"action_model.state_dim ({state_dim}) = loader state dim + {len(dims)}")
        self.prev_action_dims = dims

    def _with_prev(self, examples: List[dict], training: bool) -> List[dict]:
        """state [T_s, D] -> [T_s, P + D] with the previous command in front; training also drops action row 0."""
        dims, out = self.prev_action_dims, []
        for ex in examples:
            ex = dict(ex)
            st = np.asarray(ex["state"], dtype=np.float32)
            st = st[None] if st.ndim == 1 else st
            if training:
                a = np.asarray(ex["action"], dtype=np.float32)
                if a.shape[0] != self.action_horizon + 1:
                    raise ValueError(f"Helix_VLA_PrevCmd: example['action'] has {a.shape[0]} rows, expected a[t - 1 .. t + "
                                     f"{self.action_horizon - 1}] = {self.action_horizon + 1} (data config action indices "
                                     "from -1)")
                pa, ex["action"] = a[0, dims], a[1:]
            else:
                if "prev_action" not in ex:
                    raise KeyError("Helix_VLA_PrevCmd: every example needs 'prev_action' (the previous command's dims "
                                   f"{dims}, action-normalised) at inference")
                pa = np.asarray(ex["prev_action"], dtype=np.float32).reshape(-1)
                if pa.shape != (len(dims),):
                    raise ValueError(f"prev_action of shape {pa.shape}, expected ({len(dims)},)")
            ex["state"] = np.concatenate([np.repeat(pa[None], st.shape[0], 0), st], axis=-1)
            out.append(ex)
        return out

    def forward(self, examples: List[dict] = None, **kwargs):
        return super().forward(self._with_prev(examples, training=True), **kwargs)

    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        examples = examples if isinstance(examples, list) else [examples]
        return super().predict_action(self._with_prev(examples, training=False), **kwargs)
