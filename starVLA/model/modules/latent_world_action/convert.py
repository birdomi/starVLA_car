"""Checkpoint conversion: QwenGR00T_TriSystem System 1.5 -> QwenLatentWorldAction System 1.5.

The TriSystem ChangeFlowTransformer is the same network as LatentFlowDiT with system15.adaln_cond "tau+s2" and
cond_dim == d_model (same blocks, same conditioning, FinalLayer = FinalAdaNorm + velocity head), so its weights map
1:1. Only the System 1.5 weights and the C-RADIO feature statistics are taken; Qwen / C-RADIO weights are the
pretrained ones in both frameworks and the TriSystem action DiT has no counterpart here.
"""

from typing import Dict

import torch


def convert_trisystem_system15(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("system15.net."):
            r = k[len("system15.net."):]
            if r.startswith("transformer.blocks."):
                r = r[len("transformer."):]
            elif r.startswith("final.ada."):
                r = "final_norm.ada." + r[len("final.ada."):]
            elif r.startswith("final.out."):
                r = "velocity_head." + r[len("final.out."):]
            out["system15." + r] = v
        elif k.startswith("feature_stats."):
            out["radio_encoder.stats." + k[len("feature_stats."):]] = v
    if not any(k.startswith("system15.") for k in out):
        raise ValueError("no system15.net.* keys: not a QwenGR00T_TriSystem checkpoint")
    return out
