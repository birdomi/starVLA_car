# Thin façade over torch.optim.Muon + torch.optim.AdamW.
#
# All update math comes from PyTorch (torch>=2.12 ships torch.optim.Muon);
# this class only routes param groups so the trainer / lr scheduler /
# DeepSpeed see ONE optimizer object. torch.optim.Muon accepts only 2D
# matrix params, so non-matrix params (biases, norms, embeddings) go to
# AdamW — the standard Muon recipe.
#
# NOTE: Muon needs the FULL gradient of each matrix on every rank, so run
# under DDP or DeepSpeed ZeRO stage 0 (ACCELERATE_DEEPSPEED_ZERO_STAGE=0),
# never ZeRO 1/2/3 (they shard/flatten gradients).

import torch


class MuonAdamW(torch.optim.Optimizer):
    """Single-optimizer facade: Muon for matrix groups, AdamW for the rest.

    Each param group must carry ``use_muon: bool``. The child optimizers are
    built over the SAME group dicts as this facade, so scheduler updates to
    ``group["lr"]`` are seen by the children automatically.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group, "each param group needs use_muon"
        super().__init__(param_groups, defaults=dict(lr=1e-3))

        muon_groups = [g for g in self.param_groups if g["use_muon"]]
        adam_groups = [g for g in self.param_groups if not g["use_muon"]]
        self._muon = torch.optim.Muon(muon_groups) if muon_groups else None
        self._adam = torch.optim.AdamW(adam_groups) if adam_groups else None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._muon is not None:
            self._muon.step()
        if self._adam is not None:
            self._adam.step()
        return loss

    def zero_grad(self, set_to_none: bool = True):
        if self._muon is not None:
            self._muon.zero_grad(set_to_none=set_to_none)
        if self._adam is not None:
            self._adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self._muon.state_dict() if self._muon is not None else None,
            "adam": self._adam.state_dict() if self._adam is not None else None,
        }

    def load_state_dict(self, state_dict):
        if self._muon is not None and state_dict.get("muon") is not None:
            self._muon.load_state_dict(state_dict["muon"])
        if self._adam is not None and state_dict.get("adam") is not None:
            self._adam.load_state_dict(state_dict["adam"])
