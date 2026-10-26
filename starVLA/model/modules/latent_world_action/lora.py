"""Minimal LoRA for fine-tuning System 2 (Qwen) without extra dependencies (peft is not installed in this env).

apply_lora(model, r, alpha, dropout, target_modules) replaces every nn.Linear child whose attribute name is in
target_modules by LoRALinear(base frozen + B @ A, B zero-initialised => the model is unchanged at step 0) and
freezes every other parameter of `model`. Parameter names become <...>.<name>.base.weight / .lora_A / .lora_B, so a
checkpoint trained with LoRA is reloaded by building the framework with the same system2.lora settings.
"""

import math
from typing import Iterable

import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, r, bias=False, device=base.weight.device)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, device=base.weight.device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = float(alpha) / float(r)
        self.drop = nn.Dropout(dropout)

    @property
    def weight(self):  # some HF code paths read .weight (dtype / device checks)
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.drop(x))) * self.scaling


def apply_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, dropout: float = 0.05,
               target_modules: Iterable[str] = ("q_proj", "k_proj", "v_proj", "o_proj")) -> int:
    """Inject LoRA adapters in place; returns the number of wrapped Linear layers."""
    targets = set(target_modules)
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, int(r), float(alpha), float(dropout)))
                n += 1
    if n == 0:
        raise ValueError(f"system2.lora.target_modules {sorted(targets)} matched no nn.Linear layer")
    return n
