"""LatentFlowSampler: ODE integration of System 1.5 from noise (tau = 0) to the latent change (tau = 1).

    X_0 ~ N(0, I),   dX/dtau = v_theta(X, tau | Z_t, S),   X_1 = dZ_hat (scaled)

Fixed-step solvers from tri_system.flow_matching.ode_solve: euler (1 NFE / step), heun / midpoint (2), rk4 (4).
The flow timestep tau is generation time only; it is unrelated to the video time between Z_t and Z_f.

Gradients flow through the solver only if the caller enables them (the joint stage's `action_grad: full`);
otherwise wrap the call in torch.no_grad().
"""

from typing import Optional, Tuple

import torch

from starVLA.model.modules.tri_system.flow_matching import ode_solve

from .latent_flow_dit import FlowConditions, LatentFlowDiT

SOLVERS = ("euler", "heun", "midpoint", "rk4")


class LatentFlowSampler:
    def __init__(self, steps: int = 4, solver: str = "euler"):
        if solver not in SOLVERS:
            raise ValueError(f"system15.solver must be one of {SOLVERS}, got {solver!r}")
        self.steps, self.solver = int(steps), solver

    def initial_noise(self, cond: FlowConditions, feat_dim: int, num_tokens: int, device, dtype=torch.float32,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return torch.randn(cond.batch_size, num_tokens, feat_dim, device=device, dtype=dtype, generator=generator)

    def sample(self, model: LatentFlowDiT, cond: FlowConditions, noise: Optional[torch.Tensor] = None,
               steps: Optional[int] = None, solver: Optional[str] = None, generator: Optional[torch.Generator] = None,
               record_last_hidden: bool = False):
        """-> X_1 [B, N, D_c] (scaled latent change). With record_last_hidden also returns the hidden state of the
        LAST velocity evaluation inside the solver (diagnostics only: System 1 never reads it, see
        QwenLatentWorldAction.extract_final_hidden)."""
        steps = self.steps if steps is None else int(steps)
        solver = self.solver if solver is None else solver
        if solver not in SOLVERS:
            raise ValueError(f"unknown solver {solver!r}")
        device = cond.contexts["zt"][0].device
        x0 = noise if noise is not None else self.initial_noise(cond, model.feat_dim, model.num_tokens, device,
                                                                generator=generator)
        last: dict = {}

        def velocity(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
            if record_last_hidden:
                v, last["h"] = model(x, tau, cond, return_hidden_states=True)
                return v
            return model(x, tau, cond)

        x1 = ode_solve(velocity, x0, steps=steps, method=solver)
        return (x1, last["h"]) if record_last_hidden else x1
