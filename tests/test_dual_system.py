"""Unit tests of the Dual-System VLA modules (CPU, no pretrained weights).

    PYTHONPATH=. python tests/test_dual_system.py        (unittest runner)
"""

import math
import threading
import time
import unittest

import torch

from starVLA.model.modules.dual_system import (AsyncDualSystemRunner, DenseTrajectoryLoss, FlowMatchingScheduler,
                                               KeyframeTimeline, PlanNoise, System1ActionPolicy,
                                               System2KeyframePlanner, TrajectoryInterpolator, VisualFeatures,
                                               RegressionScheduler, X0DiffusionScheduler)
from starVLA.model.modules.dual_system.layers import ContinuousRoPE

D_A, D_V, D_C, D_ST = 5, 24, 32, 6
GRID = (2, 3)


def approx(a, b, abs_tol=1e-6):
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=abs_tol)


class raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        assert et is not None and issubclass(et, self.exc), f"expected {self.exc.__name__}"
        assert self.match is None or self.match in str(ev), str(ev)
        return True


def _randomize_zero_init(module):
    """adaLN-Zero / zero-init output layers make an untrained DiT ignore its inputs; randomize them for flow tests."""
    torch.manual_seed(0)
    for p in module.parameters():
        if p.dim() >= 1 and torch.count_nonzero(p) == 0:
            torch.nn.init.normal_(p, std=0.05)


# ---------------------------------------------------------------- timeline / interpolation
def test_timeline_61_keyframes():
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    assert tl.num == 61
    assert tl.counts() == {"past": 20, "current": 1, "future": 40}
    assert approx(tl.times[0], -2.0) and approx(tl.times[-1], 4.0)
    assert tl.times[20].item() == 0.0
    with raises(ValueError):
        KeyframeTimeline(-2.0, 4.05, 0.1)


def test_linear_interpolation_is_exact_on_lines_and_differentiable():
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    b, q = 3, 50
    slope, icpt = torch.randn(b, 1, D_A), torch.randn(b, 1, D_A)
    kf = (slope * tl.times[None, :, None] + icpt).requires_grad_(True)
    qt = torch.rand(b, q) * 6.0 - 2.0
    out, ok = TrajectoryInterpolator(D_A)(kf, tl.times, qt)
    assert ok.all()
    torch.testing.assert_close(out, slope * qt[..., None] + icpt, atol=1e-5, rtol=1e-5)
    out[0, 0].sum().backward()
    nz = (kf.grad[0].abs().sum(-1) > 0).nonzero().flatten().tolist()
    assert 1 <= len(nz) <= 2 and kf.grad[1:].abs().sum() == 0  # only the bracketing keyframes get gradient


def test_no_extrapolation_and_strict_loss():
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    interp = TrajectoryInterpolator(D_A, strict=True)
    kf = torch.randn(1, tl.num, D_A)
    qt = torch.tensor([[-2.5, -2.0, 0.0, 4.0, 4.2]])
    out, ok = interp(kf, tl.times, qt)
    assert ok.tolist() == [[False, True, True, True, False]]
    assert out[0, 0].abs().sum() == 0 and out[0, -1].abs().sum() == 0
    loss = DenseTrajectoryLoss(interp)
    gt = torch.randn(1, 5, D_A)
    with raises(ValueError, match="outside the keyframe span"):
        loss(kf, tl.times, gt, qt, torch.ones(1, 5, dtype=torch.bool))
    l, info = loss(kf, tl.times, gt, qt, torch.tensor([[False, True, True, True, False]]))
    assert torch.isfinite(l) and approx(info["valid_frac"], 0.6)


def test_discrete_angle_and_quaternion_modes():
    times = torch.tensor([0.0, 1.0])
    groups = [{"dims": [0, 1], "mode": "step"}, {"dims": [1, 2], "mode": "nearest"},
              {"dims": [2, 3], "mode": "angle", "period": 2.0}, {"dims": [3, 7], "mode": "slerp"}]
    interp = TrajectoryInterpolator(8, groups)  # dim 7 -> default linear
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q1 = torch.tensor([0.0, 0.0, 0.0, 1.0])
    kf = torch.stack([torch.cat([torch.tensor([0.0, 0.0, 0.9]), q0, torch.tensor([0.0])]),
                      torch.cat([torch.tensor([1.0, 1.0, -0.9]), q1, torch.tensor([2.0])])])[None]
    out, _ = interp(kf, times, torch.tensor([[0.3, 0.7, 1.0]]))
    assert out[0, :, 0].tolist() == [0.0, 0.0, 1.0]  # zero-order hold
    assert out[0, :, 1].tolist() == [0.0, 1.0, 1.0]  # nearest
    # angle: 0.9 -> -0.9 with period 2 goes forward through the wrap point (shortest arc, +0.2), not back through 0
    assert approx(out[0, 0, 2], 0.9 + 0.3 * 0.2, 1e-5)
    assert approx(out[0, 1, 7], 1.4, 1e-5)  # linear
    quat = out[0, :, 3:7]
    torch.testing.assert_close(quat.norm(dim=-1), torch.ones(3))
    half = interp(kf, times, torch.tensor([[0.5]]))[0][0, 0, 3:7]
    torch.testing.assert_close(half, torch.tensor([2 ** -0.5, 0.0, 0.0, 2 ** -0.5]), atol=1e-5, rtol=1e-5)
    with raises(ValueError):
        TrajectoryInterpolator(4, [{"dims": [0, 3], "mode": "slerp"}])
    with raises(ValueError):
        TrajectoryInterpolator(4, [{"dims": [0, 2], "mode": "angle"}])


# ---------------------------------------------------------------- diffusion
def test_x0_ddim_with_oracle_recovers_x0():
    for eta in (0.0, 1.0):
        _check_ddim(eta)


def _check_ddim(eta):
    sch = X0DiffusionScheduler({"num_train_timesteps": 100, "sample_steps": 7, "eta": eta})
    x0 = torch.randn(4, 61, D_A)
    out = sch.sample(lambda x, t: x0, x0.shape, "cpu")
    torch.testing.assert_close(out, x0)
    xt, eps = sch.add_noise(x0, torch.full((4,), 99))
    a = sch.alphas_cumprod[99]
    torch.testing.assert_close(xt, a.sqrt() * x0 + (1 - a).sqrt() * eps)
    assert sch.inference_timesteps(7)[0] == 99 and sch.inference_timesteps(7)[-1] == 0


def test_gr00t_flow_matching_scheduler():
    sch = FlowMatchingScheduler({"sample_steps": 4})
    torch.manual_seed(0)
    tau = sch.sample_tau(20000, "cpu")
    assert float(tau.min()) >= 0 and float(tau.max()) <= 1 and float(tau.mean()) < 0.5  # GR00T: more weight near noise
    x1 = torch.randn(3, 61, D_A)
    x_tau, t, aux = sch.training_inputs(x1)
    tt = aux["tau"][:, None, None]
    torch.testing.assert_close(x_tau, (1 - tt) * aux["noise"] + tt * x1)
    torch.testing.assert_close(aux["target"], x1 - aux["noise"])
    assert t.dtype == torch.long and torch.equal(t, (aux["tau"] * 1000).long())
    # Euler with the exact velocity field of a point mass, (x1 - x) / (1 - tau), lands on x1 at the last step
    out = sch.sample(lambda x, t: (x1 - x) / (1 - t.float()[:, None, None] / 1000), x1.shape, "cpu")
    torch.testing.assert_close(out, x1, atol=1e-5, rtol=1e-5)


def test_regression_scheduler_and_system1_regression():
    sch = RegressionScheduler()
    x1 = torch.randn(3, 20, D_A)
    x, t, aux = sch.training_inputs(x1)
    assert sch.prediction == "x0" and not x.any() and not t.any() and t.dtype == torch.long and aux["target"] is x1
    assert torch.equal(sch.sample(lambda x, t: x + 1.5, x1.shape, "cpu"), torch.full_like(x1, 1.5))
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    cfg = {"d_model": 32, "depth": 2, "num_heads": 4, "use_state": True, "diffusion": {"type": "regression"}}
    s1 = System1ActionPolicy(D_C, GRID, D_ST, D_A, horizon=20, action_rate=20.0, timeline=tl, cfg=cfg,
                             num_prefix=N_PRE, prefix_dim=D_C)
    _randomize_zero_init(s1)
    b = 2
    kf, radio, st, delay = torch.randn(b, 61, D_A), _radio(b), torch.randn(b, 1, D_ST), torch.tensor([0.0, 0.3])
    a, valid = torch.randn(b, 20, D_A), torch.ones(b, 20, dtype=torch.bool)
    valid[1, 15:] = False
    s1.eval()
    out = s1.sample(kf, delay, radio, st)
    torch.testing.assert_close(out, s1.sample(kf, delay, radio, st, noise=torch.randn(b, 20, D_A), steps=7))  # deterministic
    loss, logs = s1.loss(kf, delay, radio, st, a, valid)
    m = valid[..., None].float()
    assert "system1/mse" in logs and approx(loss, ((out - a) ** 2 * m).sum() / (m.sum() * D_A), 1e-5)
    s1.train()
    s1.loss(kf, delay, radio, st, a, valid)[0].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in s1.tokenizer.parameters())


def test_smooth_time():
    from starVLA.model.modules.dual_system.system1 import smooth_time
    x = torch.randn(2, 30, D_A)
    assert torch.equal(smooth_time(x, 0.0), x)
    c = torch.full((2, 30, D_A), 0.7)
    torch.testing.assert_close(smooth_time(c, 6.0), c)
    lin = torch.arange(30.0)[None, :, None].expand(2, -1, D_A) * 0.1
    out = smooth_time(lin, 3.0)  # a Gaussian keeps a line away from the (replicated) edges
    assert out.shape == lin.shape
    torch.testing.assert_close(out[:, 9:21], lin[:, 9:21], atol=1e-5, rtol=1e-5)
    assert float(out[0, 0, 0]) > float(lin[0, 0, 0]) and float(out[0, -1, 0]) < float(lin[0, -1, 0])
    assert float(smooth_time(x, 3.0).diff(dim=1).abs().mean()) < float(x.diff(dim=1).abs().mean())


def test_system1_plan_as_action_input():
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    cfg = {"d_model": 32, "depth": 2, "num_heads": 4, "use_state": True, "diffusion": {"type": "regression"},
           "plan_input": "action", "plan_chunk_s": 0.5}
    s1 = System1ActionPolicy(D_C, GRID, D_ST, D_A, horizon=20, action_rate=20.0, timeline=tl, cfg=cfg,
                             num_prefix=N_PRE, prefix_dim=D_C)
    _randomize_zero_init(s1)
    assert s1.tokenizer is None and s1.plan_chunk_steps == 10 and s1.dit.noplan_embed is not None
    assert "10 with the smoothed plan chunk" in s1.describe_sequence()
    b = 2
    kf, radio, st, delay = torch.randn(b, 61, D_A), _radio(b), torch.randn(b, 1, D_ST), torch.zeros(b)
    chunk, a, valid = torch.randn(b, 10, D_A), torch.randn(b, 20, D_A), torch.ones(b, 20, dtype=torch.bool)
    s1.eval()
    out = s1.sample(kf, delay, radio, st, plan_chunk=chunk)
    assert out.shape == (b, 20, D_A)
    torch.testing.assert_close(out, s1.sample(torch.randn(b, 61, D_A), delay, radio, st, plan_chunk=chunk))  # kf unused
    assert not torch.allclose(out, s1.sample(kf, delay, radio, st, plan_chunk=chunk + 1.0))
    loss, logs = s1.loss(kf, delay, radio, st, a, valid, plan_chunk=chunk)
    assert "system1/mse" in logs and approx(loss, ((out - a) ** 2).mean(), 1e-5)
    with raises(ValueError, match="plan_chunk"):
        s1.sample(kf, delay, radio, st)
    s1.train()
    s1.loss(kf, delay, radio, st, a, valid, plan_chunk=chunk)[0].backward()
    assert s1.dit.noplan_embed.grad is not None and s1.dit.action_in.weight.grad.abs().sum() > 0
    with raises(ValueError, match="regression"):
        System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, dict(cfg, diffusion={"type": "flow_matching"}))
    with raises(ValueError, match="trajectory tokens"):
        System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, dict(cfg, traj_token_dropout=0.2))
    full = System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, dict(cfg, plan_chunk_s=1.0))
    assert full.dit.noplan_embed is None


def test_dense_velocity_loss_is_zero_for_the_true_velocity():
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    loss_fn = DenseTrajectoryLoss(TrajectoryInterpolator(D_A))
    b = 2
    x1, eps = torch.randn(b, tl.num, D_A), torch.randn(b, tl.num, D_A)
    dense_t = (torch.arange(-40, 81) / 20.0)[None].expand(b, -1)
    dense = TrajectoryInterpolator(D_A)(x1, tl.times, dense_t)[0]  # a dense trajectory the keyframes represent exactly
    valid = torch.ones(b, 121, dtype=torch.bool)
    l0, _ = loss_fn.velocity(x1 - eps, eps, tl.times, dense, dense_t, valid)
    l1, _ = loss_fn.velocity(x1 - eps + 0.1, eps, tl.times, dense, dense_t, valid)
    assert float(l0) < 1e-10 and approx(l1, 0.01, 1e-5)


# ---------------------------------------------------------------- RoPE
def test_continuous_rope_is_relative():
    rope = ContinuousRoPE(16, time_scale=20.0, base=1000.0)
    q, k = torch.randn(1, 2, 5, 16), torch.randn(1, 2, 5, 16)
    t = torch.rand(1, 5) * 3
    def scores(tt):
        c, s = rope.cos_sin(tt)
        return ContinuousRoPE.apply(q, c, s) @ ContinuousRoPE.apply(k, c, s).transpose(-1, -2)
    torch.testing.assert_close(scores(t), scores(t + 1.234), atol=1e-4, rtol=1e-4)
    assert not torch.allclose(scores(t), scores(t * 1.5), atol=1e-3)


# ---------------------------------------------------------------- systems
N_PRE = 2


def _radio(b):
    return VisualFeatures(torch.randn(b, GRID[0] * GRID[1], D_C), torch.ones(b, GRID[0] * GRID[1], dtype=torch.bool), GRID,
                          torch.randn(b, N_PRE, D_C))


def _s2(tl, **over):
    cfg = {"d_model": 32, "depth": 2, "num_heads": 4, "use_state": True, "num_noise_samples": 2,
           "diffusion": {"num_train_timesteps": 50, "sample_steps": 3}}
    cfg.update(over)
    return System2KeyframePlanner(D_V, D_C, GRID, D_ST, D_A, tl, cfg, num_prefix=N_PRE, prefix_dim=D_C)


def test_system2_context_concat_on_token_dim_loss_and_sample():
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    s2 = _s2(tl)
    b, lv = 2, 7
    vmask = torch.ones(b, lv, dtype=torch.bool)
    vmask[0, :2] = False  # left padding
    radio = _radio(b)
    ctx, cmask = s2.conditioner(torch.randn(b, lv, D_V), vmask, radio, torch.randn(b, 1, D_ST))
    # token-dim concat: [vlm (7), radio summary (2), radio patches (6), state (1)]
    assert ctx.shape == (b, lv + N_PRE + 6 + 1, 32) and cmask.shape == (b, lv + N_PRE + 6 + 1)
    assert cmask[0, :2].tolist() == [False, False] and bool(cmask[:, lv:].all())
    radio2 = VisualFeatures(radio.tokens, radio.valid, GRID, radio.prefix + torch.randn_like(radio.prefix))
    ctx2, _ = s2.conditioner(torch.randn(b, lv, D_V), vmask, radio2, torch.randn(b, 1, D_ST))
    assert not torch.allclose(ctx2[:, lv:lv + N_PRE], ctx[:, lv:lv + N_PRE])  # the prefix slots carry the summary tokens
    with raises(ValueError, match="prefix tokens"):
        s2.conditioner(torch.randn(b, lv, D_V), vmask, VisualFeatures(radio.tokens, radio.valid, GRID), torch.randn(b, 1, D_ST))
    rate = 20.0
    dense_t = (torch.arange(-40, 81) / rate)[None].expand(b, -1)
    dense = torch.randn(b, 121, D_A)
    valid = torch.ones(b, 121, dtype=torch.bool)
    valid[1, :10] = False  # before the episode start
    kf_gt = s2.extract_keyframes(dense, dense_t)
    torch.testing.assert_close(kf_gt, dense[:, ::2])  # 0.1 s keyframes = every 2nd step at 20 Hz
    loss, logs = s2.loss(ctx, cmask, dense, dense_t, valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert s2.conditioner.vlm_projector[1].weight.grad is not None
    assert s2.conditioner.state_projector[0].weight.grad is not None
    kf = s2.sample(ctx.detach(), cmask)
    assert kf.shape == (b, 61, D_A) and s2.scheduler.prediction == "velocity" and "system2/dense_fm_loss" in logs
    # the x0-prediction option still works
    s2x = _s2(tl, diffusion={"type": "x0", "num_train_timesteps": 50, "sample_steps": 3})
    ctx_x, cmask_x = s2x.conditioner(torch.randn(b, lv, D_V), vmask, radio, torch.randn(b, 1, D_ST))
    lx, logs_x = s2x.loss(ctx_x, cmask_x, dense, dense_t, valid)
    assert torch.isfinite(lx) and "system2/dense_mse" in logs_x and s2x.sample(ctx_x.detach(), cmask_x).shape == (b, 61, D_A)


def test_system1_full_attention_rope_delay_semantics():
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    cfg = {"d_model": 32, "depth": 2, "num_heads": 4, "abs_time_embed": False, "use_state": True,
           "diffusion": {"num_train_timesteps": 50, "sample_steps": 2}}
    s1 = System1ActionPolicy(D_C, GRID, D_ST, D_A, horizon=20, action_rate=20.0, timeline=tl, cfg=cfg,
                             num_prefix=N_PRE, prefix_dim=D_C)
    _randomize_zero_init(s1)
    s1.eval()
    b = 2
    kf, radio = torch.randn(b, 61, D_A), _radio(b)
    traj = s1.tokenizer(kf)
    st = torch.randn(b, 1, D_ST)
    ctx, cmask = s1.context(radio, st)
    assert ctx.shape == (b, N_PRE + 6 + 1, 32)  # [summary, patches, state]
    with raises(ValueError, match="state-conditioned"):
        s1.context(radio)
    x_t, t = torch.randn(b, 20, D_A), torch.full((b,), 10)
    a_t0, k_t0 = s1.timestamps(torch.zeros(b))
    base = s1.dit(x_t, t, traj, a_t0, k_t0, ctx, cmask)
    assert base.shape == (b, 20, D_A)
    # full bidirectional attention: the LAST future trajectory token changes the FIRST action token
    traj2 = traj.clone()
    traj2[:, -1] += torch.randn_like(traj2[:, -1])  # (a constant shift would be removed by the LayerNorm)
    assert not torch.allclose(s1.dit(x_t, t, traj2, a_t0, k_t0, ctx, cmask)[:, 0], base[:, 0])
    # RoPE only: shifting every timestamp by the same constant changes nothing ...
    torch.testing.assert_close(s1.dit(x_t, t, traj, a_t0 + 0.7, k_t0 + 0.7, ctx, cmask), base, atol=1e-4, rtol=1e-4)
    # ... while a plan age (trajectory tokens shifted relative to the action tokens) does
    a_t1, k_t1 = s1.timestamps(torch.full((b,), 0.4))
    torch.testing.assert_close(a_t1, a_t0)
    torch.testing.assert_close(k_t1, k_t0 - 0.4)
    assert not torch.allclose(s1.dit(x_t, t, traj, a_t1, k_t1, ctx, cmask), base, atol=1e-4)
    # the summary tokens reach the actions
    ctx_s, _ = s1.context(VisualFeatures(radio.tokens, radio.valid, GRID, radio.prefix + torch.randn_like(radio.prefix)), st)
    assert not torch.allclose(s1.dit(x_t, t, traj, a_t0, k_t0, ctx_s, cmask), base, atol=1e-4)
    # ... and so does the state
    ctx_st, _ = s1.context(radio, st + torch.randn_like(st))
    assert not torch.allclose(s1.dit(x_t, t, traj, a_t0, k_t0, ctx_st, cmask), base, atol=1e-4)
    # training loss / sampling
    s1.train()
    valid = torch.ones(b, 20, dtype=torch.bool)
    loss, _ = s1.loss(kf, torch.tensor([0.0, 0.2]), radio, st, torch.randn(b, 20, D_A), valid)
    loss.backward()
    assert s1.tokenizer.proj[0].weight.grad is not None and s1.radio_projector_s1[1].weight.grad is not None
    assert s1.state_projector[0].weight.grad is not None and s1.state_type.grad is not None
    assert s1.sample(kf, torch.zeros(b), radio, st).shape == (b, 20, D_A)


def test_cross_attention_rope_is_relative():
    from starVLA.model.modules.dual_system.layers import Attention
    torch.manual_seed(0)
    att, rope = Attention(32, 4), ContinuousRoPE(8, time_scale=20.0, base=1000.0)
    x, ctx = torch.randn(2, 5, 32), torch.randn(2, 7, 32)
    tq, tk = torch.rand(2, 5), torch.rand(2, 7) * 3 - 1
    out = lambda a, b: att(x, ctx, rope=rope.cos_sin(a), ctx_rope=rope.cos_sin(b))
    torch.testing.assert_close(out(tq, tk), out(tq + 0.9, tk + 0.9), atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out(tq, tk), out(tq, tk + 0.3), atol=1e-4)
    with raises(ValueError, match="both"):
        att(x, ctx, rope=rope.cos_sin(tq))
    with raises(ValueError, match="cross-attention only"):
        att(x, ctx_rope=rope.cos_sin(tq))


def test_system1_plan_as_condition_state_in_sequence():
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    base_cfg = {"d_model": 32, "depth": 2, "num_heads": 4, "abs_time_embed": False, "use_state": True,
                "diffusion": {"num_train_timesteps": 50, "sample_steps": 2}}
    with raises(ValueError, match="plan_input"):
        System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, dict(base_cfg, plan_input="cross"))
    cfg = dict(base_cfg, plan_input="condition", state_input="sequence")
    s1 = System1ActionPolicy(D_C, GRID, D_ST, D_A, horizon=20, action_rate=20.0, timeline=tl, cfg=cfg,
                             num_prefix=N_PRE, prefix_dim=D_C)
    _randomize_zero_init(s1)
    s1.eval()
    b = 2
    kf, radio, st = torch.randn(b, 61, D_A), _radio(b), torch.randn(b, 1, D_ST)
    traj = s1.tokenizer(kf)
    ctx, cmask = s1.context(radio, st)
    assert ctx.shape == (b, N_PRE + 6, 32)  # [summary, patches]: the state token is NOT in the memory
    assert s1.describe_sequence() == "[B, 20 + 1 state, 32]" and s1.describe_context(6) == "[B, 61 plan + 2 + 6, 32]"
    st_tok = s1.sequence_state(st)
    assert st_tok.shape == (b, 1, 32)
    lens = {}
    hooks = [s1.dit.stack.blocks[0].attn.register_forward_hook(lambda m, a, o: lens.__setitem__("self", a[0].shape[1])),
             s1.dit.stack.blocks[0].cross.register_forward_hook(lambda m, a, o: lens.__setitem__("cross", a[1].shape[1]))]
    x_t, t = torch.randn(b, 20, D_A), torch.full((b,), 10)
    a_t0, k_t0 = s1.timestamps(torch.zeros(b))
    base = s1.dit(x_t, t, traj, a_t0, k_t0, ctx, cmask, st_tok)
    for h in hooks:
        h.remove()
    assert base.shape == (b, 20, D_A)
    assert lens == {"self": 20 + 1, "cross": 61 + N_PRE + 6}  # X = [actions, state], memory = [plan, summary, patches]
    # the plan (as a condition), its age, the state token (in the sequence) and the summary tokens all reach the actions
    traj2 = traj.clone()
    traj2[:, -1] += torch.randn_like(traj2[:, -1])
    assert not torch.allclose(s1.dit(x_t, t, traj2, a_t0, k_t0, ctx, cmask, st_tok)[:, 0], base[:, 0])
    a_t1, k_t1 = s1.timestamps(torch.full((b,), 0.4))
    assert not torch.allclose(s1.dit(x_t, t, traj, a_t1, k_t1, ctx, cmask, st_tok), base, atol=1e-4)
    st_tok2 = s1.sequence_state(st + torch.randn_like(st))
    assert not torch.allclose(s1.dit(x_t, t, traj, a_t0, k_t0, ctx, cmask, st_tok2), base, atol=1e-4)
    ctx_s, _ = s1.context(VisualFeatures(radio.tokens, radio.valid, GRID, radio.prefix + torch.randn_like(radio.prefix)), st)
    assert not torch.allclose(s1.dit(x_t, t, traj, a_t0, k_t0, ctx_s, cmask, st_tok), base, atol=1e-4)
    # a masked-out patch is ignored (the prepended plan tokens do not shift the mask)
    radio_m = VisualFeatures(radio.tokens.clone(), radio.valid.clone(), GRID, radio.prefix)
    radio_m.valid[:, 3] = False
    ctx_m, cmask_m = s1.context(radio_m, st)
    ref = s1.dit(x_t, t, traj, a_t0, k_t0, ctx_m, cmask_m, st_tok)
    radio_m.tokens[:, 3] += 5.0
    ctx_m2, _ = s1.context(radio_m, st)
    torch.testing.assert_close(s1.dit(x_t, t, traj, a_t0, k_t0, ctx_m2, cmask_m, st_tok), ref, atol=1e-5, rtol=1e-5)
    # training / sampling; gradient reaches the tokenizer and the state token
    s1.train()
    loss, _ = s1.loss(kf, torch.tensor([0.0, 0.2]), radio, st, torch.randn(b, 20, D_A), torch.ones(b, 20, dtype=torch.bool))
    loss.backward()
    assert s1.tokenizer.proj[0].weight.grad is not None and s1.radio_projector_s1[1].weight.grad is not None
    assert s1.state_projector[0].weight.grad is not None and s1.state_type.grad is not None
    assert s1.sample(kf, torch.zeros(b), radio, st).shape == (b, 20, D_A)
    with raises(ValueError, match="state-conditioned"):
        s1.sample(kf, torch.zeros(b), radio, None)
    # plan AND state as conditions: X = [actions] only, memory = [plan, summary, patches, state]
    both = System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, dict(cfg, state_input="condition"),
                               num_prefix=N_PRE, prefix_dim=D_C)
    _randomize_zero_init(both)
    both.eval()
    assert both.sequence_state(st) is None and both.context(radio, st)[0].shape == (b, N_PRE + 6 + 1, 32)
    assert both.describe_sequence() == "[B, 20, 32]" and both.describe_context(6) == "[B, 61 plan + 2 + 6 + 1, 32]"
    lens = {}
    hooks = [both.dit.stack.blocks[0].attn.register_forward_hook(lambda m, a, o: lens.__setitem__("self", a[0].shape[1])),
             both.dit.stack.blocks[0].cross.register_forward_hook(lambda m, a, o: lens.__setitem__("cross", a[1].shape[1]))]
    out_b = both.sample(kf, torch.zeros(b), radio, st, noise=torch.zeros(b, 20, D_A))
    for h in hooks:
        h.remove()
    assert lens == {"self": 20, "cross": 61 + N_PRE + 6 + 1} and out_b.shape == (b, 20, D_A)
    assert not torch.allclose(both.sample(kf, torch.zeros(b), radio, st + 1.0, noise=torch.zeros(b, 20, D_A)), out_b, atol=1e-4)
    # ActionMap columns + per-column plan tokens as the condition
    am = System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, dict(cfg, action_map={"groups": ["0:2", "2:5"]}),
                             num_prefix=N_PRE, prefix_dim=D_C)
    assert am.describe_sequence().startswith("[B, 20 x 2 + 1 state, 32]") and "61 x 2 plan" in am.describe_context(6)
    assert am.sample(kf, torch.zeros(b), radio, st).shape == (b, 20, D_A)


def test_system1_action_map_grid():
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    groups = ["0:1", "1:2,4:5", [2, 3]]  # 3 columns, non-contiguous dims
    with raises(ValueError, match="partition"):
        System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, {"d_model": 32, "depth": 1, "num_heads": 4,
                                                                 "action_map": {"groups": ["0:2", "1:5"]}})
    for plan_grid in (True, False):
        cfg = {"d_model": 32, "depth": 2, "num_heads": 4, "abs_time_embed": False, "use_state": True,
               "diffusion": {"num_train_timesteps": 50, "sample_steps": 2},
               "action_map": {"groups": groups, "names": ["a", "b", "c"], "plan_grid": plan_grid}}
        s1 = System1ActionPolicy(D_C, GRID, D_ST, D_A, horizon=20, action_rate=20.0, timeline=tl, cfg=cfg,
                                 num_prefix=N_PRE, prefix_dim=D_C)
        _randomize_zero_init(s1)
        s1.eval()
        b = 2
        kf, radio, st = torch.randn(b, 61, D_A), _radio(b), torch.randn(b, 1, D_ST)
        traj = s1.tokenizer(kf)
        assert traj.shape == (b, 61 * (3 if plan_grid else 1), 32)
        ctx, cmask = s1.context(radio, st)
        x_t, t = torch.randn(b, 20, D_A), torch.full((b,), 10)
        a_t0, k_t0 = s1.timestamps(torch.zeros(b))
        assert s1.dit.embed_actions(x_t).shape == (b, 20 * 3, 32)
        base = s1.dit(x_t, t, traj, a_t0, k_t0, ctx, cmask)
        assert base.shape == (b, 20, D_A)
        # cells are row-major (step, group) and every cell of a step shares its timestamp: a common time shift is a no-op
        torch.testing.assert_close(s1.dit(x_t, t, traj, a_t0 + 0.7, k_t0 + 0.7, ctx, cmask), base, atol=1e-4, rtol=1e-4)
        a_t1, k_t1 = s1.timestamps(torch.full((b,), 0.4))
        assert not torch.allclose(s1.dit(x_t, t, traj, a_t1, k_t1, ctx, cmask), base, atol=1e-4)
        # dim d of the output comes from its own column's projection: zero one column's output layer -> only its dims vanish
        with torch.no_grad():
            s1.dit.group_final.outs[1].weight.zero_()
            s1.dit.group_final.outs[1].bias.zero_()
        out = s1.dit(x_t, t, traj, a_t0, k_t0, ctx, cmask)
        assert out[..., [1, 4]].abs().max() == 0 and out[..., [0, 2, 3]].abs().min() > 0
        if plan_grid:  # the plan of one column at one keyframe is a separate token: perturbing it changes the output
            kf2 = kf.clone()
            kf2[:, 40, 2:4] += 1.0
            assert not torch.allclose(s1.dit(x_t, t, s1.tokenizer(kf2), a_t0, k_t0, ctx, cmask), out, atol=1e-5)
        s1.train()
        loss, logs = s1.loss(kf, torch.tensor([0.0, 0.2]), radio, st, torch.randn(b, 20, D_A), torch.ones(b, 20, dtype=torch.bool))
        loss.backward()
        assert {"system1/fm_loss_a", "system1/fm_loss_b", "system1/fm_loss_c"} <= set(logs)
        # the dim-weighted mean of the column losses is the loss
        assert approx(sum(logs[f"system1/fm_loss_{n}"] * w for n, w in zip("abc", (1, 2, 2))) / D_A, loss.item(), 1e-5)
        assert s1.dit.group_in[2].weight.grad is not None and s1.dit.group_embed.grad is not None
        assert s1.sample(kf, torch.zeros(b), radio, st).shape == (b, 20, D_A)
        assert "x 3" in s1.describe_sequence()


def test_system1_action_map_noise_std():
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    groups = ["0:1", "1:2,4:5", [2, 3]]
    base = {"d_model": 32, "depth": 2, "num_heads": 4, "use_state": True, "plan_input": "condition",
            "diffusion": {"num_train_timesteps": 50, "sample_steps": 2}}
    mk = lambda **am: System1ActionPolicy(D_C, GRID, D_ST, D_A, horizon=20, action_rate=20.0, timeline=tl,
                                          cfg=dict(base, action_map=dict({"groups": groups}, **am)), num_prefix=N_PRE, prefix_dim=D_C)
    with raises(ValueError, match="noise_std"):
        mk(noise_std=[1.0, 0.5])
    with raises(ValueError, match="noise_std"):
        mk(noise_std=[1.0, 0.0, 1.0])
    with raises(ValueError, match="flow_matching"):
        System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, dict(base, diffusion={"type": "x0", "num_train_timesteps": 50},
                                                                     action_map={"groups": groups, "noise_std": [1, 1, 1]}))
    torch.manual_seed(0); ref = mk()
    torch.manual_seed(0); ones = mk(noise_std=[1.0, 1.0, 1.0])
    torch.manual_seed(0); low = mk(noise_std=[1.0, 0.2, 0.5])
    assert ref.noise_std is None and "noise_std" not in ref.state_dict() and "noise_std" not in low.state_dict()
    torch.testing.assert_close(low.noise_std, torch.tensor([1.0, 0.2, 0.5, 0.5, 0.2]))
    for m in (ones, low):  # same parameters at init (the buffer uses no RNG)
        for (k, p), (_, q) in zip(ref.state_dict().items(), m.state_dict().items()):
            assert torch.equal(p, q), k
    for m in (ref, ones, low):
        _randomize_zero_init(m) if m is ref else m.load_state_dict(ref.state_dict())
    b = 3
    kf, radio, st = torch.randn(b, 61, D_A), _radio(b), torch.randn(b, 1, D_ST)
    a, v, delay = torch.randn(b, 20, D_A), torch.ones(b, 20, dtype=torch.bool), torch.zeros(b)
    outs = {}
    for name, m in (("ref", ref), ("ones", ones), ("low", low)):
        m.eval()
        torch.manual_seed(1); loss = m.loss(kf, delay, radio, st, a, v)[0].item()
        torch.manual_seed(2); smp = m.sample(kf, delay, radio, st)
        outs[name] = (loss, smp)
    # sigma = 1 everywhere is the default path exactly (same RNG draws)
    assert outs["ones"][0] == outs["ref"][0] and torch.equal(outs["ones"][1], outs["ref"][1])
    assert outs["low"][0] != outs["ref"][0]
    # training: x_tau = (1 - tau) sigma eps + tau a, target a - sigma eps (dims of a sigma = 1 column untouched)
    seen = {}
    orig = low.scheduler.training_inputs

    def spy(x1, noise=None, tau=None):
        seen["n"] = noise
        return orig(x1, noise, tau)

    low.scheduler.training_inputs = spy
    torch.manual_seed(1); low.loss(kf, delay, radio, st, a, v)
    torch.manual_seed(1); eps = torch.randn(b, 20, D_A)
    torch.testing.assert_close(seen["n"], eps * low.noise_std)
    low.scheduler.training_inputs = orig
    # inference: the Euler start is sigma-scaled (a zero velocity field returns the start itself)
    with torch.no_grad():
        for p in low.dit.group_final.outs.parameters():
            p.zero_()
    torch.manual_seed(2); x0 = low.sample(kf, delay, radio, st)
    torch.manual_seed(2); eps = torch.randn(b, 20, D_A)
    torch.testing.assert_close(x0, eps * low.noise_std)


def test_multi_view_context_keeps_views_apart():
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    v, b, lv, n = 3, 2, 7, GRID[0] * GRID[1]
    cfg = {"d_model": 32, "depth": 2, "num_heads": 4, "use_state": True, "diffusion": {"num_train_timesteps": 50, "sample_steps": 2}}
    s2 = System2KeyframePlanner(D_V, D_C, GRID, D_ST, D_A, tl, cfg, num_prefix=N_PRE, prefix_dim=D_C, num_views=v)
    s1 = System1ActionPolicy(D_C, GRID, D_ST, D_A, horizon=15, action_rate=15.0, timeline=tl, cfg=cfg,
                             num_prefix=N_PRE, prefix_dim=D_C, num_views=v)
    # views concatenated on the token dim (view-major); the SAME image in every view still gives distinct tokens
    tok, pre = torch.randn(b, n, D_C), torch.randn(b, N_PRE, D_C)
    radio = VisualFeatures(tok.repeat(1, v, 1), torch.ones(b, v * n, dtype=torch.bool), GRID, pre.repeat(1, v, 1))
    vmask = torch.ones(b, lv, dtype=torch.bool)
    ctx, cmask = s2.conditioner(torch.randn(b, lv, D_V), vmask, radio, torch.randn(b, 1, D_ST))
    assert ctx.shape == (b, lv + v * N_PRE + v * n + 1, 32) and cmask.shape == ctx.shape[:2]
    p0 = lv + v * N_PRE
    views = ctx[:, p0:p0 + v * n].reshape(b, v, n, 32)
    assert not torch.allclose(views[:, 0], views[:, 1]) and not torch.allclose(views[:, 1], views[:, 2])
    assert not torch.allclose(ctx[:, lv:lv + N_PRE], ctx[:, lv + N_PRE:lv + 2 * N_PRE])  # per-(view, slot) prefix embedding
    with raises(ValueError, match="prefix tokens"):  # one view's prefix only
        s2.conditioner(torch.randn(b, lv, D_V), vmask, VisualFeatures(radio.tokens, radio.valid, GRID, pre), torch.randn(b, 1, D_ST))
    st = torch.randn(b, 1, D_ST)
    c1, _ = s1.context(radio, st)
    assert c1.shape == (b, v * N_PRE + v * n + 1, 32)
    loss, _ = s1.loss(torch.randn(b, 61, D_A), torch.tensor([0.0, 0.2]), radio, st, torch.randn(b, 15, D_A),
                      torch.ones(b, 15, dtype=torch.bool))
    loss.backward()
    assert s1.radio_pos.view_embed.grad is not None
    # single view keeps the old state dict (no per-view parameters)
    assert not any("view_embed" in k for k in _s2(tl).state_dict())
    assert s2.conditioner.radio_prefix_projector_s2.slot_embed.shape == (1, v * N_PRE, 32)


def test_framework_frame_layout_is_camera_major():
    """The loader packs "image" camera-major (every frame of camera 0, then camera 1, ...); the framework regroups it to
    [B, T, V, 3, H, W], gives Qwen / System 1 the first frame of every camera and merges frame_valid over cameras."""
    import numpy as np
    from PIL import Image

    from starVLA.model.framework.VLM4A.QwenGR00T_DualSystem import Qwen_GR00T_DualSystem

    m = Qwen_GR00T_DualSystem.__new__(Qwen_GR00T_DualSystem)
    torch.nn.Module.__init__(m)
    m.register_buffer("dense_times", torch.zeros(1), persistent=False)
    m.num_views, m.frame_size, m.video_offsets = 3, (4, 4), [0, 3]
    m.cur = m.video_offsets.index(0)  # as __init__ (frame t = the video offset 0)
    tag = lambda c, t: Image.fromarray(np.full((4, 4, 3), 10 * c + t, dtype=np.uint8))  # pixel value = 10 * camera + time
    ex = {"image": [tag(c, t) for c in range(3) for t in range(2)],
          "frame_valid": np.array([1, 0, 1, 1, 1, 0], dtype=bool)}  # camera 1 claims t=1 valid; the others do not
    f = m._frames([ex], 2)
    assert f.shape == (1, 2, 3, 3, 4, 4)
    for t in range(2):
        for c in range(3):
            assert round(float(f[0, t, c, 0, 0, 0]) * 255) == 10 * c + t
    assert [np.asarray(im)[0, 0, 0] for im in m._current_images(ex)] == [0, 10, 20]
    assert m._current_images({"image": [tag(c, 0) for c in range(3)]})[2].getpixel((0, 0))[0] == 20  # inference: V images
    assert m._frame_valid([ex]).tolist() == [[True, False]]
    with raises(ValueError, match="views"):
        m._frames([{"image": ex["image"][:5]}], 2)


def test_vlm_train_last_layers_freezes_all_but_the_last():
    """dual_system.vlm_train_last_layers: only the last n language-model layers + its final norm stay trainable, cast
    to float32; the rest of qwen_vl_interface frozen at its dtype; refused when trainer.freeze_modules lists the VLM."""
    from omegaconf import OmegaConf

    from starVLA.model.framework.VLM4A.QwenGR00T_DualSystem import Qwen_GR00T_DualSystem

    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(5, 4)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(6)])
            self.norm = torch.nn.LayerNorm(4)

    def build(freeze):
        iface = torch.nn.Module()
        iface.model = torch.nn.Module()
        iface.model.model = torch.nn.Module()
        iface.model.model.language_model = LM()
        iface.model.model.visual = torch.nn.Linear(4, 4)
        m = Qwen_GR00T_DualSystem.__new__(Qwen_GR00T_DualSystem)
        torch.nn.Module.__init__(m)
        m.qwen_vl_interface = iface.to(torch.bfloat16)
        m.config = OmegaConf.create({"trainer": {"freeze_modules": freeze}})
        return m

    m = build("radio_encoder")
    m._train_vlm_last_layers(2)
    tr = {n for n, p in m.qwen_vl_interface.named_parameters() if p.requires_grad}
    lm = "model.model.language_model"
    assert tr == {f"{lm}.layers.{i}.{w}" for i in (4, 5) for w in ("weight", "bias")} | {f"{lm}.norm.weight", f"{lm}.norm.bias"}
    for n, p in m.qwen_vl_interface.named_parameters():
        assert p.dtype == (torch.float32 if n in tr else torch.bfloat16), n
    with raises(ValueError, match="freeze_modules"):
        build("radio_encoder, qwen_vl_interface")._train_vlm_last_layers(2)
    with raises(ValueError, match="1..6"):
        build("")._train_vlm_last_layers(7)


def test_memory_from_window_is_the_training_unroll():
    """predict_action's memory window (memory_from_window) = the training unroll (_memory_context): same memory
    context; frames flagged invalid skip their update (episode start -> the initial memory); System 2 gets the VLM
    tokens of video offset index vlm_main_idx."""
    import numpy as np
    from PIL import Image

    from starVLA.model.framework.VLM4A.QwenGR00T_DualSystem import Qwen_GR00T_DualSystem
    from starVLA.model.modules.dual_system import EpisodeMemory

    torch.manual_seed(0)
    m = Qwen_GR00T_DualSystem.__new__(Qwen_GR00T_DualSystem)
    torch.nn.Module.__init__(m)
    m.register_buffer("dense_times", torch.zeros(1), persistent=False)
    m.num_views, m.frame_size, m.video_offsets = 1, (4, 4), [-6, -3, 0]
    m.cur, m.mem_idx, m.vlm_main_idx, m.check_shapes = 2, [0, 1], 1, False
    m.episode_memory = EpisodeMemory(8, 16, num_tokens=4, d_model=16, depth=1, num_heads=2, progress_bins=5).eval()

    def encode_vlm(images, instructions, sc, with_grad):  # 3 tokens per image, token value = its pixel value + channel
        v = torch.tensor([float(np.asarray(ims[0])[0, 0, 0]) for ims in images])
        return v[:, None, None] + torch.arange(8.0).expand(len(images), 3, 8), torch.ones(len(images), 3, dtype=torch.bool)

    m.encode_vlm = encode_vlm
    img = lambda val: Image.fromarray(np.full((4, 4, 3), val, dtype=np.uint8))
    exs = [{"image": [img(10), img(20), img(30)], "lang": "a", "frame_valid": np.array([True, True, True])},
           {"image": [img(30), img(30), img(30)], "lang": "b", "frame_valid": np.array([False, False, True])}]
    mem, (vt, vm) = m.memory_from_window(exs)
    ctx, logits, _ = m._memory_context(exs, None)
    ref_ctx, ref_logits = m.episode_memory.readout(mem)
    assert torch.allclose(ctx, ref_ctx) and torch.allclose(logits, ref_logits)
    assert torch.equal(mem[1], m.memory_init(2)[1])
    assert not torch.equal(mem[0], m.memory_init(2)[0])
    assert vt[:, 0, 0].tolist() == [20.0, 30.0] and vm.all()


def test_gated_memory_attention():
    """GatedAttention = nn.MultiheadAttention (same weights) times a sigmoid gate on every head channel before the
    output projection; padded VLM tokens never reach the memory; the gated EpisodeMemory runs and trains end to end."""
    from starVLA.model.modules.dual_system import EpisodeMemory
    from starVLA.model.modules.dual_system.memory import GatedAttention

    torch.manual_seed(0)
    d, h = 16, 4
    ga, mha = GatedAttention(d, h), torch.nn.MultiheadAttention(d, h, batch_first=True)
    with torch.no_grad():
        for i, p in enumerate((ga.q_proj, ga.k_proj, ga.v_proj)):
            p.weight.copy_(mha.in_proj_weight[i * d:(i + 1) * d]); p.bias.copy_(mha.in_proj_bias[i * d:(i + 1) * d])
        ga.out_proj.load_state_dict(mha.out_proj.state_dict())
        ga.gate_proj.weight.zero_(); ga.gate_proj.bias.fill_(30.0)  # gate = 1
    q, kv = torch.randn(2, 3, d), torch.randn(2, 7, d)
    pad = torch.zeros(2, 7, dtype=torch.bool); pad[1, 4:] = True
    ref = mha(q, kv, kv, key_padding_mask=pad, need_weights=False)[0]
    assert torch.allclose(ga(q, kv, key_padding_mask=pad), ref, atol=1e-5)
    with torch.no_grad():
        ga.gate_proj.bias.fill_(-30.0)  # gate = 0 -> only the output bias is left
    assert torch.allclose(ga(q, kv, key_padding_mask=pad), ga.out_proj.bias.expand(2, 3, d), atol=1e-5)

    mem = EpisodeMemory(8, 12, num_tokens=4, d_model=16, depth=2, num_heads=4, progress_bins=5, attention="gated")
    assert all(blk.gated and isinstance(blk.attn, GatedAttention) for blk in mem.blocks)
    x, xm = torch.randn(2, 6, 8), torch.ones(2, 6, dtype=torch.bool)
    xm[1, 3:] = False
    m1 = mem.update(mem.init(2), x, xm)
    x2 = x.clone(); x2[1, 3:] = 1e3  # padded tokens changed -> the same memory
    assert torch.allclose(mem.update(mem.init(2), x2, xm), m1, atol=1e-5)
    ctx, logits = mem.readout(m1)
    assert ctx.shape == (2, 5, 12) and logits.shape == (2, 5)
    (ctx.square().mean() + logits.square().mean()).backward()
    assert all(blk.attn.gate_proj.weight.grad.abs().sum() > 0 for blk in mem.blocks)
    assert mem.init_tokens.grad.abs().sum() > 0
    assert not EpisodeMemory(8, 12, num_tokens=4, d_model=16, depth=1, num_heads=4).blocks[0].gated  # default: standard
    with raises(ValueError):
        EpisodeMemory(8, 12, d_model=16, num_heads=4, attention="gate")


# ---------------------------------------------------------------- async runtime
def test_plan_noise_matches_its_error_model():
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    assert not PlanNoise(None, tl.num, 6).enabled  # off by default
    ts = torch.linspace(0.5, 1.5, tl.num)
    ds = torch.tensor([0.1, 0.2, 0.3, 0.3, 0.3, 0.05])
    pn = PlanNoise({"scale_max": 3 ** 0.5, "white_frac": 0.35, "rho": 0.91, "time_std": ts.tolist(),
                    "dim_std": ds.tolist(), "shared_dims": [[2, 3, 4]]}, tl.num, 6)
    g = torch.Generator().manual_seed(0)
    n = pn.sample(20000, "cpu", generator=g)
    assert n.shape == (20000, tl.num, 6)
    std = n.pow(2).mean(0).sqrt()  # E[s^2] = 1 -> std = time_std x dim_std
    assert ((std / (ts[:, None] * ds[None]) - 1).abs() < 0.06).all()
    z = n / (ts[:, None] * ds[None])
    lag1 = (z[:, 1:] * z[:, :-1]).mean() / z.pow(2).mean()
    lag10 = (z[:, 10:] * z[:, :-10]).mean() / z.pow(2).mean()
    assert approx(float(lag1), 0.65 * 0.91, 0.02) and approx(float(lag10), 0.65 * 0.91 ** 10, 0.02)
    assert torch.equal(z[..., 2], z[..., 3]) and torch.equal(z[..., 2], z[..., 4])  # one shared error per group
    assert float((z[..., 0] * z[..., 1]).mean().abs()) < 0.02  # other dims independent
    assert float(PlanNoise({"scale_max": 1.0, "prob": 0.0}, tl.num, 6).sample(4, "cpu").abs().max()) == 0.0
    with raises(ValueError):
        PlanNoise({"scale_max": 1.0, "dim_std": [0.1] * 5}, tl.num, 6)


def test_trajectory_token_dropout():
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    cfg = {"d_model": 32, "depth": 1, "num_heads": 4, "use_state": True, "num_noise_samples": 2,
           "diffusion": {"num_train_timesteps": 50, "sample_steps": 2}}
    assert System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, cfg, num_prefix=N_PRE,
                               prefix_dim=D_C).traj_mask_token is None  # off by default: no new parameter
    with raises(ValueError):
        System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, dict(cfg, traj_token_dropout=1.0))
    with raises(ValueError, match="plan_grid"):
        System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, dict(cfg, traj_token_dropout=0.2,
                                                                     action_map={"groups": ["0:2", "2:5"]}))
    s1 = System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, dict(cfg, traj_token_dropout=0.3),
                             num_prefix=N_PRE, prefix_dim=D_C)
    _randomize_zero_init(s1)
    traj = s1.tokenizer(torch.randn(400, 61, D_A))
    out, drop = s1.drop_trajectory_tokens(traj)
    assert approx(float(drop.float().mean()), 0.3, 0.02)
    masked = s1.traj_mask_token[0] + s1.tokenizer.type_embed(s1.tokenizer.types)  # [K, d], same for every sample
    torch.testing.assert_close(out[drop], masked.expand(400, -1, -1)[drop])
    assert torch.equal(out[~drop], traj[~drop])
    # training: the mask token gets a gradient; inference never drops (sampling is deterministic given the noise)
    b = 2
    kf, radio, st = torch.randn(b, 61, D_A), _radio(b), torch.randn(b, 1, D_ST)
    s1.train()
    loss, logs = s1.loss(kf, torch.tensor([0.0, 0.4]), radio, st, torch.randn(b, 20, D_A), torch.ones(b, 20, dtype=torch.bool))
    loss.backward()
    assert s1.traj_mask_token.grad is not None and "system1/traj_drop_frac" in logs
    s1.eval()
    noise = torch.randn(b, 20, D_A)
    torch.testing.assert_close(s1.sample(kf, torch.zeros(b), radio, st, noise=noise),
                               s1.sample(kf, torch.zeros(b), radio, st, noise=noise), atol=0, rtol=0)


def test_system1_state_drop_dims():
    tl = KeyframeTimeline(-2.0, 4.0, 0.1)
    cfg = {"d_model": 32, "depth": 1, "num_heads": 4, "use_state": True,
           "diffusion": {"num_train_timesteps": 50, "sample_steps": 2}}
    with raises(ValueError, match="state_drop_dims"):
        System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, dict(cfg, state_drop_dims=[D_ST]))
    torch.manual_seed(0)
    full = System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, cfg, num_prefix=N_PRE, prefix_dim=D_C)
    torch.manual_seed(0)
    s1 = System1ActionPolicy(D_C, GRID, D_ST, D_A, 20, 20.0, tl, dict(cfg, state_drop_dims=[0, 1]),
                             num_prefix=N_PRE, prefix_dim=D_C)
    assert full.state_keep is None and "state_keep" not in s1.state_dict()  # same parameters / checkpoint keys
    assert full.state_dict().keys() == s1.state_dict().keys()
    st = torch.randn(3, 1, D_ST)
    st2 = st.clone()
    st2[..., :2] = torch.randn(3, 1, 2)
    torch.testing.assert_close(s1.state_token(st), s1.state_token(st2), atol=0, rtol=0)  # dims 0, 1 are not seen
    st3 = st.clone()
    st3[..., 2] += 1.0
    assert not torch.allclose(s1.state_token(st), s1.state_token(st3))  # the others are
    zeroed = st.clone()
    zeroed[..., :2] = 0
    torch.testing.assert_close(s1.state_token(st), full.state_token(zeroed), atol=0, rtol=0)


class _FakeModel:
    """System 2 takes 0.3 s; System 1 is instant; keyframes encode the observation id."""

    _device = torch.device("cpu")
    max_plan_age_s, train_delay_range_s = 3.0, (0.0, 0.4)

    def __init__(self):
        self.s2_calls = 0
        self.s2_threads = set()

    def encode_visual(self, examples):
        return VisualFeatures(torch.zeros(len(examples), 1, 1), torch.ones(len(examples), 1, dtype=torch.bool), (1, 1))

    def plan_keyframes(self, examples, radio=None):
        self.s2_threads.add(threading.current_thread().name)
        time.sleep(0.3)
        self.s2_calls += 1
        return torch.full((len(examples), 61, 1), float(examples[0]["id"]))

    def act_from_plan(self, examples, keyframes, delay_s, radio=None):
        return keyframes[:, :20] + 0 * delay_s[:, None, None]


def test_async_runner_never_waits_for_system2():
    clock = {"t": 0.0}
    model = _FakeModel()
    runner = AsyncDualSystemRunner(model, clock=lambda: clock["t"])
    with runner:
        a, info = runner.act([{"id": 0}], t_now=0.0)  # first call: waits for the first plan only
        assert info["plan_seq"] == 1 and info["delay_s"] == 0.0 and a[0, 0, 0].item() == 0.0
        lat = []
        for i in range(1, 6):
            t0 = time.perf_counter()
            a, info = runner.act([{"id": i}], t_now=0.1 * i)
            lat.append(time.perf_counter() - t0)
        assert max(lat) < 0.1  # System 1 returns immediately while System 2 (0.3 s) is in flight
        assert info["delay_s"] >= 0.0 and info["delay_in_training_range"] in (True, False)
        time.sleep(0.8)
        a, info = runner.act([{"id": 9}], t_now=2.0, submit=False)
        assert info["plan_seq"] >= 2 and approx(info["delay_s"], 2.0 - info["plan_obs_time"])
    assert model.s2_threads == {"dual-system-s2"}
    assert runner.dropped >= 1  # stale submissions are replaced by newer ones, not queued


class DualSystemTests(unittest.TestCase):
    pass


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(DualSystemTests, _name, staticmethod(_fn))
del _name, _fn

if __name__ == "__main__":
    unittest.main(verbosity=2)
