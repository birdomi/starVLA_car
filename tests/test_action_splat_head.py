"""Coverage, differentiability and single-chunk conditional action learning."""

import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

from starVLA.model.modules.action_model.ActionSplatHead import ActionSplatHead
from starVLA.model.modules.action_model.GR00T_RegressorHeader import get_action_model
from starVLA.model.modules.action_model.temporal_action_splat import TemporalActionSplat


def tiny_config(**extra):
    return OmegaConf.create({"framework": {"action_model": {
        "head_type": "action_splat", "action_model_type": "DiT-B", "action_dim": 3,
        "state_dim": 3, "action_horizon": 7, "num_splat_points": 4,
        "hidden_size": 32, "num_target_vision_tokens": 0,
        "moe_expert_layers": 0, "loss_type": "l1", **extra,
        "diffusion_model_cfg": {"num_attention_heads": 2, "attention_head_dim": 16,
            "output_dim": 32, "num_layers": 2, "cross_attention_dim": 12,
            "dropout": 0.0, "final_dropout": False, "positional_embeddings": None,
            "interleave_self_attention": True, "norm_elementwise_affine": True},
    }}})


class ActionSplatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def test_constant_values_preserved_with_extreme_geometry_and_scores(self):
        for count in (1, 2, 32):
            splat = TemporalActionSplat(2, 101, count)
            points = torch.randn(2, 4, count, splat.parameter_dim) * 1000
            points[..., :2] = torch.tensor([2.0, -3.0])
            points[..., 3] = -1000  # Minimum radii; saturated time offsets.
            actual = splat(points)
            expected = torch.tensor([2.0, -3.0]).expand(2, 4, 101, 2)
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected)

    def test_triangular_overlap_and_importance(self):
        splat = TemporalActionSplat(1, 5, 2, initial_radius=1.0)
        points = torch.zeros(1, 2, 4)
        points[0, :, 0] = torch.tensor([-1.0, 1.0])
        expected = torch.tensor([-1., -1., 0., 1., 1.])[None, :, None]
        torch.testing.assert_close(splat(points), expected)
        points[0, 1, -1] = torch.log(torch.tensor(3.0))
        self.assertAlmostEqual(splat(points)[0, 2, 0].item(), 0.5, places=6)
        # The high-importance right point has zero support at t=0.
        points[0, 1, -1] = 10000
        self.assertAlmostEqual(splat(points)[0, 0, 0].item(), -1.0, places=6)

    def test_splat_gradcheck_and_all_point_fields_receive_gradient(self):
        splat = TemporalActionSplat(2, 7, 3).double()
        points = (torch.randn(1, 3, 5, dtype=torch.double) * 0.3).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(splat, (points,), eps=1e-6, atol=1e-4))
        (splat(points) - torch.randn(1, 7, 2, dtype=torch.double)).square().mean().backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertTrue((points.grad.abs().sum((0, 1)) > 0).all())

    def test_autocast_stability(self):
        splat = TemporalActionSplat(3, 7, 4)
        points = torch.randn(2, 3, 4, 6, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            prediction = splat(points)
            loss = prediction.square().mean()
        loss.backward()
        self.assertEqual(prediction.dtype, torch.float32)
        self.assertTrue(torch.isfinite(points.grad).all())

    def test_single_chunk_inference_and_direct_loss(self):
        head = get_action_model(tiny_config()).eval()
        self.assertEqual(head.point_bank.shape, (4, 6))
        self.assertFalse(hasattr(head, "score_head"))
        points = torch.zeros(1, 4, 6)
        points[..., :3] = 1
        with mock.patch.object(head, "_predict_points", return_value=points):
            context = torch.zeros(1, 2, 12)
            torch.testing.assert_close(head.predict_action(context), torch.ones(1, 7, 3))
            self.assertAlmostEqual(head(context, torch.zeros(1, 7, 3)).item(), 1.0, places=6)

    def test_masked_nan_targets_and_empty_samples(self):
        head = get_action_model(tiny_config())
        context, state = torch.randn(2, 6, 12), torch.randn(2, 1, 3)
        actions = torch.randn(2, 7, 3)
        mask = torch.ones_like(actions, dtype=torch.bool)
        mask[0] = False
        mask[1, -2:] = False
        mask[1, 1, 0] = False
        actions[~mask] = float("nan")
        loss = head(context, actions, state, action_mask=mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for p in head.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
        zero = head(context, actions, state, action_mask=torch.zeros(2, 7, dtype=torch.bool))
        self.assertEqual(zero.item(), 0)

    def test_framework_contract_context_gradients_and_checkpoint(self):
        cfg = tiny_config()
        head = get_action_model(cfg).eval()
        self.assertIsInstance(head, ActionSplatHead)
        context = torch.randn(2, 6, 12, requires_grad=True)
        state = torch.randn(2, 1, 3, requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
        loss = head(context, torch.randn(2, 7, 3), state, mask)
        loss.backward()
        for gradient in (context.grad, state.grad, head.point_bank.grad, head.point_decoder.layer2.weight.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertTrue((head.point_bank.grad.abs().sum(0) > 0).all())
        self.assertEqual(context.grad[~mask].abs().sum().item(), 0)
        prediction = head.predict_action(context.detach(), state.detach(), mask)
        self.assertEqual(prediction.shape, (2, 7, 3))
        restored = get_action_model(cfg).eval()
        restored.load_state_dict(head.state_dict(), strict=True)
        altered = context.detach().clone()
        altered[~mask] = 1000
        torch.testing.assert_close(prediction, restored.predict_action(altered, state.detach(), mask))

    def test_reconstruction_loss_reaches_geometry_without_score_head(self):
        head = get_action_model(tiny_config())
        head(torch.randn(2, 6, 12), torch.randn(2, 7, 3), torch.randn(2, 1, 3)).backward()
        self.assertFalse(hasattr(head, "score_head"))
        self.assertTrue((head.point_bank.grad[..., -3:].abs().sum(0) > 0).all())

    def test_observation_conditioned_actions_can_be_learned(self):
        head = get_action_model(tiny_config())
        context, state = torch.randn(2, 6, 12), torch.randn(2, 1, 3)
        target = torch.tensor([-0.6, 0.6])[:, None, None].expand(-1, 7, 3)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.002, weight_decay=0)
        initial = head(context, target, state).item()
        for _ in range(160):
            optimizer.zero_grad()
            loss = head(context, target, state)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10)
            optimizer.step()
        head.eval()
        self.assertLess(head(context, target, state).item(), initial * 0.2)
        self.assertLess((head.predict_action(context, state) - target).square().mean().sqrt().item(), 0.06)

    def test_invalid_geometry_config(self):
        for kwargs in ({"num_points": 0}, {"max_shift": 0.5}, {"temperature": 0}, {"initial_radius": 0.5}):
            with self.assertRaises(ValueError):
                TemporalActionSplat(3, 7, **kwargs)
        cfg = tiny_config()
        cfg.framework.action_model.moe_expert_layers = 1
        with self.assertRaisesRegex(ValueError, "moe_expert_layers"):
            get_action_model(cfg)

    # ------------------------------------------------------------ adaptive K
    def test_static_geometry_unchanged_by_capacity_api(self):
        splat = TemporalActionSplat(2, 9, 6)
        self.assertEqual(splat.initial_points, 6)
        self.assertTrue(splat.active.all())
        torch.testing.assert_close(splat.anchors, (torch.arange(6) + 0.5) / 6)

    def test_inactive_points_ignored_and_uncovered_slots_fall_back(self):
        splat = TemporalActionSplat(1, 11, num_points=8, initial_points=2)
        points = torch.zeros(1, 8, 4)
        points[0, :, 0] = torch.arange(8).float()
        points[0, :, 2] = -100  # minimum radii (0.8/8 = 0.1): coverage holes around the two grid anchors
        anchors = torch.zeros(8); anchors[:2] = torch.tensor([0.25, 0.75])
        active = torch.zeros(8, dtype=torch.bool); active[:2] = True
        out = splat(points, anchors=anchors, active=active)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(((out == 0) | (out == 1)).all())  # only active values appear
        self.assertEqual(out[0, 0, 0].item(), 0.0); self.assertEqual(out[0, -1, 0].item(), 1.0)  # nearest fallback
        out2, w = splat(points, anchors=anchors, active=active, return_weights=True)
        torch.testing.assert_close(w.sum(-1), torch.ones(1, 11)); self.assertEqual(w[..., 2:].abs().sum().item(), 0)

    def test_adaptive_densify_prune_and_checkpoint(self):
        cfg = tiny_config(num_splat_points=4, num_splat_points_max=12, splat_densify_from=1,
                          splat_densify_until=100, splat_densify_interval=1, splat_densify_quantile=0.5,
                          splat_max_new_per_interval=2, splat_prune_usage_ratio=0.0, splat_min_points=2)
        head = get_action_model(cfg)
        self.assertTrue(head.adaptive)
        self.assertEqual(head.point_active.sum().item(), 4)
        context, state, target = torch.randn(3, 6, 12), torch.randn(3, 1, 3), torch.randn(3, 7, 3)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
        for step in (1, 2):
            optimizer.zero_grad(); head(context, target, state).backward(); optimizer.step()
            self.assertGreater(head._stat_count.item(), 0)
            stats = head.adaptive_step(step, optimizer)
            self.assertEqual(head._stat_count.item(), 0)
            self.assertEqual(stats["splat/n_split"] + stats["splat/n_clone"], 2)
        self.assertEqual(head.point_active.sum().item(), 8)
        self.assertTrue(((head.point_anchor >= 0) & (head.point_anchor <= 1)).all())
        # optimizer moments of rewritten rows were reset
        exp_avg = optimizer.state[head.point_bank]["exp_avg"]
        self.assertGreater(exp_avg.abs().sum().item(), 0)
        loss = head(context, target, state); loss.backward(); self.assertTrue(torch.isfinite(loss))
        # prune everything down to min_points with an impossible usage threshold
        head.prune_usage_ratio = 1e9
        head.max_new_per_interval = 0
        head.adaptive_step(3, optimizer)
        self.assertEqual(head.point_active.sum().item(), 2)
        head.eval(); pred = head.predict_action(context, state); self.assertTrue(torch.isfinite(pred).all())
        restored = get_action_model(cfg); restored.load_state_dict(head.state_dict(), strict=True); restored.eval()
        torch.testing.assert_close(restored.point_active, head.point_active)
        torch.testing.assert_close(restored.predict_action(context, state), pred)
        # outside the densify window nothing happens
        head.adaptive_step(1000, optimizer); self.assertEqual(head.point_active.sum().item(), 2)

    def test_non_adaptive_head_never_changes_population(self):
        head = get_action_model(tiny_config())
        self.assertFalse(head.adaptive)
        before = head.point_active.clone()
        head(torch.randn(2, 6, 12), torch.randn(2, 7, 3), torch.randn(2, 1, 3)).backward()
        self.assertEqual(head.adaptive_step(500), head.last_wta_stats)
        torch.testing.assert_close(head.point_active, before)


if __name__ == "__main__":
    unittest.main()
