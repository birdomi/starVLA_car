"""CPU checks for action-space density fitting and mode-path extraction."""

import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

from starVLA.model.modules.action_model.gaussian_action_field import GaussianActionField
from starVLA.model.modules.action_model.Gaussian_ActionHeader import GaussianActionHead
from starVLA.model.modules.action_model.GR00T_RegressorHeader import get_action_model


def tiny_config():
    return OmegaConf.create({"framework": {"action_model": {
        "head_type": "gaussian", "action_model_type": "DiT-B", "action_dim": 3,
        "state_dim": 3, "action_horizon": 7, "num_gaussians": 4,
        "hidden_size": 32, "num_hypotheses": 1, "num_target_vision_tokens": 0,
        "diffusion_model_cfg": {"num_attention_heads": 2, "attention_head_dim": 16,
            "output_dim": 32, "num_layers": 2, "cross_attention_dim": 12,
            "dropout": 0.0, "final_dropout": False, "positional_embeddings": None,
            "interleave_self_attention": True, "norm_elementwise_affine": True},
    }}})


class GaussianFieldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def test_density_matches_torch_distribution(self):
        field = GaussianActionField(3, 7, 4)
        p = torch.randn(2, 4, field.parameter_dim) * 0.2
        target = torch.randn(2, 7, 3)
        weight, mean, std = field.components(p)
        mixture = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=weight),
            torch.distributions.Independent(torch.distributions.Normal(mean, std[:, None]), 1),
        )
        torch.testing.assert_close(field.log_prob(p, target), mixture.log_prob(target))

    def test_single_primitive_recovers_linear_action(self):
        field = GaussianActionField(1, 5, 1)
        p = torch.zeros(1, 1, field.parameter_dim)
        p[..., 3], p[..., 4] = 2.0, 3.0
        expected = (2 + 3 * (field.times - 0.5))[None, :, None]
        torch.testing.assert_close(field.select_actions(p), expected)

    def test_modes_are_selected_not_averaged_and_gate_changes_action(self):
        field = GaussianActionField(1, 9, 2, initial_time_width=1.0)
        p = torch.zeros(1, 2, field.parameter_dim)
        p[..., 0] = -field.center_logits  # Both modes cover identical times.
        p[0, :, 3] = torch.tensor([-1.0, 1.0])
        result = field.select_actions(p)
        self.assertTrue((result.abs() > 0.99).all())
        self.assertTrue((result[:, 1:] == result[:, :-1]).all())
        p[0, 1, 2] = 5.0
        torch.testing.assert_close(field.select_actions(p), torch.ones(1, 9, 1))
        p[0, :, 2] = torch.tensor([5.0, 0.0])
        torch.testing.assert_close(field.select_actions(p), -torch.ones(1, 9, 1))

    def test_continuity_penalty_changes_mode_path(self):
        field = GaussianActionField(1, 7, 2, initial_time_width=0.2)
        p = torch.zeros(1, 2, field.parameter_dim)
        p[0, :, 3] = torch.tensor([-1.0, 1.0])
        independent = field.select_actions(p, continuity_weight=0)
        continuous = field.select_actions(p, continuity_weight=100)
        self.assertLess(independent[0, 0, 0], 0)
        self.assertGreater(independent[0, -1, 0], 0)
        self.assertTrue((continuous[:, 1:] == continuous[:, :-1]).all())

    def test_masked_nan_targets_have_finite_gradients(self):
        field = GaussianActionField(2, 4, 3)
        p = torch.randn(1, 3, field.parameter_dim, requires_grad=True)
        target = torch.randn(1, 4, 2)
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[:, -1] = False
        mask[:, 0, 1] = False
        target[~mask] = float("nan")
        likelihood = field.log_prob(p, target, mask)
        self.assertEqual(likelihood[0, -1].item(), 0)
        (-likelihood.sum()).backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        zero = field.log_prob(p, target, torch.zeros(1, 4, dtype=torch.bool))
        torch.testing.assert_close(zero, torch.zeros_like(zero))

    def test_autocast_keeps_density_and_actions_fp32(self):
        field = GaussianActionField(2, 5, 3)
        p = torch.zeros(1, 3, field.parameter_dim, dtype=torch.bfloat16, requires_grad=True)
        p.data[..., 1] = -100  # Time width hits its lower bound.
        p.data[..., -2:] = -100  # Action std hits its lower bound.
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = -field.log_prob(p, torch.ones(1, 5, 2)).mean()
            actions = field.select_actions(p)
        loss.backward()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(actions.dtype, torch.float32)
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_fit_shared_bank_to_two_demonstration_modes(self):
        field = GaussianActionField(1, 8, 2, initial_time_width=1.0)
        bank = torch.nn.Parameter(torch.zeros(1, 2, field.parameter_dim))
        with torch.no_grad():
            bank[..., 0] = -field.center_logits
            bank[0, :, 3] = torch.tensor([-0.4, 0.4])
        target = torch.tensor([-0.8, 0.8])[:, None, None].expand(-1, 8, 1)
        optimizer = torch.optim.Adam([bank], lr=0.03)
        initial = -field.log_prob(bank.expand(2, -1, -1), target).mean().item()
        for _ in range(120):
            optimizer.zero_grad()
            loss = -field.log_prob(bank.expand(2, -1, -1), target).mean()
            loss.backward()
            optimizer.step()
        final = -field.log_prob(bank.expand(2, -1, -1), target).mean().item()
        self.assertLess(final, initial - 1.0)
        self.assertLess((field.select_actions(bank).abs() - 0.8).abs().max().item(), 0.05)

    def test_head_context_bank_gradients_mask_and_checkpoint(self):
        cfg = tiny_config()
        head = get_action_model(cfg).eval()
        self.assertIsInstance(head, GaussianActionHead)
        context = torch.randn(2, 6, 12, requires_grad=True)
        state = torch.randn(2, 1, 3, requires_grad=True)
        target = torch.randn(2, 7, 3)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.bool)
        loss = head(context, target, state, attention_mask)
        loss.backward()
        for grad in (context.grad, state.grad, head.gaussian_bank.grad,
                     head.gaussian_queries.weight.grad, head.gaussian_decoder.layer2.weight.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
        self.assertEqual(context.grad[~attention_mask].abs().sum().item(), 0)
        action = head.predict_action(context.detach(), state.detach(), attention_mask)
        self.assertEqual(action.shape, target.shape)
        restored = get_action_model(cfg).eval()
        restored.load_state_dict(head.state_dict(), strict=True)
        torch.testing.assert_close(action, restored.predict_action(context.detach(), state.detach(), attention_mask))
        changed = context.detach().clone()
        changed[~attention_mask] = 1000
        torch.testing.assert_close(action, head.predict_action(changed, state.detach(), attention_mask))

    def test_unsupported_config_and_default_factory(self):
        cfg = tiny_config()
        cfg.framework.action_model.num_hypotheses = 8
        with self.assertRaisesRegex(ValueError, "num_hypotheses"):
            get_action_model(cfg)
        del cfg.framework.action_model.head_type
        with mock.patch("starVLA.model.modules.action_model.GR00T_RegressorHeader.RegressionActionHead") as old:
            self.assertIs(get_action_model(cfg), old.return_value)
        with self.assertRaises(ValueError):
            GaussianActionField(2, 3, initial_action_std=0.01, min_action_std=0.03)

    def test_observation_conditioned_head_learns_distinct_actions(self):
        torch.manual_seed(42)
        head = get_action_model(tiny_config())
        context = torch.randn(2, 6, 12)
        state = torch.randn(2, 1, 3)
        target = torch.tensor([-0.6, 0.6])[:, None, None].expand(-1, 7, 3)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0)
        initial = head(context, target, state).item()
        for _ in range(200):
            optimizer.zero_grad()
            loss = head(context, target, state)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10)
            optimizer.step()
        head.eval()
        prediction = head.predict_action(context, state)
        self.assertLess(head(context, target, state).item(), initial - 1)
        self.assertLess((prediction - target).square().mean().sqrt().item(), 0.08)


if __name__ == "__main__":
    unittest.main()
