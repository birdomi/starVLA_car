"""Unit tests for PatchRegressionActionHead (CPU only, small DiT)."""
import unittest

import torch
from omegaconf import OmegaConf

from starVLA.model.modules.action_model.GR00T_PatchRegressorHeader import PatchRegressionActionHead


def _head(**over):
    am = {
        "action_model_type": "DiT-B", "hidden_size": 256, "action_dim": 29, "state_dim": 58, "action_horizon": 16,
        "num_past_steps": 16, "patch_t": 4, "patch_groups": [7, 7, 6, 6, 3], "past_repr": "from_start", "use_rope": True,
        "loss_type": "l1",
        "diffusion_model_cfg": {"cross_attention_dim": 64, "dropout": 0.0, "final_dropout": True, "interleave_self_attention": True,
                                "num_layers": 2, "output_dim": 128, "positional_embeddings": None, "norm_elementwise_affine": True},
    }
    am.update(over)
    return PatchRegressionActionHead(OmegaConf.create({"framework": {"action_model": am}})).eval()


class TestPatchRegressorHead(unittest.TestCase):
    def test_patchify_roundtrip(self):
        h = _head()
        x = torch.randn(3, 16, 29)
        self.assertTrue(torch.equal(h.unpatchify(h.patchify(x, 4), 4), x))
        self.assertEqual(len(h.patchify(x, 4)), 4 * 5)

    def test_past_from_start(self):
        h = _head()
        a = torch.randn(2, 17, 29)
        d = a[:, :16] - a[:, 1:17]  # loader convention for past slots
        rel = h.past_to_from_start(d)
        self.assertTrue(torch.allclose(rel, a[:, :16] - a[:, :1], atol=1e-5))

    def test_forward_and_predict_shapes(self):
        h = _head()
        vl = torch.randn(2, 12, 64)
        actions, state = torch.randn(2, 32, 29), torch.randn(2, 1, 58)
        loss = h(vl, actions, state)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(tuple(h.predict_action(vl, state, past_actions=actions[:, :16]).shape), (2, 16, 29))
        self.assertEqual(tuple(h.predict_action(vl, state).shape), (2, 16, 29))
        self.assertEqual(h.rope.cos.shape[-2], 1 + 20 + 20)

    def test_rope_time_shift_invariance(self):
        """Axial RoPE depends on relative time only: shifting every time coordinate must not change outputs."""
        h = _head()
        vl, state, actions = torch.randn(1, 8, 64), torch.randn(1, 1, 58), torch.randn(1, 32, 29)
        y0 = h.predict_action(vl, state, past_actions=actions[:, :16])
        h.t_coords = h.t_coords + 37.0
        y1 = h.predict_action(vl, state, past_actions=actions[:, :16])
        self.assertTrue(torch.allclose(y0, y1, atol=1e-4), float((y0 - y1).abs().max()))

    def test_no_past_config(self):
        h = _head(num_past_steps=0)
        vl, state, actions = torch.randn(2, 8, 64), torch.randn(2, 1, 58), torch.randn(2, 16, 29)
        self.assertEqual(h.n_past_tokens, 0)
        self.assertEqual(h(vl, actions, state).ndim, 0)

    def test_image2image_tokens_and_grid_mapping(self):
        """Past patch tokens are the queries: sequence = state + 16 grid cells, each decoded to the same (pt, g) cell."""
        head = _head(token_mode="image2image", past_repr="raw", target_mode="abs")
        self.assertIsNone(head.future_query)
        vl = torch.randn(2, 6, 64); st = torch.randn(2, 1, 58); past = torch.randn(2, 16, 29)
        toks = head._build_tokens(2, st, past, vl.dtype, vl.device)
        self.assertEqual(toks.shape[1], 1 + head.n_past_tokens)
        self.assertEqual(head.t_coords.shape[0], 1 + head.n_past_tokens)
        pred = head.predict_action(vl, st, past_actions=past)
        self.assertEqual(pred.shape, (2, 16, 29))
        # locality: perturbing past cell (pt=1, g=0) must change the output; with rope the mapping is position-aware
        past2 = past.clone(); past2[:, 4:8, :7] += 3.0
        self.assertGreater((head.predict_action(vl, st, past_actions=past2) - pred).abs().sum().item(), 0)
        loss = head.train()(vl, torch.cat([past, torch.randn(2, 16, 29)], 1), st); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None))

    def test_image2image_requires_equal_windows(self):
        with self.assertRaises(AssertionError):
            _head(token_mode="image2image", num_past_steps=8)


if __name__ == "__main__":
    unittest.main()


class TestStateHistoryImage2Image(unittest.TestCase):
    def _h(self):
        return _head(action_dim=14, state_dim=14, patch_groups=[6, 6, 1, 1], token_mode="image2image",
                     past_source="state", past_repr="raw", output_residual=True, target_mode="abs")

    def test_residual_zero_init_repeats_state_grid(self):
        h = self._h()
        vl, st = torch.randn(2, 8, 64), torch.randn(2, 16, 14)
        p = h.predict_action(vl, st)
        self.assertEqual(tuple(p.shape), (2, 16, 14))
        self.assertTrue(torch.allclose(p, st, atol=1e-5))
        self.assertTrue(torch.allclose(h.predict_action(vl, st[:, None]), p, atol=1e-5))  # (B,1,T,D) from the server
        self.assertTrue(torch.allclose(h.predict_action(vl, st[:, -1:]), st[:, -1:].expand(-1, 16, -1), atol=1e-5))

    def test_loss_uses_future_target_only(self):
        h = self._h()
        vl, st, act = torch.randn(2, 8, 64), torch.randn(2, 16, 14), torch.randn(2, 16, 14)
        self.assertAlmostEqual(float(h(vl, act, st)), float((st - act).abs().mean()), places=4)
