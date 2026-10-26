"""CPU unit / smoke tests of QwenLatentWorldAction with MOCK System 2 and C-RADIO encoders (no checkpoints).

These check shapes, losses, gradient routing, the inference data path and the cache / conversion utilities.
They say nothing about the pretrained Qwen3.5 / C-RADIOv4 behaviour; that is covered by
tests/integration_latent_world_action.py (GPU + real checkpoints).

    CUDA_VISIBLE_DEVICES= python -m unittest tests.test_latent_world_action -v
"""

import copy
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework.VLM4A.QwenLatentWorldAction import QwenLatentWorldAction
from starVLA.model.modules.latent_world_action import (ActionQueryDecoder, LatentFlowDiT, LatentFlowSampler, TensorCache,
                                                        apply_lora, regression_loss)
from starVLA.model.modules.latent_world_action.convert import convert_trisystem_system15
from starVLA.model.modules.latent_world_action.latent_flow_dit import one_step_endpoint

IMG = 64  # loader frame size (W = H); mock C-RADIO: 16-px patches -> 4 x 4 = 16 tokens
H_A, D_A, D_ST = 4, 7, 5


# ----------------------------------------------------------------------------- mocks
class MockVLM(nn.Module):
    """Stand-in for the repo's Qwen interface: 2x2 colour tokens per image + one token per character, a 1-layer
    Transformer, LEFT padding (as the Qwen processor). Same call signature as _QWen3_5_VL_Interface."""

    def __init__(self, hidden: int = 48):
        super().__init__()
        self.model = nn.Module()
        self.model.config = SimpleNamespace(hidden_size=hidden)
        self.img = nn.Linear(3, hidden)
        self.txt = nn.Embedding(128, hidden)
        self.enc = nn.TransformerEncoderLayer(hidden, 4, 2 * hidden, dropout=0.0, batch_first=True)
        self.autocast_dtype = torch.bfloat16
        self.seen_images = []

    def build_qwenvl_inputs(self, images, instructions, **kw):
        rows = []
        for ims, text in zip(images, instructions):
            self.seen_images.append(list(ims))
            feats, ids, is_img = [], [], []
            for im in ims:
                a = torch.from_numpy(np.asarray(im.convert("RGB").resize((2, 2)))).float().div(255).reshape(4, 3)
                feats += list(a)
                ids += [0] * 4
                is_img += [True] * 4
            for ch in text:
                feats.append(torch.zeros(3))
                ids.append(ord(ch) % 128)
                is_img.append(False)
            rows.append((torch.stack(feats), torch.tensor(ids), torch.tensor(is_img)))
        L = max(r[0].shape[0] for r in rows)
        pad = lambda t, v: torch.cat([torch.full((L - t.shape[0], *t.shape[1:]), v, dtype=t.dtype), t])
        return {
            "pixel_feats": torch.stack([pad(r[0], 0.0) for r in rows]),
            "input_ids": torch.stack([pad(r[1], 0) for r in rows]),
            "is_image": torch.stack([pad(r[2], False) for r in rows]),
            "attention_mask": torch.stack([pad(torch.ones(r[0].shape[0], dtype=torch.long), 0) for r in rows]),
        }

    def forward(self, pixel_feats, input_ids, is_image, attention_mask, output_hidden_states=True, return_dict=True,
                **kw):
        x = torch.where(is_image[..., None], self.img(pixel_feats), self.txt(input_ids))
        h = self.enc(x, src_key_padding_mask=~attention_mask.bool())
        return SimpleNamespace(hidden_states=(x, h))


class MockRadio(nn.Module):
    """Stand-in for RadioPatchEncoder: resize to input_size, 16-px patches in row-major order, fixed random
    projection; one fixed high-norm artifact token (like C-RADIO's top-row outlier) flagged invalid."""

    feature = "backbone"

    def __init__(self, input_size=(64, 64), feat_dim: int = 32, artifact=(0, 3)):
        super().__init__()
        self.input_size, self.patch_size, self.feat_dim, self.artifact = tuple(input_size), 16, feat_dim, artifact
        g = torch.Generator().manual_seed(0)
        self.proj = nn.Linear(3 * 16 * 16, feat_dim)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(feat_dim, 768, generator=g) * 0.05)
            self.proj.bias.zero_()
        self.calls = 0

    @property
    def grid(self):
        return self.input_size[0] // 16, self.input_size[1] // 16

    def forward(self, images):
        self.calls += images.shape[0]
        if tuple(images.shape[-2:]) != self.input_size:
            images = F.interpolate(images, self.input_size, mode="bilinear", antialias=True, align_corners=False)
        b, (gh, gw) = images.shape[0], self.grid
        p = images.reshape(b, 3, gh, 16, gw, 16).permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, 768)
        z = self.proj(p)
        if self.artifact is not None:
            z[:, self.artifact[0] * gw + self.artifact[1], 0] += 1000.0
        return z.float(), z.norm(dim=-1) <= 100.0


def make_config(stage="A", **framework_over):
    """stage=None keeps the framework default (stage C, endpoint joint training)."""
    fw = {
        "name": "QwenLatentWorldAction",
        "radio": {"input_size": [IMG, IMG], "normalize_features": True},
        "system15": {"d_model": 32, "depth": 2, "num_heads": 4, "sample_steps": 3, "solver": "euler"},
        "system1": {"d_model": 32, "depth": 2, "num_heads": 4, "dropout": 0.0},
        "action_model": {"action_dim": D_A, "state_dim": D_ST, "action_horizon": H_A},
        "latent_world": {"image_layout": ["t", "f"], "future_offset": H_A,
                         "eval": {"num_samples": 4, "batch_size": 2, "seed": 0}},
    }
    if stage is not None:
        fw["latent_world"]["stage"] = stage
    cfg = OmegaConf.create({"framework": fw, "datasets": {"vla_data": {"obs_image_size": [IMG, IMG]}},
                            "trainer": {"freeze_modules": "qwen_vl_interface,radio_encoder", "logging_frequency": 100,
                                        "learning_rate": {"base": 1e-5, "system15": 1e-3, "system1": 2e-3}}})
    if framework_over:
        cfg.framework = OmegaConf.merge(cfg.framework, OmegaConf.create(framework_over))
    return cfg


def make_model(stage="A", seed=0, **over):
    torch.manual_seed(seed)
    m = QwenLatentWorldAction(make_config(stage, **over), vlm_interface=MockVLM(), radio_backbone=MockRadio())
    st = m.radio_encoder.stats  # fake "loaded" normalization statistics
    st.z_mean.copy_(torch.randn(32) * 0.1)
    st.z_std.copy_(torch.rand(32) + 0.5)
    st.dz_rms.copy_(torch.rand(32) + 0.5)
    st.loaded.fill_(True)
    return m


def rand_image(rng, size=IMG):
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def make_examples(n=3, seed=0, future=True, with_meta=True, langs=None):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ex = {"image": [rand_image(rng), rand_image(rng)] if future else [rand_image(rng)],
              "lang": (langs or ["pick the cup", "open the drawer now", "go"])[i % 3],
              "action": rng.normal(size=(H_A, D_A)).astype(np.float16),
              "state": rng.normal(size=(1, D_ST)).astype(np.float16)}
        if with_meta and future:
            ex.update({"frame_offsets": np.array([0, H_A]), "future_valid": True, "sample_key": f"ds/{seed}/{i}",
                       "frame_keys": [f"ds/cam/{seed}/{i}", f"ds/cam/{seed}/{i + H_A}"]})
        out.append(ex)
    return out


class NoActionDict(dict):
    """An example that fails loudly if anything reads its GT action."""

    def __getitem__(self, k):
        if k == "action":
            raise AssertionError("GT action was read at inference")
        return super().__getitem__(k)

    def get(self, k, default=None):
        if k == "action":
            raise AssertionError("GT action was read at inference")
        return super().get(k, default)


def grads(module):
    return [p.grad for p in module.parameters() if p.requires_grad or p.grad is not None]


def has_grad(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


# ----------------------------------------------------------------------------- tests
class TestSystem2AndRadio(unittest.TestCase):
    def test_01_system2_full_token_sequence(self):
        m = make_model()
        ex = make_examples(3)
        s2 = m.encode_system2([[e["image"][0]] for e in ex], [e["lang"] for e in ex])
        lens = [4 + len(e["lang"]) for e in ex]  # 4 image tokens + one per character, nothing pooled
        self.assertEqual(tuple(s2.tokens.shape), (3, max(lens), 48))
        self.assertEqual(s2.mask.sum(1).tolist(), lens)
        self.assertFalse(bool(s2.mask[2, 0]))  # left padding masked
        self.assertTrue(bool(s2.mask[:, -1].all()))

    def test_02_radio_patch_tokens_shape_order_mask(self):
        m = make_model()
        img = torch.zeros(1, 3, IMG, IMG)
        img[:, :, 16:32, 32:48] = 1.0  # patch (row 1, col 2)
        base = m.radio_encoder.backbone(torch.zeros(1, 3, IMG, IMG))[0]
        z, valid = m.radio_encoder.backbone(img)
        self.assertEqual(tuple(z.shape), (1, 16, 32))
        changed = (z - base).norm(dim=-1)[0]
        self.assertEqual(int(changed.argmax()), 1 * 4 + 2)  # row-major order
        self.assertEqual(int((changed > 1e-6).sum()), 1)
        ps = m.radio_encoder(img)
        self.assertFalse(bool(ps.valid[0, 3]))  # artifact token flagged ...
        self.assertTrue(torch.all(ps.z[0, 3] == 0))  # ... and zeroed
        st = m.radio_encoder.stats
        torch.testing.assert_close(ps.z[0, 5], (z[0, 5] - st.z_mean) / st.z_std)

    def test_03_latent_change_target(self):
        m = make_model()
        ex = make_examples(2)
        obs = m.encode_observation(ex, with_future=True)
        x1 = m.latent_change_target(obs)
        manual = (obs.z_f - obs.z_t) / m.radio_encoder.stats.dz_rms
        v = obs.valid & obs.valid_f
        torch.testing.assert_close(x1[v], manual[v])
        self.assertTrue(torch.all(x1[~v] == 0))
        same = copy.deepcopy(ex)
        for e in same:
            e["image"][1] = e["image"][0]
        self.assertTrue(torch.all(m.latent_change_target(m.encode_observation(same, with_future=True)) == 0))
        delta, z_f_hat = m.latent_change_to_features(x1, obs)
        torch.testing.assert_close(z_f_hat[v], obs.z_f[v])  # Z_t + dZ reconstructs the future latent


class TestSystem15(unittest.TestCase):
    def setUp(self):
        self.m = make_model()
        self.obs = self.m.encode_observation(make_examples(2), with_future=True)
        self.cond = self.m.flow_conditions(self.obs)

    def test_04_dit_shapes(self):
        x = torch.randn(2, 16, 32)
        v, h = self.m.system15(x, torch.rand(2), self.cond, return_hidden_states=True)
        self.assertEqual(tuple(v.shape), (2, 16, 32))  # same as the input latent change
        self.assertEqual(tuple(h.shape), (2, 16, 32))  # [B, N, d_model]
        self.assertEqual(tuple(self.cond.contexts["s2"][0].shape[:2]), (2, 1 + self.obs.s2.tokens.shape[1]))

    def test_05_fm_loss_finite_and_grad(self):
        m, obs = self.m, self.obs
        with torch.no_grad():  # zero-initialised velocity head / adaLN gates: only the head gets gradient at step 0
            for p in m.system15.parameters():
                p.add_(torch.randn_like(p) * 0.02)
        cond = m.flow_conditions(obs)
        x1 = m.latent_change_target(obs)
        loss, logs = m.system15.flow_matching_loss(x1, cond, obs.valid, {"schedule": "uniform"}, 2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(has_grad(m.system15.velocity_head))
        self.assertTrue(has_grad(m.system15.blocks[0].cross_attn["zt"]))
        self.assertTrue(has_grad(m.system15.s2_proj))

    def test_05b_mask_attention_padding(self):
        """Padded Qwen tokens do not change the DiT output."""
        m = self.m
        x, tau = torch.randn(2, 16, 32), torch.rand(2)
        v1 = m.system15(x, tau, self.cond)
        s2 = self.obs.s2.tokens.clone()
        s2[~self.obs.s2.mask] = 1e3
        cond2 = m.system15.encode_conditions(self.obs.z_t, self.obs.valid, s2, self.obs.s2.mask)
        torch.testing.assert_close(v1, m.system15(x, tau, cond2))

    def test_06_ode_sampling(self):
        m, obs = self.m, self.obs
        with torch.no_grad():
            m.system15.velocity_head.weight.normal_(0, 0.1)  # non-trivial velocity
            noise = torch.randn(2, 16, 32)
            x_e = m.sample_latent_change(obs, self.cond, noise=noise, steps=4, solver="euler")
            x_h = m.sample_latent_change(obs, self.cond, noise=noise, steps=4, solver="heun")
            x_1 = m.sample_latent_change(obs, self.cond, noise=noise, steps=1, solver="euler")
            v0 = m.system15(noise, torch.zeros(2), self.cond)
        self.assertEqual(tuple(x_e.shape), (2, 16, 32))
        self.assertTrue(torch.isfinite(x_e).all() and torch.isfinite(x_h).all())
        self.assertFalse(torch.allclose(x_e, x_h))
        torch.testing.assert_close(x_1[obs.valid], (noise + v0)[obs.valid])  # one Euler step from tau 0 to 1
        self.assertTrue(torch.all(x_e[~obs.valid] == 0))

    def test_07_final_hidden_is_separate_tau1_pass(self):
        m, obs = self.m, self.obs
        with torch.no_grad():
            for p in m.system15.parameters():  # break the zero init so every term matters
                p.add_(torch.randn_like(p) * 0.02)
            x1, last_h = m.sample_latent_change(obs, self.cond, steps=3, record_last_hidden=True)
            h_f = m.extract_final_hidden(obs, x1, self.cond)
            ref = m.system15.backbone(x1, torch.ones(2), self.cond)
        torch.testing.assert_close(h_f, ref)
        self.assertEqual(tuple(h_f.shape), (2, 16, m.system15.d_model))
        self.assertFalse(torch.allclose(h_f, last_h))  # not the solver's last-step hidden state
        # the velocity head is not part of the extraction
        head = m.system15.velocity_head
        m.system15.velocity_head = None
        try:
            with torch.no_grad():
                torch.testing.assert_close(m.extract_final_hidden(obs, x1, self.cond), h_f)
        finally:
            m.system15.velocity_head = head
        # tau = 1 is used: a different tau gives a different hidden state
        with torch.no_grad():
            self.assertFalse(torch.allclose(h_f, m.system15.backbone(x1, torch.full((2,), 0.5), self.cond)))


class TestSystem1(unittest.TestCase):
    def test_08_decoder_cross_attention_shapes_and_mask(self):
        dec = ActionQueryDecoder(D_A, H_A, 32, (4, 4), state_dim=D_ST, d_model=24, depth=2, num_heads=4, dropout=0.0)
        h, state = torch.randn(2, 16, 32), torch.randn(2, 1, D_ST)
        mask = torch.ones(2, 16, dtype=torch.bool)
        mask[:, 3] = False
        self.assertEqual(tuple(dec.queries.shape), (1, H_A, 24))
        self.assertEqual(tuple(dec.embed_memory(h).shape), (2, 16, 24))  # projected H_f = K / V
        a = dec(h, mask, state)
        self.assertEqual(tuple(a.shape), (2, H_A, D_A))
        h2 = h.clone()
        h2[:, 3] += 100.0  # masked token cannot influence the actions
        torch.testing.assert_close(dec(h2, mask, state), a)
        dec_add = ActionQueryDecoder(D_A, H_A, 32, (4, 4), state_dim=D_ST, d_model=24, depth=1, num_heads=4,
                                     state_conditioning="add")
        self.assertEqual(tuple(dec_add(h, mask, state).shape), (2, H_A, D_A))

    def test_09_stage_b_regression_loss_and_grad_routing(self):
        m = make_model("B")
        out = m(make_examples(3))
        self.assertTrue(torch.isfinite(out["loss"]) and "fm_loss" not in out)
        torch.testing.assert_close(out["loss"], out["action_loss"])
        out["loss"].backward()
        self.assertTrue(has_grad(m.system1))
        for mod in (m.system15, m.qwen_vl_interface, m.radio_encoder):
            self.assertTrue(all(p.grad is None for p in mod.parameters()))
            self.assertTrue(all(not p.requires_grad for p in mod.parameters()))

    def test_09b_loss_types(self):
        for kind in ("smooth_l1", "l1", "mse"):
            m = make_model("B", system1={"loss": kind})
            self.assertTrue(torch.isfinite(m(make_examples(2))["loss"]))


class TestFramework(unittest.TestCase):
    def test_10_forward_all_stages_and_predict(self):
        ex = make_examples(3)
        m = make_model("A")
        out = m(ex)
        self.assertIn("system15/fm_loss", out)
        self.assertNotIn("action_loss", out)
        out["loss"].backward()
        self.assertTrue(has_grad(m.system15) and not has_grad(m.system1))
        self.assertFalse(m.use_action_head)
        m = make_model("C")
        out = m(ex)
        self.assertIn("fm_loss", out)
        self.assertIn("action_loss", out)
        lam = out["joint/action_weight"]
        self.assertAlmostEqual(lam, 0.1)  # step 0 of the linear warmup
        torch.testing.assert_close(out["loss"], out["fm_loss"] + lam * out["action_loss"])
        m.eval()
        pred = m.predict_action(make_examples(2, future=False), return_latent=True)
        self.assertEqual(pred["normalized_actions"].shape, (2, H_A, D_A))
        self.assertEqual(pred["latent_change"].shape, (2, 16, 32))
        self.assertTrue(np.isfinite(pred["normalized_actions"]).all())

    def test_10b_joint_gradient_paths(self):
        ex = make_examples(3)

        def s15_grad(mode):
            m = make_model("C", seed=1, latent_world={"loss_weights": {"fm": 0.0, "action": 1.0},
                                                      "joint": {"action_grad": mode, "action_path": "ode",
                                                                "gt_prob_schedule": "fixed", "gt_prob_end": 0.0}})
            with torch.no_grad():
                for p in m.system15.parameters():
                    p.add_(torch.randn_like(p) * 0.02)
            torch.manual_seed(5)
            m(ex)["loss"].backward()
            return m, torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                                 for p in m.system15.parameters()])

        m, g_none = s15_grad("none")
        self.assertEqual(float(g_none.abs().sum()), 0.0)  # default: action loss does not reach System 1.5
        self.assertTrue(has_grad(m.system1))
        _, g_final = s15_grad("final_pass")
        _, g_full = s15_grad("full")
        self.assertGreater(float(g_final.abs().sum()), 0.0)
        self.assertGreater(float(g_full.abs().sum()), 0.0)
        self.assertFalse(torch.allclose(g_final, g_full))  # full also differentiates through the solver steps

    def test_10c_teacher_forcing_and_clamped_future(self):
        ex = make_examples(3)
        m = make_model("B", system1={"hidden_source": "gt"})
        out = m(ex)
        self.assertEqual(out["system1/gt_hidden_frac"], 1.0)
        ex[1]["future_valid"] = False  # clamped future -> predicted path for that sample
        self.assertAlmostEqual(m(ex)["system1/gt_hidden_frac"], 2 / 3)
        a = make_model("A")
        for e in ex:
            e["future_valid"] = False
        out = a(ex)
        self.assertEqual(float(out["loss"]), 0.0)  # every target clamped -> no FM signal
        self.assertEqual(out["system15/future_valid_frac"], 0.0)

    def test_10d_frame_offset_check(self):
        m = make_model("A")
        ex = make_examples(2)
        for e in ex:
            e["frame_offsets"] = np.array([0, 8])  # data config says t+8, model expects t+4
        with self.assertRaises(ValueError):
            m(ex)

    def test_11_inference_never_uses_future_or_actions(self):
        m = make_model("B", seed=3)
        m.eval()
        ex = make_examples(2, seed=7)
        rng = np.random.default_rng(1)
        current_only = [NoActionDict(image=[e["image"][0]], lang=e["lang"], state=e["state"]) for e in ex]
        other_future = [NoActionDict(image=[e["image"][0], rand_image(rng)], lang=e["lang"], state=e["state"])
                        for e in ex]
        calls = m.radio_encoder.backbone.calls
        a = m.predict_action(current_only, seed=0)["normalized_actions"]
        self.assertEqual(m.radio_encoder.backbone.calls - calls, 2)  # only the 2 current frames were encoded
        b = m.predict_action(other_future, seed=0)["normalized_actions"]
        np.testing.assert_array_equal(a, b)
        m.qwen_vl_interface.seen_images.clear()
        m.predict_action(other_future, seed=0)
        self.assertTrue(all(len(v) == 1 for v in m.qwen_vl_interface.seen_images))  # Qwen saw the current view only
        c = m.predict_action(current_only, seed=1)["normalized_actions"]
        self.assertFalse(np.allclose(a, c))  # the ODE noise matters (sanity check of the comparison)

    def test_11b_state_dict_roundtrip(self):
        m = make_model("C", seed=2)
        m.eval()
        ex = make_examples(2, future=False)
        a = m.predict_action(ex, seed=0)["normalized_actions"]
        m2 = make_model("C", seed=9)
        m2.load_state_dict(m.state_dict(), strict=True)
        m2.eval()
        np.testing.assert_allclose(m2.predict_action(ex, seed=0)["normalized_actions"], a, rtol=1e-5, atol=1e-6)

    def test_11c_eval_hook(self):
        m = make_model("B")
        with tempfile.TemporaryDirectory() as d:
            ex = make_examples(4)
            metrics = m.trainer_eval_hook(lambda: ex, 10, d, is_main=True)
        for k in ("eval/system15/rel_mse", "eval/system15/cos", "eval/system1/rmse_pred", "eval/system1/rmse_gt_hidden"):
            self.assertIn(k, metrics)
            self.assertTrue(np.isfinite(metrics[k]))
        self.assertTrue(m.system1.training)  # train mode restored

    def test_12_registry_keeps_existing_frameworks(self):
        from starVLA.model.framework.base_framework import _auto_import_framework_modules
        from starVLA.model.tools import FRAMEWORK_REGISTRY

        _auto_import_framework_modules()
        names = set(FRAMEWORK_REGISTRY._registry)
        for n in ("QwenGR00T", "QwenOFT", "QwenGR00T_TriSystem", "QwenGR00T_Regressor", "QwenLatentWorldAction"):
            self.assertIn(n, names)
        self.assertIs(FRAMEWORK_REGISTRY["QwenLatentWorldAction"], QwenLatentWorldAction)


class TestUtilities(unittest.TestCase):
    def test_hidden_cache(self):
        with tempfile.TemporaryDirectory() as d:
            m = make_model("B", system1={"hidden_source": "cache", "hidden_cache_dir": d})
            ex = make_examples(3)
            out1 = m(ex)
            self.assertEqual(out1["system1/cache_miss_frac"], 1.0)
            calls = m.radio_encoder.backbone.calls
            out2 = m(ex)
            self.assertEqual(out2["system1/cache_miss_frac"], 0.0)
            self.assertEqual(m.radio_encoder.backbone.calls, calls)  # hits skip System 2 / C-RADIO
            torch.testing.assert_close(out1["action_loss"], out2["action_loss"], rtol=2e-2, atol=1e-3)  # bf16 storage
            other = make_model("B", seed=4, system1={"hidden_source": "cache", "hidden_cache_dir": d})
            with self.assertRaises(ValueError):  # other System 1.5 weights -> stale cache refused
                other(ex)

    def test_radio_feature_cache(self):
        with tempfile.TemporaryDirectory() as d:
            m = make_model("A", radio={"cache_dir": d})
            ex = make_examples(2)
            o1 = m.encode_observation(ex, with_future=True)
            calls = m.radio_encoder.backbone.calls
            o2 = m.encode_observation(ex, with_future=True)
            self.assertEqual(m.radio_encoder.backbone.calls, calls)
            torch.testing.assert_close(o1.z_t, o2.z_t, rtol=1e-2, atol=1e-2)

    def test_tensor_cache_keys(self):
        with tempfile.TemporaryDirectory() as d:
            c = TensorCache(d, {"a": 1})
            c.put("x/1/2", {"t": torch.ones(2)})
            self.assertTrue(torch.equal(c.get("x/1/2")["t"], torch.ones(2)))
            self.assertIsNone(c.get("x/1/3"))
            with self.assertRaises(ValueError):
                c.get("../escape")
            with self.assertRaises(ValueError):
                TensorCache(d, {"a": 2})

    def test_lora(self):
        vlm = MockVLM()
        x = vlm.build_qwenvl_inputs([[Image.new("RGB", (8, 8))]], ["hi"])
        ref = vlm(**x).hidden_states[-1]
        vlm.enc.linear1.__class__  # nn.Linear children named linear1 / linear2
        n = apply_lora(vlm, r=4, alpha=8, target_modules=["linear1", "linear2"])
        self.assertEqual(n, 2)
        torch.testing.assert_close(vlm(**x).hidden_states[-1], ref)  # B = 0 at init
        trainable = [k for k, p in vlm.named_parameters() if p.requires_grad]
        self.assertTrue(trainable and all("lora_" in k for k in trainable))

    def test_trisystem_conversion(self):
        from starVLA.model.modules.tri_system.system15_flow import ChangeFlowTransformer

        torch.manual_seed(0)
        tri = ChangeFlowTransformer(32, 48, (4, 4), d_model=32, depth=2, num_heads=4)
        for p in tri.parameters():
            p.data.normal_(0, 0.05)
        sd = {"system15.net." + k: v for k, v in tri.state_dict().items()}
        sd.update({"feature_stats.z_mean": torch.zeros(32), "feature_stats.z_std": torch.ones(32),
                   "feature_stats.dz_rms": torch.ones(32), "feature_stats.loaded": torch.tensor(True),
                   "system1.net.action_in.weight": torch.zeros(3)})
        conv = convert_trisystem_system15(sd)
        dit = LatentFlowDiT(32, 48, (4, 4), d_model=32, depth=2, num_heads=4, adaln_cond="tau+s2")
        dit.load_state_dict({k[len("system15."):]: v for k, v in conv.items() if k.startswith("system15.")}, strict=True)
        self.assertIn("radio_encoder.stats.dz_rms", conv)
        z, s2 = torch.randn(2, 16, 32), torch.randn(2, 5, 48)
        zv, sm = torch.ones(2, 16, dtype=torch.bool), torch.ones(2, 5, dtype=torch.bool)
        x, tau = torch.randn(2, 16, 32), torch.rand(2)
        ref = tri(x, tau, z, zv, s2, sm)
        torch.testing.assert_close(dit(x, tau, dit.encode_conditions(z, zv, s2, sm)), ref)

    def test_sampler_rejects_unknown_solver(self):
        with self.assertRaises(ValueError):
            LatentFlowSampler(4, "dopri5")


# ----------------------------------------------------------------------------- stage C only (end-to-end joint)
def grad_norm(params):
    gs = [p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
    return float(torch.stack(gs).sum().sqrt()) if gs else 0.0


def backbone_params(m):
    head = {id(p) for p in m.system15.velocity_head.parameters()}
    return [p for p in m.system15.parameters() if id(p) not in head]


# the endpoint path without the scheduled GT replacement (the default starts at p_gt = 1: System 1 reads GT X_1)
PURE_ENDPOINT = {"gt_prob_schedule": "fixed", "gt_prob_end": 0.0}


class TestJointEndpointTraining(unittest.TestCase):
    """Default training = stage C from scratch with the one-step endpoint path."""

    def joint_tensors(self, m, ex):
        """The pieces of one joint forward, built with the framework's own methods."""
        obs = m.encode_observation(ex, with_future=True)
        cond = m.flow_conditions(obs)
        x1 = m.latent_change_target(obs)
        loss_fm, _, inter = m.system15.flow_matching_loss(x1, cond, obs.valid, {"schedule": "uniform"},
                                                          return_intermediates=True)
        return obs, cond, x1, loss_fm, inter

    def test_j01_default_is_joint_from_scratch(self):
        m = make_model(stage=None)  # framework defaults, no checkpoint
        self.assertEqual((m.stage, m.action_path, m.action_grad, m.endpoint_tau), ("C", "endpoint", "full", "shared"))
        self.assertEqual((m.endpoint_gt_prob(0), m.endpoint_gt_prob(10_000)), (1.0, 0.05))  # GT first -> dZ~ >= 0.95
        self.assertEqual(m.trainable_modules, ["system15", "system1"])
        self.assertTrue(m.use_action_head)
        out = m(make_examples(3))
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_j02_j03_both_losses(self):
        m = make_model(stage=None)
        out = m(make_examples(3))
        for k in ("fm_loss", "action_loss"):
            self.assertTrue(torch.isfinite(out[k]) and float(out[k]) > 0, k)
            self.assertTrue(out[k].requires_grad)
        self.assertIn("system15/endpoint_rel_mse", out)
        self.assertTrue(0.0 <= out["joint/endpoint_tau_mean"] <= 1.0)

    def test_j04_endpoint_formula_and_shape(self):
        m = make_model(stage=None)
        with torch.no_grad():
            for p in m.system15.parameters():
                p.add_(torch.randn_like(p) * 0.02)
        obs, cond, x1, _, inter = self.joint_tensors(m, make_examples(2))
        calls = []
        orig = m.system15.backbone
        m.system15.backbone = lambda x, tau, c: (calls.append(tau.detach().clone()), orig(x, tau, c))[1]
        try:
            h, dz = m.one_step_endpoint_hidden(obs, cond, inter)
        finally:
            m.system15.backbone = orig
        x_tau, tau, v = inter["x_tau"], inter["tau"], inter["v"]
        manual = (x_tau + (1 - tau)[:, None, None] * v).masked_fill(~obs.valid[..., None], 0.0)
        torch.testing.assert_close(dz, manual)
        self.assertEqual(tuple(dz.shape), (2, 16, 32))
        self.assertEqual(tuple(h.shape), (2, 16, m.system15.d_model))
        self.assertTrue(dz.requires_grad and dz.grad_fn is not None)  # not detached
        self.assertEqual(len(calls), 1)  # one extra DiT forward (the hidden pass) ...
        self.assertTrue(torch.all(calls[0] == 1.0))  # ... at tau = 1
        with torch.no_grad():
            torch.testing.assert_close(h, m.extract_final_hidden(obs, dz.detach(), cond))
        # at tau = 0 the projection is one Euler step from the noise; at tau = 1 it returns X_1
        eps = torch.randn(2, 16, 32)
        vv = torch.randn(2, 16, 32)
        torch.testing.assert_close(one_step_endpoint(eps, torch.zeros(2), vv), eps + vv)
        torch.testing.assert_close(one_step_endpoint(x1, torch.ones(2), vv), x1)

    def test_j05_j07_action_loss_alone_reaches_system15(self):
        m = make_model(stage=None, latent_world={"joint": PURE_ENDPOINT})  # random init (zero velocity head, adaLN gates)
        out = m(make_examples(3))
        out["action_loss"].backward()  # ONLY the action loss
        n1, nb, nh = grad_norm(m.system1.parameters()), grad_norm(backbone_params(m)), \
            grad_norm(m.system15.velocity_head.parameters())
        for name, n in (("system1", n1), ("system15 backbone", nb), ("velocity head", nh)):
            self.assertTrue(np.isfinite(n) and n > 0, f"{name}: {n}")
        self.assertGreater(grad_norm(m.system15.zt_proj.parameters()), 0)  # conditioning projectors too
        self.assertGreater(grad_norm(m.system15.s2_proj.parameters()), 0)

    def test_j06b_gradient_goes_through_velocity(self):
        """L_action -> H_f -> dZ~ -> v^ : the velocity tensor of the FM pass receives action gradient, and cutting
        that link (action_grad final_pass) removes the velocity-head gradient."""
        m = make_model(stage=None)
        with torch.no_grad():
            for p in m.system15.parameters():
                p.add_(torch.randn_like(p) * 0.02)
        ex = make_examples(3)
        obs, cond, _, _, inter = self.joint_tensors(m, ex)
        inter["v"].retain_grad()
        h, dz = m.one_step_endpoint_hidden(obs, cond, inter)
        dz.retain_grad()
        h.retain_grad()
        loss, _ = regression_loss(m.system1(h, obs.valid, obs.state), m._actions(ex))
        loss.backward()
        for t in (h, dz, inter["v"]):
            g = float(t.grad.norm())
            self.assertTrue(np.isfinite(g) and g > 0)
        torch.testing.assert_close(inter["v"].grad, (1 - inter["tau"])[:, None, None] * dz.grad)  # chain rule of dZ~

        def head_grad(mode):
            mm = make_model(stage=None, seed=1, latent_world={"joint": {"action_grad": mode, **PURE_ENDPOINT}})
            mm(make_examples(3))["action_loss"].backward()
            return grad_norm(mm.system15.velocity_head.parameters()), grad_norm(backbone_params(mm))
        (h_full, b_full), (h_fp, b_fp), (h_none, b_none) = head_grad("full"), head_grad("final_pass"), head_grad("none")
        self.assertGreater(h_full, 0)
        self.assertEqual(h_fp, 0.0)  # dZ~ detached -> no gradient into v^ / the velocity head
        self.assertGreater(b_fp, 0)  # ... but still via the tau = 1 hidden pass
        self.assertEqual((h_none, b_none), (0.0, 0.0))

    def test_j08_frozen_encoders_get_no_grad(self):
        m = make_model(stage=None)
        m(make_examples(3))["loss"].backward()
        for mod in (m.qwen_vl_interface, m.radio_encoder):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in mod.parameters()))

    def test_j09_one_optimizer_step_updates_both(self):
        from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

        m = make_model(stage=None)
        groups = build_param_lr_groups(m, m.config)  # the trainer's own param-group builder
        self.assertEqual(sorted(g["name"] for g in groups), ["system1", "system15"])
        self.assertEqual({g["name"]: g["lr"] for g in groups}, {"system15": 1e-3, "system1": 2e-3})
        n_opt = sum(p.numel() for g in groups for p in g["params"])
        self.assertEqual(n_opt, sum(p.numel() for p in m.system15.parameters()) + sum(p.numel() for p in m.system1.parameters()))
        opt = torch.optim.AdamW(groups)
        before = {k: v.detach().clone() for k, v in m.named_parameters()}
        m(make_examples(3))["loss"].backward()
        opt.step()
        changed = lambda prefix: any(not torch.equal(before[k], v) for k, v in m.named_parameters() if k.startswith(prefix))
        self.assertTrue(changed("system15.velocity_head"))
        self.assertTrue(changed("system15.blocks"))
        self.assertTrue(changed("system1.queries") and changed("system1.blocks") and changed("system1.head"))
        self.assertFalse(changed("qwen_vl_interface") or changed("radio_encoder"))

    def test_j10_j11_inference_is_ode(self):
        m = make_model(stage=None, system15={"sample_steps": 3})
        m.eval()
        calls = {"sample": 0, "endpoint": 0, "taus": []}
        orig_sample, orig_ep, orig_bb = m.sampler.sample, m.one_step_endpoint_hidden, m.system15.backbone
        m.sampler.sample = lambda *a, **k: (calls.__setitem__("sample", calls["sample"] + 1), orig_sample(*a, **k))[1]
        m.one_step_endpoint_hidden = lambda *a, **k: (calls.__setitem__("endpoint", calls["endpoint"] + 1), orig_ep(*a, **k))[1]
        m.system15.backbone = lambda x, tau, c: (calls["taus"].append(float(tau[0])), orig_bb(x, tau, c))[1]
        out = m.predict_action(make_examples(2, future=False), use_ddim=True, num_ddim_steps=20)  # trainer-style kwargs
        self.assertEqual((calls["sample"], calls["endpoint"]), (1, 0))
        np.testing.assert_allclose(calls["taus"], [0.0, 1 / 3, 2 / 3, 1.0], rtol=1e-6)  # 3 Euler steps + tau = 1 pass
        self.assertEqual(set(out), {"normalized_actions"})
        self.assertIsInstance(out["normalized_actions"], np.ndarray)
        self.assertEqual(out["normalized_actions"].shape, (2, H_A, D_A))

    def test_action_loss_warmup(self):
        m = make_model(stage=None, latent_world={"joint": {"action_loss_warmup_steps": 100, "action_loss_weight_start": 0.1},
                                                 "loss_weights": {"fm": 1.0, "action": 2.0}})
        self.assertAlmostEqual(m.action_loss_weight(0), 0.1)
        self.assertAlmostEqual(m.action_loss_weight(50), 1.05)
        self.assertAlmostEqual(m.action_loss_weight(100), 2.0)
        self.assertAlmostEqual(m.action_loss_weight(10_000), 2.0)
        m.set_training_step(50)  # what the trainer does before each step
        out = m(make_examples(2))
        self.assertAlmostEqual(out["joint/action_weight"], 1.05)
        torch.testing.assert_close(out["loss"], out["fm_loss"] + 1.05 * out["action_loss"])
        fixed = make_model(stage=None, latent_world={"joint": {"action_loss_schedule": "fixed"}})
        self.assertEqual(fixed.action_loss_weight(0), 1.0)
        fb = make_model(stage=None, latent_world={"joint": {"action_loss_warmup_steps": 10}})  # no trainer: count forwards
        for _ in range(5):
            fb(make_examples(2))
        self.assertAlmostEqual(fb.action_loss_weight(), 0.1 + 0.9 * 5 / 10)

    def test_per_loss_grad_norm_logging(self):
        m = make_model(stage=None, latent_world={"joint": {"grad_norm_log_interval": 1, **PURE_ENDPOINT}})
        out = m(make_examples(3))
        keys = ["grad_norm/fm->system15", "grad_norm/action->system15", "grad_norm/action->system15_velocity_head",
                "grad_norm/action->system1"]
        for k in keys:
            self.assertTrue(np.isfinite(out[k]) and out[k] > 0, (k, out.get(k)))
        self.assertEqual(out["grad_norm/fm->system1"], 0.0)
        self.assertTrue(all(p.grad is None for p in m.parameters()))  # .grad untouched by the probe
        out["loss"].backward()  # the graph is still there for the real backward
        self.assertGreater(grad_norm(m.system15.parameters()), 0)
        m2 = make_model(stage=None)  # default interval = trainer.logging_frequency (100): not at step 0
        self.assertNotIn("grad_norm/action->system15", m2(make_examples(2)))

    def test_endpoint_tau_zero_option(self):
        m = make_model(stage=None, latent_world={"joint": {"endpoint_tau": "zero", **PURE_ENDPOINT}})
        out = m(make_examples(3))
        self.assertEqual(out["joint/endpoint_tau_mean"], 0.0)
        out["action_loss"].backward()
        self.assertGreater(grad_norm(m.system15.velocity_head.parameters()), 0)

    def test_gt_prob_schedule(self):
        lin = make_model(stage=None, latent_world={"joint": {"gt_prob_decay_steps": 100}})
        for step, p in ((0, 1.0), (50, 0.525), (100, 0.05), (10_000, 0.05)):
            self.assertAlmostEqual(lin.endpoint_gt_prob(step), p)
        cos = make_model(stage=None, latent_world={"joint": {"gt_prob_schedule": "cosine", "gt_prob_decay_steps": 100}})
        self.assertAlmostEqual(cos.endpoint_gt_prob(0), 1.0)
        self.assertAlmostEqual(cos.endpoint_gt_prob(50), 0.525)
        self.assertGreater(cos.endpoint_gt_prob(25), lin.endpoint_gt_prob(25))  # cosine stays near GT longer
        self.assertAlmostEqual(cos.endpoint_gt_prob(100), 0.05)
        fixed = make_model(stage=None, latent_world={"joint": {"gt_prob_schedule": "fixed", "gt_prob_end": 0.1}})
        self.assertEqual(fixed.endpoint_gt_prob(0), 0.1)
        lin.set_training_step(50)  # the trainer's step drives the schedule
        out = lin(make_examples(2))
        self.assertAlmostEqual(out["joint/gt_prob"], 0.525)
        for bad in ({"gt_prob_schedule": "step"}, {"gt_prob_end": 1.5}):
            with self.assertRaises(ValueError):
                make_model(stage=None, latent_world={"joint": bad})

    def test_scheduled_gt_replaces_endpoint_input(self):
        """p_gt = 1 (step 0 of the default): System 1 reads the GT X_1 for every example with a valid future, the
        clamped one keeps dZ~; the action loss then reaches System 1 and the backbone (tau = 1 pass), and v^ only
        through the clamped example. p_gt = 0: dZ~ for all."""
        m = make_model(stage=None)
        with torch.no_grad():
            for p in m.system15.parameters():
                p.add_(torch.randn_like(p) * 0.02)
        ex = make_examples(3)
        seen = []
        orig = m.extract_final_hidden
        m.extract_final_hidden = lambda obs, x, cond=None: (seen.append((obs, x.detach().clone())), orig(obs, x, cond))[1]
        try:
            out = m(ex)  # step 0 -> p_gt = 1
            self.assertEqual((out["joint/gt_prob"], out["joint/gt_frac"]), (1.0, 1.0))
            obs, x_in = seen[-1]
            torch.testing.assert_close(x_in, m.latent_change_target(obs))
            out["action_loss"].backward()
            self.assertGreater(grad_norm(m.system1.parameters()), 0)
            self.assertGreater(grad_norm(backbone_params(m)), 0)
            self.assertEqual(grad_norm(m.system15.velocity_head.parameters()), 0.0)  # GT rows: no path into v^
            m.zero_grad(set_to_none=True)
            ex[1]["future_valid"] = False  # clamped future: its GT is wrong -> always dZ~
            out = m(ex)
            self.assertAlmostEqual(out["joint/gt_frac"], 2 / 3)
            obs, x_in = seen[-1]
            gt = m.latent_change_target(obs)
            torch.testing.assert_close(x_in[[0, 2]], gt[[0, 2]])
            self.assertFalse(torch.allclose(x_in[1], gt[1]))
            out["action_loss"].backward()
            self.assertGreater(grad_norm(m.system15.velocity_head.parameters()), 0)
        finally:
            m.extract_final_hidden = orig
        out = make_model(stage=None, latent_world={"joint": PURE_ENDPOINT})(make_examples(3))
        self.assertEqual(out["joint/gt_prob"], 0.0)
        self.assertNotIn("joint/gt_frac", out)  # no replacement drawn at p_gt = 0

    def test_eval_and_inference_never_read_gt_dz(self):
        m = make_model(stage=None)  # p_gt = 1 at step 0 in training
        m.eval()
        with torch.no_grad():
            out = m(make_examples(2))
        self.assertEqual(out["joint/gt_prob"], 0.0)
        self.assertNotIn("joint/gt_frac", out)

    def test_eval_reports_ode_and_endpoint(self):
        m = make_model(stage=None)
        with tempfile.TemporaryDirectory() as d:
            metrics = m.trainer_eval_hook(lambda: make_examples(4), 1, d, is_main=True)
        for k in ("eval/system1/rmse_pred", "eval/system1/rmse_endpoint", "eval/system15/endpoint_rel_mse"):
            self.assertTrue(np.isfinite(metrics[k]), k)

    def test_ode_steps_multi_sample_path(self):
        """action_path ode + ode_steps [1, 4]: both samples from the same noise (= sample_latent_change with that noise),
        euler shares the tau = 0 velocity (4 evaluations), one tau = 1 pass over 2 B rows (= extract_final_hidden),
        L_action = mean of the two losses, gradient through every solver step."""
        joint = {"action_path": "ode", "ode_steps": [1, 4], **PURE_ENDPOINT}
        m = make_model(stage=None, system15={"sample_steps": 4}, latent_world={"joint": joint})
        with torch.no_grad():
            for p in m.system15.parameters():
                p.add_(torch.randn_like(p) * 0.02)
        self.assertEqual(m.ode_steps, [1, 4])
        ex = make_examples(3)
        obs = m.encode_observation(ex, with_future=True)
        cond = m.flow_conditions(obs)
        calls = []
        orig = m.system15.velocity_head.forward
        m.system15.velocity_head.forward = lambda h: (calls.append(h.shape[0]), orig(h))[1]
        try:
            torch.manual_seed(7)
            h, dz = m.ode_samples_hidden(obs, cond)
        finally:
            m.system15.velocity_head.forward = orig
        self.assertEqual(calls, [3, 3, 3, 3])  # tau 0 (shared), .25, .5, .75
        self.assertEqual((h.shape[0], dz.shape[0]), (6, 6))
        torch.manual_seed(7)
        x0 = m.sampler.initial_noise(cond, m.feat_dim, m.system15.num_tokens, obs.z_t.device)
        with torch.no_grad():
            for k, n in enumerate([1, 4]):
                ref = m.sample_latent_change(obs, cond, steps=n, noise=x0)
                torch.testing.assert_close(dz[3 * k:3 * k + 3].detach(), ref, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(h[3 * k:3 * k + 3].detach(), m.extract_final_hidden(obs, ref, cond),
                                           rtol=1e-5, atol=1e-5)
        self.assertFalse(torch.allclose(dz[:3], dz[3:]))
        out = m(ex)
        self.assertAlmostEqual(float(out["action_loss"]),
                               (out["system1/action_loss_ode1"] + out["system1/action_loss_ode4"]) / 2, places=5)
        for k in ("system15/ode1_rel_mse", "system15/ode4_rel_mse"):
            self.assertTrue(np.isfinite(out[k]), k)
        out["action_loss"].backward()
        self.assertGreater(grad_norm(m.system15.velocity_head.parameters()), 0)  # through the solver steps
        self.assertGreater(grad_norm(m.system1.parameters()), 0)

    def test_ode_steps_teacher_forcing_and_eval(self):
        m = make_model(stage=None, system15={"sample_steps": 4},
                       latent_world={"joint": {"action_path": "ode", "ode_steps": [1, 4]}})  # p_gt = 1 at step 0
        with torch.no_grad():
            for p in m.system15.parameters():
                p.add_(torch.randn_like(p) * 0.02)
        ex = make_examples(3)
        ex[1]["future_valid"] = False  # clamped: keeps its ODE samples
        out = m(ex)
        self.assertEqual(out["joint/gt_prob"], 1.0)
        self.assertAlmostEqual(out["joint/gt_frac"], 2 / 3)
        out["action_loss"].backward()
        self.assertGreater(grad_norm(m.system15.velocity_head.parameters()), 0)  # only via the clamped row
        m.zero_grad(set_to_none=True)
        for e in ex:
            e["future_valid"] = True
        out = m(ex)
        self.assertEqual(out["joint/gt_frac"], 1.0)
        self.assertAlmostEqual(out["system1/action_loss_ode1"], out["system1/action_loss_ode4"], places=5)  # same GT
        out["action_loss"].backward()
        self.assertEqual(grad_norm(m.system15.velocity_head.parameters()), 0.0)
        with tempfile.TemporaryDirectory() as d:
            metrics = m.trainer_eval_hook(lambda: make_examples(4), 1, d, is_main=True)
        for k in ("eval/system1/rmse_pred", "eval/system1/rmse_pred_ode1", "eval/system15/cos_ode1"):
            self.assertTrue(np.isfinite(metrics[k]), k)
        self.assertNotIn("eval/system1/rmse_pred_ode4", metrics)  # = rmse_pred (sample_steps 4)
        self.assertNotIn("eval/system1/rmse_endpoint", metrics)

    def test_ode_steps_config(self):
        self.assertEqual(make_model("C").ode_steps, [3])  # legacy: [sample_steps]
        self.assertEqual(make_model("C", latent_world={"joint": {"full_grad_steps": 2}}).ode_steps, [2])
        self.assertEqual(make_model("C", latent_world={"joint": {"ode_steps": 2}}).ode_steps, [2])
        for bad in ([0], [4, 4], []):
            with self.assertRaises(ValueError):
                make_model("C", latent_world={"joint": {"ode_steps": bad}})

    def test_staged_recipe_still_available(self):
        for stage in ("A", "B"):
            m = make_model(stage)
            self.assertTrue(torch.isfinite(m(make_examples(2))["loss"]))
        m = make_model("C", latent_world={"joint": {"action_path": "ode", "action_grad": "none"}})
        self.assertTrue(torch.isfinite(m(make_examples(2))["loss"]))


if __name__ == "__main__":
    unittest.main()
