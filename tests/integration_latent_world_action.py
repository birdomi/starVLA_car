"""GPU integration tests of QwenLatentWorldAction with the REAL pretrained checkpoints and RoboCasa data
(Qwen/Qwen3.5-0.8B, C-RADIOv4-SO400M, the QwenGR00T_TriSystem stage-1 System 1.5 converted by
`latent_world_action_tools.py convert-trisystem`). Skipped when CUDA / the files are missing.

    CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python -m unittest tests.integration_latent_world_action -v

The mock-encoder CPU tests are tests/test_latent_world_action.py.
"""

import os
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
Y = ROOT / "examples/simBenchmarks/Robocasa_tabletop/train_files/latent_world_action/stageB_system1_rc100_from_trisystem.yaml"
CONVERTED = ROOT / "playground/Checkpoints/lwa_from_trisystem_s1p5_rc100/system15_steps_30000.pt"
TRI_CKPT = ROOT / "playground/Checkpoints/starvla_trisystem_s1p5_rc100_robocasa_gr1/checkpoints/steps_30000_pytorch_model.pt"
DATA = ROOT / "playground/Datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
READY = torch.cuda.is_available() and CONVERTED.exists() and DATA.exists()


class NoActionDict(dict):
    def __getitem__(self, k):
        if k == "action":
            raise AssertionError("GT action was read at inference")
        return super().__getitem__(k)

    def get(self, k, default=None):
        if k == "action":
            raise AssertionError("GT action was read at inference")
        return super().get(k, default)


def _cfg(**over):
    os.chdir(ROOT)
    cfg = OmegaConf.load(Y)
    for k, v in over.items():
        OmegaConf.update(cfg, k, v, force_add=True)
    return cfg


@unittest.skipUnless(READY, "needs CUDA, the converted System 1.5 checkpoint and the RoboCasa data")
class TestRealCheckpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset
        from starVLA.model.framework.base_framework import build_framework
        from starVLA.training.trainer_utils.trainer_tools import TrainerUtils

        cls.cfg = _cfg()
        model = build_framework(cls.cfg)
        model = TrainerUtils.load_pretrained_backbones(model, str(CONVERTED), reload_modules="system15,radio_encoder.stats")
        cls.model = model.cuda().eval()
        vc = cls.cfg.datasets.val_data
        vc.data_root_dir = Path(vc.data_root_dir)
        cls.val = get_vla_dataset(data_cfg=vc)
        rng = np.random.default_rng(0)
        cls.examples = [cls.val[int(i)] for i in rng.choice(len(cls.val), 8, replace=False)]

    def _batch(self, n=4, valid_only=True):
        ex = [e for e in self.examples if e["future_valid"] or not valid_only]
        return ex[:n]

    # 1
    def test_01_qwen_full_hidden_sequence(self):
        m = self.model
        ex = self._batch(2)
        ex[1] = dict(ex[1], lang=ex[1]["lang"] + " and then do it again carefully please")
        iface = m.qwen_vl_interface
        inputs = iface.build_qwenvl_inputs(images=[[e["image"][0]] for e in ex], instructions=[e["lang"] for e in ex])
        with torch.no_grad():
            s2 = m.encode_system2([[e["image"][0]] for e in ex], [e["lang"] for e in ex])
        L = inputs["input_ids"].shape[1]
        self.assertEqual(tuple(s2.tokens.shape), (2, L, 1024))  # every token, nothing pooled
        torch.testing.assert_close(s2.mask, inputs["attention_mask"].bool())
        self.assertFalse(bool(s2.mask[0, 0]))  # shorter prompt is left padded and masked
        pad_id = iface.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        n_img = (inputs["input_ids"] == pad_id).sum(1).tolist()
        self.assertEqual(n_img, [64, 64])  # 256x256 frame, 16-px patches, 2x2 merge -> 64 vision tokens per sample
        self.assertGreater(int(s2.mask[0].sum()), 64)  # vision AND text tokens
        self.assertTrue(torch.isfinite(s2.tokens[s2.mask]).all())

    # 2
    def test_02_cradio_patch_tokens(self):
        enc = self.model.radio_encoder
        self.assertEqual((enc.feat_dim, enc.grid), (1152, (16, 16)))
        base = torch.full((1, 3, 256, 256), 0.5, device="cuda")
        r, c = 5, 11
        img = base.clone()
        img[:, :, r * 16:(r + 1) * 16, c * 16:(c + 1) * 16] = torch.tensor([1.0, 0.0, 0.0], device="cuda")[:, None, None]
        with torch.no_grad():
            z0, v0 = enc.encode_raw(base)
            z1, v1 = enc.encode_raw(img)
        self.assertEqual(tuple(z1.shape), (1, 256, 1152))
        change = (z1 - z0).norm(dim=-1)[0].masked_fill(~(v0 & v1)[0], 0)
        self.assertEqual(int(change.argmax()), r * 16 + c)  # 2D row-major token order preserved
        self.assertLessEqual(int((~v1).sum()), 6)  # only C-RADIO's artifact tokens are masked
        bad = torch.nonzero(~v1[0]).flatten().tolist()
        self.assertTrue(all(i < 16 for i in bad), bad)  # ... all in the top row

    # 3
    def test_03_latent_change_real(self):
        m = self.model
        ex = self._batch(4)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            obs = m.encode_observation(ex, with_future=True)
            x1 = m.latent_change_target(obs)
            same = [dict(e, image=[e["image"][0], e["image"][0]]) for e in ex]
            x1_same = m.latent_change_target(m.encode_observation(same, with_future=True))
        self.assertEqual(tuple(x1.shape), (4, 256, 1152))
        v = obs.valid & obs.valid_f
        rms = x1[v].float().pow(2).mean().sqrt().item()
        self.assertTrue(0.2 < rms < 5.0, rms)  # unit-scale data after delta_norm rms
        self.assertEqual(float(x1_same.abs().max()), 0.0)

    def test_03b_future_frame_alignment_and_clamping(self):
        ds = self.val.datasets[0]
        traj = ds.trajectory_ids[0]
        T = int(ds.trajectory_lengths[0])
        pos = {(int(t), int(b)): i for i, (t, b) in enumerate(ds.all_steps) if int(t) == int(traj)}
        a = ds[pos[(int(traj), 10)]]
        b = ds[pos[(int(traj), 26)]]
        self.assertEqual(list(a["frame_offsets"]), [0, 16])
        self.assertTrue(a["future_valid"])
        np.testing.assert_array_equal(np.asarray(a["image"][1]), np.asarray(b["image"][0]))  # t+16 of the SAME episode
        self.assertTrue(a["frame_keys"][1].endswith(f"/{int(traj)}/26"))
        last = ds[pos[(int(traj), T - 4)]]
        self.assertFalse(last["future_valid"])  # t + 16 > T - 1 -> clamped, flagged
        self.assertTrue(last["frame_keys"][1].endswith(f"/{int(traj)}/{T - 1}"))  # clamped inside the episode
        edge = ds[pos[(int(traj), T - 17)]]
        self.assertTrue(edge["future_valid"])

    # 4-7
    def test_04_07_system15_with_trained_weights(self):
        m = self.model
        ex = self._batch(4)
        torch.manual_seed(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            obs = m.encode_observation(ex, with_future=True)
            cond = m.flow_conditions(obs)
            x1 = m.latent_change_target(obs)
            mask = obs.valid & obs.valid_f
            v, h = m.system15(x1, torch.full((4,), 0.5, device="cuda"), cond, return_hidden_states=True)
            self.assertEqual((tuple(v.shape), tuple(h.shape)), ((4, 256, 1152), (4, 256, 768)))
            losses = [m.system15.flow_matching_loss(x1, cond, mask, {"schedule": "uniform"}, 4)[0].item() for _ in range(3)]
            x_hat, last_h = m.sample_latent_change(obs, cond, record_last_hidden=True)
            h_f = m.extract_final_hidden(obs, x_hat, cond)
        fm = float(np.mean(losses))
        self.assertTrue(np.isfinite(fm))
        self.assertLess(fm, 1.4)  # trained (TriSystem stage-1 train fm_loss ~0.76); zero velocity would give ~2.0
        self.assertEqual(tuple(x_hat.shape), (4, 256, 1152))
        self.assertEqual(tuple(h_f.shape), (4, 256, 768))
        self.assertFalse(torch.allclose(h_f.float(), last_h.float(), atol=1e-2))
        cos = torch.nn.functional.cosine_similarity((x_hat * mask[..., None]).flatten(1).float(),
                                                    (x1 * mask[..., None]).flatten(1).float(), dim=1)
        self.assertGreater(float(cos.mean()), 0.1)  # predicted change correlates with the real one

    @unittest.skipUnless(TRI_CKPT.exists(), "TriSystem stage-1 checkpoint missing")
    def test_04b_equivalent_to_trisystem_system15(self):
        from starVLA.model.modules.tri_system.system15_flow import ChangeFlowTransformer

        sd = torch.load(TRI_CKPT, map_location="cpu", weights_only=True, mmap=True)
        tri = ChangeFlowTransformer(1152, 1024, (16, 16), 768, 12, 12)
        tri.load_state_dict({k[len("system15.net."):]: v for k, v in sd.items() if k.startswith("system15.net.")})
        tri = tri.cuda().eval()
        m = self.model
        ex = self._batch(2)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                obs = m.encode_observation(ex)
            x = torch.randn(2, 256, 1152, device="cuda")
            tau = torch.tensor([0.25, 0.8], device="cuda")
            ref = tri(x, tau, obs.z_t, obs.valid, obs.s2.tokens.float(), obs.s2.mask)
            out = m.system15(x, tau, m.system15.encode_conditions(obs.z_t, obs.valid, obs.s2.tokens.float(), obs.s2.mask))
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)

    # 8-10
    def test_08_10_stage_b_forward_backward_predict(self):
        m = self.model
        m.train()
        try:
            ex = self._batch(4)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(ex)
            self.assertTrue(torch.isfinite(out["loss"]))
            out["loss"].backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.system1.parameters()))
            for mod in (m.system15, m.qwen_vl_interface, m.radio_encoder):
                self.assertTrue(all(p.grad is None for p in mod.parameters()))
            m.zero_grad(set_to_none=True)
        finally:
            m.eval()
        pred = m.predict_action([{"image": [e["image"][0]], "lang": e["lang"], "state": e["state"]} for e in ex])
        self.assertEqual(pred["normalized_actions"].shape, (4, 16, 29))
        self.assertTrue(np.isfinite(pred["normalized_actions"]).all())

    # 11
    def test_11_inference_ignores_future_and_actions(self):
        m = self.model
        ex = self._batch(2)
        cur = [NoActionDict(image=[e["image"][0]], lang=e["lang"], state=e["state"]) for e in ex]
        rng = np.random.default_rng(3)
        from PIL import Image

        noise_future = [NoActionDict(image=[e["image"][0], Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))],
                                     lang=e["lang"], state=e["state"]) for e in ex]
        a = m.predict_action(cur, seed=0)["normalized_actions"]
        b = m.predict_action(noise_future, seed=0)["normalized_actions"]
        np.testing.assert_array_equal(a, b)

    # 12
    def test_12_loader_unchanged_without_frame_meta(self):
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset

        vc = _cfg().datasets.val_data
        vc.data_root_dir = Path(vc.data_root_dir)
        vc.emit_frame_meta = False
        ds = get_vla_dataset(data_cfg=vc)
        self.assertEqual(set(ds[0].keys()), {"action", "image", "lang", "robot_tag", "state"})

    def test_12b_existing_frameworks_still_run(self):
        """QwenGR00T (rc100 baseline yaml) still builds, trains a step and predicts with the shared Qwen3.5 interface."""
        from starVLA.model.framework.base_framework import build_framework

        cfg = OmegaConf.load(ROOT / "examples/simBenchmarks/Robocasa_tabletop/train_files/"
                                    "starvla_qwen35_gr00t_rc100_50k_dit_v2_robocasa.yaml")
        gr = build_framework(cfg).cuda()
        ex = [{"image": [e["image"][0].resize((224, 224))], "lang": e["lang"], "state": e["state"], "action": e["action"]}
              for e in self._batch(2)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = gr(ex)
        self.assertTrue(torch.isfinite(out["action_loss"]))
        self.assertEqual(gr.predict_action(examples=ex)["normalized_actions"].shape, (2, 16, 29))
        del gr
        torch.cuda.empty_cache()


JOINT_Y = ROOT / "examples/simBenchmarks/Robocasa_tabletop/train_files/latent_world_action/joint_rc100.yaml"


@unittest.skipUnless(torch.cuda.is_available() and DATA.exists(), "needs CUDA and the RoboCasa data")
class TestRealJointFromScratch(unittest.TestCase):
    """Default recipe: stage C only, System 1.5 / System 1 randomly initialised, real frozen Qwen3.5 + C-RADIOv4."""

    @classmethod
    def setUpClass(cls):
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset
        from starVLA.model.framework.base_framework import build_framework

        os.chdir(ROOT)
        cls.cfg = OmegaConf.load(JOINT_Y)
        assert not cls.cfg.trainer.get("pretrained_checkpoint", None)
        torch.manual_seed(0)
        cls.model = build_framework(cls.cfg).cuda().train()  # no checkpoint of System 1.5 / System 1
        vc = cls.cfg.datasets.val_data
        vc.data_root_dir = Path(vc.data_root_dir)
        ds = get_vla_dataset(data_cfg=vc)
        cls.examples = [ds[int(i)] for i in np.random.default_rng(1).choice(len(ds), 4, replace=False)]

    @staticmethod
    def _norm(params):
        g = [p.grad.float().pow(2).sum() for p in params if p.grad is not None]
        return float(torch.stack(g).sum().sqrt()) if g else 0.0

    def test_joint_action_loss_reaches_system15_and_one_step_updates_both(self):
        from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

        m = self.model
        self.assertEqual((m.stage, m.action_path, m.ode_steps, m.action_grad), ("C", "ode", [1, 4], "full"))
        j = m.config.framework.latent_world.joint
        saved = (j.gt_prob_schedule, j.gt_prob_end)
        j.gt_prob_schedule, j.gt_prob_end = "fixed", 0.0  # the pure ODE path (the yaml starts at p_gt = 1 -> GT)
        m.zero_grad(set_to_none=True)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(self.examples)
        finally:
            j.gt_prob_schedule, j.gt_prob_end = saved
        self.assertTrue(torch.isfinite(out["fm_loss"]) and torch.isfinite(out["action_loss"]))
        out["action_loss"].backward(retain_graph=True)  # action loss ONLY
        head = list(m.system15.velocity_head.parameters())
        backbone = [p for p in m.system15.parameters() if all(p is not h for h in head)]
        norms = {"system1": self._norm(m.system1.parameters()), "system15_backbone": self._norm(backbone),
                 "velocity_head": self._norm(head)}
        print("\n[real joint] action-only grad norms:", norms, "fm_loss", out["fm_loss"].item(),
              "action_loss", out["action_loss"].item())
        for k, v in norms.items():
            self.assertTrue(np.isfinite(v) and v > 0, (k, v))
        for mod in (m.qwen_vl_interface, m.radio_encoder):
            self.assertTrue(all(p.grad is None for p in mod.parameters()))
        # one full optimizer step with the trainer's param groups
        m.zero_grad(set_to_none=True)
        groups = build_param_lr_groups(m, self.cfg)
        self.assertEqual(sorted(g["name"] for g in groups), ["system1", "system15"])
        opt = torch.optim.AdamW(groups)
        snap = {k: p.detach().clone() for k, p in m.named_parameters()
                if k.startswith(("system15.velocity_head", "system15.blocks.11", "system1.head", "system1.queries"))}
        out["loss"].backward()
        opt.step()
        for k, before in snap.items():
            self.assertFalse(torch.equal(before, dict(m.named_parameters())[k]), k)
        m.zero_grad(set_to_none=True)

    def test_joint_scheduled_gt(self):
        """yaml schedule: p_gt = 1 at step 0 (System 1 reads GT X_1 for every valid future) -> 0.05 at decay end."""
        m = self.model
        j = m.config.framework.latent_world.joint
        fv = float(np.mean([bool(e.get("future_valid", True)) for e in self.examples]))
        try:
            for step, p in ((0, 1.0), (int(j.gt_prob_decay_steps), 0.05)):
                m.set_training_step(step)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out = m(self.examples)
                self.assertAlmostEqual(out["joint/gt_prob"], p)
                if p == 1.0:
                    self.assertAlmostEqual(out["joint/gt_frac"], fv)
                print(f"\n[real joint] step {step}: p_gt {out['joint/gt_prob']:.3f}, gt_frac {out.get('joint/gt_frac', 0.0)}")
        finally:
            m._trainer_step = None

    def test_joint_inference_is_ode(self):
        m = self.model
        m.eval()
        try:
            calls = []
            orig = m.sampler.sample
            m.sampler.sample = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
            out = m.predict_action([{"image": [e["image"][0]], "lang": e["lang"], "state": e["state"]}
                                    for e in self.examples])
        finally:
            m.sampler.sample = orig
            m.train()
        self.assertEqual(len(calls), 1)
        self.assertEqual(out["normalized_actions"].shape, (4, 16, 29))


if __name__ == "__main__":
    unittest.main()
