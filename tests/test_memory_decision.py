"""Unit tests of the memory pretraining's Joint Decision Transformer (memory_pretrain.py) and its training step (CPU, no
pretrained weights: a tiny random T5 and a word-hash tokenizer stand in for flan-t5).

    PYTHONPATH=. python tests/test_memory_decision.py        (unittest runner)
"""
import os
import unittest

import torch
import torch.nn.functional as F

from starVLA.model.modules.dual_system.memory import EpisodeMemory
from starVLA.model.modules.dual_system.memory_pretrain import (NUM_QUERIES, QUERY_DONE, QUERY_SUBTASK,
                                                               FrozenT5Encoder, MultiVLMMemory, done_loss,
                                                               expected_progress, masked_mean, progress_bin_targets,
                                                               progress_loss, subtask_loss)

DIMS = {"vlm_a": 24, "vlm.b": 40}
TOK, D, N, T5D = 32, 48, 4, 16


def make_model(**kw):
    torch.manual_seed(0)
    return MultiVLMMemory(DIMS, token_dim=TOK, num_tokens=N, d_model=D, depth=2, num_heads=4, mlp_ratio=2.0,
                          progress_bins=10, attention="gated", t5_dim=T5D, decision_layers=2,
                          decision_heads=4, decision_mlp_ratio=4.0, decision_progress_bins=20, **kw)


class WordTokenizer:
    """Whitespace words -> ids by hash; right padding; T5-style trailing EOS (id 1)."""

    def __call__(self, texts, padding=True, truncation=True, max_length=64, return_tensors="pt"):
        rows = [[2 + (hash(w) % 97) for w in t.split()][: max_length - 1] + [1] for t in texts]
        n = max(len(r) for r in rows)
        ids = torch.tensor([r + [0] * (n - len(r)) for r in rows])
        att = torch.tensor([[1] * len(r) + [0] * (n - len(r)) for r in rows])
        return {"input_ids": ids, "attention_mask": att}


def make_t5(cache_size=100):
    from transformers import T5Config, T5EncoderModel

    torch.manual_seed(1)
    cfg = T5Config(vocab_size=100, d_model=T5D, d_kv=4, d_ff=32, num_layers=2, num_heads=4, dropout_rate=0.0)
    return FrozenT5Encoder(T5EncoderModel(cfg), WordTokenizer(), torch.device("cpu"), cache_size=cache_size)


def tokens(b, length, name="vlm_a", seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(b, length, DIMS[name], generator=g)
    mask = torch.ones(b, length, dtype=torch.bool)
    return h, mask


class TestDecision(unittest.TestCase):
    def test_memory_encoder_checkpoint_compat(self):
        # (1) the shared memory keeps EpisodeMemory's exact layout; the export loads strictly into the action model's one
        model = make_model()
        ref = EpisodeMemory(TOK, TOK, num_tokens=N, d_model=D, depth=2, num_heads=4, mlp_ratio=2.0, progress_bins=10,
                            attention="gated")
        self.assertEqual(set(model.memory.state_dict()), set(ref.state_dict()))
        for name, h in DIMS.items():
            car = EpisodeMemory(h, TOK, token_dim=TOK, num_tokens=N, d_model=D, depth=2, num_heads=4, mlp_ratio=2.0,
                                progress_bins=10, attention="gated")
            sd = {k[len("episode_memory."):]: v for k, v in model.export_for_vlm(name).items()}
            car.load_state_dict(sd, strict=True)
            x, mask = tokens(2, 5, name)
            m0 = model.init(2)
            torch.testing.assert_close(car.update(m0, x, mask), model.update(name, m0, x, mask))

    def test_shapes_and_batch_one(self):
        # (2, 3, 8, 10, 13) query / feature / head shapes, B = 1 and token lengths changing between calls
        model, t5 = make_model(), make_t5()
        self.assertEqual(NUM_QUERIES, 4)
        self.assertEqual(tuple(model.decision.decision_queries.shape), (4, D))
        m = model.init(1)
        for length in (3, 7, 5):
            x, mask = tokens(1, length)
            feats, valid = t5.candidates([["grasp the cup", "lift it"]])
            out = model("vlm_a", m, x, mask, cand_feats=feats, cand_valid=valid)
            m = out["memory"]
            self.assertEqual(tuple(out["memory"].shape), (1, N, D))
            self.assertEqual(tuple(out["decision_features"].shape), (1, 4, D))
            self.assertEqual(tuple(out["subtask_logits"].shape), (1, 2))
            self.assertEqual(tuple(out["done_logits"].shape), (1, 2))
            for k in ("progress", "subtask_progress"):
                self.assertEqual(tuple(out[f"{k}_logits"].shape), (1, 20))
                self.assertEqual(tuple(out[k].shape), (1,))
            self.assertNotIn("elapsed_time", out)
            self.assertNotIn("retrieval", out)

    def test_variable_candidates_and_padding(self):
        # (4) different candidate counts per sample: padded candidates get no probability, an empty list is all padding
        model, t5 = make_model(), make_t5()
        lists = [["a b", "c d e", "f"], ["g h"], []]
        feats, valid = t5.candidates(lists)
        self.assertEqual(tuple(feats.shape), (3, 3, T5D))
        self.assertEqual(valid.tolist(), [[True, True, True], [True, False, False], [False, False, False]])
        z = torch.randn(3, 4, D)
        logits = model.subtask_logits(z, feats, valid)
        probs = logits[:2].softmax(-1)
        self.assertTrue(torch.all(probs[1, 1:] == 0))
        torch.testing.assert_close(probs.sum(-1), torch.ones(2))
        # only rows with a target count; none at all keeps the loss in the graph at 0
        has = torch.tensor([True, True, False])
        loss = subtask_loss(logits, torch.tensor([2, 0, 0]), has)
        self.assertTrue(torch.isfinite(loss))
        zero = subtask_loss(logits, torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.bool))
        self.assertEqual(float(zero), 0.0)
        # finished subtasks: BCE over real candidates of rows with a target only
        dl = model.done_logits(z, feats)
        done = torch.tensor([[True, False, False], [False, False, False], [False, False, False]])
        ref = F.binary_cross_entropy_with_logits(torch.cat([dl[0], dl[1, :1]]), torch.tensor([1.0, 0, 0, 0]))
        torch.testing.assert_close(done_loss(dl, done, valid, has), ref)
        self.assertEqual(float(done_loss(dl, done, valid, torch.zeros(3, dtype=torch.bool))), 0.0)

    def test_t5_masked_mean_per_sentence(self):
        # (5) each sentence pooled over its own tokens: the same as alone, whatever it is batched (padded) with
        t5 = make_t5(cache_size=0)
        alone = t5.encode(["pick up the cup"])[0]
        batched = t5.encode(["pick up the cup", "open the drawer and take out the red marker slowly"])[0]
        torch.testing.assert_close(alone, batched, atol=1e-5, rtol=1e-5)
        enc = t5.tok(["pick up the cup"])
        h = t5.model(**enc).last_hidden_state
        manual = h[0, : int(enc["attention_mask"].sum())].mean(0)
        torch.testing.assert_close(alone, manual.float(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(masked_mean(h, enc["attention_mask"].bool())[0], manual)

    def test_t5_cache(self):
        t5 = make_t5(cache_size=2)
        a = t5.encode(["x y", "z"])
        self.assertEqual(set(t5.cache), {"x y", "z"})
        b = t5.encode(["z", "x y", "w"])
        torch.testing.assert_close(a[0], b[1])
        self.assertEqual(len(t5.cache), 2)

    def test_gradients(self):
        # (6) frozen T5 gets none; candidate projection, subtask query projection, decision queries and memory do
        model, t5 = make_model(), make_t5()
        feats, valid = t5.candidates([["a", "b c"], ["d e f", "g", "h"]])
        x, mask = tokens(2, 6)
        out = model("vlm_a", model.init(2), x, mask, cand_feats=feats, cand_valid=valid)
        has = torch.tensor([True, True])
        (subtask_loss(out["subtask_logits"], torch.tensor([1, 2]), has)
         + done_loss(out["done_logits"], torch.tensor([[True, False, False], [True, True, False]]), valid, has)).backward()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in t5.model.parameters()))
        self.assertGreater(float(model.done_bias.grad.abs()), 0)
        for mod in (model.candidate_proj, model.subtask_query_proj, model.done_query_proj):
            self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters()))
        self.assertGreater(float(model.decision.decision_queries.grad.abs().sum()), 0)
        self.assertGreater(float(model.memory.init_tokens.grad.abs().sum()), 0)

    def test_similarity(self):
        # (7) logits = cos(subtask_query_proj(z_subtask), candidate_proj(c)) / temperature
        model = make_model(subtask_temperature=0.5)
        z, c = torch.randn(2, 4, D), torch.randn(2, 4, T5D)
        valid = torch.ones(2, 4, dtype=torch.bool)
        q = model.subtask_query_proj(z[:, QUERY_SUBTASK])
        ce = model.candidate_proj(c)
        ref = F.cosine_similarity(q[:, None], ce, dim=-1) / 0.5
        torch.testing.assert_close(model.subtask_logits(z, c, valid), ref, atol=1e-5, rtol=1e-5)
        qd = model.done_query_proj(z[:, QUERY_DONE])
        refd = F.cosine_similarity(qd[:, None], ce, dim=-1) / 0.5 + model.done_bias
        torch.testing.assert_close(model.done_logits(z, c), refd, atol=1e-5, rtol=1e-5)
        with self.assertRaises(ValueError):
            make_model(subtask_temperature=0.0)

    def test_progress_bins_and_expectation(self):
        # (9) 0 -> bin 0, 1 -> bin 19, edges to the upper bin; expected progress over the bin centres
        p = torch.tensor([0.0, 0.049, 0.05, 0.5, 0.95, 0.999, 1.0, 1.3, -0.2])
        self.assertEqual(progress_bin_targets(p, 20).tolist(), [0, 0, 1, 10, 19, 19, 19, 19, 0])
        logits = torch.full((1, 20), -1e4)
        logits[0, 19] = 0
        probs, e = expected_progress(logits)
        self.assertAlmostEqual(float(e), 0.975, places=4)
        torch.testing.assert_close(probs.sum(-1), torch.ones(1))
        loss = progress_loss(torch.randn(3, 20), torch.tensor([0, 5, 19]), torch.tensor([1.0, 0.0, 1.0]))
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):
            progress_loss(torch.randn(1, 20), torch.tensor([0]), torch.ones(1), kind="ordinal")

    def test_progress_heads_learn(self):
        # (10) episode / subtask progress heads fit their bins on a fixed batch, each from its own query
        model = make_model()
        x, mask = tokens(4, 5)
        tgt = progress_bin_targets(torch.tensor([0.0, 0.3, 0.6, 1.0]), 20)
        stgt = progress_bin_targets(torch.tensor([0.9, 0.1, 0.5, 0.2]), 20)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        first = None
        for _ in range(60):
            out = model("vlm_a", model.init(4), x, mask)
            loss = (progress_loss(out["progress_logits"], tgt, torch.ones(4))
                    + progress_loss(out["subtask_progress_logits"], stgt, torch.ones(4)))
            first = float(loss.detach()) if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(float(loss), 0.2 * first)
        self.assertEqual(out["subtask_progress_logits"].argmax(-1).tolist(), stgt.tolist())

    def test_episode_progress_needs_no_candidates(self):
        # unannotated episodes: the progress heads work without candidates, the subtask heads are absent
        model = make_model()
        x, mask = tokens(2, 5)
        out = model("vlm_a", model.init(2), x, mask)
        self.assertNotIn("subtask_logits", out)
        self.assertNotIn("done_logits", out)
        self.assertTrue(torch.isfinite(out["progress"]).all())

    def test_mixed_precision(self):
        # (12) bf16 autocast: finite outputs, float32 losses
        model, t5 = make_model(), make_t5()
        feats, valid = t5.candidates([["a b"], ["c", "d"]])
        x, mask = tokens(2, 5)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = model("vlm_a", model.init(2), x, mask, cand_feats=feats, cand_valid=valid)
            loss = subtask_loss(out["subtask_logits"], torch.tensor([0, 1]), torch.tensor([True, True]))
        self.assertTrue(all(torch.isfinite(out[k].float()).all()
                            for k in ("progress_logits", "subtask_progress_logits", "done_logits")))
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()

    def test_no_future_or_padding_leak(self):
        # (14) decisions at t depend on frames <= t only, and not on the values under the padding mask
        model = make_model()
        frames = [tokens(2, 5, seed=s) for s in range(4)]
        def run(fr):
            m, outs = model.init(2), []
            for x, mask in fr:
                o = model("vlm_a", m, x, mask)
                m = o["memory"]
                outs.append(o["decision_features"])
            return outs
        base = run(frames)
        changed = list(frames)
        changed[3] = (frames[3][0] + 5.0, frames[3][1])
        new = run(changed)
        for t in range(3):
            torch.testing.assert_close(base[t], new[t])
        self.assertFalse(torch.allclose(base[3], new[3]))
        x, mask = tokens(2, 6, seed=9)
        mask[:, 4:] = False
        x2 = x.clone()
        x2[:, 4:] = 100.0
        m0 = model.init(2)
        torch.testing.assert_close(model("vlm_a", m0, x, mask)["decision_features"],
                                   model("vlm_a", m0, x2, mask)["decision_features"])
        # the decisions read the memory tokens only: the same memory gives the same decisions, whatever the frame
        out = model("vlm_a", m0, x, mask)
        torch.testing.assert_close(out["decision_features"], model.decide(out["memory"]))


def _batch(step):
    return [
        {"image": None, "lang": "t", "progress": step / 4, "elapsed": float(step), "stream": torch.tensor([0, step, 0]),
         "subtasks": ["reach", "grasp", "lift"], "sub_cur": min(step // 2, 2), "sub_progress": (step % 2) / 2,
         "sub_done": [step >= 2, step >= 4, False]},
        {"image": None, "lang": "t", "progress": step / 4, "elapsed": float(step), "stream": torch.tensor([1, step, 0]),
         "subtasks": [], "sub_cur": 0, "sub_progress": 0.0, "sub_done": []},
        {"image": None, "lang": "t", "progress": 0.0, "elapsed": 0.0, "stream": torch.tensor([2, 0, step]),
         "subtasks": ["open", "pour"], "sub_cur": 0, "sub_progress": 0.1, "sub_done": [False, False]},
    ]


class TestFusedStep(unittest.TestCase):
    def test_fused_equals_mean_of_per_vlm(self):
        # all VLMs stacked in one unroll == the mean over the VLMs of each one alone (loss and gradients), over steps
        import copy

        import starVLA.training.train_memory_pretrain as T

        model, t5 = make_model(), make_t5()
        w = {"subtask": 1.0, "subtask_done": 1.0, "progress": 1.0, "subtask_progress": 1.0}
        fused = {n: T.Streams(tbptt=3) for n in DIMS}
        alone = {n: T.Streams(tbptt=3) for n in DIMS}
        for step in range(5):
            batch = _batch(step)
            hidden = {}
            for k, n in enumerate(DIMS):  # different token counts per VLM, some padding
                h, mask = tokens(3, 4 + k + step % 2, n, seed=10 * step + k)
                mask[1, -1] = False
                hidden[n] = (h, mask)
            model.zero_grad()
            loss_f, logs = T.step_all(model, hidden, fused, batch, t5, w, torch.device("cpu"))
            loss_f.backward()
            g_f = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
            model.zero_grad()
            parts = [T.step_all(model, {n: hidden[n]}, {n: alone[n]}, batch, t5, w, torch.device("cpu"))[0]
                     for n in DIMS]
            loss_a = sum(parts) / len(parts)
            loss_a.backward()
            torch.testing.assert_close(loss_f, loss_a, atol=1e-5, rtol=1e-5)
            for k, p in model.named_parameters():
                if p.grad is not None:
                    torch.testing.assert_close(g_f[k], p.grad, atol=1e-5, rtol=1e-4, msg=k)
            self.assertIn("vlm.b/subtask_done_acc", logs)
        for n in DIMS:  # the same stream states afterwards
            for slot in fused[n].s:
                a, b = fused[n].s[slot], alone[n].s[slot]
                self.assertEqual(len(a["frames"]), len(b["frames"]))
                if a["base"] is not None:
                    torch.testing.assert_close(a["base"], b["base"], atol=1e-5, rtol=1e-5)


class TestTrainStep(unittest.TestCase):
    def test_step_vlm_streams_and_losses(self):
        # the training step: TBPTT window, losses present, subtask only where annotated; a gloo all-reduce of world 1
        import starVLA.training.train_memory_pretrain as T

        model, t5 = make_model(), make_t5()
        streams = T.Streams(tbptt=3)
        w = {"subtask": 1.0, "subtask_done": 1.0, "progress": 1.0, "subtask_progress": 1.0}
        calls = {"n": 0}

        def fake_frames(iface, images, langs):
            calls["n"] += 1
            h, mask = tokens(len(images), 4 + calls["n"] % 3, seed=calls["n"])
            mask[0, -1] = False
            return h, mask

        T.vlm_frames, orig = fake_frames, T.vlm_frames
        try:
            for step in range(5):
                batch = [
                    {"image": None, "lang": "t", "progress": step / 4, "elapsed": float(step),
                     "stream": torch.tensor([0, step, 0]), "subtasks": ["reach", "grasp", "lift"],
                     "sub_cur": min(step // 2, 2), "sub_progress": (step % 2) / 2,
                     "sub_done": [step >= 2, step >= 4, False]},
                    {"image": None, "lang": "t", "progress": step / 4, "elapsed": float(step),
                     "stream": torch.tensor([1, step, 0]), "subtasks": [], "sub_cur": 0, "sub_progress": 0.0,
                     "sub_done": []},
                ]
                loss, logs = T.step_vlm(model, "vlm_a", None, streams, batch, t5, w, torch.device("cpu"))
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(logs["vlm_a/subtask_frac"], 0.5)
                for k in ("progress_mae", "subtask_acc", "subtask_done_acc", "subtask_progress_mae"):
                    self.assertIn(f"vlm_a/{k}", logs)
                model.zero_grad()
                loss.backward()
            self.assertEqual(len(streams.s[0]["frames"]), 2)  # tbptt - 1 cached
            self.assertIsNotNone(streams.s[0]["base"])
        finally:
            T.vlm_frames = orig
        import torch.distributed as dist

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29581")
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            params = [p for p in model.parameters() if p.requires_grad]
            before = [None if p.grad is None else p.grad.clone() for p in params]
            T.allreduce_grads(params, 1)
            for p, g in zip(params, before):
                torch.testing.assert_close(p.grad, torch.zeros_like(p) if g is None else g)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
