"""
QwenGR00T_EpisodeMemory: QwenGR00T + the episode memory of QwenGR00T_DualSystem (starVLA/model/modules/dual_system/
memory.py, EpisodeMemory; attention "gated" = a sigmoid output gate per head channel, as in Qwen3.5's full-attention
layers), read by the action DiT.

  * Frames: the data config's video delta indices must equal framework.episode_memory.update_offsets (ascending, the
    last one 0 = the current frame), camera-major (per camera oldest first). Default [-60, -45, -30, -15, 0] at 30 Hz
    = every 0.5 s over the last 2 s (the car DualSystem memory window).
  * VLM: the current frame gets exactly QwenGR00T's call (same batch, same inputs), so its last hidden states (the DiT's
    vl_embs) are those of the checkpoint the run starts from; the past frames run in one extra batched call without
    gradient (episode_memory.past_grad: true gives them gradient when the VLM is trainable).
  * Memory: M = init; M = update(M, H_frame) for every frame, oldest first; a frame before the episode start (loader
    frame_valid False, datasets.vla_data.emit_action_valid) keeps M. Readout = the memory tokens + one progress token
    -> the DiT's memory cross-attention (framework.action_model.diffusion_model_cfg.memory_cond, memory_dim = the VLM
    hidden size), whose output projection starts at zero: a QwenGR00T checkpoint (trainer.pretrained_checkpoint, the
    memory keys missing) predicts exactly as before until training moves it.
  * Progress: t / (T - 1) classified into progress_bins bins from the memory, cross-entropy x progress_loss_weight
    (datasets.vla_data.emit_progress: true).

Two training modes:
  * window (default): the memory is rebuilt from its initial state over the K frames of every sample.
    Inference: example["image"] = the same K frames per camera, camera-major (an episode start repeats the first frame,
    as the training loader clamps; example["frame_valid"] [V * K], False for those, makes the memory skip them).
  * stream (episode_memory.stream: {tbptt: 4}; update_offsets [0]; datasets.vla_data.stream: {stride: S}, in_order):
    the memory runs CONTINUOUSLY from the episode start, as deployed. The loader walks every batch position through
    episodes from frame 0 in steps of S frames (sample["stream"] = [slot, step, episode counter, step in the episode,
    grid steps]); each position carries its memory across training steps, one update per step on the current frame
    (the VLM tokens the DiT reads, no extra VLM pass); gradient through the last tbptt updates (truncated BPTT over
    cached VLM tokens, the older memory detached). A position whose stream lost its history (its first episode joined
    mid-way by random_start, reordered batches) is left out of every loss until its next episode starts. Needs a
    frozen VLM. Inference: predict_action(examples, memory=M) with M = memory_init(B) at the episode start and then the
    returned "memory", one call per S frames (the training update interval).
Returns "normalized_actions" and "progress" (expected episode progress), plus "memory" in stream mode.
"""

import contextlib
import time
from typing import List, Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.modules.dual_system.memory import EpisodeMemory
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images

_DEFAULTS = {
    "update_offsets": [-60, -45, -30, -15, 0],
    "num_tokens": 16,
    "d_model": 768,
    "depth": 2,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "progress_bins": 50,
    "progress_loss_weight": 0.1,
    "attention": "gated",
    "past_grad": False,
    "stream": None,
}


@FRAMEWORK_REGISTRY.register("QwenGR00T_EpisodeMemory")
class Qwen_GR00T_EpisodeMemory(Qwen_GR00T):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config)
        fw = self.config.framework
        mc = {**_DEFAULTS, **dict(fw.get("episode_memory", None) or {})}
        self.offsets = [int(o) for o in mc["update_offsets"]]
        if not self.offsets or self.offsets[-1] != 0 or any(a >= b for a, b in zip(self.offsets, self.offsets[1:])):
            raise ValueError(f"episode_memory.update_offsets must be ascending and end at 0 (the current frame), "
                             f"got {self.offsets}")
        vl_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        dit = fw.action_model.diffusion_model_cfg
        if not dit.get("memory_cond", False) or int(dit.get("memory_dim", 0) or 0) != vl_dim:
            raise ValueError("QwenGR00T_EpisodeMemory needs framework.action_model.diffusion_model_cfg.memory_cond: true "
                             f"and memory_dim: {vl_dim} (the VLM hidden size)")
        self.num_views = int(fw.get("num_views", 0) or 0)  # 0 = infer from the image count
        self.progress_weight = float(mc["progress_loss_weight"])
        self.past_grad = bool(mc["past_grad"])
        st = mc["stream"]
        self.mem_tbptt = int(st.get("tbptt", 4)) if st else 0
        if self.mem_tbptt:
            if self.offsets != [0]:
                raise ValueError(f"episode_memory.stream: update_offsets must be [0] (one update per step on the current "
                                 f"frame; the loader's stream stride sets the interval), got {self.offsets}")
            if self.mem_tbptt < 1:
                raise ValueError(f"episode_memory.stream.tbptt must be >= 1, got {self.mem_tbptt}")
        self._mem_streams = {}  # stream slot -> carried memory state (training, stream mode)
        self.episode_memory = EpisodeMemory(
            vl_dim, vl_dim, num_tokens=int(mc["num_tokens"]), d_model=int(mc["d_model"]), depth=int(mc["depth"]),
            num_heads=int(mc["num_heads"]), mlp_ratio=float(mc["mlp_ratio"]), progress_bins=int(mc["progress_bins"]),
            attention=str(mc["attention"]))

    # ------------------------------------------------------------------
    def _split_frames(self, examples: List[dict]):
        """camera-major images -> (current-frame images [B][V], past-frame images [(K - 1) * B][V] frame-major,
        frame_valid [B, K] bool)."""
        k = len(self.offsets)
        cur, past, fv = [], [[] for _ in range(k - 1)], []
        for ex in examples:
            imgs = list(ex["image"])
            if len(imgs) % k or (self.num_views and len(imgs) != self.num_views * k):
                raise ValueError(f"QwenGR00T_EpisodeMemory: {len(imgs)} images per example, expected "
                                 f"{self.num_views or 'V'} camera(s) x {k} frames (offsets {self.offsets}), camera-major")
            v = len(imgs) // k
            per = [[imgs[c * k + j] for c in range(v)] for j in range(k)]
            cur.append(per[-1])
            for j in range(k - 1):
                past[j].append(per[j])
            if "frame_valid" in ex:
                f = np.asarray(ex["frame_valid"], dtype=bool).reshape(v, k).all(0)
            else:
                f = np.ones(k, dtype=bool)
            fv.append(f)
        return cur, [im for frame in past for im in frame], torch.as_tensor(np.stack(fv))

    def _vlm(self, images, instructions, grad: bool):
        """One VLM call -> (last hidden states [N, L, H], attention mask [N, L] bool)."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
        with (contextlib.nullcontext() if grad else torch.no_grad()), \
                torch.autocast("cuda", dtype=getattr(self.qwen_vl_interface, "autocast_dtype", torch.bfloat16)):
            out = self.qwen_vl_interface(**qwen_inputs, output_attentions=False, output_hidden_states=True,
                                         return_dict=True)
        mask = qwen_inputs.get("attention_mask", None)
        h = out.hidden_states[-1]
        mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device) if mask is None else mask.to(torch.bool)
        return h, mask

    def _memory(self, cur_h, cur_mask, past_h, past_mask, fv, grad: bool):
        """Unroll the memory over the past frames (oldest first) and the current one -> (context [B, N + 1, H],
        progress logits [B, bins])."""
        b, k = cur_h.shape[0], len(self.offsets)
        fv = fv.to(cur_h.device)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.float32):
            m = self.episode_memory.init(b).to(cur_h.device)
            if k > 1:
                ph, pm = past_h.view(k - 1, b, *past_h.shape[1:]), past_mask.view(k - 1, b, -1)
                for j in range(k - 1):
                    m = self.episode_memory.update(m, ph[j], pm[j], fv[:, j])
            m = self.episode_memory.update(m, cur_h, cur_mask, fv[:, -1])
            return self.episode_memory.readout(m)

    def _memory_stream(self, examples: List[dict], tok, mask, train: bool):
        """stream mode (training): every sample continues the memory of its stream slot by one update on its current
        frame (tokens tok [B, L, H], mask [B, L]); gradient through the last mem_tbptt updates (cached tokens) from the
        detached older memory. A stream restarts from the initial memory at its episode's step 0; a stream that does not
        continue its previous step is invalid until then. -> (context [B, N + 1, H], progress logits, valid [B] bool)."""
        if self._vlm_trainable():
            raise ValueError("episode_memory.stream needs a frozen VLM (trainer.freeze_modules: qwen_vl_interface): "
                             "it caches the VLM tokens across steps")
        if "stream" not in examples[0]:
            raise KeyError("episode_memory.stream: examples need 'stream' (datasets.vla_data.stream + in_order: true)")
        b, kmax, dev = len(examples), self.mem_tbptt, tok.device
        info = [np.asarray(ex["stream"], dtype=np.int64) for ex in examples]
        states, seqs = [], []
        for i in range(b):
            slot, k, j, es = (int(x) for x in info[i][:4])
            st = self._mem_streams.get(slot)
            if es == 0 or st is None or not st["valid"] or st["key"] != (k - 1, j, es - 1):
                st = {"base": None, "frames": [], "valid": es == 0}
            st["key"] = (k, j, es)
            states.append(st)
            seqs.append((st["frames"] + [tok[i][mask[i]].detach()])[-kmax:])  # valid tokens only
        n = max(len(q) for q in seqs)
        length = max(t_.shape[0] for q in seqs for t_ in q)
        tk = tok.new_zeros(n, b, length, tok.shape[-1])
        mk = torch.zeros(n, b, length, dtype=torch.bool, device=dev)
        vk = torch.zeros(n, b, dtype=torch.bool, device=dev)
        for i, q in enumerate(seqs):  # right-aligned: the current frame last; missing slots skip their update
            for s_, t_ in enumerate(q, start=n - len(q)):
                tk[s_, i, : t_.shape[0]], mk[s_, i, : t_.shape[0]], vk[s_, i] = t_, True, True
        init = self.episode_memory.init(1).to(dev)[0]
        with torch.set_grad_enabled(train and torch.is_grad_enabled()), torch.autocast("cuda", dtype=torch.float32):
            m = torch.stack([init if st["base"] is None else st["base"] for st in states])
            first = m
            for s_ in range(n):
                m = self.episode_memory.update(m, tk[s_], mk[s_], vk[s_])
                if s_ == 0:
                    first = m
            ctx, logits = self.episode_memory.readout(m)
        for i, (st, q) in enumerate(zip(states, seqs)):
            if len(q) == kmax:  # the oldest cached frame leaves the window -> folded into the detached memory
                st["base"], st["frames"] = first[i].detach(), q[1:]
            else:
                st["frames"] = q
            self._mem_streams[int(info[i][0])] = st
        return ctx, logits, torch.tensor([st["valid"] for st in states], device=dev)

    def memory_init(self, b: int = 1) -> torch.Tensor:
        """Initial episode memory [B, N, d] (stream-mode inference: pass it to predict_action at the episode start)."""
        return self.episode_memory.init(b).to(next(self.episode_memory.parameters()).device)

    def _vlm_trainable(self) -> bool:
        return any(p.requires_grad for p in self.qwen_vl_interface.parameters())

    def _encode(self, examples: List[dict], train: bool, resize: bool = False):
        """-> (vl_embs [B, L, H], mask [B, L], memory context [B, N + 1, H], progress logits [B, bins], timings)."""
        cur, past, fv = self._split_frames(examples)
        instructions = [ex["lang"] for ex in examples]
        if resize:
            size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
            cur = [to_pil_preserve(x) for x in cur]
            past = [to_pil_preserve(x) for x in past]
            if size:
                cur, past = resize_images(cur, target_size=size), resize_images(past, target_size=size)
        vlm_grad = train and self._vlm_trainable() and torch.is_grad_enabled()
        t0 = time.perf_counter()
        cur_h, cur_mask = self._vlm(cur, instructions, grad=vlm_grad)
        t1 = time.perf_counter()
        past_h = past_mask = None
        if past:
            past_h, past_mask = self._vlm(past, [s for _ in range(len(self.offsets) - 1) for s in instructions],
                                          grad=vlm_grad and self.past_grad)
        t2 = time.perf_counter()
        ctx, logits = self._memory(cur_h, cur_mask, past_h, past_mask, fv, grad=train and torch.is_grad_enabled())
        timings = {"timing/vlm_fwd": t1 - t0, "timing/vlm_past_fwd": t2 - t1, "timing/memory": time.perf_counter() - t2}
        return cur_h, cur_mask, ctx, logits, fv, timings

    # ------------------------------------------------------------------
    def forward(self, examples: List[dict] = None, **kwargs):
        if isinstance(examples, dict):
            raise ValueError("QwenGR00T_EpisodeMemory does not support datasets.vla_data.preprocess_in_workers")
        actions = [ex["action"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None
        if self._progress_of(examples) is not None:
            raise ValueError("QwenGR00T_EpisodeMemory: a progress-conditioned DiT is not supported")

        if self.mem_tbptt:
            cur, _, fv = self._split_frames(examples)
            t0 = time.perf_counter()
            last_hidden, mask = self._vlm(cur, [ex["lang"] for ex in examples], grad=False)
            t1 = time.perf_counter()
            ctx, logits, ok = self._memory_stream(examples, last_hidden, mask, train=True)
            timings = {"timing/vlm_fwd": t1 - t0, "timing/memory": time.perf_counter() - t1}
        else:
            last_hidden, mask, ctx, logits, fv, timings = self._encode(examples, train=True)
            ok = torch.ones(len(examples), dtype=torch.bool, device=last_hidden.device)
        # samples without a valid memory history (stream mode) leave every loss; none valid -> all, weighted 0
        sel = ok.nonzero(as_tuple=True)[0] if bool(ok.any()) else torch.arange(len(examples), device=ok.device)
        w = 1.0 if bool(ok.any()) else 0.0

        t_head = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -self.action_horizon:, :]
            r = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            state_t = None if state is None else torch.tensor(np.array(state), device=last_hidden.device,
                                                              dtype=last_hidden.dtype)
            action_loss = w * self.action_model(
                last_hidden[sel].repeat(r, 1, 1), actions_target[sel].repeat(r, 1, 1),
                None if state_t is None else state_t[sel].repeat(r, 1, 1),
                encoder_attention_mask=mask[sel].repeat(r, 1), progress=None, memory_states=ctx[sel].repeat(r, 1, 1),
            )
            out = {"action_loss": action_loss}
            loss = action_loss
            if self.progress_weight > 0:
                if "progress" not in examples[0]:
                    raise KeyError("QwenGR00T_EpisodeMemory: examples need 'progress' (datasets.vla_data.emit_progress: "
                                   "true) or episode_memory.progress_loss_weight: 0")
                prog = torch.as_tensor(np.array([ex["progress"] for ex in examples], dtype=np.float32))
                ce, logs = self.episode_memory.progress_loss(logits, prog, valid=ok)
                loss = loss + self.progress_weight * ce
                out.update(logs)
            out["loss"] = loss
        head_time = time.perf_counter() - t_head

        with torch.no_grad():  # free eval metric on the same features, as QwenGR00T
            was_training = self.action_model.training
            self.action_model.eval()
            with torch.autocast("cuda", dtype=torch.float32):
                pred = self.action_model.predict_action(last_hidden.detach(), state_t, encoder_attention_mask=mask,
                                                        memory_states=ctx.detach())
            if was_training:
                self.action_model.train()
            diff = pred.float() - actions_target.float()
            out["mse_score"] = (torch.linalg.norm(diff) / pred.numel()).item()
            out["action_rmse"] = diff.pow(2).mean().sqrt().item()
        out.update(timings)
        out["timing/head_fwd"] = head_time
        out["memory/frame_valid_frac"] = float(fv.float().mean())
        if self.mem_tbptt:
            out["memory/stream_valid_frac"] = float(ok.float().mean())
            out["memory/stream_mean_step"] = float(np.mean([int(ex["stream"][3]) for ex in examples]))
        return out

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], memory: Optional[torch.Tensor] = None, **kwargs) -> dict:
        """stream mode: memory = the carried episode memory [B, N, d] (memory_init(B) at the episode start; None = the
        initial memory, i.e. this call is the episode's first); the updated memory is returned under "memory"."""
        if not isinstance(examples, list):
            examples = [examples]
        if self._progress_of(examples) is not None:
            raise ValueError("QwenGR00T_EpisodeMemory: a progress-conditioned DiT is not supported")
        if self.mem_tbptt:
            cur, _, _ = self._split_frames(examples)
            cur = [to_pil_preserve(x) for x in cur]
            size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
            if size:
                cur = resize_images(cur, target_size=size)
            last_hidden, mask = self._vlm(cur, [ex["lang"] for ex in examples], grad=False)
            m = self.memory_init(len(examples)) if memory is None else memory.to(last_hidden.device).float()
            with torch.autocast("cuda", dtype=torch.float32):
                m = self.episode_memory.update(m, last_hidden, mask)
                ctx, logits = self.episode_memory.readout(m)
        else:
            if memory is not None:
                raise ValueError("predict_action(memory=...) is for episode_memory.stream; window mode rebuilds the memory "
                                 "from the example's frames")
            last_hidden, mask, ctx, logits, _, _ = self._encode(examples, train=False, resize=True)
            m = None
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None
        state = None if state is None else torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
        with torch.autocast("cuda", dtype=torch.float32):
            pred = self.action_model.predict_action(last_hidden, state, encoder_attention_mask=mask, memory_states=ctx)
        out = {"normalized_actions": pred.detach().cpu().numpy(),
               "progress": self.episode_memory.expected_progress(logits).cpu().numpy()}
        if m is not None:
            out["memory"] = m
        return out
