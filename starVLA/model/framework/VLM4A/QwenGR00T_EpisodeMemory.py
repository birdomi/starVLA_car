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
  * token_dim (optional): an MLP resizes every VLM token to token_dim before the memory (EpisodeMemory.token_resize), and
    the readout / the DiT's memory cross-attention use token_dim too (memory_dim: token_dim), so the memory path is the
    same architecture for any VLM size (Qwen3.5-2B 2048, -4B 2560). Unset: memory_dim = the VLM hidden size.
  * init_from (optional): a pretrained memory for this VLM (starVLA/training/train_memory_pretrain.py,
    checkpoints/steps_<k>_memory_<vlm>.pt, episode_memory.* keys; needs the same token_dim / num_tokens / d_model / depth
    / num_heads / mlp_ratio / progress_bins / attention). Loaded at construction; trainer.pretrained_checkpoint (a
    QwenGR00T baseline without episode_memory.* keys) leaves it in place.
  * condition: how the DiT reads the memory context. "cross_attn" (default): its own zero-initialised cross-attention
    in every cross-attention block (diffusion_model_cfg.memory_cond: true, memory_dim = ctx_dim). "concat": the memory
    context, projected to the VLM hidden size, is appended to the VLM tokens (the DiT's existing cross-attention reads
    both as one sequence; diffusion_model_cfg.memory_cond: false). With concat the projections into the context
    (mem_proj, the progress projection) start at zero, so the 33 appended tokens carry no content until trained; the
    pretrained readout tensors whose width changes with it are not loaded from init_from (unused by the pretraining).
  * decision (optional, {init_from, ...}): the pretraining's Joint Decision Transformer (memory_pretrain.py, from its full
    checkpoint steps_<k>.pt: decision.* + progress_head.*) reads the memory instead of the EpisodeMemory's own progress
    head: of its 4 decision queries only the episode-progress output is used. It gives the progress logits (the
    pretraining's bins, CE x progress_loss_weight) and, through a new projection (d_model -> memory_dim), the progress
    token of the DiT's memory context (the memory tokens stay EpisodeMemory's out_norm -> mem_proj). The episode-progress
    query is a parameter of its own (ProgressDecision.progress_query); the other 3 queries, the decision blocks and
    out_norm are frozen with trainer.freeze_modules: decision.jdt (EpisodeMemory.progress_head / progress_embed are
    unused then: freeze them too).
  * geometry (optional, framework.geometry {path, image_size, grid, pool, dtype, cameras, condition}; stream mode +
    concat): a frozen MoGe encoder (MoGe-3 ViT-L model.pt) on the current frame of the chosen cameras, pooled per camera
    (GeometryTokens). condition concat: projected (zero-initialised) to the VLM hidden size and appended to the DiT
    context after the memory tokens; cross_attn: read by the DiT's separate cross-attention (diffusion_model_cfg
    memory_cond: true, memory_dim: the MoGe width; zero-initialised output in every cross-attention block). The memory
    still reads the VLM tokens only.

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
    frozen VLM, unless stream.vlm_grad: true (end-to-end): the current frame's VLM tokens then keep their gradient (the
    DiT's and the memory's last update), while the cached tokens of the earlier updates in the window stay detached
    (computed with the VLM weights of 1 to tbptt - 1 steps before). Inference: predict_action(examples, memory=M)
    with M = memory_init(B) at the episode start and then the
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
from starVLA.model.modules.dual_system.memory_pretrain import QUERY_PROGRESS, JointDecisionTransformer
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
    "token_dim": None,
    "init_from": None,
    "decision": None,
    "condition": "cross_attn",
}
_DECISION_DEFAULTS = {"init_from": None, "num_layers": 2, "num_heads": 12, "mlp_ratio": 4.0, "progress_bins": 20}
_GEOMETRY_DEFAULTS = {"path": None, "image_size": [224, 224], "grid": [16, 16], "pool": 2, "dtype": "bfloat16",
                      "cameras": None, "condition": "concat"}


class GeometryTokens(torch.nn.Module):
    """framework.geometry: a frozen MoGe-2 / MoGe-3 encoder (dual_system/moge_encoder.py: the DINOv2 features MoGe's
    geometry neck reads) on the current frame of the `cameras` (indices into the example's views; None = all) -> token
    grid area-pooled by `pool` per camera -> LayerNorm. MoGe-3 ViT-L at 224 x 224: 16 x 16 x 1024 per camera, pool 2 ->
    64 tokens per camera (the Qwen3.5 visual token count of the same image). By `condition`:
      * concat: -> Linear to the DiT context width (zero-initialised) + a source embedding, appended to the DiT context
        after the memory tokens (they start content-free like the memory's); out_dim = ctx_dim
      * cross_attn: the normed features themselves go to the DiT's own geometry cross-attention (diffusion_model_cfg
        memory_cond: true, memory_dim = the MoGe feature width; one per cross-attention block, output projection
        zero-initialised), apart from the VLM / memory tokens; out_dim = the MoGe feature width."""

    def __init__(self, path: str, ctx_dim: int, image_size, grid, pool: int, dtype: str, cameras=None,
                 condition: str = "concat"):
        super().__init__()
        from starVLA.model.modules.dual_system.moge_encoder import MoGeEncoderBackbone

        if condition not in ("concat", "cross_attn"):
            raise ValueError(f"geometry.condition must be concat or cross_attn, got {condition!r}")
        self.condition = condition
        self.cameras = None if cameras is None else [int(c) for c in cameras]
        self.encoder = MoGeEncoderBackbone(path, input_size=tuple(image_size), grid=tuple(grid), dtype=dtype, trainable=False)
        self.image_size = tuple(int(v) for v in image_size)
        self.pool = int(pool)
        if self.encoder.grid[0] % self.pool or self.encoder.grid[1] % self.pool:
            raise ValueError(f"geometry.pool {self.pool} must divide the MoGe grid {self.encoder.grid}")
        d = self.encoder.feat_dim
        self.norm = torch.nn.LayerNorm(d)
        if condition == "concat":
            self.proj = torch.nn.Linear(d, ctx_dim)
            torch.nn.init.zeros_(self.proj.weight)
            torch.nn.init.zeros_(self.proj.bias)
            self.source_embed = torch.nn.Parameter(torch.randn(1, 1, ctx_dim) * 0.02)
        self.out_dim = ctx_dim if condition == "concat" else d

    def forward(self, images: List[list]) -> torch.Tensor:
        """images [B][V] (PIL or HWC uint8) -> geometry tokens [B, len(cameras) * n, out_dim] float32."""
        if self.cameras is not None:
            images = [[cams[c] for c in self.cameras] for cams in images]
        b, v = len(images), len(images[0])
        w, h = self.image_size[1], self.image_size[0]
        arr = []
        for cams in images:
            for im in cams:
                im = to_pil_preserve(im).convert("RGB")
                arr.append(np.asarray(im if im.size == (w, h) else im.resize((w, h))))
        x = torch.from_numpy(np.stack(arr)).to(self.norm.weight.device).permute(0, 3, 1, 2).float() / 255.0
        f = self.encoder(x).tokens  # [B * V, gh * gw, D]
        gh, gw = self.encoder.grid
        d = f.shape[-1]
        f = torch.nn.functional.avg_pool2d(f.transpose(1, 2).reshape(b * v, d, gh, gw), self.pool)  # [B * V, D, gh/p, gw/p]
        f = f.flatten(2).transpose(1, 2).reshape(b, -1, d)  # [B, V * n, D], camera-major
        with torch.autocast("cuda", dtype=torch.float32):
            f = self.norm(f.float())
            return self.proj(f) + self.source_embed if self.condition == "concat" else f


class ProgressDecision(torch.nn.Module):
    """The pretraining's Joint Decision Transformer reduced to its episode-progress output: all 4 decision queries pass
    the decision blocks as in the pretraining (self-attention among them included), the episode-progress one taken from
    progress_query (trainable on its own), the others from jdt.decision_queries. -> (progress token [B, ctx_dim],
    progress logits [B, bins])."""

    def __init__(self, d: int, ctx_dim: int, num_layers: int, num_heads: int, mlp_ratio: float, bins: int):
        super().__init__()
        self.jdt = JointDecisionTransformer(d, num_layers=num_layers, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.progress_query = torch.nn.Parameter(self.jdt.decision_queries[QUERY_PROGRESS].detach().clone())
        self.progress_head = torch.nn.Linear(d, bins)
        self.progress_proj = torch.nn.Linear(d, ctx_dim)  # new: the DiT's progress token
        self.num_bins = int(bins)
        self.register_buffer("bin_centers", (torch.arange(self.num_bins, dtype=torch.float32) + 0.5) / self.num_bins,
                             persistent=False)

    def load_pretrained(self, path: str) -> None:
        """decision.* -> jdt, progress_head.* -> progress_head (the pretraining's full checkpoint, or its 'model')."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        jdt = {k[len("decision."):]: v for k, v in sd.items() if k.startswith("decision.")}
        head = {k[len("progress_head."):]: v for k, v in sd.items() if k.startswith("progress_head.")}
        self.jdt.load_state_dict(jdt, strict=True)
        self.progress_head.load_state_dict(head, strict=True)
        with torch.no_grad():
            self.progress_query.copy_(self.jdt.decision_queries[QUERY_PROGRESS])
        print(f"[QwenGR00T_EpisodeMemory] decision transformer + progress head from {path} "
              f"({len(jdt)} + {len(head)} tensors)")

    def forward(self, memory: torch.Tensor):
        q = self.jdt.decision_queries.to(memory.dtype)
        q = torch.cat([q[:QUERY_PROGRESS], self.progress_query[None].to(memory.dtype), q[QUERY_PROGRESS + 1:]])
        q = q[None].expand(memory.shape[0], -1, -1)
        for blk in self.jdt.blocks:
            q = blk(q, memory)
        z = self.jdt.out_norm(q)[:, QUERY_PROGRESS]
        return self.progress_proj(z), self.progress_head(z)

    def expected_progress(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits.float().softmax(-1) * self.bin_centers).sum(-1)

    def progress_loss(self, logits: torch.Tensor, p: torch.Tensor, valid: Optional[torch.Tensor] = None):
        """Same terms and metric names as EpisodeMemory.progress_loss, on this head's bins."""
        p = p.to(logits.device).float()
        tgt = (p * self.num_bins).long().clamp(0, self.num_bins - 1)
        w = torch.ones_like(p) if valid is None else valid.to(logits.device).float()
        n = w.sum().clamp_min(1)
        ce = (torch.nn.functional.cross_entropy(logits.float(), tgt, reduction="none") * w).sum() / n
        with torch.no_grad():
            acc = ((logits.argmax(-1) == tgt).float() * w).sum() / n
            mae = ((self.expected_progress(logits) - p).abs() * w).sum() / n
        return ce, {"memory/progress_ce": float(ce.detach()), "memory/progress_acc": float(acc),
                    "memory/progress_mae": float(mae)}


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
        token_dim = int(mc["token_dim"]) if mc["token_dim"] else None
        self.condition = str(mc["condition"])
        if self.condition not in ("cross_attn", "concat"):
            raise ValueError(f"episode_memory.condition must be cross_attn or concat, got {self.condition!r}")
        dit = fw.action_model.diffusion_model_cfg
        gc = {**_GEOMETRY_DEFAULTS, **dict(fw.geometry)} if fw.get("geometry", None) else None
        geo_xattn = gc is not None and gc["condition"] == "cross_attn"
        if self.condition == "concat":
            ctx_dim = vl_dim  # appended to the VLM tokens
            # the DiT's extra cross-attention (memory_cond) is free under concat: geometry cross_attn takes it
            if bool(dit.get("memory_cond", False)) != geo_xattn:
                raise ValueError("episode_memory.condition: concat needs diffusion_model_cfg.memory_cond: false (the memory "
                                 "goes through the DiT's VLM cross-attention), or true with geometry.condition: cross_attn")
        else:
            ctx_dim = token_dim or vl_dim
            if not dit.get("memory_cond", False) or int(dit.get("memory_dim", 0) or 0) != ctx_dim:
                raise ValueError("QwenGR00T_EpisodeMemory needs framework.action_model.diffusion_model_cfg.memory_cond: "
                                 f"true and memory_dim: {ctx_dim} (episode_memory.token_dim, else the VLM hidden size)")
        self.num_views = int(fw.get("num_views", 0) or 0)  # 0 = infer from the image count
        self.progress_weight = float(mc["progress_loss_weight"])
        self.past_grad = bool(mc["past_grad"])
        st = mc["stream"]
        self.mem_tbptt = int(st.get("tbptt", 4)) if st else 0
        self.mem_vlm_grad = bool(st.get("vlm_grad", False)) if st else False
        if self.mem_tbptt:
            if self.offsets != [0]:
                raise ValueError(f"episode_memory.stream: update_offsets must be [0] (one update per step on the current "
                                 f"frame; the loader's stream stride sets the interval), got {self.offsets}")
            if self.mem_tbptt < 1:
                raise ValueError(f"episode_memory.stream.tbptt must be >= 1, got {self.mem_tbptt}")
        self._mem_streams = {}  # stream slot -> carried memory state (training, stream mode)
        self.episode_memory = EpisodeMemory(
            vl_dim, ctx_dim, token_dim=token_dim, num_tokens=int(mc["num_tokens"]), d_model=int(mc["d_model"]), depth=int(mc["depth"]),
            num_heads=int(mc["num_heads"]), mlp_ratio=float(mc["mlp_ratio"]), progress_bins=int(mc["progress_bins"]),
            attention=str(mc["attention"]))
        if mc["init_from"]:
            sd = torch.load(mc["init_from"], map_location="cpu")
            sd = {k[len("episode_memory."):]: v for k, v in sd.items() if k.startswith("episode_memory.")}
            own = self.episode_memory.state_dict()
            resized = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
            if [k for k in resized if not k.startswith(("mem_proj.", "progress_embed.", "source_embed."))]:
                raise ValueError(f"episode_memory.init_from {mc['init_from']}: shape mismatch in {resized}")
            sd = {k: v for k, v in sd.items() if k not in resized}  # readout widths differ under concat: keep own
            missing, unexpected = self.episode_memory.load_state_dict(sd, strict=False)
            missing = [k for k in missing if k not in resized]
            if unexpected or [k for k in missing if not k.startswith("token_resize")]:
                raise ValueError(f"episode_memory.init_from {mc['init_from']}: missing {missing}, unexpected {unexpected}")
            print(f"[QwenGR00T_EpisodeMemory] memory initialised from {mc['init_from']} ({len(sd)} tensors)")
        self.decision = None
        if mc["decision"]:
            dc = {**_DECISION_DEFAULTS, **dict(mc["decision"])}
            self.decision = ProgressDecision(int(mc["d_model"]), ctx_dim, int(dc["num_layers"]), int(dc["num_heads"]),
                                             float(dc["mlp_ratio"]), int(dc["progress_bins"]))
            if dc["init_from"]:
                self.decision.load_pretrained(dc["init_from"])
        if self.condition == "concat":  # appended tokens start content-free (source embeddings only)
            projs = [self.episode_memory.mem_proj] + ([self.decision.progress_proj] if self.decision is not None else [])
            for lin in projs:
                torch.nn.init.zeros_(lin.weight)
                torch.nn.init.zeros_(lin.bias)
        self.geometry = None
        if gc is not None:
            if self.condition != "concat" or not self.mem_tbptt:
                raise ValueError("framework.geometry needs episode_memory.condition: concat and episode_memory.stream")
            self.geometry = GeometryTokens(str(gc["path"]), ctx_dim, gc["image_size"], gc["grid"], int(gc["pool"]),
                                           str(gc["dtype"]), cameras=gc["cameras"], condition=str(gc["condition"]))
            if geo_xattn and int(dit.get("memory_dim", 0) or 0) != self.geometry.out_dim:
                raise ValueError(f"geometry.condition: cross_attn needs diffusion_model_cfg.memory_dim: "
                                 f"{self.geometry.out_dim} (the MoGe feature width)")

    def _geo_xattn(self) -> bool:
        return self.geometry is not None and self.geometry.condition == "cross_attn"

    def _dit_inputs(self, vl: torch.Tensor, mask: torch.Tensor, ctx: torch.Tensor, geo: Optional[torch.Tensor] = None):
        """-> (encoder tokens, their mask, memory_states) for the action DiT, by episode_memory.condition; concat appends
        the memory context to the VLM tokens, and the geometry tokens (framework.geometry) after it (geometry concat) or
        as the DiT's separate geometry cross-attention input (geometry cross_attn)."""
        if self.condition == "concat":
            xattn = None
            if geo is not None and self.geometry.condition == "cross_attn":
                xattn, geo = geo.float(), None
            extra = ctx if geo is None else torch.cat([ctx.float(), geo.float()], dim=1)
            ones = torch.ones(extra.shape[:2], dtype=mask.dtype, device=mask.device)
            return torch.cat([vl, extra.to(vl.dtype)], dim=1), torch.cat([mask, ones], dim=1), xattn
        return vl, mask, ctx

    def _readout(self, m: torch.Tensor):
        """memory [B, N, d] -> (DiT memory context [B, N + 1, ctx_dim], progress logits): EpisodeMemory.readout, or with
        a decision transformer its progress token / logits in place of the EpisodeMemory's own."""
        em = self.episode_memory
        if self.decision is None:
            return em.readout(m)
        mem = em.mem_proj(em.out_norm(m)) + em.source_embed["memory"]
        tok, logits = self.decision(m)
        return torch.cat([mem, (tok[:, None] + em.source_embed["progress"]).to(mem.dtype)], dim=1), logits

    def _progress_head(self):
        return self.episode_memory if self.decision is None else self.decision

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
            return self._readout(m)

    def _memory_stream(self, examples: List[dict], tok, mask, train: bool):
        """stream mode (training): every sample continues the memory of its stream slot by one update on its current
        frame (tokens tok [B, L, H], mask [B, L]); gradient through the last mem_tbptt updates (cached tokens) from the
        detached older memory. A stream restarts from the initial memory at its episode's step 0; a stream that does not
        continue its previous step is invalid until then. -> (context [B, N + 1, H], progress logits, valid [B] bool)."""
        if self._vlm_trainable() and not self.mem_vlm_grad:
            raise ValueError("episode_memory.stream needs a frozen VLM (trainer.freeze_modules: qwen_vl_interface): "
                             "it caches the VLM tokens across steps; stream.vlm_grad: true trains it end-to-end through "
                             "the current frame")
        if "stream" not in examples[0]:
            raise KeyError("episode_memory.stream: examples need 'stream' (datasets.vla_data.stream + in_order: true)")
        b, kmax, dev = len(examples), self.mem_tbptt, tok.device
        info = [np.asarray(ex["stream"], dtype=np.int64) for ex in examples]
        states, seqs, cached = [], [], []
        for i in range(b):
            slot, k, j, es = (int(x) for x in info[i][:4])
            st = self._mem_streams.get(slot)
            if es == 0 or st is None or not st["valid"] or st["key"] != (k - 1, j, es - 1):
                st = {"base": None, "frames": [], "valid": es == 0}
            st["key"] = (k, j, es)
            states.append(st)
            cur = tok[i][mask[i]]  # valid tokens only
            cached.append((st["frames"] + [cur.detach()])[-kmax:])  # what the next steps reuse: no gradient
            seqs.append((st["frames"] + [cur if self.mem_vlm_grad else cur.detach()])[-kmax:])
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
            ctx, logits = self._readout(m)
        for i, (st, q) in enumerate(zip(states, cached)):
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
            vlm_grad = self.mem_vlm_grad and self._vlm_trainable() and torch.is_grad_enabled()
            last_hidden, mask = self._vlm(cur, [ex["lang"] for ex in examples], grad=vlm_grad)
            t1 = time.perf_counter()
            ctx, logits, ok = self._memory_stream(examples, last_hidden, mask, train=True)
            t2 = time.perf_counter()
            geo = None if self.geometry is None else self.geometry(cur)
            timings = {"timing/vlm_fwd": t1 - t0, "timing/memory": t2 - t1}
            if geo is not None:
                timings["timing/geometry"] = time.perf_counter() - t2
        else:
            geo = None
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
            enc, enc_mask, mem_states = self._dit_inputs(last_hidden[sel], mask[sel], ctx[sel],
                                                         None if geo is None else geo[sel])
            action_loss = w * self.action_model(
                enc.repeat(r, 1, 1), actions_target[sel].repeat(r, 1, 1),
                None if state_t is None else state_t[sel].repeat(r, 1, 1),
                encoder_attention_mask=enc_mask.repeat(r, 1), progress=None,
                # geometry cross_attn: one key / value set per sample, shared by its r noise draws inside the DiT
                memory_states=None if mem_states is None else mem_states if self._geo_xattn() else mem_states.repeat(r, 1, 1),
            )
            out = {"action_loss": action_loss}
            loss = action_loss
            if self.progress_weight > 0:
                if "progress" not in examples[0]:
                    raise KeyError("QwenGR00T_EpisodeMemory: examples need 'progress' (datasets.vla_data.emit_progress: "
                                   "true) or episode_memory.progress_loss_weight: 0")
                prog = torch.as_tensor(np.array([ex["progress"] for ex in examples], dtype=np.float32))
                ce, logs = self._progress_head().progress_loss(logits, prog, valid=ok)
                loss = loss + self.progress_weight * ce
                out.update(logs)
            out["loss"] = loss
        head_time = time.perf_counter() - t_head

        with torch.no_grad():  # free eval metric on the same features, as QwenGR00T
            was_training = self.action_model.training
            self.action_model.eval()
            with torch.autocast("cuda", dtype=torch.float32):
                enc, enc_mask, mem_states = self._dit_inputs(last_hidden.detach(), mask, ctx.detach(),
                                                             None if geo is None else geo.detach())
                pred = self.action_model.predict_action(enc, state_t, encoder_attention_mask=enc_mask,
                                                        memory_states=mem_states)
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
                ctx, logits = self._readout(m)
            geo = None if self.geometry is None else self.geometry(cur)
        else:
            geo = None
            if memory is not None:
                raise ValueError("predict_action(memory=...) is for episode_memory.stream; window mode rebuilds the memory "
                                 "from the example's frames")
            last_hidden, mask, ctx, logits, _, _ = self._encode(examples, train=False, resize=True)
            m = None
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None
        state = None if state is None else torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
        with torch.autocast("cuda", dtype=torch.float32):
            enc, enc_mask, mem_states = self._dit_inputs(last_hidden, mask, ctx, geo)
            pred = self.action_model.predict_action(enc, state, encoder_attention_mask=enc_mask, memory_states=mem_states)
        out = {"normalized_actions": pred.detach().cpu().numpy(),
               "progress": self._progress_head().expected_progress(logits).cpu().numpy()}
        if m is not None:
            out["memory"] = m
        return out
