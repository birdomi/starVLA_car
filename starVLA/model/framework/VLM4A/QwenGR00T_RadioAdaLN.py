"""
QwenGR00T_RadioAdaLN: Qwen-VL output vision tokens modulated by C-RADIO (AdaLN) + one GR00T flow-matching DiT.

System 2 = current-frame feature extractor (no DiT): Qwen3.5-0.8B and C-RADIOv4-SO400M both see frame t.
  * C-RADIOv4 (frozen by default, radio.trainable as in QwenGR00T_DualSystem): 256 x 256 -> 16 x 16 patch tokens
    [N, D_c] (the fixed top-row artifact token masked) + the summary prefix tokens (SigLIP2-g / DINOv3 CLS, 2 x D_c).
  * Qwen3.5-0.8B runs unmodified on frame t + instruction; its OUTPUT = the LM's last hidden states of all prompt tokens
    [B, L, H]. RadioAdaLN on the OUTPUT vision tokens (the positions of <|image_pad|>, 64 per 256 x 256 image, merged
    tokens row-major): each is modulated by the C-RADIO features at the same image location; text positions unchanged:
        c_tok = masked mean of the C-RADIO patches inside the token's cell (16 x 16 patch grid -> Qwen's merged grid,
                8 x 8 at 256 x 256), s = the image's summary tokens concatenated (radio_adaln.use_summary)
        (gamma, beta) = Linear_zero(SiLU(W_c LN(c_tok) + W_s LN(s)))          zero-init
        mode plain (default, standard AdaLN): x' = x_hat * (1 + gamma) + beta,  x_hat = (x - mu) / sigma per token
                                              (= LayerNorm(x) without affine at init)
        mode denorm: x' = mu + sigma * (x_hat * (1 + gamma) + beta)   (the token keeps its own mean / scale, x' = x at init)
    radio_adaln.text_norm (default true): the other positions (text) get a plain LayerNorm with a learned affine (as GR00T's
    vlln), so vision and text tokens enter System 1 at the same scale.
  * Qwen features = those last hidden states with the vision positions modulated (and the text normalised) [B, L, H].
System 1 = the only DiT: GR00T flow matching (Beta(1.5, 1) time, Euler at inference), input = the noisy action
  chunk (+ the state token with state_dim > 0), cross-attention condition = [Qwen features, C-RADIO tokens of frame t
  projected to H (LN + Linear + learned position / prefix embeddings, radio_cond)], padding / artifact masks concatenated.
Inference: one Qwen pass + one C-RADIO pass on frame t, num_inference_timesteps Euler steps -> [B, action_horizon, D_a].
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.dual_system.backbone import SharedRadioBackbone, VisualFeatures
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@dataclass
class QwenGR00TRadioAdaLNDefaultConfig(QwenGR00TDefaultConfig):
    name: str = "QwenGR00T_RadioAdaLN"
    # C-RADIOv4 backbone (same keys as QwenGR00T_DualSystem framework.radio)
    radio: dict = field(default_factory=lambda: {
        "path": "playground/Pretrained_models/C-RADIOv4-SO400M",
        "feature": "backbone",
        "input_size": [256, 256],
        "outlier_norm": 100.0,
        "mask_outliers": True,
        "dtype": "bfloat16",
        "prefix_tokens": "summary",  # SigLIP2-g + DINOv3 CLS (RADIO's summary)
        "trainable": False,          # true / n last blocks: fine-tune (drop radio_encoder from trainer.freeze_modules)
    })
    # AdaLN of Qwen's OUTPUT vision tokens (last hidden states at <|image_pad|>) by co-located C-RADIO features
    radio_adaln: dict = field(default_factory=lambda: {
        "enabled": True,
        "mode": "plain",        # plain (standard AdaLN) | denorm (identity at init)
        "hidden_dim": 1024,     # conditioning MLP width
        "use_summary": True,    # add the image's C-RADIO summary to every token's condition
        "text_norm": True,      # LayerNorm (learned affine) on the text positions -> same scale as the AdaLN'd vision tokens
    })
    # C-RADIO tokens in System 1's cross-attention condition
    radio_cond: dict = field(default_factory=lambda: {
        "enabled": True,
        "use_prefix": True,     # also the summary prefix tokens (else patches only)
    })


def _images_to_tensor(images: List, size, device) -> torch.Tensor:
    """PIL / array images -> [M, 3, H, W] float in [0, 1] at size (H, W)."""
    h, w = int(size[0]), int(size[1])
    arrs = []
    for im in images:
        im = to_pil_preserve(im).convert("RGB")
        if im.size != (w, h):
            im = im.resize((w, h))
        arrs.append(np.asarray(im))
    x = torch.from_numpy(np.stack(arrs)).to(device, non_blocking=True)
    return x.permute(0, 3, 1, 2).float().div_(255.0)


def _example_images(ex: dict) -> list:
    """An example's "image" (one image or a list, PIL / arrays) -> list of PIL images (prompt order)."""
    ims = to_pil_preserve(ex["image"])
    return list(ims) if isinstance(ims, (list, tuple)) else [ims]


def _pool_to_grid(f: VisualFeatures, gh: int, gw: int) -> torch.Tensor:
    """C-RADIO patches [M, N, D_c] (grid f.grid) -> masked mean over each cell of a (gh, gw) grid, row-major
    [M, gh * gw, D_c] (cells with no valid patch -> 0)."""
    m, _, d = f.tokens.shape
    ph, pw = f.grid
    tok = (f.tokens.float() * f.valid[..., None].float()).transpose(1, 2).reshape(m, d, ph, pw)
    val = f.valid.float().reshape(m, 1, ph, pw)
    s = F.adaptive_avg_pool2d(tok, (gh, gw))
    n = F.adaptive_avg_pool2d(val, (gh, gw))
    return (s / n.clamp_min(1e-6)).flatten(2).transpose(1, 2)


class RadioAdaLN(nn.Module):
    """Qwen output vision tokens x [n, H] modulated by per-token C-RADIO conditions c [n, D_c] (+ summary s [n, D_s])."""

    def __init__(self, d_vis: int, d_radio: int, d_summary: int, hidden: int, mode: str = "plain", eps: float = 1e-6,
                 text_norm: bool = False):
        super().__init__()
        if mode not in ("denorm", "plain"):
            raise ValueError(f"radio_adaln.mode must be denorm or plain, got {mode!r}")
        self.mode, self.eps = mode, eps
        self.c_norm = nn.LayerNorm(d_radio)
        self.c_proj = nn.Linear(d_radio, hidden)
        self.s_norm = nn.LayerNorm(d_summary) if d_summary else None
        self.s_proj = nn.Linear(d_summary, hidden) if d_summary else None
        self.mod = nn.Linear(hidden, 2 * d_vis)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        self.text_norm = nn.LayerNorm(d_vis, eps=eps) if text_norm else None  # the non-vision positions

    def forward(self, x: torch.Tensor, c: torch.Tensor, s: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.c_proj(self.c_norm(c.to(self.c_proj.weight.dtype)))
        if self.s_proj is not None:
            h = h + self.s_proj(self.s_norm(s.to(self.s_proj.weight.dtype)))
        gamma, beta = self.mod(F.silu(h)).float().chunk(2, dim=-1)
        x32 = x.float()
        mu = x32.mean(-1, keepdim=True)
        sigma = (x32.var(-1, unbiased=False, keepdim=True) + self.eps).sqrt()
        y = (x32 - mu) / sigma * (1 + gamma) + beta
        if self.mode == "denorm":
            y = mu + sigma * y
        return y.to(x.dtype)


class RadioCondition(nn.Module):
    """C-RADIO tokens of frame t -> System 1 cross-attention tokens [B, V * (N [+ P]), H] + mask."""

    def __init__(self, d_radio: int, d_prefix: int, hidden: int, n_tokens: int, n_prefix: int, n_views: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_radio)
        self.proj = nn.Linear(d_radio, hidden)
        self.pos = nn.Parameter(torch.randn(n_views * n_tokens, hidden) * 0.02)
        self.n_prefix = n_prefix
        if n_prefix:
            self.p_norm = nn.LayerNorm(d_prefix)
            self.p_proj = nn.Linear(d_prefix, hidden)
            self.p_pos = nn.Parameter(torch.randn(n_views * n_prefix, hidden) * 0.02)

    def forward(self, f: VisualFeatures, b: int):
        tok = self.proj(self.norm(f.tokens.reshape(b, -1, f.tokens.shape[-1]).to(self.proj.weight.dtype)))
        if tok.shape[1] != self.pos.shape[0]:
            raise ValueError(f"C-RADIO condition: {tok.shape[1]} patch tokens per sample, expected {self.pos.shape[0]} "
                             "(views x radio grid)")
        parts, masks = [tok + self.pos], [f.valid.reshape(b, -1)]
        if self.n_prefix:
            p = self.p_proj(self.p_norm(f.prefix.reshape(b, -1, f.prefix.shape[-1]).to(self.p_proj.weight.dtype)))
            parts.append(p + self.p_pos)
            masks.append(torch.ones(p.shape[:2], dtype=torch.bool, device=p.device))
        return torch.cat(parts, 1), torch.cat(masks, 1)


@FRAMEWORK_REGISTRY.register("QwenGR00T_RadioAdaLN")
class Qwen_GR00T_RadioAdaLN(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenGR00TRadioAdaLNDefaultConfig, config)
        fw = self.config.framework
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        h = int(self.qwen_vl_interface.model.config.hidden_size)
        fw.action_model.diffusion_model_cfg.cross_attention_dim = h

        rc = fw.radio
        self.radio_encoder = SharedRadioBackbone(
            rc.path, input_size=tuple(rc.get("input_size", [256, 256])), feature=rc.get("feature", "backbone"),
            outlier_norm=float(rc.get("outlier_norm", 100.0)), mask_outliers=bool(rc.get("mask_outliers", True)),
            dtype=rc.get("dtype", "bfloat16"), teacher_ckpt=rc.get("teacher_ckpt", None),
            prefix_tokens=str(rc.get("prefix_tokens", "summary")), trainable=rc.get("trainable", False),
        )
        if self.radio_encoder.trainable:
            tr = self.config.get("trainer", None)
            frozen = str(tr.get("freeze_modules", "") or "") if tr is not None and hasattr(tr, "get") else ""
            if "radio_encoder" in [p.strip() for p in frozen.split(",")]:
                raise ValueError("radio.trainable is set but trainer.freeze_modules lists radio_encoder")
        self.radio_size = tuple(int(v) for v in self.radio_encoder.input_size)
        d_c, n_p = int(self.radio_encoder.feat_dim), int(self.radio_encoder.num_prefix)
        d_p = int(self.radio_encoder.prefix_dim)

        qc = self.qwen_vl_interface.model.config
        self.merge = int(getattr(qc.vision_config, "spatial_merge_size", 2))
        self.image_token_id = int(qc.image_token_id)  # <|image_pad|>: one per merged vision token in the prompt
        ac = fw.radio_adaln
        self.radio_adaln = None
        if bool(ac.get("enabled", True)):
            d_s = d_p * n_p if (bool(ac.get("use_summary", True)) and n_p) else 0
            self.radio_adaln = RadioAdaLN(h, d_c, d_s, int(ac.get("hidden_dim", 1024)), str(ac.get("mode", "plain")),
                                          text_norm=bool(ac.get("text_norm", True)))

        cc = fw.radio_cond
        self.radio_cond = None
        if bool(cc.get("enabled", True)):
            self.num_views = int(fw.get("num_views", 1))
            self.radio_cond = RadioCondition(d_c, d_p, h, self.radio_encoder.num_tokens,
                                             n_p if bool(cc.get("use_prefix", True)) else 0, self.num_views)
        if self.radio_adaln is None and self.radio_cond is None:
            logger.warning("[RadioAdaLN] radio_adaln and radio_cond both disabled: C-RADIO is unused")

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(fw.action_model.action_horizon)
        logger.info(f"[RadioAdaLN] Qwen hidden {h} (merge {self.merge}), C-RADIO {self.radio_encoder.trainable_desc} "
                    f"D_c {d_c} prefix {n_p}x{d_p}, AdaLN {None if self.radio_adaln is None else self.radio_adaln.mode} "
                    f"(text LayerNorm {self.radio_adaln is not None and self.radio_adaln.text_norm is not None}) "
                    f"on Qwen's output vision tokens, S1 condition {'Qwen + C-RADIO' if self.radio_cond is not None else 'Qwen only'}")

    # ------------------------------------------------------------------ System 2: features of frame t
    def _modulate_vision(self, last: torch.Tensor, input_ids: torch.Tensor, grid_thw: torch.Tensor,
                         f: VisualFeatures) -> torch.Tensor:
        """Qwen's last hidden states [B, L, H] at the <|image_pad|> positions (the LM OUTPUT of the vision tokens: images
        in prompt order, merged tokens row-major) AdaLN-modulated by the co-located C-RADIO features of the same
        images; text positions unchanged, or LayerNorm'd with radio_adaln.text_norm."""
        pos = input_ids.to(last.device) == self.image_token_id  # [B, L]
        g = [(int(t), int(hh) // self.merge, int(ww) // self.merge) for t, hh, ww in grid_thw.tolist()]
        if len(g) != f.tokens.shape[0]:
            raise RuntimeError(f"{len(g)} Qwen images vs {f.tokens.shape[0]} C-RADIO images")
        n_img = [t * gh * gw for t, gh, gw in g]
        if sum(n_img) != int(pos.sum()):
            raise RuntimeError(f"{int(pos.sum())} <|image_pad|> positions != merged tokens of the grids {g}")
        if len(set(g)) == 1:  # one shape for the whole batch (the usual case): one pooled tensor
            t, gh, gw = g[0]
            c = _pool_to_grid(f, gh, gw).repeat(1, t, 1).flatten(0, 1)
            idx = torch.arange(len(g), device=last.device).repeat_interleave(t * gh * gw)
        else:
            cs = [_pool_to_grid(f.index(slice(i, i + 1)), gh, gw).repeat(1, t, 1)[0] for i, (t, gh, gw) in enumerate(g)]
            c = torch.cat(cs, 0)
            idx = torch.repeat_interleave(torch.arange(len(g), device=last.device), torch.tensor(n_img, device=last.device))
        s = f.prefix.flatten(1)[idx] if self.radio_adaln.s_proj is not None else None
        tn = self.radio_adaln.text_norm
        out = last.clone() if tn is None else tn(last).to(last.dtype)
        out[pos] = self.radio_adaln(last[pos], c, s).to(out.dtype)  # last[pos]: row-major over (b, l) = prompt image order
        return out

    def _radio_features(self, images_per_example: List[list]) -> VisualFeatures:
        """Every image of every example (prompt order) -> C-RADIO features [M, N, D_c] (M = sum of images)."""
        flat = [im for ims in images_per_example for im in ims]
        return self.radio_encoder(_images_to_tensor(flat, self.radio_size, self._device))

    @property
    def _device(self):
        return next(self.action_model.parameters()).device

    def _qwen_features(self, images_per_example: List[list], instructions: List[str], radio: VisualFeatures):
        """Qwen last hidden states [B, L, H] (vision positions AdaLN-modulated by radio) + attention mask [B, L] bool."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images_per_example, instructions=instructions)
        with torch.autocast("cuda", dtype=getattr(self.qwen_vl_interface, "autocast_dtype", torch.bfloat16)):
            out = self.qwen_vl_interface(**qwen_inputs, output_attentions=False, output_hidden_states=True,
                                         return_dict=True)
            last = out.hidden_states[-1]
            if self.radio_adaln is not None:
                last = self._modulate_vision(last, qwen_inputs["input_ids"], qwen_inputs["image_grid_thw"], radio)
        mask = qwen_inputs.get("attention_mask", None)
        mask = torch.ones(last.shape[:2], dtype=torch.bool, device=last.device) if mask is None else mask.to(last.device).bool()
        return last, mask

    def _condition(self, images_per_example: List[list], instructions: List[str]):
        """System 2 on frame t -> System 1 cross-attention condition [B, L + L_r, H] + mask, and timings."""
        t0 = time.perf_counter()
        radio = self._radio_features(images_per_example)
        t1 = time.perf_counter()
        last, mask = self._qwen_features(images_per_example, instructions, radio)
        t2 = time.perf_counter()
        if self.radio_cond is not None:
            r_tok, r_mask = self.radio_cond(radio, len(images_per_example))
            last = torch.cat([last, r_tok.to(last.dtype)], 1)
            mask = torch.cat([mask, r_mask], 1)
        return last, mask, {"timing/radio_fwd": t1 - t0, "timing/vlm_fwd": t2 - t1}

    def _progress_of(self, examples) -> Optional[np.ndarray]:
        if getattr(getattr(self.action_model, "model", None), "progress_encoder", None) is None:
            return None
        return np.asarray([example["progress"] for example in examples], dtype=np.float32)

    # ------------------------------------------------------------------ training / inference
    def forward(self, examples: List[dict] = None, **kwargs):
        images = [_example_images(ex) for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        cond, mask, timing = self._condition(images, instructions)
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None
        progress = self._progress_of(examples)

        t_head = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array([ex["action"] for ex in examples]), device=cond.device, dtype=cond.dtype)
            target = actions[:, -self.action_horizon:, :]
            r = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            state_t = None if state is None else torch.tensor(np.array(state), device=cond.device, dtype=cond.dtype)
            progress_t = None if progress is None else torch.as_tensor(progress, device=cond.device)
            loss = self.action_model(
                cond.repeat(r, 1, 1), target.repeat(r, 1, 1), None if state_t is None else state_t.repeat(r, 1, 1),
                encoder_attention_mask=mask.repeat(r, 1), progress=None if progress_t is None else progress_t.repeat(r),
            )
        head_time = time.perf_counter() - t_head

        with torch.no_grad():  # free per-step metric: 4-step sampling on the same features (head in eval mode)
            was_training = self.action_model.training
            self.action_model.eval()
            with torch.autocast("cuda", dtype=torch.float32):
                pred = self.action_model.predict_action(cond.detach(), state_t, encoder_attention_mask=mask, progress=progress_t)
            if was_training:
                self.action_model.train()
            diff = pred.float() - target.float()
        return {"action_loss": loss, **timing, "timing/head_fwd": head_time,
                "mse_score": (torch.linalg.norm(diff) / pred.numel()).item(),
                "action_rmse": diff.pow(2).mean().sqrt().item()}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        examples = examples if isinstance(examples, list) else [examples]
        images = [_example_images(ex) for ex in examples]
        size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if size:  # obs_image_size [H, W] -> PIL (W, H)
            images = resize_images(images, target_size=(int(size[1]), int(size[0])))
        cond, mask, _ = self._condition(images, [ex["lang"] for ex in examples])
        state = None
        if "state" in examples[0]:
            state = torch.from_numpy(np.array([ex["state"] for ex in examples])).to(cond.device, dtype=cond.dtype)
        progress = self._progress_of(examples)
        progress = None if progress is None else torch.as_tensor(progress, device=cond.device)
        with torch.autocast("cuda", dtype=torch.float32):
            pred = self.action_model.predict_action(cond, state, encoder_attention_mask=mask, progress=progress)
        return {"normalized_actions": pred.detach().cpu().numpy()}
