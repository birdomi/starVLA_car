"""Multi-VLM pretraining of the episode memory (EpisodeMemory, memory.py) without an action head.

One memory transformer shared by several frozen VLMs: every VLM has its own token_resize MLP (LayerNorm -> Linear(H,
token_dim) -> GELU -> Linear(token_dim, token_dim)), everything after it is shared, so a pretrained memory drops into
QwenGR00T_EpisodeMemory (episode_memory.token_dim = token_dim) for any of the VLMs (export_for_vlm).

Every VLM runs its own memory stream over the same frames (same weights, separate states). From the memory M_t after
the update on frame t, a Joint Decision Transformer (CLEF-style): 4 learned decision queries (current subtask, finished
subtasks, episode progress, subtask progress; the same for every VLM and every step, never carried over in time) pass
`decision.num_layers` pre-LN blocks of cross-attention over the memory tokens M_t only (the current frame reaches the
decisions through the memory update alone), self-attention among the queries and an FFN. From the 4 outputs:
  - current subtask: which of the sample's candidate subtasks is in progress. The candidates (any number per sample) are
    encoded by a frozen T5 encoder (FrozenT5Encoder: masked mean of the last hidden state over each sentence's tokens)
    -> candidate_proj; logits = cos(subtask_query_proj(z), candidate) / temperature, padded candidates masked;
    cross-entropy on the local index of the true candidate.
  - finished subtasks: the same candidates scored by cos(done_query_proj(z), candidate) / temperature + a learned bias;
    binary cross-entropy per real candidate (finished at or before t).
  - episode progress: t / (T - 1) in `progress_bins` (20) equal bins, cross-entropy.
  - subtask progress: (t - start) / (end - start) of the subtask in progress, in the same bins, cross-entropy.
Subtask terms need an annotated episode (no made-up targets); episode progress needs none.
The EpisodeMemory's own readout layers (progress_head, progress_embed, mem_proj) are not used here; they keep their
initialisation and are exported with the memory for the action model to train."""
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.dual_system.memory import EpisodeMemory

QUERY_SUBTASK, QUERY_DONE, QUERY_PROGRESS, QUERY_SUBTASK_PROGRESS = 0, 1, 2, 3
NUM_QUERIES = 4


def vlm_key(name: str) -> str:
    """ModuleDict-safe VLM name (no dots)."""
    return name.replace(".", "_").replace("-", "_")


def token_resize(vlm_dim: int, token_dim: int) -> nn.Sequential:
    # same layout as EpisodeMemory.token_resize, so the weights export 1:1
    return nn.Sequential(nn.LayerNorm(vlm_dim), nn.Linear(vlm_dim, token_dim), nn.GELU(approximate="tanh"),
                         nn.Linear(token_dim, token_dim))


# ----------------------------------------------------------------------------------------------------------------------
class DecisionBlock(nn.Module):
    """Pre-LN, each step residual: cross-attention (queries -> context) -> self-attention among the queries -> FFN."""

    def __init__(self, d: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_ctx = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.norm_self = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, q: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """q [B, Q, d], ctx [B, S, d]."""
        c = self.norm_ctx(ctx)
        q = q + self.cross_attn(self.norm_q(q), c, c, need_weights=False)[0]
        s = self.norm_self(q)
        q = q + self.self_attn(s, s, s, need_weights=False)[0]
        return q + self.ffn(self.norm_ffn(q))


class JointDecisionTransformer(nn.Module):
    """Learned decision queries over the memory tokens -> decision features [B, Q, d]."""

    def __init__(self, d: int, num_layers: int = 2, num_heads: int = 12, mlp_ratio: float = 4.0,
                 num_queries: int = NUM_QUERIES):
        super().__init__()
        if num_queries < 1:
            raise ValueError(f"JointDecisionTransformer: need at least one decision query, got {num_queries}")
        self.decision_queries = nn.Parameter(torch.randn(num_queries, d) * 0.02)
        self.blocks = nn.ModuleList([DecisionBlock(d, num_heads, mlp_ratio) for _ in range(int(num_layers))])
        self.out_norm = nn.LayerNorm(d)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """memory [B, N, d] -> [B, Q, d]."""
        q = self.decision_queries[None].expand(memory.shape[0], -1, -1).to(memory.dtype)
        for blk in self.blocks:
            q = blk(q, memory)
        return self.out_norm(q)


# ----------------------------------------------------------------------------------------------------------------------
def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """h [..., T, D], mask [..., T] -> mean over the masked-in positions [..., D] (0 where none)."""
    m = mask.to(h.dtype)[..., None]
    return (h * m).sum(-2) / m.sum(-2).clamp_min(1)


class FrozenT5Encoder:
    """Frozen T5 encoder for subtask candidate sentences: each sentence alone -> masked mean of the encoder's last
    hidden state over its own tokens [D]. Kept outside the trainable module (like the frozen VLMs: no gradient, not in
    the checkpoints); an LRU cache keeps the features of sentences already seen (cache_size 0 = off)."""

    def __init__(self, model, tokenizer, device=None, cache_size: int = 50000, max_length: int = 64):
        self.model, self.tok = model.eval(), tokenizer
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device if device is not None else next(model.parameters()).device
        self.cache_size, self.max_length = int(cache_size), int(max_length)
        self.cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.dim = int(model.config.d_model)

    @classmethod
    def from_pretrained(cls, path: str, device, dtype=torch.bfloat16, **kw) -> "FrozenT5Encoder":
        from transformers import AutoTokenizer, T5EncoderModel

        model = T5EncoderModel.from_pretrained(path, dtype=dtype).to(device)
        return cls(model, AutoTokenizer.from_pretrained(path), device, **kw)

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """-> [N, D] float32 sentence features, one per text."""
        texts = list(texts)
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        feats = {}
        if new:
            enc = self.tok(new, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            ids, att = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
            h = self.model(input_ids=ids, attention_mask=att).last_hidden_state.float()
            feats = dict(zip(new, masked_mean(h, att.bool())))
        out = []
        for t in texts:
            if t in feats:
                v = feats[t]
                if self.cache_size:
                    self.cache[t] = v
            else:
                v = self.cache[t]
                self.cache.move_to_end(t)
            out.append(v)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return torch.stack(out) if out else torch.zeros(0, self.dim, device=self.device)

    def candidates(self, lists: Sequence[Sequence[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-sample candidate lists (any lengths, empty allowed) -> (features [B, K, D] float32 zero-padded,
        valid [B, K] bool), K = the longest list (at least 1)."""
        b, k = len(lists), max([len(c) for c in lists] + [1])
        feats = torch.zeros(b, k, self.dim, device=self.device)
        valid = torch.zeros(b, k, dtype=torch.bool, device=self.device)
        flat = self.encode([t for c in lists for t in c])
        i = 0
        for j, c in enumerate(lists):
            feats[j, :len(c)], valid[j, :len(c)] = flat[i:i + len(c)], True
            i += len(c)
        return feats, valid


# ----------------------------------------------------------------------------------------------------------------------
def progress_bin_targets(p: torch.Tensor, bins: int) -> torch.Tensor:
    """progress in [0, 1] -> bin floor(p * bins); bin `bins - 1` closed at 1 (p = 1 -> the last bin)."""
    return (p.float().clamp(0.0, 1.0) * bins).long().clamp(max=bins - 1)


def expected_progress(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> (probabilities [B, bins], expected progress over the bin centres [B])."""
    probs = logits.float().softmax(-1)
    bins = logits.shape[-1]
    centres = (torch.arange(bins, device=logits.device, dtype=probs.dtype) + 0.5) / bins
    return probs, (probs * centres).sum(-1)


def progress_loss(logits: torch.Tensor, target_bins: torch.Tensor, weight: torch.Tensor, kind: str = "ce") -> torch.Tensor:
    """Progress classification loss, averaged over the samples with weight 1 (others 0). `kind` "ce" only for now;
    ordinal / soft-label variants go here."""
    if kind != "ce":
        raise ValueError(f"progress_loss: unknown kind {kind!r}")
    w = weight.float()
    return (F.cross_entropy(logits.float(), target_bins, reduction="none") * w).sum() / w.sum().clamp_min(1)


def subtask_loss(logits: torch.Tensor, target: torch.Tensor, has: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the sample's own candidates (local index), only the rows with `has`; 0 (in the graph) when
    none."""
    if not bool(has.any()):
        return (logits * 0.0).sum()  # zero first: the padded logits sit at the dtype's minimum and would sum to -inf
    return F.cross_entropy(logits[has].float(), target[has])


def done_loss(logits: torch.Tensor, done: torch.Tensor, cand_valid: torch.Tensor, has: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy per real candidate of the rows with `has` (done [B, K] bool: finished at or before t); 0 (in
    the graph) when none."""
    w = (cand_valid.bool() & has[:, None]).float()
    if not bool(w.any()):
        return (logits * 0.0).sum()
    bce = F.binary_cross_entropy_with_logits(logits.float(), done.float(), reduction="none")
    return (bce * w).sum() / w.sum()


# ----------------------------------------------------------------------------------------------------------------------
class MultiVLMMemory(nn.Module):
    def __init__(self, vlm_dims: Dict[str, int], token_dim: int = 1024, num_tokens: int = 32, d_model: int = 768,
                 depth: int = 8, num_heads: int = 12, mlp_ratio: float = 4.0, progress_bins: int = 50,
                 attention: str = "gated", t5_dim: int = 768, decision_layers: int = 2, decision_heads: int = 12,
                 decision_mlp_ratio: float = 4.0, decision_progress_bins: int = 20, subtask_temperature: float = 0.07):
        super().__init__()
        self.names = list(vlm_dims)
        self.token_dim = int(token_dim)
        if not subtask_temperature > 0:
            raise ValueError(f"MultiVLMMemory: subtask_temperature must be > 0, got {subtask_temperature}")
        self.token_resize = nn.ModuleDict({vlm_key(n): token_resize(int(h), self.token_dim) for n, h in vlm_dims.items()})
        # the shared memory works on token_dim tokens (its own token_resize stays off); progress_bins: its own readout
        # head, kept for the action model's checkpoint layout
        self.memory = EpisodeMemory(self.token_dim, self.token_dim, num_tokens=num_tokens, d_model=d_model, depth=depth,
                                    num_heads=num_heads, mlp_ratio=mlp_ratio, progress_bins=progress_bins,
                                    attention=attention)
        d = self.memory.d_model
        # joint decision transformer and its heads
        self.decision = JointDecisionTransformer(d, decision_layers, decision_heads, decision_mlp_ratio, NUM_QUERIES)
        self.candidate_proj = nn.Sequential(nn.LayerNorm(int(t5_dim)), nn.Linear(int(t5_dim), d))
        self.subtask_query_proj = nn.Linear(d, d)
        self.done_query_proj = nn.Linear(d, d)
        self.done_bias = nn.Parameter(torch.zeros(()))
        self.progress_bins = int(decision_progress_bins)
        self.progress_head = nn.Linear(d, self.progress_bins)
        self.subtask_progress_head = nn.Linear(d, self.progress_bins)
        self.subtask_temperature = float(subtask_temperature)

    # ------------------------------------------------------------------
    def init(self, b: int) -> torch.Tensor:
        return self.memory.init(b)

    def resize(self, name: str, tokens: torch.Tensor) -> torch.Tensor:
        rs = self.token_resize[vlm_key(name)]
        return rs(tokens.to(rs[1].weight.dtype))

    def update(self, name: str, m: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor,
               valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """M_t = f(M_{t-1}, token_resize_vlm(H_t)); valid [B] False keeps M_{t-1}."""
        return self.memory.update(m, self.resize(name, tokens), mask, valid)

    # ------------------------------------------------------------------
    def decide(self, m: torch.Tensor) -> torch.Tensor:
        """M_t [B, N, d] -> decision features [B, 4, d] (current subtask, finished subtasks, episode progress, subtask
        progress)."""
        return self.decision(m)

    def _candidates(self, cand_feats: torch.Tensor) -> torch.Tensor:
        cp = self.candidate_proj
        return F.normalize(cp(cand_feats.to(cp[1].weight.dtype)).float(), dim=-1)

    def subtask_logits(self, z: torch.Tensor, cand_feats: torch.Tensor, cand_valid: torch.Tensor) -> torch.Tensor:
        """z [B, 4, d]; frozen T5 candidate features [B, K, D_t5], valid [B, K] -> current-subtask logits [B, K]
        (padding at the dtype's minimum)."""
        q = F.normalize(self.subtask_query_proj(z[:, QUERY_SUBTASK]).float(), dim=-1)
        logits = torch.einsum("bd,bkd->bk", q, self._candidates(cand_feats)) / self.subtask_temperature
        return logits.masked_fill(~cand_valid.to(logits.device).bool(), torch.finfo(logits.dtype).min)

    def done_logits(self, z: torch.Tensor, cand_feats: torch.Tensor) -> torch.Tensor:
        """-> finished-subtask logits [B, K] (independent per candidate; mask padding in the loss)."""
        q = F.normalize(self.done_query_proj(z[:, QUERY_DONE]).float(), dim=-1)
        return torch.einsum("bd,bkd->bk", q, self._candidates(cand_feats)) / self.subtask_temperature + self.done_bias

    def decision_heads(self, z: torch.Tensor, cand_feats: Optional[torch.Tensor] = None,
                       cand_valid: Optional[torch.Tensor] = None) -> dict:
        """-> progress_logits / subtask_progress_logits [B, bins] with their probabilities and expectations [B] and,
        with candidates, subtask_logits / done_logits [B, K]."""
        out = {"progress_logits": self.progress_head(z[:, QUERY_PROGRESS]),
               "subtask_progress_logits": self.subtask_progress_head(z[:, QUERY_SUBTASK_PROGRESS])}
        out["progress_probs"], out["progress"] = expected_progress(out["progress_logits"])
        out["subtask_progress_probs"], out["subtask_progress"] = expected_progress(out["subtask_progress_logits"])
        if cand_feats is not None:
            out["subtask_logits"] = self.subtask_logits(z, cand_feats, cand_valid)
            out["done_logits"] = self.done_logits(z, cand_feats)
        return out

    def forward(self, name: str, prev_memory: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor,
                valid: Optional[torch.Tensor] = None, cand_feats: Optional[torch.Tensor] = None,
                cand_valid: Optional[torch.Tensor] = None) -> dict:
        """One stream step: VLM `name`'s tokens of the current frame [B, L, H] (mask [B, L]) update prev_memory
        [B, N, d]; the decision queries read the new memory only. The frozen VLM runs outside (as the frozen T5 does:
        cand_feats from FrozenT5Encoder.candidates)."""
        m = self.update(name, prev_memory, tokens, mask, valid)
        z = self.decide(m)
        out = {"memory": m, "decision_features": z}
        out.update(self.decision_heads(z, cand_feats, cand_valid))
        return out

    # ------------------------------------------------------------------
    def export_for_vlm(self, name: str) -> Dict[str, torch.Tensor]:
        """State dict in QwenGR00T_EpisodeMemory's layout (episode_memory.*, token_dim = self.token_dim) for one VLM."""
        out = {f"episode_memory.{k}": v.detach().cpu().clone() for k, v in self.memory.state_dict().items()}
        for k, v in self.token_resize[vlm_key(name)].state_dict().items():
            out[f"episode_memory.token_resize.{k}"] = v.detach().cpu().clone()
        return out
