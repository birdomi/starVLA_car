"""CPU check of the moment-token continuation for Qwen3.5 GatedDeltaNet (QwenGR00T_Memory, moment_pass=kv_cache).

prefill(prefix, cache) + _gdn_continue_forward(tail) must equal one pass over prefix + tail. Guards against
transformers changing the GatedDeltaNet internals or its cache layout.
"""
import unittest

import torch
from transformers import DynamicCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as Q
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from starVLA.model.framework.VLM4A.QwenGR00T_Memory import _gdn_continue_forward


def _layer(num_k_heads=2, num_v_heads=2):
    cfg = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=1, layer_types=["linear_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, linear_conv_kernel_dim=4,
        linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=num_k_heads,
        linear_num_value_heads=num_v_heads,
    )
    torch.manual_seed(0)
    layer = Q.Qwen3_5GatedDeltaNet(cfg, layer_idx=0).eval()
    layer.chunk_gated_delta_rule = Q.torch_chunk_gated_delta_rule  # CPU kernels for prefill and continuation
    layer.causal_conv1d_fn = None
    return cfg, layer


class TestGatedDeltaNetContinuation(unittest.TestCase):
    def _check(self, prefix_len, tail_len=4, num_k_heads=2, num_v_heads=2, left_pad=0):
        cfg, layer = _layer(num_k_heads, num_v_heads)
        x = torch.randn(2, prefix_len + tail_len, cfg.hidden_size)
        mask = torch.ones(2, prefix_len + tail_len, dtype=torch.long)
        mask[0, :left_pad] = 0
        with torch.no_grad():
            full = layer(x, attention_mask=mask if left_pad else None)
            cache = DynamicCache(config=cfg)
            layer(x[:, :prefix_len], cache_params=cache, attention_mask=mask[:, :prefix_len] if left_pad else None)
            tail = _gdn_continue_forward(layer, x[:, prefix_len:], cache_params=cache)
        torch.testing.assert_close(tail, full[:, prefix_len:], atol=1e-4, rtol=1e-4)

    def test_long_prefix(self):
        self._check(prefix_len=70)  # crosses the 64-token chunk boundary

    def test_prefix_shorter_than_conv_kernel(self):
        self._check(prefix_len=2)

    def test_grouped_value_heads(self):
        self._check(prefix_len=20, num_k_heads=2, num_v_heads=4)

    def test_left_padded_prefix(self):
        self._check(prefix_len=20, left_pad=5)

    def test_needs_cache(self):
        cfg, layer = _layer()
        with self.assertRaises(RuntimeError):
            _gdn_continue_forward(layer, torch.randn(1, 4, cfg.hidden_size), cache_params=None)


if __name__ == "__main__":
    unittest.main()
