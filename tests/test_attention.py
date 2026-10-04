"""Run with ComfyUI on PYTHONPATH, using its Python environment."""
import importlib.util
from pathlib import Path
import unittest

import torch

from comfy.cli_args import args

args.cpu = True

import comfy.sd
from comfy.ldm.cosmos.predict2 import Attention
from comfy.ldm.modules import attention


spec = importlib.util.spec_from_file_location("safe_pag", Path(__file__).parents[1] / "__init__.py")
pag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pag)


class AttentionCompatibilityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.attn = Attention(query_dim=32, n_heads=4, head_dim=8, operations=torch.nn)
        self.x = torch.randn(6, 5, 32)

    def test_forward_with_pag_and_normal_rows(self):
        original = self.attn.compute_attention
        normal = self.attn(self.x)
        q, k, v = self.attn.compute_qkv(self.x)
        raw = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        for heads in (None, [1, 3]):
            with self.subTest(heads=heads):
                weak = raw.clone()
                if heads is None:
                    weak = raw.lerp(v, 0.75)
                else:
                    weak[:, :, heads] = raw[:, :, heads].lerp(v[:, :, heads], 0.75)
                expected = normal.clone()
                expected[4:] = self.attn.output_proj(weak.flatten(-2))[4:]
                self.attn.compute_attention = pag._make_pag_compute_attention(
                    self.attn, original, 2, 0.75, heads
                )
                actual = self.attn(self.x, transformer_options={"cond_or_uncond": [0, 1, 2]})
                torch.testing.assert_close(actual, expected)

    def test_plain_tensors_and_container_consumption(self):
        compute = pag._make_pag_compute_attention(self.attn, self.attn.compute_attention, 2, 1.0, None)
        q, k, v = self.attn.compute_qkv(self.x)
        expected = self.attn.output_proj(v.flatten(-2))
        options = {"cond_or_uncond": [2]}
        torch.testing.assert_close(compute(q, k, v, options), expected)
        container = getattr(attention, "AttentionTensorContainer", None)
        if container is not None:
            inputs = [container(t) for t in (q, k, v)]
            torch.testing.assert_close(compute(*inputs, options), expected)
            self.assertTrue(all(t.tensor is None for t in inputs))

    def test_missing_labels_non_pag_and_zero_strength(self):
        original = self.attn.compute_attention
        expected = self.attn(self.x)
        for strength, options in ((0.75, {}), (0.75, {"cond_or_uncond": [0, 1]}),
                                  (0.0, {"cond_or_uncond": [2]})):
            with self.subTest(strength=strength, options=options):
                self.attn.compute_attention = pag._make_pag_compute_attention(
                    self.attn, original, 2, strength, None
                )
                torch.testing.assert_close(self.attn(self.x, transformer_options=options), expected)


if __name__ == "__main__":
    unittest.main()
