"""Regression for the FP16 boundary captured at 400M update 24935."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.yoco_official.backend import chunk_gate_retention
from runtime.yoco_official.rms_norm import RMSNorm


class CrossChunkOverflowTest(unittest.TestCase):
    def test_normalize_large_cross_chunk_results_before_fp16_cast(self):
        # Each token adds [12288, -6144]. Even the largest four-token
        # within-chunk sum fits in half, while the third chunk's prefix does not.
        q = torch.full((1, 1, 12, 4), 4.0, requires_grad=True)
        k = q.detach().clone().requires_grad_()
        v = torch.tensor([384.0, -192.0], dtype=torch.float16).repeat(1, 1, 12, 1).requires_grad_()
        g = torch.zeros(1, 1, 12, dtype=torch.float16, requires_grad=True)
        norm = RMSNorm(2, elementwise_affine=False, eps=1e-5)
        with torch.autocast("cpu", dtype=torch.float16):
            raw = chunk_gate_retention(q, k, v, g, chunk_size=4)
            activation = norm(raw).to(v.dtype)
        expected = torch.arange(1, 13)[:, None] * torch.tensor([12288.0, -6144.0])
        torch.testing.assert_close(raw[0, 0], expected, rtol=0, atol=0)
        torch.testing.assert_close(activation[0, 0], norm(expected).half(), rtol=0, atol=0)
        activation[..., 0].float().mean().backward()
        for tensor in (q, k, v, g):
            self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertGreater(v.grad.abs().max().item(), 0)


if __name__ == "__main__":
    unittest.main()
