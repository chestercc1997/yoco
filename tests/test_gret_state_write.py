import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.train import verify_fork
from runtime.yoco_official.backend import _retention_sum, chunk_gate_retention
from runtime.yoco_official.rms_norm import RMSNorm


class StateWriteTests(unittest.TestCase):
    def test_mixed_safe_and_overflowing_entries_have_the_dot_product_gradient(self):
        left = torch.tensor([[16., -16.], [1., -1.]], requires_grad=True)
        right = torch.tensor([[2048., 1.], [-2048., 0.]], dtype=torch.float16, requires_grad=True)
        # Choose exactly representable partial gradients in both paths.
        weight = torch.tensor([[0.5, 1.], [-1., 2.]])
        with torch.autocast("cpu", torch.float16):
            result = _retention_sum(left, right, torch.float32)
        grad_left, grad_right = torch.autograd.grad((result * weight).sum(), (left, right))
        torch.testing.assert_close(result.double(), left.double() @ right.double(), atol=0, rtol=0)
        torch.testing.assert_close(grad_left.double(), weight.double() @ right.double().T, atol=0, rtol=0)
        torch.testing.assert_close(grad_right.double(), left.double().T @ weight.double(), atol=0, rtol=0)

    def test_only_state_write_overflows_before_normalization(self):
        q = torch.full((1, 1, 8, 4), 1 / 16, requires_grad=True)
        k = torch.full_like(q, 32.0, requires_grad=True)
        v = torch.tensor([2048.0, -1024.0], dtype=torch.float16).repeat(1, 1, 8, 1).requires_grad_()
        g = torch.zeros(1, 1, 8, dtype=torch.float16, requires_grad=True)
        with torch.autocast("cpu", torch.float16):
            old_write = (k[:, :, :4] / 2).transpose(-1, -2) @ v[:, :, :4]
            raw = chunk_gate_retention(q, k, v, g, chunk_size=4)
        self.assertTrue(torch.isinf(old_write).any())
        expected = torch.arange(1, 9)[:, None] * torch.tensor([8192.0, -4096.0])
        torch.testing.assert_close(raw[0, 0], expected, rtol=0, atol=0)
        RMSNorm(2, elementwise_affine=False)(raw).to(v.dtype)[..., 0].float().sum().backward()
        for tensor in (q, k, v, g):
            self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertGreater(v.grad[..., :4, :].abs().max().item(), 0)

    def test_backend_only_fork_requires_matching_manifest_and_full_config(self):
        config = {"training": {"precision": "fp16_mixed_fp32_params"}}
        name = "yoco_official/backend.py"
        identity = {"model_source_sha256": {name: "frozen-backend-hash"}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            cp = path / "checkpoint.pt"
            cp.write_bytes(b"checkpoint")
            cp_hash = hashlib.sha256(cp.read_bytes()).hexdigest()
            fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            cp.with_suffix(".json").write_text(json.dumps(dict(file=cp.name, fingerprint=fingerprint,
                                                                sha256=cp_hash, step=5004)))
            (path / "resolved_config.json").write_text(json.dumps(dict(identity=identity, config=config,
                synthetic_smoke=False, data={}, world_size=8)))
            proof = path / "repair.json"
            current = hashlib.sha256((ROOT / "runtime" / name).read_bytes()).hexdigest()
            proof.write_text(json.dumps(dict(passed=True, source_checkpoint_sha256=cp_hash,
                model_source_changes={name: dict(before="frozen-backend-hash", after=current)})))
            self.assertEqual(verify_fork(cp, config, {}, 8, proof)["source_step"], 5004)
            with self.assertRaises(RuntimeError):
                verify_fork(cp, config, {}, 8)
            with self.assertRaises(RuntimeError):
                verify_fork(cp, dict(config, execution={"compile": True}), {}, 8, proof)
            proof.write_text(proof.read_text().replace(current, "different-source"))
            with self.assertRaises(RuntimeError):
                verify_fork(cp, config, {}, 8, proof)


if __name__ == "__main__":
    unittest.main()
