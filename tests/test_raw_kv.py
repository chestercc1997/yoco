import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.model import build_model
from runtime.yoco_official.backend import flash_attn_func
from test_plan_ab_kv import config


class RawKVTests(unittest.TestCase):
    def test_routing_and_shared_backbone(self):
        for window in (None, 8):
            outputs = []
            for mode in (None, 'raw_learned_kv', 'raw_conditional_kv', 'depth_attention_kv'):
                c = config(mode)
                c['model']['sliding_window'] = window
                c['tied_embeddings'] = False
                torch.manual_seed(42)
                m = build_model(c)
                if mode is None:
                    base = m.state_dict()
                    continue
                for name, tensor in base.items():
                    self.assertTrue(torch.equal(tensor, m.state_dict()[name]))
                ids = torch.arange(11)[None]
                out = m(ids)[0]
                outputs.append(out.detach())
                out.square().mean().backward()
                for param in m.memory_fusion.parameters():
                    self.assertTrue(torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0)
                f = m.memory_fusion
                states = [torch.randn(1, 11, 32) for _ in range(6)]
                if mode == 'depth_attention_kv':
                    with torch.no_grad():
                        f.key_query.normal_()
                        f.value_query.copy_(-f.key_query)
                    k, v = f.depth_attention_weights(states)
                    self.assertFalse(torch.equal(k, v))
                    norm = torch.stack([torch.nn.functional.rms_norm(s, (32,), eps=f.eps) for s in states], -2)
                    for q, weights in ((f.key_query, k), (f.value_query, v)):
                        torch.testing.assert_close(weights, ((norm @ q) / math.sqrt(32)).softmax(-1))
                    # The two routing tensors must independently influence the decoder output.
                    args = dict(x=states[-1], memory_sources=states, memory_weights=k)
                    routed = m.cross_decoder(**args, value_weights=v)
                    self.assertFalse(torch.allclose(routed, m.cross_decoder(**args, value_weights=k)))
                else:
                    with torch.no_grad():
                        bias = f.weights if mode == 'raw_learned_kv' else f.layer_bias
                        bias.copy_(torch.tensor([-2., 0., 1., 0., 0., 3.]))
                    expected = bias.expand(1, 11, -1)
                    torch.testing.assert_close(f.conditional_weights(states), expected)
            for candidate in outputs[1:]:
                torch.testing.assert_close(candidate, outputs[0], rtol=0, atol=0)

    def test_training_window_matches_dense_forward_and_backward(self):
        torch.manual_seed(42)
        length, window, dim = 1030, 1024, 8
        q, k, v = [torch.randn(1, length, 1, dim, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        result = flash_attn_func(q, k, v, window_size=(window - 1, 0))
        pos = torch.arange(length)
        distance = pos[:, None] - pos[None, :]
        scores = q[0, :, 0] @ k[0, :, 0].T / math.sqrt(dim)
        scores = scores.masked_fill((distance < 0) | (distance >= window), -torch.inf)
        expected = (scores.softmax(-1) @ v[0, :, 0])[None, :, None, :]
        grad = torch.randn_like(expected)
        torch.testing.assert_close(result, expected)
        for a, b in zip(torch.autograd.grad(result, (q,k,v), grad),
                        torch.autograd.grad(expected, (q,k,v), grad)):
            torch.testing.assert_close(a, b)
