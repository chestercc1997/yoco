"""Contract checks for the plan's conditional six-source K/V fusion."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from runtime.model import build_model


def config(dynamic=False):
    c = dict(architecture='yoco', tied_embeddings=True, model=dict(
        dim=32, n_layers=6, hidden_dim=96, n_self_heads=2,
        n_attn_heads=4, n_attn_kv_heads=4, vocab_size=97,
        max_seq_len=512, norm_eps=1e-5))
    if dynamic:
        c['memory_fusion'] = dict(mode='conditional_kv', source_layers=[1, 2, 3], temperature=1.)
    return c


class ConditionalKVTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_formula_and_gradient(self):
        model = build_model(config(True))
        f = model.memory_fusion
        with torch.no_grad():
            f.score.normal_(); f.layer_bias.normal_()
        states = [torch.randn(2, 8, 32, requires_grad=True) for _ in range(3)]
        norm = torch.stack([x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + f.eps) for x in states], -2)
        expected = ((norm * f.score).sum(-1) / 32**.5 + f.layer_bias).softmax(-1)
        actual = f.conditional_weights(states)
        torch.testing.assert_close(actual, expected)
        self.assertGreater((actual[:, 0] - actual[:, 1]).abs().max().item(), 1e-4)
        decoder = model.cross_decoder
        collected = []
        handle = decoder.layers[0].register_forward_pre_hook(lambda m, a, kw: collected.append((kw['key'], kw['value'])), with_kwargs=True)
        try:
            decoder(states[-1], memory_sources=states, memory_weights=actual)
        finally:
            handle.remove()
        from runtime.yoco_official.yoco import apply_rotary_emb
        k = sum(decoder.k_proj(decoder.kv_layer_norm(x)) * expected[..., i, None] for i, x in enumerate(states)).view(2, 8, 4, 8)
        v = sum(decoder.v_proj(decoder.kv_layer_norm(x)) * expected[..., i, None] for i, x in enumerate(states)).view(2, 8, 4, 8)
        k = apply_rotary_emb(k, *decoder.build_rel_pos(states[-1], 0), interleaved=True)
        torch.testing.assert_close(collected[0][0], k)
        torch.testing.assert_close(collected[0][1], v)
        grads = torch.autograd.grad(collected[0][0].square().mean() + collected[0][1].square().mean(), states + list(f.parameters()))
        self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads))

    def test_baseline_parameters_and_causal_cache(self):
        torch.manual_seed(42); base = build_model(config())
        torch.manual_seed(42); model = build_model(config(True)).eval()
        for name, p in base.state_dict().items():
            torch.testing.assert_close(p, model.state_dict()[name], rtol=0, atol=0)
        with torch.no_grad():
            model.memory_fusion.score.normal_()
            ids = torch.randint(0, 97, (2, 9))
            full = model(ids)[0]
            changed = ids.clone(); changed[:, 5:] = (changed[:, 5:] + 1) % 97
            torch.testing.assert_close(full[:, :5], model(changed)[0][:, :5], atol=2e-5, rtol=2e-5)
            state = {}
            pieces = [model(ids[:, i:i+1], start_pos=i, incremental_state=state, is_prefilling=False)[0] for i in range(9)]
            torch.testing.assert_close(full, torch.cat(pieces, 1), atol=2e-5, rtol=2e-5)

    def test_activation_checkpoint_and_optimizer_resume(self):
        plain = build_model(config(True))
        checked = build_model(config(True), checkpoint_activations=True)
        checked.load_state_dict(plain.state_dict())
        ids = torch.randint(0, 97, (2, 9))
        for model in (plain, checked):
            model(ids)[0].square().mean().backward()
        for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
            self.assertIsNotNone(q.grad, name)
            torch.testing.assert_close(p.grad, q.grad, atol=5e-6, rtol=1e-4)
        self.assertGreater(checked.memory_fusion.score.grad.abs().sum().item(), 0)
        import io
        opt = torch.optim.AdamW(plain.parameters(), lr=1e-3)
        opt.step(); opt.zero_grad()
        stream = io.BytesIO(); torch.save({'model':plain.state_dict(), 'optimizer':opt.state_dict()}, stream)
        stream.seek(0); saved = torch.load(stream, weights_only=True)
        restored = build_model(config(True)); restored.load_state_dict(saved['model'])
        opt2 = torch.optim.AdamW(restored.parameters(), lr=1e-3); opt2.load_state_dict(saved['optimizer'])
        for model, optimizer in ((plain, opt), (restored, opt2)):
            model(ids)[0].square().mean().backward(); optimizer.step()
        for p, q in zip(plain.parameters(), restored.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
