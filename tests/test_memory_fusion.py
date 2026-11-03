"""CPU contract checks for experimental YOCO memory fusion."""
from pathlib import Path
import io
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from runtime.model import build_model


def config(mode=None):
    c = dict(architecture='yoco', tied_embeddings=True, model=dict(
        dim=32, n_layers=6, hidden_dim=96, n_self_heads=2,
        n_attn_heads=4, n_attn_kv_heads=4, vocab_size=97,
        max_seq_len=512, norm_eps=1e-5))
    if mode is not None:
        c['memory_fusion'] = dict(mode=mode, source_layers=[1, 2, 3], temperature=1.0)
    return c


class MemoryFusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        self.ids = torch.randint(0, 97, (2, 9))

    def test_disabled_exact_forward_and_gradient_parity(self):
        base, last = build_model(config()), build_model(config('last'))
        last.load_state_dict(base.state_dict(), strict=True)
        self.assertEqual(set(base.state_dict()), set(last.state_dict()))
        a, b = base(self.ids)[0], last(self.ids)[0]
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        a.square().mean().backward(); b.square().mean().backward()
        for p, q in zip(base.parameters(), last.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)

    def test_uniform_controls_and_reference_gradients(self):
        states = [torch.randn(2, 9, 32, requires_grad=True) for _ in range(3)]
        for mode in ('mean', 'static', 'token'):
            with self.subTest(mode=mode):
                fusion = build_model(config(mode)).memory_fusion
                torch.testing.assert_close(fusion(states), torch.stack(states).mean(0))
        fusion = build_model(config('token')).memory_fusion
        with torch.no_grad():
            fusion.router.weight.normal_(0, .15)
        weights = fusion.routing_weights(states[-1])
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 9))
        self.assertGreater((weights[:, 0] - weights[:, 1]).abs().max().item(), 1e-3)
        reference = (torch.stack(states, dim=-2) * weights.unsqueeze(-1)).sum(-2)
        output = fusion(states)
        torch.testing.assert_close(output, reference)
        variables = states + list(fusion.parameters())
        a = torch.autograd.grad(output.square().sum(), variables, retain_graph=True)
        b = torch.autograd.grad(reference.square().sum(), variables)
        for p, q in zip(a, b):
            torch.testing.assert_close(p, q)

    def test_selected_memory_and_unchanged_query_stream(self):
        c = config('mean')
        c['memory_fusion']['source_layers'] = [1, 3]
        model = build_model(c)
        hidden, inputs = [], {}
        handles = [layer.register_forward_hook(lambda m, a, out: hidden.append(out))
                   for layer in model.self_decoder.layers]
        def capture(name):
            def hook(module, args):
                inputs[name] = args[0]
            return hook
        handles.append(model.cross_decoder.kv_layer_norm.register_forward_pre_hook(capture('memory')))
        handles.append(model.cross_decoder.layers[0].register_forward_pre_hook(capture('query')))
        try:
            model(self.ids)
            length = self.ids.shape[1]
            torch.testing.assert_close(inputs['memory'], (hidden[0][:, :length] + hidden[2][:, :length]) / 2)
            torch.testing.assert_close(inputs['query'], hidden[2][:, :length], atol=0, rtol=0)
        finally:
            for handle in handles:
                handle.remove()

    def test_paired_initialization_and_160m_parameter_counts(self):
        torch.manual_seed(42)
        baseline = build_model(config()).state_dict()
        for mode in ('last', 'mean', 'static', 'token'):
            torch.manual_seed(42)
            state = build_model(config(mode)).state_dict()
            for key, value in baseline.items():
                torch.testing.assert_close(value, state[key], atol=0, rtol=0)
        import json
        root = Path(__file__).resolve().parents[1]
        counts = {'last': 166708224, 'mean': 166708224,
                  'static': 166708227, 'token': 166710531}
        for mode, expected in counts.items():
            c = json.loads((root / f'configs/yoco_160m_fusion_{mode}_4gpu.json').read_text())
            with torch.device('meta'):
                model = build_model(c)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)

    def test_causality_and_incremental_cache(self):
        model = build_model(config('token')).eval()
        with torch.no_grad():
            # Exercise genuinely token-dependent weights, beyond zero initialization.
            model.memory_fusion.router.weight.normal_(0, .15)
            full = model(self.ids)[0]
            changed = self.ids.clone(); changed[:, 5:] = (changed[:, 5:] + 1) % 97
            torch.testing.assert_close(full[:, :5], model(changed)[0][:, :5], atol=2e-5, rtol=2e-5)
            state = {}
            parts = [model(self.ids[:, i:i+1], start_pos=i, incremental_state=state,
                           is_prefilling=False)[0] for i in range(9)]
            torch.testing.assert_close(full, torch.cat(parts, 1), atol=2e-5, rtol=2e-5)
            self.assertEqual(set(state), {0, 1, 2, 'prev_key', 'prev_value'})
            self.assertEqual(state['prev_key'].shape, (2, 512, 4, 8))
            state = {}; parts = []
            for start, end in ((0, 4), (4, 8), (8, 9)):
                parts.append(model(self.ids[:, start:end], start_pos=start,
                                   incremental_state=state, is_prefilling=end != 9)[0])
            torch.testing.assert_close(full, torch.cat(parts, 1), atol=2e-5, rtol=2e-5)
            state = {}
            model(self.ids[:, :8], incremental_state=state, skip_cross_decoder=True)
            last = model(self.ids[:, 8:], start_pos=8, incremental_state=state,
                         is_prefilling=False)[0]
            torch.testing.assert_close(full[:, 8:], last, atol=2e-5, rtol=2e-5)

    def test_activation_checkpoint_gradients(self):
        plain = build_model(config('token'))
        checked = build_model(config('token'), checkpoint_activations=True)
        checked.load_state_dict(plain.state_dict(), strict=True)
        for model in (plain, checked):
            model(self.ids)[0].square().mean().backward()
        for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
            self.assertIsNotNone(q.grad, name)
            self.assertTrue(torch.isfinite(q.grad).all().item(), name)
            torch.testing.assert_close(p.grad, q.grad, atol=5e-6, rtol=1e-4)
        self.assertGreater(checked.memory_fusion.router.weight.grad.abs().sum().item(), 0)

    def test_model_optimizer_roundtrip_continuation(self):
        model = build_model(config('token'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        def step(m, o):
            o.zero_grad(); loss = m(self.ids)[0].square().mean(); loss.backward(); o.step()
        step(model, optimizer)
        stream = io.BytesIO()
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream)
        stream.seek(0); state = torch.load(stream, weights_only=True)
        restored = build_model(config('token'))
        restored.load_state_dict(state['model'], strict=True)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_optimizer.load_state_dict(state['optimizer'])
        step(model, optimizer); step(restored, restored_optimizer)
        for p, q in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_invalid_configs(self):
        invalid = [dict(mode='typo'), dict(source_layers=[2, 1, 3]),
                   dict(source_layers=[1, 1, 3]), dict(source_layers=[1, 2]),
                   dict(source_layers=[0, 3]), dict(temperature=0),
                   dict(temperature=float('nan')), dict(unknown=True),
                   dict(initial_weights=[0, 0, 1]), dict(initial_weights=[.1, .9]),
                   dict(initial_weights=[.1, .1, .9]), dict(initial_weights=[.1, float('nan'), .9]),
                   dict(mode='mean', initial_weights=[.05, .05, .9])]
        for override in invalid:
            c = config('token'); c['memory_fusion'].update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                build_model(c)
        c = config('token'); c['architecture'] = 'transformer'
        with self.assertRaises(ValueError):
            build_model(c)

    def test_last_biased_prior_changes_only_router_initialization(self):
        c = config('token')
        c['memory_fusion'].update(initial_weights=[.05, .05, .90], temperature=.7)
        torch.manual_seed(42)
        model = build_model(c)
        torch.manual_seed(42)
        original = build_model(config('token'))
        for name, p in original.named_parameters():
            if name != 'memory_fusion.router.bias':
                torch.testing.assert_close(p, dict(model.named_parameters())[name], atol=0, rtol=0)
        states = [torch.randn(2, 9, 32) for _ in range(3)]
        weights = model.memory_fusion.routing_weights(states[-1])
        torch.testing.assert_close(weights, weights.new_tensor([.05, .05, .90]).expand_as(weights))
        torch.testing.assert_close(model.memory_fusion(states), .05*states[0]+.05*states[1]+.90*states[2])
        checked = build_model(c, checkpoint_activations=True)
        checked.load_state_dict(model.state_dict())
        for m in (model, checked):
            m(self.ids)[0].square().mean().backward()
        for p, q in zip(model.parameters(), checked.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=5e-6, rtol=1e-4)
        self.assertGreater(model.memory_fusion.router.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            full = model(self.ids)[0]
            state = {}
            pieces = [model(self.ids[:, i:i+1], start_pos=i, incremental_state=state,
                            is_prefilling=False)[0] for i in range(9)]
            torch.testing.assert_close(full, torch.cat(pieces, 1), atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
