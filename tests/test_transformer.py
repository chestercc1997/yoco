"""Check attention against explicit equations, causality, recomputation and loading."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from runtime.model import build_model
from runtime.yoco_official.transformer import Attention, ModelArgs


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    args = ModelArgs(dim=32, n_layers=2, head_dim=8, hidden_dim=80,
                     n_heads=4, n_kv_heads=4, norm_eps=1e-5,
                     vocab_size=97, max_seq_len=64, max_batch_size=2)
    attn = Attention(args).double()
    x = torch.randn(2, 7, 32, dtype=torch.double, requires_grad=True)
    angles = torch.randn(7, 4, dtype=torch.double)
    cos, sin = angles.cos(), angles.sin()
    actual = attn(x, (cos, sin), 0)
    q, k, v = [linear(x).reshape(2, 7, 4, 8) for linear in (attn.wq, attn.wk, attn.wv)]
    def rotate(t):
        phase = torch.cat((cos, cos), -1)[None, :, None, :]
        sine = torch.cat((sin, sin), -1)[None, :, None, :]
        return t * phase + torch.cat((-t[..., 4:], t[..., :4]), -1) * sine
    q, k, v = rotate(q).transpose(1, 2), rotate(k).transpose(1, 2), v.transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / (8 ** 0.5)
    scores = scores.masked_fill(torch.ones(7, 7, dtype=torch.bool).triu(1), -torch.inf)
    reference = attn.wo((scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 7, 32))
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
    inputs = (x, *attn.parameters())
    ga = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    gb = torch.autograd.grad(reference.square().sum(), inputs)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
    c = {'architecture': 'transformer', 'model': vars(args), 'tied_embeddings': True}
    model = build_model(c).eval()
    assert model.output.weight is model.tok_embeddings.weight
    ids = torch.randint(0, 97, (2, 11))
    changed = ids.clone(); changed[:, 6:] = (changed[:, 6:] + 1) % 97
    with torch.no_grad():
        logits = model(ids)[0]
        torch.testing.assert_close(logits[:, :6], model(changed)[0][:, :6])
        state = {}
        cached = torch.cat([model(ids[:, i:i+1], start_pos=i, incremental_state=state)[0]
                            for i in range(ids.shape[1])], dim=1)
        torch.testing.assert_close(logits, cached, atol=2e-6, rtol=2e-5)
    checked = build_model(c, checkpoint_activations=True)
    checked.load_state_dict(model.state_dict(), strict=True)
    model.train(); checked.train()
    for m in (model, checked):
        m(ids)[0].square().mean().backward()
    for a, b in zip(model.parameters(), checked.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)
    bad = copy.copy(model.state_dict()); bad.pop('layers.0.attention.wq.weight')
    try:
        checked.load_state_dict(bad, strict=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Missing weight accepted')
    proof = {'passed': True, 'tests': ['explicit_attention_forward_and_gradients',
             'non_interleaved_rotary', 'full_model_causality', 'cached_decode_equivalence',
             'activation_checkpoint_gradients', 'tied_embeddings', 'strict_checkpoint_loading']}
    Path('provenance/transformer_tests.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
