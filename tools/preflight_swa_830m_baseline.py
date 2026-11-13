"""Validate a fresh SWA 830M baseline without changing the established recipe."""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from runtime.model import build_model
from runtime.train import atomic_json, sha256
from runtime.yoco_official.backend import flash_attn_func
from runtime.yoco_official.sliding_window_attention import SlidingWindowAttention


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initialization-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = json.loads((ROOT/'configs/yoco_830m_swa_baseline_8gpu.json').read_text())
    reference = json.loads((ROOT/'configs/yoco_400m_swa_baseline_8gpu.json').read_text())
    for key in ('architecture', 'training', 'data', 'seed', 'tied_embeddings', 'execution'):
        assert cfg[key] == reference[key], key
    expected = copy.deepcopy(reference['model'])
    expected.update(dim=1536, hidden_dim=4608, n_self_heads=12,
                    n_attn_heads=12, n_attn_kv_heads=12)
    assert cfg['model'] == expected and 'memory_fusion' not in cfg
    torch.manual_seed(cfg['seed'])
    model = build_model(cfg).eval()
    assert all(isinstance(layer.mixer, SlidingWindowAttention) for layer in model.self_decoder.layers)
    parameters = sum(p.numel() for p in model.parameters())
    state = model.state_dict()
    assert all(torch.isfinite(v).all() for v in state.values())
    args.initialization_dir.mkdir(parents=True, exist_ok=True)
    path = args.initialization_dir/'baseline.pt'
    if path.exists():
        saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        assert saved['model'].keys() == state.keys()
        assert all(torch.equal(v, saved['model'][k]) for k, v in state.items())
        del saved
    else:
        temp = path.with_suffix('.tmp')
        torch.save(dict(model=state, model_config=cfg['model'], seed=cfg['seed'],
                        tied_embeddings=True, arm='baseline'), temp)
        temp.replace(path)
    torch.manual_seed(cfg['seed'])
    repeat = build_model(cfg).eval()
    assert all(torch.equal(v, repeat.state_dict()[k]) for k, v in state.items())
    del repeat
    gc.collect()
    ids = torch.arange(10, 26).unsqueeze(0)
    changed = ids.clone(); changed[:, 8:] += 31
    with torch.no_grad():
        output = model(ids)[0]
        altered = model(changed)[0]
        causal_error = float((output[:, :8]-altered[:, :8]).abs().max())
        assert torch.isfinite(output).all() and causal_error < 2e-5
    model.zero_grad(set_to_none=True)
    logits = model(ids[:, :8])[0]
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids[:, 1:9].flatten())
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    probe_loss = float(loss.detach())
    del loss, logits, output, altered, state, model
    gc.collect()
    q = torch.zeros(1, 1030, 1, 4); v = q.clone(); v[:, 0] = 1
    with torch.no_grad(): local = flash_attn_func(q, q, v, causal=True, window_size=(1023, 0))
    torch.testing.assert_close(local[0, 1023], torch.full((1, 4), 1/1024), rtol=1e-5, atol=1e-7)
    assert torch.equal(local[0, 1024:], torch.zeros_like(local[0, 1024:]))
    tiny = copy.deepcopy(cfg)
    tiny['model'].update(dim=32, n_layers=4, hidden_dim=64, n_self_heads=4,
                         n_attn_heads=4, n_attn_kv_heads=4, vocab_size=256,
                         max_seq_len=32, sliding_window=8)
    torch.manual_seed(42); model = build_model(tiny).eval()
    tokens = torch.arange(10, 30).repeat(2, 1)
    with torch.no_grad():
        full = model(tokens)[0]; cache = {}; parts = []
        for pos in range(tokens.shape[1]):
            parts.append(model(tokens[:, pos:pos+1], start_pos=pos,
                               incremental_state=cache, is_prefilling=False)[0])
        tokenwise = torch.cat(parts, 1); cache = {}; parts = []
        for start, end in ((0, 11), (11, 16), (16, 20)):
            parts.append(model(tokens[:, start:end], start_pos=start,
                               incremental_state=cache, is_prefilling=start == 0)[0])
        chunkwise = torch.cat(parts, 1)
    torch.testing.assert_close(full, tokenwise, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(full, chunkwise, rtol=2e-5, atol=2e-5)
    assert cache[0]['prev_key'].shape == (2, 8, 4, 8)
    proof = dict(passed=True, architecture='YOCO-SWA', scale='830M',
                 model=cfg['model'], parameters={'baseline': parameters},
                 common_weights_bitwise_equal=True, repeated_seed_initialization_bitwise_equal=True,
                 initialization_origin='fresh random seed42; no trained checkpoint loaded',
                 training_data_execution_match_swa400m=True, all_gradients_finite=True,
                 probe_loss=probe_loss, causality_prefix_max_abs=causal_error,
                 window_boundary_1024_passed=True,
                 incremental_cache_checks=dict(tokenwise_max_abs=float((full-tokenwise).abs().max()),
                                               chunkwise_max_abs=float((full-chunkwise).abs().max())),
                 initializations={'baseline': dict(path=str(path.resolve()), sha256=sha256(path))},
                 official_setting_source='https://arxiv.org/html/2405.05254v2#A4',
                 reproduction_boundary='Official 830M architecture dimensions and 3d FFN; existing approved FineWeb-Edu recipe, not original paper corpus.')
    atomic_json(args.output, proof)
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == '__main__': main()
