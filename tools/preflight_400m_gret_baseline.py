"""Prepare the original random gRet 400M baseline without using a GPU."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from runtime.model import build_model
from tools.run_swa_ab import read, digest, atomic_json, now


def main():
    torch.set_num_threads(4)
    cfg = read(ROOT / 'configs/yoco_400m_gret_baseline_8gpu.json')
    b_run = Path(read(ROOT / 'runs/setup/400m-B-active.json')['run_dir'])
    proof = read(b_run / 'release/preflight.json')
    b_cfg = read(b_run / 'release/configs/B.json')
    a_run = Path(proof['source_A_run'])
    a_cfg = read(a_run / 'resolved_config.json')['config']
    for key in ('architecture', 'model', 'training', 'data', 'seed', 'tied_embeddings', 'execution'):
        assert cfg[key] == a_cfg[key] == b_cfg[key], key
    assert 'memory_fusion' not in cfg and cfg['model'].get('sliding_window') is None
    assert not cfg['execution']['compile'] and proof['passed']
    for name, expected in proof['repaired_runtime_sha256'].items():
        assert digest(ROOT / name) == expected, name
    source_a = Path(proof['source_A_initialization'])
    source_b = b_run / 'release/initializations/B.pt'
    assert digest(source_a) == proof['source_A_initialization_sha256']
    assert digest(source_b) == proof['initialization_sha256']
    a = torch.load(source_a, map_location='cpu', weights_only=False, mmap=True)['model']
    b = torch.load(source_b, map_location='cpu', weights_only=False, mmap=True)['model']
    torch.manual_seed(cfg['seed']);model = build_model(cfg).eval()
    state = model.state_dict()
    assert len(state) == 222
    assert set(a) - set(state) == {'memory_fusion.logits'}
    assert set(b) - set(state) == {'memory_fusion.score', 'memory_fusion.layer_bias'}
    assert all(torch.equal(v, a[name]) and torch.equal(v, b[name]) for name, v in state.items())
    assert sum(p.numel() for p in model.parameters()) == 400590848
    assert all('GateRetention' in type(layer.mixer).__name__ for layer in model.self_decoder.layers)
    # Independently run the original A release as a plain baseline, using its untrained weights.
    reference_output = ROOT / 'runs/setup/yoco-400m-gret-baseline-legacy-forward.pt'
    code = '''
import json,pathlib,sys,torch
torch.set_num_threads(4)
from runtime.model import build_model
run,initial,output=map(pathlib.Path,sys.argv[1:])
cfg=json.loads((run/'resolved_config.json').read_text())['config'];cfg.pop('memory_fusion')
torch.manual_seed(cfg['seed']);model=build_model(cfg).eval()
source=torch.load(initial,map_location='cpu',weights_only=False,mmap=True)['model']
model.load_state_dict({k:v for k,v in source.items() if not k.startswith('memory_fusion.')},strict=True)
ids=torch.arange(10,170,10).reshape(1,-1)
with torch.no_grad():logits=model(ids)[0]
torch.save({'ids':ids,'logits':logits},output)
'''
    subprocess.run([sys.executable, '-c', code, str(a_run), str(source_a), str(reference_output)],
                   cwd=a_run / 'release', env=dict(os.environ, PYTHONPATH=str(a_run / 'release'), CUDA_VISIBLE_DEVICES=''), check=True)
    old = torch.load(reference_output, map_location='cpu', weights_only=False)
    with torch.no_grad():
        logits = model(old['ids'])[0]
        difference = float((logits - old['logits']).abs().max())
        torch.testing.assert_close(logits, old['logits'], rtol=2e-5, atol=2e-5)
        future = old['ids'].clone();future[:, 8:] += 17
        causal = float((logits[:, :8] - model(future)[0][:, :8]).abs().max())
        assert causal < 2e-5
    initial = ROOT / 'runs/setup/yoco-400m-gret-baseline-initialization.pt'
    if initial.exists():
        saved = torch.load(initial, map_location='cpu', weights_only=False, mmap=True)
        assert saved['model'].keys() == state.keys()
        assert all(torch.equal(v, saved['model'][name]) for name, v in state.items())
    else:
        temporary = initial.with_suffix('.tmp')
        torch.save(dict(model=state, model_config=cfg['model'], tied_embeddings=cfg['tied_embeddings'], seed=cfg['seed']), temporary)
        temporary.replace(initial)
    del a, b, state, old, logits
    ids = torch.arange(10,170,10).reshape(1,-1)
    logits = model(ids)[0]
    loss = F.cross_entropy(logits.flatten(0,1), ids.roll(-1,1).flatten())
    loss.backward()
    assert math.isfinite(float(loss.detach()))
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    record = dict(passed=True, architecture='YOCO-gRet', parameters={'baseline':400590848},
        common_state_tensors_verified=222, common_weights_bitwise_equal=True,
        common_weights_match_original_A_initialization=True, common_weights_match_B_initialization=True,
        old_baseline_initial_forward_max_abs=difference, causality_prefix_max_abs=causal,
        all_gradients_finite=True, source_A_run=str(a_run), source_B_run=str(b_run),
        source_A_initialization_sha256=digest(source_a), source_B_initialization_sha256=digest(source_b),
        initializations={'baseline':{'path':str(initial),'sha256':digest(initial)}},
        repaired_runtime_sha256=proof['repaired_runtime_sha256'], created_at=now(),
        note='CPU preparation only. Full eight-GPU restore checks wait for predecessor completion.')
    atomic_json(ROOT / 'runs/setup/yoco-400m-gret-baseline-preflight.json', record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':main()
