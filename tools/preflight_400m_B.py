"""Verify gRet 400M B against the completed A recipe and original shared initialization."""
import argparse
import hashlib
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
from runtime.train import atomic_json, sha256

REFERENCE = ROOT / 'runs/yoco-400m-A-8gpu-retentionfix-20261106'
SOURCE_INITIAL = ROOT / 'runs/setup/yoco-400m-A-initialization.pt'
INITIAL = ROOT / 'runs/setup/yoco-400m-B-initialization.pt'
PROOF = ROOT / 'runs/setup/yoco-400m-B-preflight.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=PROOF)
    p.add_argument('--initialization', type=Path, default=INITIAL)
    args = p.parse_args()
    torch.set_num_threads(4)
    a = json.loads((REFERENCE / 'resolved_config.json').read_text())['config']
    config = json.loads((ROOT / 'configs/yoco_400m_plan_B_8gpu.json').read_text())
    old = json.loads((ROOT / 'runs/setup/yoco-400m-A-preflight.json').read_text())
    for key in ('architecture', 'model', 'training', 'data', 'seed', 'tied_embeddings', 'execution'):
        assert config[key] == a[key], key
    assert config['model'].get('sliding_window') is None
    assert config['memory_fusion'] == dict(mode='conditional_kv', source_layers=list(range(1, 13)), temperature=1.)
    assert not config['execution']['compile']
    assert sha256(SOURCE_INITIAL) == old['initialization_sha256']
    # These are the repaired gRet arithmetic and AMP implementations already used by A.
    manifest = json.loads((REFERENCE / 'release-manifest.json').read_text())['files']
    repaired = ('runtime/train.py', 'runtime/yoco_official/backend.py', 'runtime/yoco_official/gate_retention.py')
    for name in repaired:
        assert sha256(ROOT / name) == manifest[name], name

    reference_output = ROOT / 'runs/setup/yoco-400m-B-legacy-A-forward.pt'
    reference_code = r'''
import json,pathlib,sys,torch
torch.set_num_threads(4)
from runtime.model import build_model
reference=pathlib.Path(sys.argv[1]);initial=pathlib.Path(sys.argv[2]);output=pathlib.Path(sys.argv[3])
config=json.loads((reference/'resolved_config.json').read_text())['config']
torch.manual_seed(config['seed']);model=build_model(config).eval()
state=torch.load(initial,map_location='cpu',weights_only=False,mmap=True)
model.load_state_dict(state['model'],strict=True)
ids=torch.tensor([[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160]])
with torch.no_grad():logits=model(ids)[0]
torch.save({'ids':ids,'logits':logits,'reference_config':config},output)
'''
    subprocess.run([sys.executable, '-c', reference_code, str(REFERENCE), str(SOURCE_INITIAL), str(reference_output)],
                   cwd=REFERENCE / 'release', env=dict(os.environ, PYTHONPATH=str(REFERENCE / 'release')), check=True)
    reference = torch.load(reference_output, map_location='cpu', weights_only=False)
    ids = reference['ids']
    baseline = {k: v for k, v in a.items() if k != 'memory_fusion'}
    torch.manual_seed(config['seed']);base = build_model(baseline).eval()
    torch.manual_seed(config['seed']);model = build_model(config).eval()
    source = torch.load(SOURCE_INITIAL, map_location='cpu', weights_only=False, mmap=True)['model']
    common = base.state_dict();state = model.state_dict()
    assert len(common) == old['common_state_tensors_verified'] == 222
    assert set(source) - set(common) == {'memory_fusion.logits'}
    assert all(torch.equal(v, source[name]) and torch.equal(v, state[name]) for name, v in common.items())
    assert set(state) - set(common) == {'memory_fusion.score', 'memory_fusion.layer_bias'}
    assert all(torch.count_nonzero(state[name]) == 0 for name in set(state) - set(common))
    count = sum(p.numel() for p in model.parameters());base_count = sum(p.numel() for p in base.parameters())
    assert base_count == 400590848 and count == 400591884 and count - base_count == 1036
    assert all('GateRetention' in type(layer.mixer).__name__ for layer in model.self_decoder.layers)
    with torch.no_grad():
        uniform = model.memory_fusion.conditional_weights([torch.ones(1, 16, 1024)] * 12)
        torch.testing.assert_close(uniform, torch.full_like(uniform, 1 / 12), rtol=0, atol=0)
        actual = model(ids)[0]
        old_difference = float((actual - reference['logits']).abs().max())
        torch.testing.assert_close(actual, reference['logits'], rtol=2e-5, atol=2e-5)
    args.initialization.parent.mkdir(parents=True, exist_ok=True)
    if args.initialization.exists():
        saved = torch.load(args.initialization, map_location='cpu', weights_only=False, mmap=True)
        assert saved['memory_fusion'] == config['memory_fusion']
        assert all(torch.equal(v, saved['model'][name]) for name, v in state.items())
        del saved
    else:
        temporary = args.initialization.with_suffix('.tmp')
        torch.save(dict(model=state, model_config=config['model'], tied_embeddings=config['tied_embeddings'],
                        memory_fusion=config['memory_fusion'], seed=config['seed'],
                        shared_A_initialization_sha256=old['initialization_sha256']), temporary)
        temporary.replace(args.initialization)
    with torch.no_grad():
        expected = base(ids)[0]
        model.memory_fusion.layer_bias.fill_(-1000);model.memory_fusion.layer_bias[-1] = 0
        limited = model(ids)[0]
        last_difference = float((limited - expected).abs().max())
        torch.testing.assert_close(limited, expected, rtol=2e-5, atol=2e-5)
        model.memory_fusion.layer_bias.zero_()
        original = model(ids)[0]
        future = ids.clone();future[:, 8:] += 17
        perturbed = model(future)[0]
        causal_difference = float((original[:, :8] - perturbed[:, :8]).abs().max())
        assert causal_difference < 2e-5
    del base, source, common, reference, actual, expected, limited, original, perturbed
    model.zero_grad(set_to_none=True)
    logits = model(ids)[0]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids.roll(-1, 1).reshape(-1))
    loss.backward()
    assert math.isfinite(float(loss))
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    gradients = {name: float(getattr(model.memory_fusion, name).grad.norm()) for name in ('score', 'layer_bias')}
    assert all(math.isfinite(v) and v > 0 for v in gradients.values())
    record = dict(passed=True, architecture='YOCO-gRet', baseline_parameters=base_count, B_parameters=count,
                  extra_parameters=1036, common_state_tensors_verified=222, common_weights_bitwise_equal=True,
                  common_weights_match_original_A_initialization=True, model=config['model'],
                  memory_fusion=config['memory_fusion'], training_and_data_match_completed_A=True,
                  execution_matches_completed_A_final_native_segment=True, precision=config['training']['precision'],
                  initial_weights=[1 / 12] * 12, old_A_initial_forward_max_abs=old_difference,
                  last_source_limit_max_abs=last_difference, causality_prefix_max_abs=causal_difference,
                  full_size_control_gradients=gradients, all_gradients_finite=True,
                  source_A_run=str(REFERENCE), source_A_initialization=str(SOURCE_INITIAL),
                  source_A_initialization_sha256=old['initialization_sha256'],
                  initialization=str(args.initialization), initialization_sha256=sha256(args.initialization),
                  reference_output_sha256=sha256(reference_output),
                  repaired_runtime_sha256={name: sha256(ROOT / name) for name in repaired},
                  note='New gRet 400M B pretraining from the shared random initialization; historical A has an earlier compiled/repaired execution lineage.')
    args.output.parent.mkdir(parents=True, exist_ok=True);atomic_json(args.output, record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
