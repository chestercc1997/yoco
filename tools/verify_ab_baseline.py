"""Verify three-way evidence, full baseline replacement and release hashes."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--refresh-manifest', action='store_true', help='Refresh hashes after numerical checks pass')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]/'evaluation/ab_baseline_20261104'
s = json.loads((root/'summary.json').read_text())


def rows(name):
    with (root/name).open() as handle:
        return list(csv.DictReader(handle))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


assert s['schema_version'] == 2 and s['baseline_replacement']['kind'] == 'verified_local_eight_gpu_training'
assert s['training']['baseline']['world_size'] == 8 and 'A100' in s['training']['baseline']['hardware']
audit = s['comparability_audit']
assert audit['passed'] and audit['training_and_model_configs_matched']
assert audit['ab_initial_outputs_identical'] and audit['last_source_limit_matches_baseline']
assert audit['full_baseline_checkpoint_payload_verified']
assert audit['extra_parameters'] == {'A': 6, 'B': 774}
configs = json.loads((root/'training_protocol.json').read_text())
for arm in 'AB':
    for key in ('architecture', 'model', 'training', 'seed', 'tied_embeddings'):
        assert configs['baseline'][key] == configs[arm][key], (arm, key)
assert 'memory_fusion' not in configs['baseline']
curves = rows('validation_curves.csv')
train = rows('training_curves.csv')
ab_train = rows('ab_training_curves.csv')
scores = rows('ab_paired_scores.csv')
baseline_scores = rows('baseline_sample_scores.csv')
assert len(scores) == len(baseline_scores) == 21465
assert len(curves) == 480 and len(train) == 120000 and len(ab_train) == 80000
assert [row for row in train if row['arm'] in 'AB'] == ab_train
for filename, expected in s['retained_public_AB_sha256'].items():
    assert hashlib.sha256((root/filename).read_bytes()).hexdigest() == expected
for arm, t in s['training'].items():
    assert t['final_step'] == 40000 and t['tokens'] == 40000*262144
    assert t['parameters'] == audit['parameters'][arm]
    assert t['trunk_parameters'] == t['parameters']-77021184
    close(t['compute_trunk_6nd_pflop_days'], 6*t['trunk_parameters']*t['tokens']/8.64e19)
    close(t['compute_output_inclusive_6nd_pflop_days'], 6*t['parameters']*t['tokens']/8.64e19)
    extra = 0 if arm == 'baseline' else 6*5*2*768**2*t['tokens']/8.64e19
    close(t['extra_repeated_kv_projection_pflop_days'], extra)
    close(t['output_inclusive_plus_repeated_kv_proxy_pflop_days'], t['compute_output_inclusive_6nd_pflop_days']+extra)
    close(t['active_gpu_hours'], t['active_seconds']*t['world_size']/3600)
    cr = [row for row in curves if row['arm'] == arm]
    tr = [row for row in train if row['arm'] == arm]
    assert [int(row['step']) for row in cr] == list(range(250,40001,250))
    assert [int(row['step']) for row in tr] == list(range(1,40001))
    assert all(int(row['tokens']) == int(row['step'])*262144 for row in cr+tr)
    close(cr[-1]['validation_loss'], t['validation_loss'])
    close(cr[-1]['perplexity'], t['validation_perplexity'])
    close(statistics.mean(float(row['loss']) for row in tr[-100:]), t['last100_training_loss'])
    close(sum(float(row['seconds']) for row in tr), t['active_seconds'])
    close(statistics.mean(row[arm] for row in s['downstream']), s['macro'][arm])
    assert s['checkpoint_metadata'][arm]['step'] == 40000
for task in s['downstream']:
    name = task['task']
    rs = [row for row in scores if row['task'] == name]
    bs = [row for row in baseline_scores if row['task'] == name]
    assert len(rs) == len(bs) == task['examples']
    assert {row['doc_id'] for row in rs} == {row['doc_id'] for row in bs}
    assert len({row['doc_id'] for row in bs}) == len(bs)
    assert all(len(row[key]) == 64 for row in bs for key in ('doc_hash', 'prompt_hash', 'target_hash'))
    close(statistics.mean(float(row['baseline_correct']) for row in bs), task['baseline'])
    for arm in 'AB':
        close(statistics.mean(float(row[arm+'_correct']) for row in rs), task[arm])
        close(task[arm+'_delta_pp'], 100*(task[arm]-task['baseline']))
    close(task['B_minus_A_pp'], 100*(task['B']-task['A']))
    for key in ('task_yaml_sha256', 'seed', 'num_fewshot'):
        assert s['evaluation_protocol'][name][key] == s['baseline_evaluation_protocol'][name][key]
close(s['AB_paired_bootstrap']['B_minus_A_pp'], 100*(s['macro']['B']-s['macro']['A']))
thresholds = rows('loss_thresholds.csv')
for row in thresholds:
    target = float(row['target_loss'])
    cr = [value for value in curves if value['arm'] == row['arm']]
    hit = next((value for value in cr if float(value['validation_loss']) <= target), None)
    assert row['first_observed_step'] == (hit['step'] if hit else '')
    sustained = next((value for i, value in enumerate(cr) if all(float(q['validation_loss']) <= target for q in cr[i:])), None)
    assert row['sustained_observed_step'] == (sustained['step'] if sustained else '')
for row in rows('figures/compute_at_loss.csv'):
    if not row['baseline_first_observed_step']:
        assert row['status'] == 'baseline_not_reached' and not row['relative_change_percent'] and not row['baseline_proxy_pflop_days']
    else:
        assert row['status'] == 'compared'
        close(row['relative_change_percent'], 100*(float(row['corrected_proxy_pflop_days'])/float(row['baseline_proxy_pflop_days'])-1))
for name in ('01_quality_summary', '02_downstream_comparison', '03_validation_convergence', '04_compute_tradeoff'):
    for extension in ('png', 'pdf', 'svg'):
        assert (root/'figures'/f'{name}.{extension}').stat().st_size > 1000
for name in ('loss_compute', 'downstream', 'training_loss'):
    for extension in ('png', 'pdf', 'svg'):
        assert (root/f'{name}.{extension}').stat().st_size > 1000
if args.refresh_manifest:
    files = sorted(path for path in root.rglob('*') if path.is_file() and path.name != 'MANIFEST.sha256')
    (root/'MANIFEST.sha256').write_text(''.join(
        hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.relative_to(root).as_posix()+'\n' for path in files))
for line in (root/'MANIFEST.sha256').read_text().splitlines():
    expected, name = line.split('  ', 1)
    path = Path(name)
    assert not path.is_absolute() and '..' not in path.parts
    assert hashlib.sha256((root/path).read_bytes()).hexdigest() == expected, name
print('Verified: new 8-A100 baseline, common-backbone audit, 24 scores, 480 validation points, 120,000 training points, 21,465 baseline and A/B sample scores, thresholds, 21 figure files and manifest')
