"""Replace the public historical baseline with a verified local eight-GPU run.

A/B public evidence is retained unchanged. No task text or raw logs are exported.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PF_DAY = 8.64e19


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def csvwrite(path, data):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)


def audit_backbone(configs, resolved):
    import gc
    import torch
    from runtime.model import build_model
    torch.set_num_threads(4)
    base = resolved['config']
    keys = ('architecture', 'model', 'training', 'seed', 'tied_embeddings')
    assert all(base[key] == configs[arm][key] for arm in 'AB' for key in keys)
    assert 'memory_fusion' not in base
    assert configs['A']['memory_fusion']['mode'] == 'learned_kv'
    assert configs['B']['memory_fusion']['mode'] == 'conditional_kv'
    source_match = all(sha(ROOT/'runtime'/name) == value
                       for name, value in resolved['identity']['model_source_sha256'].items())
    assert source_match
    torch.manual_seed(base['seed'])
    model = build_model(base).eval()
    common = {key: value.detach().clone() for key, value in model.state_dict().items()}
    params = {'baseline': sum(p.numel() for p in model.parameters())}
    counts = {}
    outputs = []
    ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
    with torch.no_grad():
        expected_baseline = model(ids)[0]
    del model
    gc.collect()
    for arm, extra in (('A', 6), ('B', 774)):
        torch.manual_seed(base['seed'])
        model = build_model(configs[arm]).eval()
        state = model.state_dict()
        assert set(common) <= set(state)
        assert all(torch.equal(value, state[key]) for key, value in common.items()), arm
        assert all(key.startswith('memory_fusion.') for key in set(state)-set(common))
        params[arm] = sum(p.numel() for p in model.parameters())
        assert params[arm] == params['baseline'] + extra
        counts[arm] = len(common)
        with torch.no_grad():
            weights = model.memory_fusion.conditional_weights([torch.ones(1, 8, 768)]*6)
            torch.testing.assert_close(weights, torch.full_like(weights, 1/6))
            outputs.append(model(ids)[0].clone())
            bias = model.memory_fusion.logits if arm == 'A' else model.memory_fusion.layer_bias
            bias.fill_(-1000)
            bias[-1] = 0
            torch.testing.assert_close(model(ids)[0], expected_baseline, rtol=2e-5, atol=2e-5)
        del model, state
        gc.collect()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    return dict(passed=True, configuration_fields_matched=list(keys), parameters=params,
                extra_parameters={'A': 6, 'B': 774}, matching_common_state_tensors=counts,
                baseline_runtime_matches_repository=source_match, seed=base['seed'],
                uniform_initial_weights=[1/6]*6, ab_initial_outputs_identical=True,
                last_source_limit_matches_baseline=True,
                scope='Full-size models reconstructed locally from published run configs; all common tensors match exactly. Remote A/B step-zero payloads were not supplied.',
                checks_not_performed=['Remote A/B step-zero payload equality', 'Remote A/B token-shard SHA256 equality'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--baseline-pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation/ab_baseline_20261104')
    args = parser.parse_args()
    run, pilot, out = args.baseline_run, args.baseline_pilot, args.output
    summary = read(out/'summary.json')
    configs = read(out/'training_protocol.json')
    before_ab = {name: sha(out/name) for name in ('ab_training_curves.csv', 'ab_paired_scores.csv')}
    preserved = json.dumps({key: {a: summary[key][a] for a in 'AB'} for key in ('training', 'macro')}, sort_keys=True)
    assert (run/'PIPELINE_SUCCESS').exists()
    assert read(run/'stage_complete.json') == dict(step=40000, formal_complete=True, synthetic=False)
    resolved = read(run/'resolved_config.json')
    meta = read(run/'checkpoint_00040000.json')
    fingerprint = hashlib.sha256(json.dumps(resolved['identity'], sort_keys=True).encode()).hexdigest()
    assert meta['step'] == 40000 and meta['fingerprint'] == fingerprint
    assert sha(run/meta['file']) == meta['sha256']
    audit = audit_backbone(configs, resolved)
    print('Full-size common-backbone, initialization and last-source checks passed', flush=True)
    logpaths = [('baseline_pilot_rank0', pilot/'rank0.jsonl'), ('baseline_full_rank0', run/'rank0.jsonl')]
    logs = [[json.loads(line) for line in path.read_text().splitlines()] for _, path in logpaths]
    train = [row for phase in logs for row in phase if row['event'] == 'train']
    assert [row['step'] for row in train] == list(range(1, 40001))
    assert all(row['tokens'] == row['step']*262144 for row in train)
    all_valid = [row for phase in logs for row in phase if row['event'] == 'validation']
    valid = [row for row in all_valid if row['step'] % 250 == 0]
    assert [row['step'] for row in valid] == list(range(250, 40001, 250))
    start = next(row for row in logs[0] if row['event'] == 'started')
    assert start['parameters'] == audit['parameters']['baseline'] and start['world_size'] == 8
    active = sum(row['seconds'] for row in train)
    endhours = (datetime.fromisoformat(valid[-1]['time'])-datetime.fromisoformat(start['time'])).total_seconds()/3600
    t = dict(validation_loss=valid[-1]['loss'], validation_perplexity=valid[-1]['perplexity'],
             last100_training_loss=statistics.mean(row['loss'] for row in train[-100:]),
             tokens=train[-1]['tokens'], parameters=start['parameters'], final_step=40000,
             world_size=8, hardware='8 x A100 80GB PCIe', active_seconds=active,
             active_gpu_hours=active*8/3600, end_to_end_hours=endhours,
             overflow_events=sum(row['event'] == 'overflow' for phase in logs for row in phase),
             peak_rank0_memory_gib=max(row['peak_memory_bytes'] for row in train)/2**30)
    emb = resolved['config']['model']['dim']*resolved['config']['model']['vocab_size']
    t.update(shared_embedding_output_parameters=emb, trunk_parameters=t['parameters']-emb)
    t['compute_trunk_6nd_pflop_days'] = 6*t['trunk_parameters']*t['tokens']/PF_DAY
    t['compute_output_inclusive_6nd_pflop_days'] = 6*t['parameters']*t['tokens']/PF_DAY
    t['extra_repeated_kv_projection_pflop_days'] = 0.0
    t['output_inclusive_plus_repeated_kv_proxy_pflop_days'] = t['compute_output_inclusive_6nd_pflop_days']
    t['mean_active_tokens_per_second'] = t['tokens']/active
    old_baseline = summary.get('baseline_replacement', {}).get('superseded_baseline',
        dict(hardware=summary['training']['baseline']['hardware'], validation_loss=summary['training']['baseline']['validation_loss'], macro_accuracy=summary['macro']['baseline']))
    summary['training']['baseline'] = t
    summary['checkpoint_metadata']['baseline'] = meta
    summary['baseline_replacement'] = dict(kind='verified_local_eight_gpu_training', source_run=run.name,
        source_pilot=pilot.name, resumed_at_step=100, original_schedule_preserved=True,
        superseded_baseline=old_baseline, timing_note='Baseline wall time includes pilot/resume pauses; successful-update active time excludes those pauses and overflow retries.')
    summary['baseline_evaluation_protocol'] = {}
    sample_rows = []
    score_ids = {}
    for row in rows(out/'ab_paired_scores.csv'):
        score_ids.setdefault(row['task'], set()).add(int(row['doc_id']))
    hashes = summary['source_sha256']
    # Retire the four historical-baseline source bindings, retaining every A/B binding.
    for key in ('baseline_training_summary', 'baseline_downstream', 'baseline_validation_curve', 'baseline_timing'):
        hashes.pop(key, None)
    for label, path in logpaths + [('baseline_resolved_config', run/'resolved_config.json'),
        ('baseline_stage_complete', run/'stage_complete.json'), ('baseline_checkpoint_metadata', run/'checkpoint_00040000.json'),
        ('baseline_pipeline_summary', run/'summary.json')]:
        hashes[label] = sha(path)
    for row in summary['downstream']:
        task, metric = row['task'], row['metric']
        folder = run/'evaluation'/task
        result, provenance = read(folder/'results.json'), read(folder/'provenance.json')
        assert result['model_step'] == 40000 and not result['smoke_only'] and provenance['limit'] is None
        assert provenance['checkpoint_sha256'] == meta['sha256']
        samples = result['samples'][task]
        assert len(samples) == row['examples'] == result['n-samples'][task]['effective']
        assert {sample['doc_id'] for sample in samples} == score_ids[task]
        for field in ('task_yaml_sha256', 'seed', 'num_fewshot'):
            assert provenance[field] == summary['evaluation_protocol'][task][field], (task, field)
        key = metric.split(',')[0]
        score = statistics.mean(sample[key] for sample in samples)
        assert abs(score-result['results'][task][metric]) < 1e-12
        row['baseline'] = score
        row['baseline_stderr'] = result['results'][task][metric.replace(',', '_stderr,')]
        for arm in 'AB':
            row[arm+'_delta_pp'] = 100*(row[arm]-score)
        for sample in samples:
            sample_rows.append(dict(task=task, doc_id=sample['doc_id'], baseline_correct=sample[key],
                **{k: sample[k] for k in ('doc_hash', 'prompt_hash', 'target_hash')}))
        summary['baseline_evaluation_protocol'][task] = {k: provenance[k] for k in ('dataset_lock', 'task_yaml_sha256', 'seed', 'num_fewshot')}
        for name in ('results.json', 'provenance.json'):
            hashes['baseline_'+task+'_'+name] = sha(folder/name)
    summary['macro']['baseline'] = statistics.mean(row['baseline'] for row in summary['downstream'])
    audit.update(training_and_model_configs_matched=True, evaluation_task_yaml_seed_fewshot_matched_all_8=True,
                 full_baseline_checkpoint_payload_verified=True, evaluation_counts_and_document_ids_matched_all_8=True,
                 ab_dataset_document_prompt_target_hashes_available_in_public_export=False,
                 baseline_validation_points=160, baseline_training_updates=40000,
                 note='Document-id sets match A/B public scores. Remote A/B sample hashes and token manifests are not present in this checkout; no baseline paired significance is claimed.')
    summary['comparability_audit'] = audit
    summary['retained_public_AB_sha256'] = before_ab
    summary['schema_version'] = 2
    summary['boundary'] = ('Baseline is the completed 8-A100 rerun with checkpoint payload rehashed and all 8 full evaluations verified. '
        'A/B are the published 8-A800 runs. Full-size reconstruction confirms the same backbone and initial common tensors; only the fusion modules add parameters. '
        'Model/training configs and all task YAMLs, seeds and few-shot counts match. Remote A/B step-zero payloads, sample hashes and token-shard hashes are not supplied. '
        'A/B public evidence and paired bootstrap are unchanged; baseline differences are descriptive. 6ND is a compute proxy, not measured device FLOPs.')
    curves = [row for row in rows(out/'validation_curves.csv') if row['arm'] != 'baseline']
    basecurves = []
    for row in valid:
        count = row['step']*262144
        basecurves.append(dict(arm='baseline', step=row['step'], tokens=count,
            validation_loss=row['loss'], perplexity=row['perplexity'],
            compute_trunk_6nd_pflop_days=6*t['trunk_parameters']*count/PF_DAY,
            compute_output_inclusive_6nd_pflop_days=6*t['parameters']*count/PF_DAY,
            output_inclusive_plus_repeated_kv_proxy_pflop_days=6*t['parameters']*count/PF_DAY))
    curves = basecurves + curves
    ab_train = rows(out/'ab_training_curves.csv')
    all_train = [dict(arm='baseline', **{k: row[k] for k in ('step', 'tokens', 'loss', 'lr', 'seconds')}) for row in train] + ab_train
    thresholds = []
    for target in (4.0, 3.5, 3.3, 3.2, 3.17, t['validation_loss']):
        for arm in ('baseline', 'A', 'B'):
            cr = [row for row in curves if row['arm'] == arm]
            hits = [row for row in cr if float(row['validation_loss']) <= target]
            first = hits[0] if hits else None
            sustained = next((row for i, row in enumerate(cr) if all(float(q['validation_loss']) <= target for q in cr[i:])), None)
            thresholds.append(dict(target_loss=target, arm=arm,
                first_observed_step=first['step'] if first else None,
                sustained_observed_step=sustained['step'] if sustained else None,
                tokens_at_first=first['tokens'] if first else None,
                trunk_6nd_at_first=first['compute_trunk_6nd_pflop_days'] if first else None,
                output_inclusive_6nd_at_first=first['compute_output_inclusive_6nd_pflop_days'] if first else None))
    configs['baseline'] = {k: resolved['config'][k] for k in ('architecture', 'model', 'tied_embeddings', 'seed', 'training')}
    assert preserved == json.dumps({key: {a: summary[key][a] for a in 'AB'} for key in ('training', 'macro')}, sort_keys=True)
    assert all(sha(out/name) == expected for name, expected in before_ab.items())
    write(out/'summary.json', summary)
    write(out/'training_protocol.json', configs)
    write(out/'comparability_audit.json', audit)
    csvwrite(out/'training_summary.csv', [dict(arm=arm, **summary['training'][arm]) for arm in ('baseline', 'A', 'B')])
    csvwrite(out/'downstream.csv', summary['downstream'])
    csvwrite(out/'validation_curves.csv', curves)
    csvwrite(out/'training_curves.csv', all_train)
    csvwrite(out/'baseline_sample_scores.csv', sample_rows)
    csvwrite(out/'loss_thresholds.csv', thresholds)
    print(json.dumps({'validation_loss': {arm: summary['training'][arm]['validation_loss'] for arm in ('baseline', 'A', 'B')},
                      'macro': summary['macro'], 'AB_evidence_unchanged': True}, indent=2), flush=True)


if __name__ == '__main__':
    main()
