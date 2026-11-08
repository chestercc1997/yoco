"""Train only gRet 400M conditional-K/V B with the completed A recipe."""
import argparse
import concurrent.futures
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_swa_ab import atomic_json, check_resume, digest, now, read, training_review, METRICS

ACTIVE = ROOT / 'runs/setup/400m-B-active.json'
CONFIG = ROOT / 'configs/yoco_400m_plan_B_8gpu.json'
PREFLIGHT = ROOT / 'runs/setup/yoco-400m-B-preflight.json'
REFERENCE = ROOT / 'runs/yoco-400m-A-8gpu-retentionfix-20261106'
FROZEN_TOOLS = ('run_gret_400m_B.py', 'watch_gret_400m_B.py', 'preflight_400m_B.py',
                'export_gret_400m_B.py', 'run_swa_ab.py', 'watch_swa_training.py',
                'prepare_downstream_data.py', 'check_400m_restore.py', 'diagnose_training_failure.py')


def prepare(run):
    if ACTIVE.exists():
        assert Path(read(ACTIVE)['run_dir']).resolve() == run, 'Inspect the existing B run before launching another'
    cfg = read(CONFIG);a = read(REFERENCE / 'resolved_config.json')['config'];proof = read(PREFLIGHT)
    assert proof['passed'] and proof['architecture'] == 'YOCO-gRet' and proof['B_parameters'] == 400591884
    assert proof['common_weights_match_original_A_initialization'] and proof['common_weights_bitwise_equal']
    assert proof['old_A_initial_forward_max_abs'] < 2e-5 and proof['causality_prefix_max_abs'] < 2e-5
    assert proof['last_source_limit_max_abs'] < 2e-5 and proof['all_gradients_finite']
    for key in ('architecture', 'model', 'training', 'data', 'seed', 'tied_embeddings', 'execution'):
        assert cfg[key] == a[key], key
    assert cfg['model'].get('sliding_window') is None and cfg['model']['n_self_heads'] == 4
    assert cfg['memory_fusion'] == dict(mode='conditional_kv', source_layers=list(range(1, 13)), temperature=1.)
    assert cfg['training']['precision'] == 'fp16_mixed_fp32_params' and cfg['training']['keep_last_checkpoints'] is None
    assert not cfg['execution']['compile']
    for name, expected in proof['repaired_runtime_sha256'].items():assert digest(ROOT / name) == expected, name
    run.mkdir(parents=True, exist_ok=True);release = run / 'release';manifest_path = run / 'release-manifest.json'
    if not manifest_path.exists():
        assert not release.exists(), 'Incomplete release; inspect before continuing'
        release.mkdir()
        for folder in ('runtime', 'evaluation'):
            shutil.copytree(ROOT / folder, release / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for folder in ('configs', 'tools', 'initializations'):(release / folder).mkdir()
        for name in FROZEN_TOOLS:shutil.copyfile(ROOT / 'tools' / name, release / 'tools' / name)
        shutil.copyfile(ROOT / 'requirements.a100-py312.lock.txt', release / 'requirements.lock.txt')
        atomic_json(release / 'configs/B.json', cfg)
        atomic_json(release / 'configs/A-reference.json', a)
        baseline = {k: v for k, v in a.items() if k != 'memory_fusion'}
        atomic_json(release / 'configs/baseline-reference.json', baseline)
        for name, path, expected in (('B', proof['initialization'], proof['initialization_sha256']),
                                     ('A-reference', proof['source_A_initialization'], proof['source_A_initialization_sha256'])):
            assert digest(path) == expected
            shutil.copyfile(path, release / f'initializations/{name}.pt')
        atomic_json(release / 'preflight.json', proof)
        atomic_json(release / 'A-reference-summary.json', read(REFERENCE / 'summary.json'))
        files = {str(p.relative_to(release)): digest(p) for p in sorted(release.rglob('*')) if p.is_file()}
        atomic_json(manifest_path, dict(created_at=now(), files=files, architecture='YOCO-gRet',
            data_manifest_sha256=digest(cfg['data']['manifest']), source_A_run=str(REFERENCE),
            root_config=str(CONFIG), parameters={'baseline_reference':400590848, 'A_reference':400590860, 'B':400591884}))
    manifest = read(manifest_path)
    for name, expected in manifest['files'].items():assert digest(release / name) == expected, name
    assert read(release / 'configs/B.json') == cfg
    assert digest(cfg['data']['manifest']) == manifest['data_manifest_sha256']
    return release


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args();run = args.run_dir.resolve()
    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    lock = (ACTIVE.parent / '400m-B-8gpu.lock').open('a')
    try:fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('The gRet 400M B controller is already running')
    release = prepare(run)
    if args.prepare_only:print(f'Prepared immutable gRet 400M B: {run}', flush=True);return
    if (run / 'PIPELINE_SUCCESS').exists():print('This gRet 400M B pipeline is complete; no rerun', flush=True);return
    os.environ.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        NCCL_SOCKET_IFNAME='lo', GLOO_SOCKET_IFNAME='lo', TORCHINDUCTOR_COMPILE_THREADS='2',
        TORCHINDUCTOR_CACHE_DIR=str(ROOT / 'data/cache/torchinductor'), HF_HOME=str(ROOT / 'data/cache/huggingface'),
        TIKTOKEN_CACHE_DIR=str(ROOT / 'data/cache/tiktoken'), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', MPLCONFIGDIR=str(ROOT / 'data/cache/matplotlib'),
        YOCO_ENV_PREFIX=str(Path(sys.executable).parent.parent))
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
    output = ROOT / 'evaluation' / ('gret_400m_B_' + run.name.rsplit('-', 1)[-1])
    status = dict(state='starting', run_dir=str(run), pid=os.getpid(), started_at=now(), stages={},
                  architecture='YOCO-gRet', world_size=8, arms=['B'], target_steps_per_arm=40000)
    atomic_json(ACTIVE, dict(run_dir=str(run), release=str(release), status=str(run / 'pipeline-status.json'),
        tmux_session='yoco-gret-400m-B', architecture='YOCO-gRet', method='B', parameters=400591884,
        world_size=8, max_updates=40000, checkpoint_every=5000, keep_all_checkpoints=True,
        report_output=str(output), reference_A_run=str(REFERENCE), recorded_at=now()))
    mutex = threading.Lock()
    def update(stage, state, **fields):
        with mutex:
            if state == 'running':status['state'] = stage
            status['updated_at'] = now();status['stages'][stage] = dict(state=state, **fields)
            atomic_json(run / 'pipeline-status.json', status)
            print(json.dumps(dict(time=now(), stage=stage, state=state, **fields)), flush=True)
    def command(stage, argv, env=None):
        path = run / f'{stage}.log';update(stage, 'running', log=str(path))
        with path.open('a') as stream:
            result = subprocess.run(argv, cwd=release, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            update(stage, 'failed', returncode=result.returncode)
            raise RuntimeError(f'{stage} failed; inspect {path}')
        update(stage, 'passed', log=str(path))
    try:
        usage = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        assert len(usage.splitlines()) == 8 and all(int(x.strip()) < 1000 for x in usage.splitlines()), 'GPUs are busy'
        import torch
        assert torch.cuda.is_available() and torch.cuda.device_count() == 8
        status['devices'] = [torch.cuda.get_device_name(i) for i in range(8)]
        if not (run / 'downstream-offline.json').exists():
            command('offline-data', [sys.executable, 'tools/prepare_downstream_data.py', '--offline', '--output', str(run / 'downstream-offline.json')])
        offline = read(run / 'downstream-offline.json');assert offline['ready'] and offline['offline']
        check_resume(run, 'B', command, release)
        update('B-eight-gpu-check', 'passed', proof=str(run / 'B/resume-verification.json'))
        target = run / 'B';config = release / 'configs/B.json';locks = release / 'evaluation/plan_ab_data_locks'
        smoke = target / 'eval-smoke'
        if not (smoke / 'results.json').exists():
            command('B-eval-smoke', [sys.executable, 'evaluation/run.py', '--config', str(config),
                '--checkpoint', str(target / 'checks/resumed/checkpoint_00000004.pt'), '--locks-dir', str(locks),
                '--task', 'arc_easy', '--limit', '4', '--device', 'cuda:0', '--output', str(smoke)])
        assert read(smoke / 'results.json')['smoke_only'] and read(smoke / 'results.json')['model_step'] == 4
        if not (target / 'PIPELINE_SUCCESS').exists():
            if not (target / 'stage_complete.json').exists() or not read(target / 'stage_complete.json')['formal_complete']:
                command('B-training', [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                    '-m', 'runtime.train', '--config', str(config), '--run-dir', str(target),
                    '--resume-from', str(target / 'checks/resumed/checkpoint_00000004.pt'), '--stop-after', '40000'])
            assert read(target / 'stage_complete.json') == dict(step=40000, formal_complete=True, synthetic=False)
            meta = read(target / 'latest.json');checkpoint = target / meta['file']
            assert meta['step'] == 40000 and digest(checkpoint) == meta['sha256']
            review = training_review(run, 'B')
            def evaluate(item):
                device, task = item;folder = target / 'evaluation' / task
                if not (folder / 'results.json').exists() or not (folder / 'provenance.json').exists():
                    command('B-eval-' + task, [sys.executable, 'evaluation/run.py', '--config', str(config),
                        '--checkpoint', str(checkpoint), '--locks-dir', str(locks), '--task', task,
                        '--device', 'cuda:0', '--output', str(folder)], env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(device)))
                result = read(folder / 'results.json');provenance = read(folder / 'provenance.json')
                assert result['model_step'] == 40000 and not result['smoke_only'] and provenance['limit'] is None
                assert provenance['num_fewshot'] == 0 and provenance['checkpoint_sha256'] == meta['sha256']
                assert provenance['dataset_lock'] == read(locks / f'data_lock_{task}.json')
                count = offline['tasks'][task]['evaluation_documents']
                assert result['n-samples'][task]['effective'] == result['n-samples'][task]['original'] == count
                score = result['results'][task][METRICS[task]];assert math.isfinite(score) and 0 <= score <= 1
                return dict(task=task, metric=METRICS[task], score=score, documents=count)
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:scores = list(pool.map(evaluate, enumerate(METRICS)))
            summary = dict(complete=True, arm='B', method='B', architecture='YOCO-gRet', step=40000, world_size=8,
                parameters=400591884, checkpoint=str(checkpoint), sha256=meta['sha256'], global_batch_tokens=262144,
                total_training_tokens=10485760000, tasks=scores, macro_accuracy=sum(x['score'] for x in scores) / 8,
                training_review=review, source_A_reference=str(REFERENCE), completed_at=now())
            atomic_json(target / 'summary.json', summary);(target / 'PIPELINE_SUCCESS').touch()
            update('B-pipeline', 'passed', summary=str(target / 'summary.json'))
        summary = read(target / 'summary.json')
        if not (output / 'summary.json').exists() or not read(output / 'summary.json')['complete']:
            command('export-reports', [sys.executable, 'tools/export_gret_400m_B.py', '--run-dir', str(run), '--output', str(output)])
        atomic_json(run / 'summary.json', dict(complete=True, architecture='YOCO-gRet', method='B',
            experiment=summary, report_output=str(output), rendered_agent_review_and_publication_pending=True, completed_at=now()))
        status['state'] = 'complete';update('pipeline', 'passed', summary=str(run / 'summary.json'))
        (run / 'PIPELINE_SUCCESS').touch()
    except BaseException as error:
        status['state'] = 'failed';update('pipeline', 'failed', error=str(error));raise


if __name__ == '__main__':
    main()
