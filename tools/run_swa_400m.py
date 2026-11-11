"""Run one immutable, resumable SWA baseline/A/B cohort on eight local GPUs."""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
ACTIVE=ROOT/'runs/setup/swa-400m-active.json'
ARMS=('baseline','A','B')
METRICS={'arc_easy':'acc_norm,none','arc_challenge':'acc_norm,none','boolq':'acc,none',
         'hellaswag':'acc_norm,none','openbookqa':'acc_norm,none','piqa':'acc_norm,none',
         'winogrande':'acc,none','sciq':'acc_norm,none'}


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):return json.loads(Path(path).read_text())


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


from tools.run_swa_ab import check_resume, training_review

def prepare(run):
    if ACTIVE.exists():assert Path(read(ACTIVE)['run_dir']).resolve()==run,'Inspect the active SWA400M cohort before starting another'
    preflight=read(ROOT/'runs/setup/swa-400m-preflight.json')
    assert preflight['passed'] and preflight['common_weights_bitwise_equal']
    assert preflight['window_boundary_1024_passed'] and preflight['uniform_A_B_outputs_bitwise_equal']
    configs={a:read(ROOT/f'configs/yoco_400m_swa_{a}_8gpu.json') for a in ARMS}
    reference=read(ROOT/'configs/yoco_160m_swa_baseline_8gpu.json')
    gret=read(ROOT/'configs/yoco_400m_gret_baseline_8gpu.json')
    for arm,c in configs.items():
        assert c['model']==preflight['model'] and c['model']['sliding_window']==1024
        assert c['model']['dim']==1024 and c['model']['n_layers']==24 and c['model']['hidden_dim']==2816
        assert c['model']['n_self_heads']==c['model']['n_attn_heads']==c['model']['n_attn_kv_heads']==16
        for key in ('training','data','seed','tied_embeddings'):
            assert c[key]==reference[key]==gret[key],key
        assert c['execution']==reference['execution'] and not c['execution']['compile']
        assert c['training']['precision']=='fp16_mixed_fp32_params' and c['training']['keep_last_checkpoints'] is None
        for key in ('architecture','model','seed','tied_embeddings','training','data','execution'):assert c[key]==configs['baseline'][key],(arm,key)
        if arm=='baseline':assert 'memory_fusion' not in c
        else:assert c['memory_fusion']==dict(mode='learned_kv' if arm=='A' else 'conditional_kv',source_layers=list(range(1,13)),temperature=1.)
    run.mkdir(parents=True,exist_ok=True);release=run/'release';record_path=run/'release-manifest.json'
    if not record_path.exists():
        assert not release.exists(),'Incomplete release; inspect before continuing'
        release.mkdir()
        shutil.copytree(ROOT/'runtime',release/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (release/'evaluation').mkdir()
        for path in (ROOT/'evaluation').glob('*.py'):shutil.copyfile(path,release/'evaluation'/path.name)
        for folder in ('plan_ab_data_locks','data_locks'):shutil.copytree(ROOT/'evaluation'/folder,release/'evaluation'/folder)
        for folder in ('configs','tools','initializations'):(release/folder).mkdir()
        for name in ('run_swa_400m.py','watch_swa_400m.py','preflight_swa_400m.py','export_swa_400m.py',
                     'run_swa_ab.py','watch_swa_training.py','prepare_downstream_data.py','check_400m_restore.py','diagnose_training_failure.py'):
            shutil.copyfile(ROOT/'tools'/name,release/'tools'/name)
        shutil.copyfile(ROOT/'requirements.a100-py312.lock.txt',release/'requirements.lock.txt')
        for arm,c in configs.items():
            atomic_json(release/f'configs/{arm}.json',c)
            source=Path(preflight['initializations'][arm]['path']);assert digest(source)==preflight['initializations'][arm]['sha256']
            shutil.copyfile(source,release/f'initializations/{arm}.pt')
        atomic_json(release/'configs/SWA160M-reference.json',reference)
        atomic_json(release/'configs/gRet400M-reference.json',gret)
        atomic_json(release/'preflight.json',preflight)
        files={str(p.relative_to(release)):digest(p) for p in sorted(release.rglob('*')) if p.is_file()}
        atomic_json(record_path,dict(created_at=now(),architecture='YOCO-SWA',scale='400M',files=files,
            parameters=preflight['parameters'],data_manifest_sha256=digest(configs['baseline']['data']['manifest']),
            evaluation_packaging='evaluation Python modules and both original lock directories; historical plot/data exports omitted'))
    record=read(record_path)
    for name,expected in record['files'].items():assert digest(release/name)==expected,name
    for arm in ARMS:assert read(release/f'configs/{arm}.json')==configs[arm],'Root config changed after freezing'
    assert digest(configs['baseline']['data']['manifest'])==record['data_manifest_sha256']
    return release


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();run=args.run_dir.resolve()
    ACTIVE.parent.mkdir(parents=True,exist_ok=True)
    lock=(ACTIVE.parent/'swa-400m-8gpu.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('The SWA cohort controller is already running')
    release=prepare(run)
    if args.prepare_only:print(f'Prepared and verified immutable SWA cohort: {run}',flush=True);return
    if (run/'PIPELINE_SUCCESS').exists():print('This SWA cohort is already complete',flush=True);return
    os.environ.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        NCCL_SOCKET_IFNAME='lo',GLOO_SOCKET_IFNAME='lo',TORCHINDUCTOR_COMPILE_THREADS='2',
        TORCHINDUCTOR_CACHE_DIR=str(ROOT/'data/cache/torchinductor'),HF_HOME=str(ROOT/'data/cache/huggingface'),
        TIKTOKEN_CACHE_DIR=str(ROOT/'data/cache/tiktoken'),HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',MPLCONFIGDIR=str(ROOT/'data/cache/matplotlib'),
        YOCO_ENV_PREFIX=str(Path(sys.executable).parent.parent))
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    status={'state':'starting','run_dir':str(run),'pid':os.getpid(),'started_at':now(),'stages':{},
            'architecture':'YOCO-SWA','world_size':8,'arms':list(ARMS),'target_steps_per_arm':40000}
    atomic_json(ACTIVE,{'run_dir':str(run),'release':str(release),'status':str(run/'pipeline-status.json'),
        'tmux_session':'yoco-swa-400m-8gpu','arms':list(ARMS),'max_updates_per_arm':40000,
        'world_size':8,'checkpoint_every':5000,'keep_all_checkpoints':True,'recorded_at':now(),'architecture':'YOCO-SWA','scale':'400M','report_output':str(ROOT/'evaluation'/('swa_400m_ab_'+run.name.rsplit('-',1)[-1]))})
    guard=threading.Lock()
    def update(stage,state,**fields):
        with guard:
            if state=='running':status['state']=stage
            status['updated_at']=now();status['stages'][stage]={'state':state,**fields}
            atomic_json(run/'pipeline-status.json',status)
            print(json.dumps({'time':now(),'stage':stage,'state':state,**fields}),flush=True)
    def command(stage,argv,env=None):
        path=run/f'{stage}.log';update(stage,'running',log=str(path))
        with path.open('a') as output:result=subprocess.run(argv,cwd=release,env=env,stdout=output,stderr=subprocess.STDOUT)
        if result.returncode:
            update(stage,'failed',returncode=result.returncode)
            raise RuntimeError(f'{stage} failed; inspect {path}')
        update(stage,'passed',log=str(path))
    try:
        usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        assert len(usage.splitlines())==8 and all(int(x.strip())<1000 for x in usage.splitlines()),'GPUs are busy'
        import torch
        assert torch.cuda.is_available() and torch.cuda.device_count()==8
        status['devices']=[torch.cuda.get_device_name(i) for i in range(8)]
        if not (run/'downstream-offline.json').exists():
            command('offline-data',[sys.executable,'tools/prepare_downstream_data.py','--offline',
                                   '--output',str(run/'downstream-offline.json')])
        offline=read(run/'downstream-offline.json');assert offline['ready'] and offline['offline']
        summaries={}
        for arm in ARMS:
            target=run/arm;config=release/f'configs/{arm}.json'
            if (target/'PIPELINE_SUCCESS').exists():
                summary=read(target/'summary.json');assert summary['complete'] and summary['step']==40000
                summaries[arm]=summary;continue
            check_resume(run,arm,command,release)
            update(f'{arm}-eight-gpu-check','passed',proof=str(run/arm/'resume-verification.json'))
            smoke=run/arm/'eval-smoke'
            if not (smoke/'results.json').exists():
                command(f'{arm}-eval-smoke',[sys.executable,'evaluation/run.py','--config',str(release/f'configs/{arm}.json'),
                    '--checkpoint',str(run/arm/'checks/resumed/checkpoint_00000004.pt'),
                    '--locks-dir',str(release/'evaluation/plan_ab_data_locks'),'--task','arc_easy',
                    '--limit','4','--device','cuda:0','--output',str(smoke)])
            assert read(smoke/'results.json')['smoke_only'] and read(smoke/'results.json')['model_step']==4
            if not (target/'stage_complete.json').exists() or not read(target/'stage_complete.json')['formal_complete']:
                command(f'{arm}-training',[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
                    '-m','runtime.train','--config',str(config),'--run-dir',str(target),
                    '--resume-from',str(target/'checks/resumed/checkpoint_00000004.pt'),'--stop-after','40000'])
            assert read(target/'stage_complete.json')=={'step':40000,'formal_complete':True,'synthetic':False}
            meta=read(target/'latest.json');checkpoint=target/meta['file']
            assert meta['step']==40000 and digest(checkpoint)==meta['sha256']
            review=training_review(run,arm)
            locks=release/'evaluation/plan_ab_data_locks'
            def evaluate(item):
                device,task=item;output=target/'evaluation'/task
                if not (output/'results.json').exists() or not (output/'provenance.json').exists():
                    command(f'{arm}-eval-{task}',[sys.executable,'evaluation/run.py','--config',str(config),
                        '--checkpoint',str(checkpoint),'--locks-dir',str(locks),'--task',task,
                        '--device','cuda:0','--output',str(output)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(device)))
                result=read(output/'results.json');provenance=read(output/'provenance.json')
                assert result['model_step']==40000 and not result['smoke_only'] and provenance['limit'] is None
                assert provenance['checkpoint_sha256']==meta['sha256'] and provenance['num_fewshot']==0
                assert provenance['dataset_lock']==read(locks/f'data_lock_{task}.json')
                count=offline['tasks'][task]['evaluation_documents']
                assert result['n-samples'][task]['effective']==result['n-samples'][task]['original']==count
                score=result['results'][task][METRICS[task]];assert math.isfinite(score) and 0<=score<=1
                return {'task':task,'metric':METRICS[task],'score':score,'documents':count}
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:scores=list(pool.map(evaluate,enumerate(METRICS)))
            summary={'complete':True,'arm':arm,'architecture':'YOCO-SWA','step':40000,'world_size':8,
                'parameters':read(release/'preflight.json')['parameters'][arm],
                'checkpoint':str(checkpoint),'sha256':meta['sha256'],'global_batch_tokens':262144,
                'total_training_tokens':40000*262144,'tasks':scores,'macro_accuracy':sum(x['score'] for x in scores)/8,
                'training_review':review,'completed_at':now()}
            atomic_json(target/'summary.json',summary);(target/'PIPELINE_SUCCESS').touch();summaries[arm]=summary
            update(f'{arm}-pipeline','passed',summary=str(target/'summary.json'))
        report_output=ROOT/'evaluation'/('swa_400m_ab_'+run.name.rsplit('-',1)[-1])
        if not (run/'summary.json').exists():
            atomic_json(run/'summary.json',dict(complete=True,architecture='YOCO-SWA',arms=summaries,report_output=str(report_output),rendered_review_and_publication_pending=True,completed_at=now()))
        else:
            previous=read(run/'summary.json');assert previous['arms']==summaries and previous['report_output']==str(report_output)
        if not (report_output/'summary.json').exists() or not read(report_output/'summary.json')['complete']:
            command('export-reports',[sys.executable,'tools/export_swa_400m.py','--run-dir',str(run),'--output',str(report_output)])
        status['state']='complete';update('pipeline','passed',summary=str(run/'summary.json'));(run/'PIPELINE_SUCCESS').touch()
    except BaseException as error:
        status['state']='failed';update('pipeline','failed',error=str(error));raise


if __name__=='__main__':main()
