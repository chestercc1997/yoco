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
ACTIVE=ROOT/'runs/setup/swa-160m-active.json'
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


def prepare(run):
    if ACTIVE.exists():
        assert Path(read(ACTIVE)['run_dir'])==run,'Inspect the existing SWA cohort before starting another'
    preflight=read(ROOT/'runs/setup/swa-160m-preflight.json')
    assert preflight['passed'] and preflight['common_weights_bitwise_equal']
    assert preflight['window_boundary_1024_passed'] and preflight['uniform_A_B_outputs_bitwise_equal']
    configs={arm:read(ROOT/f'configs/yoco_160m_swa_{arm}_8gpu.json') for arm in ARMS}
    reference=read(ROOT/'configs/yoco_160m_baseline_8gpu_matched.json')
    for arm,c in configs.items():
        assert c['model']==preflight['model'] and c['model']['sliding_window']==1024
        assert c['training']==reference['training'] and c['data']==reference['data']
        assert c['training']['precision']=='fp16_mixed_fp32_params'
        assert c['training']['keep_last_checkpoints'] is None
        for key in ('model','seed','tied_embeddings','training','data','execution'):
            assert c[key]==configs['baseline'][key],(arm,key)
        assert not c['execution']['compile']
    run.mkdir(parents=True,exist_ok=True);release=run/'release'
    record_path=run/'release-manifest.json'
    if not record_path.exists():
        assert not release.exists(),'Incomplete release; inspect before continuing'
        release.mkdir()
        for folder in ('runtime','evaluation'):
            shutil.copytree(ROOT/folder,release/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for folder in ('configs','tools','initializations'):(release/folder).mkdir()
        for name in ('prepare_downstream_data.py','check_400m_restore.py','preflight_swa_ab.py','run_swa_ab.py'):
            shutil.copyfile(ROOT/'tools'/name,release/'tools'/name)
        shutil.copyfile(ROOT/'requirements.a100-py312.lock.txt',release/'requirements.lock.txt')
        for arm,c in configs.items():
            atomic_json(release/f'configs/{arm}.json',c)
            source=Path(preflight['initializations'][arm]['path'])
            assert digest(source)==preflight['initializations'][arm]['sha256']
            shutil.copyfile(source,release/f'initializations/{arm}.pt')
        atomic_json(release/'preflight.json',preflight)
        files={str(p.relative_to(release)):digest(p) for p in sorted(release.rglob('*')) if p.is_file()}
        atomic_json(record_path,{'created_at':now(),'files':files,'parameters':preflight['parameters'],
            'data_manifest_sha256':digest(configs['baseline']['data']['manifest'])})
    record=read(record_path)
    for name,expected in record['files'].items():assert digest(release/name)==expected,name
    for arm in ARMS:assert read(release/f'configs/{arm}.json')==configs[arm],'Root config changed after freezing'
    assert digest(configs['baseline']['data']['manifest'])==record['data_manifest_sha256']
    return release


def check_resume(run,arm,command,release):
    import numpy as np
    import torch
    target=run/arm;target.mkdir(exist_ok=True)
    proof=target/'resume-verification.json'
    if proof.exists():
        assert read(proof)['passed'];return
    config=release/f'configs/{arm}.json'
    initial=release/f'initializations/{arm}.pt'
    train=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
           '-m','runtime.train','--config',str(config)]
    split=target/'checks/resumed';direct=target/'checks/uninterrupted';repeat=target/'checks/uninterrupted-repeat'
    def checkpoint(folder,step):return folder/f'checkpoint_{step:08d}.pt'
    for folder,step,label in ((split,2,'first'),(split,4,'resume'),(direct,4,'direct'),(repeat,4,'repeat')):
        if not checkpoint(folder,step).with_suffix('.json').exists():
            command(f'{arm}-check-{label}',train+['--run-dir',str(folder),'--init-model',str(initial),'--stop-after',str(step)])
        assert digest(checkpoint(folder,step))==read(checkpoint(folder,step).with_suffix('.json'))['sha256']
    audit=target/'checks/exact-source'
    if not (audit/'exact-restore.json').exists():
        command(f'{arm}-exact-restore',[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
            'tools/check_400m_restore.py','--release',str(release),'--config',str(config),
            '--checkpoint',str(checkpoint(split,2)),'--run-dir',str(audit)])
    exact=read(audit/'exact-restore.json')
    assert exact['passed'] and exact['world_size']==8 and exact['model_optimizer_scaler_rng_bitwise_equal']
    saved=[torch.load(checkpoint(folder,4),map_location='cpu',weights_only=False,mmap=True) for folder in (split,direct,repeat)]
    torch.set_num_threads(4)
    def compare(x,y,path):
        if isinstance(x,torch.Tensor):
            assert x.dtype==y.dtype and x.shape==y.shape,path
            if not x.is_floating_point() or path.endswith('/step'):assert torch.equal(x,y),path
            else:
                assert torch.isfinite(x).all() and torch.isfinite(y).all(),path
                assert torch.allclose(x,y,rtol=1e-3,atol=2e-5 if path.startswith('model') else 1e-6),path
        elif isinstance(x,np.ndarray):assert np.array_equal(x,y),path
        elif isinstance(x,dict):
            assert x.keys()==y.keys(),path
            for key in x:compare(x[key],y[key],path+'/'+str(key))
        elif isinstance(x,(tuple,list)):
            assert len(x)==len(y),path
            for i,(a,b) in enumerate(zip(x,y)):compare(a,b,path+'/'+str(i))
        else:assert x==y,path
    def difference(x,y):
        delta=norm=maximum=0.
        for key in x:
            d=x[key]-y[key];maximum=max(maximum,float(d.abs().max()))
            delta+=float(d.double().square().sum());norm+=float(y[key].double().square().sum())
        return {'max_absolute_difference':maximum,'relative_l2':math.sqrt(delta/norm)}
    differences={}
    for label,state in (('resumed',saved[0]),('independent_repeat',saved[2])):
        assert state['step']==saved[1]['step']==4 and state['world_size']==saved[1]['world_size']==8
        for key in ('model','optimizer','scaler','rngs'):compare(state[key],saved[1][key],key)
        differences[label]=difference(state['model'],saved[1]['model'])
        assert differences[label]['max_absolute_difference']<2e-5
        assert differences[label]['relative_l2']<2e-6
    def losses(folder):return {e['step']:e['loss'] for e in map(json.loads,(folder/'rank0.jsonl').read_text().splitlines()) if e['event']=='train'}
    direct_losses=losses(direct)
    loss_difference=max(abs(value-direct_losses[step]) for folder in (split,repeat) for step,value in losses(folder).items())
    assert loss_difference<5e-5
    for rank in range(8):
        events=[json.loads(line) for line in (split/f'rank{rank}.jsonl').read_text().splitlines()]
        assert [e['step'] for e in events if e['event']=='started']==[0,2]
        assert [e['step'] for e in events if e['event']=='train']==[1,2,3,4]
        assert not any(e['event'] in ('overflow','nonfinite_forward') for e in events)
    atomic_json(proof,{'passed':True,'world_size':8,'real_data':True,'split_resume_step':2,'final_step':4,
        'exact_pre_forward_restore':exact,'model_differences':differences,'max_loss_difference':loss_difference,
        'tolerances':{'model_absolute':2e-5,'model_relative_l2':2e-6,'optimizer_atol':1e-6,
                      'loss_absolute':5e-5,'RNG_scaler_steps_groups':'exact'},'completed_at':now()})


def training_review(run,arm):
    target=run/arm
    events=[json.loads(line) for folder in (target/'checks/resumed',target)
            for line in (folder/'rank0.jsonl').read_text().splitlines()]
    # A restarted controller may replay a suffix after the latest durable checkpoint.
    # The last occurrence is the committed training lineage, not duplicate work.
    updates={e['step']:e for e in events if e['event']=='train'}
    validations={e['step']:e for e in events if e['event']=='validation' and e['step']%250==0}
    assert sorted(updates)==list(range(1,40001)) and sorted(validations)==list(range(250,40001,250))
    assert all(math.isfinite(e['loss']) and math.isfinite(e['grad_norm']) for e in updates.values())
    best=float('inf');regression=0.
    for step in sorted(validations):
        value=validations[step]['loss'];assert math.isfinite(value)
        best=min(best,value);regression=max(regression,value-best)
    checkpoints=[]
    for step in range(5000,40001,5000):
        path=target/f'checkpoint_{step:08d}.pt';meta=read(path.with_suffix('.json'))
        assert path.is_file() and meta['step']==step
        checkpoints.append(meta)
    review={'final_validation_loss':validations[40000]['loss'],'best_validation_loss':best,
        'maximum_validation_regression':regression,'overflow_retries':sum(e['event']=='overflow' for e in events),
        'nonfinite_forward_events':sum(e['event']=='nonfinite_forward' for e in events),
        'maximum_gradient_norm':max(e['grad_norm'] for e in updates.values()),
        'convergence_warning':validations[40000]['loss']-best>.1,'checkpoints':checkpoints}
    atomic_json(target/'training-review.json',review)
    return review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();run=args.run_dir.resolve()
    ACTIVE.parent.mkdir(parents=True,exist_ok=True)
    lock=(ACTIVE.parent/'swa-160m-8gpu.lock').open('a')
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
        'tmux_session':'yoco-swa-160m-8gpu','arms':list(ARMS),'max_updates_per_arm':40000,
        'world_size':8,'checkpoint_every':5000,'keep_all_checkpoints':True,'recorded_at':now()})
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
        for arm in ARMS:
            check_resume(run,arm,command,release)
            update(f'{arm}-eight-gpu-check','passed',proof=str(run/arm/'resume-verification.json'))
            smoke=run/arm/'eval-smoke'
            if not (smoke/'results.json').exists():
                command(f'{arm}-eval-smoke',[sys.executable,'evaluation/run.py','--config',str(release/f'configs/{arm}.json'),
                    '--checkpoint',str(run/arm/'checks/resumed/checkpoint_00000004.pt'),
                    '--locks-dir',str(release/'evaluation/plan_ab_data_locks'),'--task','arc_easy',
                    '--limit','4','--device','cuda:0','--output',str(smoke)])
            assert read(smoke/'results.json')['smoke_only'] and read(smoke/'results.json')['model_step']==4
        summaries={}
        for arm in ARMS:
            target=run/arm;config=release/f'configs/{arm}.json'
            if (target/'PIPELINE_SUCCESS').exists():
                summary=read(target/'summary.json');assert summary['complete'] and summary['step']==40000
                summaries[arm]=summary;continue
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
                assert provenance['checkpoint_sha256']==meta['sha256']
                assert provenance['dataset_lock']==read(locks/f'data_lock_{task}.json')
                count=offline['tasks'][task]['evaluation_documents']
                assert result['n-samples'][task]['effective']==count
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
        atomic_json(run/'summary.json',{'complete':True,'architecture':'YOCO-SWA','arms':summaries,'completed_at':now()})
        status['state']='complete';update('pipeline','passed',summary=str(run/'summary.json'));(run/'PIPELINE_SUCCESS').touch()
    except BaseException as error:
        status['state']='failed';update('pipeline','failed',error=str(error));raise


if __name__=='__main__':main()
