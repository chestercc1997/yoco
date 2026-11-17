"""Retain the committed prefix and audit full-state continuation of the 830M repair."""
import json,math,sys
from pathlib import Path
from tools.run_swa_ab import atomic_json,digest,now,read

def checkpoint_path(run,arm,step):
    if step==5000:return Path(read(run/'recovery-source.json')['source_checkpoint'])
    return run/arm/f'checkpoint_{step:08d}.pt'

def lineage_events(run,arm,rank):
    result=[]
    for line in (run/'release/history'/f'rank{rank}.jsonl').read_text().splitlines():
        result.append(json.loads(line))
    for folder in (run/arm/'checks/resumed',run/arm):
        path=folder/f'rank{rank}.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:result.append(json.loads(line))
                except json.JSONDecodeError:continue
    return result

def check_resume(run,arm,command,release):
    import numpy as np
    import torch
    target=run/arm;target.mkdir(exist_ok=True)
    proof=target/'resume-verification.json'
    if proof.exists():
        assert read(proof)['passed'];return
    config=release/f'configs/{arm}.json'
    initial=Path(read(run/'recovery-source.json')['source_checkpoint'])
    repair=release/'configs/model-repair.json'
    train=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
           '-m','runtime.train','--config',str(config)]
    split=target/'checks/resumed';direct=target/'checks/uninterrupted';repeat=target/'checks/uninterrupted-repeat'
    def checkpoint(folder,step):return folder/f'checkpoint_{step:08d}.pt'
    for folder,step,label in ((split,5002,'first'),(split,5004,'resume'),(direct,5004,'direct'),(repeat,5004,'repeat')):
        if not checkpoint(folder,step).with_suffix('.json').exists():
            command(f'{arm}-check-{label}',train+['--run-dir',str(folder),'--resume-from',str(initial),'--model-repair',str(repair),'--stop-after',str(step)])
        assert digest(checkpoint(folder,step))==read(checkpoint(folder,step).with_suffix('.json'))['sha256']
    audit=target/'checks/exact-source'
    if not (audit/'exact-restore.json').exists():
        command(f'{arm}-exact-restore',[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
            'tools/check_400m_restore.py','--release',str(release),'--config',str(config),
            '--checkpoint',str(initial),'--model-repair',str(repair),'--run-dir',str(audit)])
    exact=read(audit/'exact-restore.json')
    assert exact['passed'] and exact['world_size']==8 and exact['model_optimizer_scaler_rng_bitwise_equal']
    saved=[torch.load(checkpoint(folder,5004),map_location='cpu',weights_only=False,mmap=True) for folder in (split,direct,repeat)]
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
        assert state['step']==saved[1]['step']==5004 and state['world_size']==saved[1]['world_size']==8
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
        assert [e['step'] for e in events if e['event']=='started']==[5000,5002]
        assert [e['step'] for e in events if e['event']=='train']==[5001,5002,5003,5004]
        assert not any(e['event'] in ('overflow','nonfinite_forward') for e in events)
    atomic_json(proof,{'passed':True,'world_size':8,'real_data':True,'source_step':5000,'split_resume_step':5002,'final_step':5004,
        'exact_pre_forward_restore':exact,'model_differences':differences,'max_loss_difference':loss_difference,
        'tolerances':{'model_absolute':2e-5,'model_relative_l2':2e-6,'optimizer_atol':1e-6,
                      'loss_absolute':5e-5,'RNG_scaler_steps_groups':'exact'},'completed_at':now()})


def training_review(run,arm):
    target=run/arm
    events=lineage_events(run,arm,0)
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
        path=checkpoint_path(run,arm,step);meta=read(path.with_suffix('.json'))
        assert path.is_file() and meta['step']==step
        checkpoints.append(meta)
    review={'final_validation_loss':validations[40000]['loss'],'best_validation_loss':best,
        'maximum_validation_regression':regression,'overflow_retries':sum(e['event']=='overflow' for e in events),
        'nonfinite_forward_events':sum(e['event']=='nonfinite_forward' for e in events),
        'maximum_gradient_norm':max(e['grad_norm'] for e in updates.values()),
        'convergence_warning':validations[40000]['loss']-best>.1,'checkpoints':checkpoints}
    atomic_json(target/'training-review.json',review)
    return review

