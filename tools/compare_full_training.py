"""Compare complete paired runs, keeping the pilot and continuation provenance."""
import argparse,json,math,statistics,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(archive,run_name,leaf):
    with tarfile.open(archive) as tar:
        return tar.extractfile('runs/'+run_name+'/'+leaf).read().decode()


def load(archive,arm):
    tag=('baseline' if arm=='baseline' else 'fusion-token')+'-2gpu'
    full_name='yoco-160m-'+tag+'-full-20261101-v2'
    pilot_name='yoco-160m-'+tag+'-20261101-v2'
    pilot=ROOT/'runs/setup'/(tag+'-20261101-v2-results.tar.gz')
    c=json.loads(read(archive,full_name,'resolved_config.json'))
    pc=json.loads(read(pilot,pilot_name,'resolved_config.json'))
    assert c['identity']==pc['identity'],'Continuation identity differs from pilot'
    done=json.loads(read(archive,full_name,'full_complete.json'))
    assert done['step']==40000 and done['full_training']
    stage=json.loads(read(archive,full_name,'stage_complete.json'))
    assert stage['formal_complete'] and stage['step']==40000
    origin=json.loads(read(archive,full_name,'resume_origin.json'))
    assert origin['step']==1000
    rows=[json.loads(line) for line in (read(pilot,pilot_name,'rank0.jsonl')+read(archive,full_name,'rank0.jsonl')).splitlines() if line.strip()]
    train=[r for r in rows if r['event']=='train']
    assert [r['step'] for r in train]==list(range(1,40001)), 'Missing, duplicate, or out-of-order training updates'
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in train)
    val={r['step']:r['loss'] for r in rows if r['event']=='validation'}
    assert sorted(val)==list(range(250,40001,250)) and all(math.isfinite(x) for x in val.values())
    return dict(config=c,train=train,validation=val,overflows=sum(r['event']=='overflow' for r in rows),
                gradient=json.loads(read(archive,full_name,'gradient_flow_probe.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--dynamic',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();b=load(a.baseline,'baseline');d=load(a.dynamic,'dynamic')
    bc=b['config'];dc=d['config']
    for key in ('model','training','seed','tied_embeddings','architecture'):
        assert bc['config'][key]==dc['config'][key],key
    for key in ('data','model_source_sha256'):
        assert bc['identity'][key]==dc['identity'][key],key
    points=[]
    for step in sorted(b['validation']):
        points.append(dict(step=step,tokens=step*262144,baseline=b['validation'][step],dynamic=d['validation'][step],
                           delta=d['validation'][step]-b['validation'][step]))
    targets={}
    for target in (6.5,6.0,5.5,4.5,4.0,3.5,3.0):
        pair={}
        for label,r in [('baseline',b),('dynamic',d)]:
            candidates=[s for s,v in sorted(r['validation'].items()) if v<=target]
            step=candidates[0] if candidates else None
            pair[label]=dict(first_observed_step=step,step_bracket=[max(0,step-250),step] if step else None,
                             tokens=step*262144 if step else None,
                             training_gpu_seconds=sum(x['seconds']*2 for x in r['train'] if step and x['step']<=step) if step else None)
        targets[str(target)]=pair
    delta=points[-1]['delta']
    verdict='LOWER_FINAL_LOSS_SINGLE_SEED' if delta<0 else 'NO_LOWER_FINAL_LOSS_SINGLE_SEED'
    result=dict(verdict=verdict,total_steps=40000,total_tokens=40000*262144,final_delta_dynamic_minus_baseline=delta,
                validation=points,target_loss_observations=targets,
                summary={label:dict(final_validation=r['validation'][40000],overflows=r['overflows'],
                                    last1000_mean_training_loss=statistics.mean(x['loss'] for x in r['train'][-1000:]),
                                    training_gpu_seconds=sum(x['seconds']*2 for x in r['train']),
                                    gradient_clip_fraction=sum(x['grad_norm']>1 for x in r['train'])/len(r['train']),
                                    gradient_flow_probe=r['gradient']) for label,r in [('baseline',b),('dynamic',d)]},
                limitations=['Single seed: no statistical significance or causal gradient-mechanism claim.',
                             'GPU-seconds are step timers only, exclude validation/checkpoints/diagnostics; not FLOPs.',
                             'Target loss crossings are observed at 250-step checkpoints, not exact crossing times.'])
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 160M 两卡完整训练对比','',f'结论：`{verdict}`。单 seed 40,000 步、每组 10,485,760,000 tokens。','',
           '| 步数 | baseline 验证 loss | dynamic 验证 loss | 差值 |','| ---: | ---: | ---: | ---: |']
    for r in points:
        if r['step'] in (1000,5000,10000,20000,30000,40000):lines.append(f"| {r['step']} | {r['baseline']:.6f} | {r['dynamic']:.6f} | {r['delta']:+.6f} |")
    lines+=['','完整160个验证点、同loss采样区间、训练GPU时间和梯度诊断见同名JSON。不能将梯度范数大小或单seed终点差异直接当作更快收敛的因果证明。']
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verdict=verdict,final_delta=delta,report=str(out))),flush=True)


if __name__=='__main__':main()
