"""Evaluate short-run convergence evidence, without equating gradients with speed."""
import argparse
import json
import math
from pathlib import Path
import statistics
import tarfile


def read_run(archive,run_name):
    with tarfile.open(archive) as tar:
        prefix='runs/'+run_name+'/'
        def read(name):return tar.extractfile(prefix+name).read().decode()
        config=json.loads(read('resolved_config.json'))
        rows=[json.loads(x) for x in read('rank0.jsonl').splitlines()]
        assert json.loads(read('pilot_complete.json'))['step']==1000
        gradient=json.loads(read('gradient_flow_probe.json'))
        routing=[json.loads(tar.extractfile(m).read()) for m in tar.getmembers() if m.name.startswith(prefix+'routing_step_') and m.name.endswith('.json')]
    train=[r for r in rows if r['event']=='train']
    assert [r['step'] for r in train]==list(range(1,1001))
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in train)
    validation={r['step']:r['loss'] for r in rows if r['event']=='validation'}
    return dict(config=config,train=train,validation=validation,gradient=gradient,routing=routing,
                overflow_count=sum(r['event']=='overflow' for r in rows))


def evaluate(b,d):
    bc,dc=b['config']['config'],d['config']['config']
    for k in ('model','seed','training','tied_embeddings','architecture'):
        assert bc[k]==dc[k],k
    assert b['config']['identity']['data']==d['config']['identity']['data']
    assert b['config']['identity']['model_source_sha256']==d['config']['identity']['model_source_sha256']
    assert bc['memory_fusion']['mode']=='last' and dc['memory_fusion']['mode']=='token'
    points=sorted(set(b['validation'])&set(d['validation']))
    assert points==[250,500,750,1000]
    delta={s:d['validation'][s]-b['validation'][s] for s in points}
    targets={}
    for target in (6.5,6.0,5.5):
        item={}
        for label,r in [('baseline',b),('dynamic',d)]:
            crossings=[s for s in points if r['validation'][s]<=target]
            upper=min(crossings) if crossings else None
            previous=max([0]+[s for s in points if upper is not None and s<upper]) if upper else None
            item[label]=dict(first_observed_step=upper,step_bracket=[previous,upper] if upper else None,
                             tokens_at_first_observation=upper*262144 if upper else None,
                             cumulative_training_gpu_seconds=sum(t['seconds']*2 for t in r['train'] if upper and t['step']<=upper) if upper else None)
        targets[str(target)]=item
    faster=all(v<=0 for v in delta.values()) and delta[1000]<0
    verdict='SHORT_RUN_SUPPORT_ONLY' if faster else 'FASTER_CONVERGENCE_NOT_ESTABLISHED'
    return dict(verdict=verdict,validation_delta_dynamic_minus_baseline=delta,target_loss_observations=targets,
                caveats=['One seed and 1000 steps; not proof of final convergence or a causal gradient mechanism.',
                         'Thresholds are bracketed by 250-step validation observations; no exact crossing time inferred.',
                         'GPU-seconds count training step timers only; exclude validation, checkpoint I/O and restarts, and are not FLOPs.',
                         'Different initial functions are intentional; backbone random initialization and training protocol are paired.'],
                summaries={label:dict(final_validation=r['validation'][1000],overflow_count=r['overflow_count'],
                                     last100_mean_training_loss=statistics.mean(t['loss'] for t in r['train'][-100:]),
                                     last100_mean_step_seconds=statistics.mean(t['seconds'] for t in r['train'][-100:]),
                                     gradient_clip_fraction=sum(t['grad_norm']>1 for t in r['train'])/len(r['train']),
                                     gradient_flow_probe=r['gradient'],routing=r['routing']) for label,r in [('baseline',b),('dynamic',d)]})


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--dynamic',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    b=read_run(a.baseline,'yoco-160m-baseline-2gpu-20261101-v2')
    d=read_run(a.dynamic,'yoco-160m-fusion-token-2gpu-20261101-v2')
    result=evaluate(b,d);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    rows=['# 两卡配对短实验：收敛验收','',f"结论：`{result['verdict']}`。单 seed 短程证据，不自动启动完整训练。",'',
          '| 步数 | baseline loss | dynamic loss | dynamic − baseline |','| ---: | ---: | ---: | ---: |']
    for s,delta in result['validation_delta_dynamic_minus_baseline'].items():
        rows.append(f"| {s} | {b['validation'][s]:.6f} | {d['validation'][s]:.6f} | {delta:+.6f} |")
    rows.extend(['','同目标 loss 的采样区间、训练 GPU 时间与梯度诊断见同名 JSON。梯度范数更大不等于梯度方向更好，也不证明收敛更快。',
                 '', '未达到某一阈值会记录为空，不进行外推；训练 GPU 时间不包含排队、验证、保存和重启，不是严格 FLOPs。'])
    out.with_suffix('.md').write_text('\n'.join(rows)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
