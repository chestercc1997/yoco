"""Export allowlisted numerical evidence; private logs and task texts stay private."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

TASKS = {'arc_easy':2376,'arc_challenge':1172,'hellaswag':10042,'piqa':1838,
         'winogrande':1267,'boolq':3270,'openbookqa':500,'sciq':1000}
PF_DAY = 8.64e19

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d): p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def csvwrite(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();root=Path(__file__).resolve().parents[1];out=a.output;out.mkdir(parents=True,exist_ok=True)
    baseline=read(root/'evaluation/results/training_comparison.json')['YOCO']
    historical=read(root/'evaluation/figures/training_efficiency_summary.json')['runs']['YOCO']
    bdown=read(root/'evaluation/results/yoco_downstream_summary.json')
    bconfig=read(root/'configs/yoco_160m.json')
    emb=bconfig['model']['dim']*bconfig['model']['vocab_size']
    hashes={};training={};curves=[];trainrows=[];configs={};provenances={};checkpoints={}
    def record(label,p): hashes[label]=sha(p)
    for label,p in [('baseline_training_summary',root/'evaluation/results/training_comparison.json'),('baseline_downstream',root/'evaluation/results/yoco_downstream_summary.json'),('baseline_validation_curve',root/'evaluation/figures/training_efficiency_validation_points.csv'),('baseline_timing',root/'evaluation/figures/training_efficiency_summary.json')]:record(label,p)
    for arm in ['baseline','A','B']:
        if arm=='baseline':
            t=dict(baseline,final_step=40000,world_size=32,hardware='32 x V100',active_seconds=historical['active_seconds'],active_gpu_hours=historical['active_gpu_hours'],end_to_end_hours=historical['end_to_end_hours'],overflow_events=None,peak_rank0_memory_gib=historical['peak_memory_gib'])
            cfg=bconfig
            with (root/'evaluation/figures/training_efficiency_validation_points.csv').open() as f:
                vals=[{'step':int(r['step']),'loss':float(r['validation_loss']),'perplexity':float(r['perplexity'])} for r in csv.DictReader(f) if r['model']=='YOCO']
        else:
            folder=a.raw/'output-a800'/arm
            for name in ['rank0.jsonl','resolved_config.json','stage_complete.json','checkpoint_00040000.json']:record(arm+'_'+name,folder/name)
            resolved=read(folder/'resolved_config.json');cfg=resolved['config'];configs[arm]=resolved
            for name,digest in resolved['identity']['model_source_sha256'].items():
                assert sha(root/'runtime'/name)==digest,('source mismatch',name)
            marker=read(folder/'stage_complete.json');assert marker=={'step':40000,'formal_complete':True,'synthetic':False}
            ck=read(folder/'checkpoint_00040000.json');assert ck['step']==40000 and ck['fingerprint']==hashlib.sha256(json.dumps(resolved['identity'],sort_keys=True).encode()).hexdigest();checkpoints[arm]=ck
            rows=[json.loads(x) for x in (folder/'rank0.jsonl').read_text().splitlines()];tr=[r for r in rows if r['event']=='train'];vals=[r for r in rows if r['event']=='validation']
            assert [r['step'] for r in tr]==list(range(1,40001));assert tr[-1]['tokens']==10485760000
            start=rows[0];assert start['world_size']==8 and not start['resumed'] and not start['synthetic']
            from datetime import datetime
            endhours=(datetime.fromisoformat(vals[-1]['time'])-datetime.fromisoformat(start['time'])).total_seconds()/3600
            seconds=sum(r['seconds'] for r in tr)
            t=dict(validation_loss=vals[-1]['loss'],validation_perplexity=vals[-1]['perplexity'],last100_training_loss=statistics.mean(r['loss'] for r in tr[-100:]),tokens=tr[-1]['tokens'],parameters=start['parameters'],final_step=40000,world_size=8,hardware='8 x A800',active_seconds=seconds,active_gpu_hours=seconds*8/3600,end_to_end_hours=endhours,overflow_events=sum(r['event']=='overflow' for r in rows),peak_rank0_memory_gib=max(r['peak_memory_bytes'] for r in tr)/2**30)
            for r in tr:trainrows.append(dict(arm=arm,step=r['step'],tokens=r['tokens'],loss=r['loss'],lr=r['lr'],seconds=r['seconds']))
        t['shared_embedding_output_parameters']=emb;t['trunk_parameters']=t['parameters']-emb
        t['compute_trunk_6nd_pflop_days']=6*t['trunk_parameters']*t['tokens']/PF_DAY
        t['compute_output_inclusive_6nd_pflop_days']=6*t['parameters']*t['tokens']/PF_DAY
        extra=0 if arm=='baseline' else 6*5*2*768**2*t['tokens']
        t['extra_repeated_kv_projection_pflop_days']=extra/PF_DAY
        t['output_inclusive_plus_repeated_kv_proxy_pflop_days']=t['compute_output_inclusive_6nd_pflop_days']+extra/PF_DAY
        t['mean_active_tokens_per_second']=t['tokens']/t['active_seconds'];training[arm]=t
        assert [r['step'] for r in vals]==list(range(250,40001,250))
        for r in vals:
            d=r['step']*262144
            curves.append(dict(arm=arm,step=r['step'],tokens=d,validation_loss=r['loss'],perplexity=r['perplexity'],compute_trunk_6nd_pflop_days=6*t['trunk_parameters']*d/PF_DAY,compute_output_inclusive_6nd_pflop_days=6*t['parameters']*d/PF_DAY,output_inclusive_plus_repeated_kv_proxy_pflop_days=t['output_inclusive_plus_repeated_kv_proxy_pflop_days']*d/t['tokens']))
    for key in ['model','training','data','seed','tied_embeddings']:assert configs['A']['config'][key]==configs['B']['config'][key]
    assert configs['A']['data']==configs['B']['data']
    down=[];paired=[]
    evalroot=a.raw/'work-a800/release/runs/eval-plan-ab';assert (evalroot/'PIPELINE_SUCCESS').exists()
    record('AB_pipeline_marker',evalroot/'PIPELINE_SUCCESS');record('AB_downstream_summary',evalroot/'summary.json')
    for task,n in TASKS.items():
        metric='acc,none' if task in ['boolq','winogrande'] else 'acc_norm,none';key=metric.split(',')[0]
        row=dict(task=task,metric=metric,examples=n,baseline=bdown[task][task][metric]);sm={};pv=[]
        row['baseline_stderr']=bdown[task][task].get(metric.replace(',','_stderr,'))
        for arm in 'AB':
            folder=evalroot/'results'/arm/task;r=read(folder/'results.json');p=read(folder/'provenance.json')
            for name in ['results.json','provenance.json']:record(arm+'_'+task+'_'+name,folder/name)
            assert r['model_step']==40000 and not r['smoke_only'] and p['limit'] is None and p['num_fewshot']==0
            assert p['checkpoint_sha256']==checkpoints[arm]['sha256']
            assert r['n-samples'][task]['effective']==n and r['n-samples'][task]['original']==n
            samples=r['samples'][task];assert len(samples)==n
            sm[arm]={s['doc_id']:s for s in samples};assert len(sm[arm])==n
            score=sum(s[key] for s in samples)/n;assert abs(score-r['results'][task][metric])<1e-12
            row[arm]=score;row[arm+'_delta_pp']=100*(score-row['baseline']);row[arm+'_stderr']=r['results'][task].get(metric.replace(',','_stderr,'))
            pv.append(p)
        for key2 in ['dataset_lock','task_yaml_sha256','seed','num_fewshot','versions']:assert pv[0][key2]==pv[1][key2],key2
        provenances[task]={k:pv[0][k] for k in ['task_yaml_sha256','seed','num_fewshot','versions']}
        assert sm['A'].keys()==sm['B'].keys()
        for doc_id,s in sm['A'].items():
            sb=sm['B'][doc_id]
            for k in ['doc_hash','prompt_hash','target_hash']:assert s[k]==sb[k]
            paired.append(dict(task=task,doc_id=doc_id,A_correct=s[key],B_correct=sb[key]))
        row['B_minus_A_pp']=100*(row['B']-row['A']);down.append(row)
    macro={arm:statistics.mean(r[arm] for r in down) for arm in ['baseline','A','B']}
    import numpy as np
    rng=np.random.default_rng(42);boot=np.zeros(10000)
    for task in TASKS:
        diffs=np.array([r['B_correct']-r['A_correct'] for r in paired if r['task']==task]);counts=np.array([(diffs==v).sum() for v in [-1,0,1]])
        draws=rng.multinomial(len(diffs),counts/len(diffs),size=10000);boot+=(draws[:,2]-draws[:,0])/len(diffs)/8
    ci=np.quantile(boot,[.025,.975])*100
    thresholds=[]
    for target in [4.0,3.5,3.3,3.2,3.17,baseline['validation_loss']]:
        for arm in training:
            cr=[r for r in curves if r['arm']==arm];hits=[r for r in cr if r['validation_loss']<=target]
            first=hits[0] if hits else None
            sustained=next((r for i,r in enumerate(cr) if all(q['validation_loss']<=target for q in cr[i:])),None)
            thresholds.append(dict(target_loss=target,arm=arm,first_observed_step=first['step'] if first else None,sustained_observed_step=sustained['step'] if sustained else None,tokens_at_first=first['tokens'] if first else None,trunk_6nd_at_first=first['compute_trunk_6nd_pflop_days'] if first else None,output_inclusive_6nd_at_first=first['compute_output_inclusive_6nd_pflop_days'] if first else None))
    result=dict(schema_version=1,training=training,downstream=down,macro=macro,AB_paired_bootstrap=dict(B_minus_A_pp=100*(macro['B']-macro['A']),ci95_pp=ci.tolist(),replicates=10000,seed=42,scope='Paired sample bootstrap within each task; equal task weights; fixed checkpoints, not training-seed robustness.'),evaluation_protocol=provenances,checkpoint_metadata=checkpoints,source_sha256=hashes,boundary='A/B complete and paired sample identities verified. Baseline is an archived 32-V100 reference; original token shard identities, task YAML and checkpoint payload not freshly verified. A/B checkpoint metadata bindings verified, payloads not downloaded or rehashed. 6ND is a proxy, not measured device FLOPs.')
    write(out/'summary.json',result);csvwrite(out/'downstream.csv',down);csvwrite(out/'training_summary.csv',[dict(arm=k,**v) for k,v in training.items()]);csvwrite(out/'validation_curves.csv',curves);csvwrite(out/'ab_training_curves.csv',trainrows);csvwrite(out/'ab_paired_scores.csv',paired);csvwrite(out/'loss_thresholds.csv',thresholds)
    write(out/'training_protocol.json',{arm:{k:cfg[k] for k in ['architecture','model','tied_embeddings','seed','training','memory_fusion'] if k in cfg} for arm,cfg in [('baseline',bconfig),('A',configs['A']['config']),('B',configs['B']['config'])]})
    print(json.dumps({'training':training,'macro':macro,'AB_paired_bootstrap':result['AB_paired_bootstrap']},indent=2))

if __name__=='__main__':main()
