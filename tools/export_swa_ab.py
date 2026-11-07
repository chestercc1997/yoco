"""Verify a completed SWA cohort and export numeric tables plus seven figures."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS=('baseline','A','B')
TASKS={'arc_easy':2376,'arc_challenge':1172,'boolq':3270,'hellaswag':10042,
       'openbookqa':500,'piqa':1838,'winogrande':1267,'sciq':1000}
METRICS={task:('acc,none' if task in ('boolq','winogrande') else 'acc_norm,none') for task in TASKS}


def read(path):return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def csvwrite(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run=args.run_dir.resolve();out=args.output.resolve()
    assert (run/'PIPELINE_SUCCESS').exists(),'All three full training/evaluation pipelines must finish first'
    assert read(run/'summary.json')['complete']
    release=run/'release';manifest=read(run/'release-manifest.json')
    for name,expected in manifest['files'].items():assert sha(release/name)==expected,name
    preflight=read(release/'preflight.json');assert preflight['passed'] and preflight['common_weights_bitwise_equal']
    configs={arm:read(release/f'configs/{arm}.json') for arm in ARMS}
    for c in configs.values():
        for key in ('model','training','data','seed','tied_embeddings','execution'):assert c[key]==configs['baseline'][key],key
    assert configs['baseline']['model']['sliding_window']==1024
    training={};trainrows=[];valrows=[];sources={};hashes={};per_doc={};down=[]
    for arm in ARMS:
        folder=run/arm;assert (folder/'PIPELINE_SUCCESS').exists()
        summary=read(folder/'summary.json');meta=read(folder/'checkpoint_00040000.json')
        assert summary['step']==meta['step']==40000 and sha(folder/meta['file'])==meta['sha256']==summary['sha256']
        assert read(folder/'stage_complete.json')=={'step':40000,'formal_complete':True,'synthetic':False}
        resolved=read(folder/'resolved_config.json');assert resolved['config']==configs[arm]
        assert meta['fingerprint']==hashlib.sha256(json.dumps(resolved['identity'],sort_keys=True).encode()).hexdigest()
        for name,expected in resolved['identity']['model_source_sha256'].items():assert sha(release/'runtime'/name)==expected
        assert read(folder/'resume-verification.json')['passed']
        events=[]
        for relative in ('checks/resumed/rank0.jsonl','rank0.jsonl'):
            path=folder/relative;hashes[f'{arm}/{relative}']=sha(path)
            events.extend(map(json.loads,path.read_text().splitlines()))
        updates={e['step']:e for e in events if e['event']=='train'}
        vals={e['step']:e for e in events if e['event']=='validation' and e['step']%250==0}
        assert sorted(updates)==list(range(1,40001)) and sorted(vals)==list(range(250,40001,250))
        assert all(math.isfinite(e['loss']) and math.isfinite(e['grad_norm']) for e in updates.values())
        seconds=sum(e['seconds'] for e in updates.values());params=summary['parameters'];tokens=40000*262144
        assert params==preflight['parameters'][arm] and updates[40000]['tokens']==tokens
        embed=configs[arm]['model']['vocab_size']*configs[arm]['model']['dim']
        extra=0 if arm=='baseline' else 6*5*2*768**2*tokens/8.64e19
        training[arm]={'parameters':params,'validation_loss':vals[40000]['loss'],
            'validation_perplexity':math.exp(vals[40000]['loss']),
            'last100_training_loss':statistics.mean(updates[s]['loss'] for s in range(39901,40001)),
            'tokens':tokens,'active_training_seconds':seconds,'active_gpu_hours':seconds*8/3600,
            'tokens_per_second':tokens/seconds,'peak_rank0_memory_gib':max(e['peak_memory_bytes'] for e in updates.values())/2**30,
            'trunk_6nd_pflop_days':6*(params-embed)*tokens/8.64e19,
            'output_inclusive_6nd_pflop_days':6*params*tokens/8.64e19,
            'extra_repeated_kv_projection_proxy_pflop_days':extra,'review':summary['training_review']}
        for step,e in sorted(updates.items()):
            trainrows.append({'arm':arm,'step':step,'tokens':e['tokens'],'loss':e['loss'],'lr':e['lr'],
                'grad_norm':e['grad_norm'],'scale':e['scale'],'seconds':e['seconds']})
        for step,e in sorted(vals.items()):
            valrows.append({'arm':arm,'step':step,'tokens':step*262144,'validation_loss':e['loss'],
                'perplexity':e['perplexity'],'trunk_6nd_pflop_days':6*(params-embed)*step*262144/8.64e19})
        sources[arm]={'checkpoint':meta,'initialization_sha256':sha(release/f'initializations/{arm}.pt'),
                     'resume_verification':read(folder/'resume-verification.json')}
        per_doc[arm]={}
    task_provenance={}
    for task,count in TASKS.items():
        metric=METRICS[task];key=metric.split(',')[0];row={'task':task,'metric':metric,'documents':count};provenances=[]
        for arm in ARMS:
            folder=run/arm/'evaluation'/task;r=read(folder/'results.json');v=read(folder/'provenance.json')
            for name in ('results.json','provenance.json'):hashes[f'{arm}/evaluation/{task}/{name}']=sha(folder/name)
            assert r['model_step']==40000 and not r['smoke_only'] and v['limit'] is None and v['num_fewshot']==0
            assert v['checkpoint_sha256']==sources[arm]['checkpoint']['sha256']
            assert v['dataset_lock']==read(release/f'evaluation/plan_ab_data_locks/data_lock_{task}.json')
            assert r['n-samples'][task]['effective']==r['n-samples'][task]['original']==count
            samples={s['doc_id']:s for s in r['samples'][task]};assert len(samples)==count
            score=sum(s[key] for s in samples.values())/count
            assert abs(score-r['results'][task][metric])<1e-12
            row[arm]=score;provenances.append(v);per_doc[arm][task]=samples
        for field in ('dataset_lock','task_yaml_sha256','seed','num_fewshot','versions'):
            assert all(v[field]==provenances[0][field] for v in provenances),field
        for arm in ('A','B'):
            assert per_doc[arm][task].keys()==per_doc['baseline'][task].keys()
            for doc_id,s in per_doc[arm][task].items():
                for field in ('doc_hash','prompt_hash','target_hash'):assert s[field]==per_doc['baseline'][task][doc_id][field]
            row[arm+'_delta_pp']=100*(row[arm]-row['baseline'])
        task_provenance[task]={k:provenances[0][k] for k in ('task_yaml_sha256','seed','num_fewshot','versions')}
        down.append(row)
    macro={arm:statistics.mean(r[arm] for r in down) for arm in ARMS}
    summary={'complete':True,'architecture':'YOCO-SWA','window':1024,'seed':42,'world_size':8,
             'training':training,'macro':macro,'downstream':down,'sources':sources,'task_provenance':task_provenance,
             'limitations':['Single seed; differences are descriptive, not a generalization claim.',
                '6ND and repeated-projection corrections are compute proxies, not measured device FLOPs.',
                'Active step timing excludes validation, checkpoint I/O, setup, and evaluation.']}
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    write(out/'summary.json',summary);write(out/'source_hashes.json',hashes);write(out/'model-preflight.json',preflight)
    csvwrite(out/'training_curves.csv',trainrows);csvwrite(out/'validation_curves.csv',valrows);csvwrite(out/'downstream.csv',down)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'baseline':'#64748b','A':'#007f86','B':'#cc5a36'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'svg.hashsalt':'yoco-swa-ab'})
    def save(fig,name):
        for ext in ('png','pdf','svg'):
            metadata={'Date':None} if ext=='svg' else ({'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
            path=out/f'figures/{name}.{ext}';fig.savefig(path,bbox_inches='tight',metadata=metadata)
            if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for ax,field,title in zip(axes,('validation_loss','last100_training_loss','macro'),('Validation loss','Final 100-step training loss','Eight-task macro (%)')):
        values=[100*macro[a] if field=='macro' else training[a][field] for a in ARMS]
        ax.scatter(values,range(3),c=[colors[a] for a in ARMS],s=65)
        ax.set_yticks(range(3),ARMS);ax.set_ylim(2.5,-.5);ax.set_title(title);ax.grid(axis='x',alpha=.2)
        for i,value in enumerate(values):ax.annotate(f'{value:.5f}',(value,i),xytext=(0,9),textcoords='offset points',ha='center')
        ax.margins(x=.35)
    save(fig,'01_quality_summary')
    for name,field,ylabel in (('02_training_loss','loss','Raw training loss'),('05_gradient_norm','grad_norm','Pre-clip gradient norm')):
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for arm in ARMS:
            rows=[r for r in trainrows if r['arm']==arm];x=np.array([r['step'] for r in rows]);y=np.array([r[field] for r in rows])
            for ax in axes:ax.plot(x,y,color=colors[arm],alpha=.55,lw=.35,label=arm)
        axes[0].set_title('All raw updates');axes[1].set_title('Late training');axes[1].set_xlim(30000,40000)
        for ax in axes:ax.set_xlabel('Updates');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        axes[0].legend();save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for arm in ARMS:
        rows=[r for r in valrows if r['arm']==arm]
        for ax in axes:ax.plot([r['tokens']/1e9 for r in rows],[r['validation_loss'] for r in rows],color=colors[arm],label=arm)
    axes[0].set_title('All scheduled validations');axes[1].set_xlim(7,10.49);axes[1].set_title('Late validation')
    late=[r['validation_loss'] for r in valrows if r['tokens']>=7e9];axes[1].set_ylim(min(late)-.02,max(late)+.02)
    for ax in axes:ax.set_xlabel('Training tokens (billions)');ax.set_ylabel('Validation loss');ax.grid(alpha=.2)
    axes[0].legend();save(fig,'03_validation_loss')
    fig,ax=plt.subplots(figsize=(12,4.5),constrained_layout=True);x=np.arange(9)
    for i,arm in enumerate(ARMS):ax.bar(x+(i-1)*.25,[r[arm]*100 for r in down]+[macro[arm]*100],.25,color=colors[arm],label=arm)
    ax.set_xticks(x,[r['task'].replace('_','\n') for r in down]+['macro']);ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,100)
    ax.set_title('Full zero-shot evaluation; acc_norm except BoolQ / WinoGrande');ax.legend(ncol=3);ax.grid(axis='y',alpha=.2);save(fig,'04_downstream')
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,field,title in zip(axes,('tokens_per_second','active_gpu_hours','peak_rank0_memory_gib'),('Active tokens / second','Active training GPU-hours','Peak rank0 memory (GiB)')):
        ax.bar(ARMS,[training[a][field] for a in ARMS],color=[colors[a] for a in ARMS]);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    save(fig,'06_execution_cost')
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for arm in ARMS:
        rows=[r for r in valrows if r['arm']==arm]
        ax.plot([r['trunk_6nd_pflop_days'] for r in rows],[r['validation_loss'] for r in rows],color=colors[arm],label=arm)
    ax.set_xlabel('Trunk 6ND proxy (PFLOP-days)');ax.set_ylabel('Validation loss');ax.set_title('Compute proxy; excludes repeated K/V and attention operations')
    ax.grid(alpha=.2);ax.legend();save(fig,'07_loss_compute_proxy')
    table='\n'.join(f"| {a} | {training[a]['parameters']} | {training[a]['validation_loss']:.6f} | {macro[a]*100:.4f} |" for a in ARMS)
    figures=['01_quality_summary','02_training_loss','03_validation_loss','04_downstream','05_gradient_norm','06_execution_cost','07_loss_compute_proxy']
    (out/'README.md').write_text('# YOCO-SWA 160M baseline / A / B\n\nAll three arms completed 40000 updates and eight full zero-shot tasks on 8×A100 each, with bitwise-equal common backbone initialization. Window1024; FP16 mixed / FP32 parameters; 10485760000 tokens per arm; seed42.\n\n| Arm | Parameters | Validation loss | Eight-task macro (%) |\n| --- | ---: | ---: | ---: |\n'+table+'\n\n[Numeric summary](summary.json) · [Raw training curves](training_curves.csv) · [Scheduled validation curves](validation_curves.csv) · [Task scores](downstream.csv) · [Source hashes](source_hashes.json)\n\n'+'\n'.join(f'![{name}](figures/{name}.png)\n\n[PDF](figures/{name}.pdf) · [SVG](figures/{name}.svg)\n' for name in figures)+'\nSingle seed: differences are descriptive. Active step timings exclude validation and checkpoint I/O. 6ND is a proxy, not measured FLOPs. Published tables contain numeric evidence; private task texts and logs remain local.\n')
    print(f'Verified and exported three complete SWA arms and seven figures to {out}')


if __name__=='__main__':main()
