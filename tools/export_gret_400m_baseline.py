"""Export complete gRet 400M baseline/A/B results without changing historical references."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.watch_swa_training import checkpoint_healthy,events
ARMS=('baseline','A','B')
TASKS={'arc_easy':2376,'arc_challenge':1172,'boolq':3270,'hellaswag':10042,
       'openbookqa':500,'piqa':1838,'winogrande':1267,'sciq':1000}
METRICS={task:('acc,none' if task in ('boolq','winogrande') else 'acc_norm,none') for task in TASKS}


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def csvwrite(path,rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def lineage(folder,through,hashes):
    origin=folder/'recovery-source.json'
    record=read(origin) if origin.exists() else None
    floor=record['source_step'] if record else 0
    prefix=lineage(Path(record['source_run']),min(through,floor),hashes) if record else []
    for name in ('checks/resumed/rank0.jsonl','rank0.jsonl'):
        path=folder/name
        if path.exists():
            hashes[str(path)]=sha(path)
            for event in events(path):
                step=event.get('step',-1)
                if (floor<=step<through if event['event']=='overflow' else floor<step<=through):
                    prefix.append(event)
    return prefix


def inventory(folder,through):
    origin=folder/'recovery-source.json';record=read(origin) if origin.exists() else None
    floor=record['source_step'] if record else 0
    result=inventory(Path(record['source_run']),min(through,floor)) if record else {}
    for meta_path in folder.glob('checkpoint_*.json'):
        meta=read(meta_path);step=meta.get('step',-1)
        if floor<step<=through and step%5000==0:result[step]=meta_path.with_suffix('.pt')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run=args.run_dir.resolve();out=args.output.resolve();release=run/'release'
    import torch
    torch.set_num_threads(4)
    assert (run/'baseline/PIPELINE_SUCCESS').exists(),'Baseline training and all full evaluations must finish first'
    manifest=read(run/'release-manifest.json');assert manifest['architecture']=='YOCO-gRet'
    hashes={'release-manifest.json':sha(run/'release-manifest.json')}
    for name,expected in manifest['files'].items():
        assert sha(release/name)==expected,name
        hashes['release/'+name]=expected
    preflight=read(release/'preflight.json');assert preflight['passed'] and preflight['common_weights_match_original_A_initialization']
    folders={'baseline':run/'baseline','A':Path(manifest['source_A_run']),'B':Path(manifest['source_B_run'])/'B'}
    configs={a:read(folders[a]/'resolved_config.json')['config'] for a in ARMS}
    for key in ('architecture','model','training','data','seed','tied_embeddings','execution'):
        assert configs['baseline'][key]==configs['A'][key]==configs['B'][key],key
    assert configs['baseline']==read(release/'configs/baseline.json')
    assert configs['B']==read(release/'configs/B-reference.json') and configs['A']==read(release/'configs/A-reference.json')
    assert 'memory_fusion' not in configs['baseline']
    assert configs['B']['model'].get('sliding_window') is None
    assert sha(configs['B']['data']['manifest'])==manifest['data_manifest_sha256']
    hashes[configs['B']['data']['manifest']]=manifest['data_manifest_sha256']
    training={};trainrows=[];valrows=[];sources={};summaries={}
    for arm in ARMS:
        folder=folders[arm];assert (folder/'PIPELINE_SUCCESS').exists()
        summary=read(folder/'summary.json');summaries[arm]=summary
        assert summary['step']==40000 and summary['world_size']==8 and summary['complete']
        checkpoints=inventory(folder,40000);assert sorted(checkpoints)==list(range(5000,40001,5000))
        verified=[]
        for step,path in sorted(checkpoints.items()):
            proof=checkpoint_healthy(path)
            assert proof['step']==step
            hashes[str(path)]=proof['sha256'];hashes[str(path.with_suffix('.json'))]=sha(path.with_suffix('.json'))
            verified.append(proof)
        assert summary['sha256']==verified[-1]['sha256']
        saved=torch.load(checkpoints[40000],map_location='cpu',weights_only=False,mmap=True)
        assert all(v.dtype==torch.float32 for v in saved['model'].values() if v.is_floating_point())
        assert all(float(v['step'])==40000 for v in saved['optimizer']['state'].values())
        cfg=configs[arm];t=cfg['training']
        for group in saved['optimizer']['param_groups']:
            assert group['lr']==t['end_lr']==0 and group['betas']==tuple(t['betas'])
            assert group['eps']==t['eps'] and group['weight_decay']==t['weight_decay']
        assert saved['scaler']['growth_factor']==2 and saved['scaler']['backoff_factor']==.5 and saved['scaler']['growth_interval']==2000
        del saved
        rows=lineage(folder,40000,hashes)
        updates={e['step']:e for e in rows if e['event']=='train'}
        vals={e['step']:e for e in rows if e['event']=='validation' and e['step']%250==0}
        assert sorted(updates)==list(range(1,40001)) and sorted(vals)==list(range(250,40001,250))
        assert all(math.isfinite(e['loss']) and math.isfinite(e['grad_norm']) for e in updates.values())
        assert all(math.isfinite(e['loss']) for e in vals.values())
        params=summary['parameters'];tokens=10485760000;seconds=sum(e['seconds'] for e in updates.values())
        assert updates[40000]['tokens']==tokens
        assert params=={'baseline':400590848,'A':400590860,'B':400591884}[arm]
        dim=cfg['model']['dim'];embed=cfg['model']['vocab_size']*dim;count=cfg['model']['n_layers']//2
        training[arm]={'parameters':params,'validation_loss':vals[40000]['loss'],'validation_perplexity':math.exp(vals[40000]['loss']),
            'last100_training_loss':statistics.mean(updates[s]['loss'] for s in range(39901,40001)),
            'tokens':tokens,'active_training_seconds':seconds,'active_gpu_hours':seconds*8/3600,'tokens_per_second':tokens/seconds,
            'peak_rank0_memory_gib':max(e['peak_memory_bytes'] for e in updates.values())/2**30,
            'trunk_6nd_pflop_days':6*(params-embed)*tokens/8.64e19,'output_inclusive_6nd_pflop_days':6*params*tokens/8.64e19,
            'extra_repeated_kv_projection_proxy_pflop_days':0. if arm=='baseline' else 6*(count-1)*2*dim**2*tokens/8.64e19,
            'amp_rejected_attempts_committed_lineage':sum(e['event']=='overflow' for e in rows),
            'review':summary['training_review']}
        for step,e in sorted(updates.items()):
            trainrows.append({'arm':arm,'step':step,'tokens':e['tokens'],'loss':e['loss'],'lr':e['lr'],
                'grad_norm':e['grad_norm'],'scale':e['scale'],'seconds':e['seconds']})
        for step,e in sorted(vals.items()):
            valrows.append({'arm':arm,'step':step,'tokens':step*262144,'validation_loss':e['loss'],'perplexity':e['perplexity'],
                'trunk_6nd_pflop_days':6*(params-embed)*step*262144/8.64e19})
        sources[arm]={'run':str(folder),'checkpoint':read(checkpoints[40000].with_suffix('.json')),
            'all_retained_checkpoints_verified':verified,'historical_recovery':summary.get('recovery_source')}
        for name in ('summary.json','resolved_config.json','resume-verification.json'):
            hashes[str(folder/name)]=sha(folder/name)
    down=[];task_provenance={}
    from lm_eval.tasks import TaskManager
    manager=TaskManager()
    for task,count in TASKS.items():
        metric=METRICS[task];key=metric.split(',')[0];row={'task':task,'metric':metric,'documents':count};paired={}
        provenance=[]
        for arm in ARMS:
            folder=folders[arm]/'evaluation'/task;result=read(folder/'results.json');v=read(folder/'provenance.json')
            assert result['model_step']==40000 and not result['smoke_only'] and v['limit'] is None and v['num_fewshot']==0 and v['seed']==42
            assert result['n-shot'][task]==0 and v['checkpoint_sha256']==sources[arm]['checkpoint']['sha256']
            assert result['n-samples'][task]['effective']==result['n-samples'][task]['original']==count
            assert v['dataset_lock']==read(release/f'evaluation/plan_ab_data_locks/data_lock_{task}.json')
            assert sha(manager.task_index[task]['yaml_path'])==v['task_yaml_sha256']
            samples={x['doc_id']:x for x in result['samples'][task]};assert len(samples)==count
            score=sum(x[key] for x in samples.values())/count
            assert abs(score-result['results'][task][metric])<1e-12
            assert abs(score-next(t['score'] for t in summaries[arm]['tasks'] if t['task']==task))<1e-12
            row[arm]=score;paired[arm]=samples;provenance.append(v)
            for name in ('results.json','provenance.json'):hashes[str(folder/name)]=sha(folder/name)
        for key in ('dataset_lock','task_yaml_sha256','seed','num_fewshot','versions'):
            assert all(v[key]==provenance[0][key] for v in provenance),key
        for arm in ('A','B'):
            assert paired[arm].keys()==paired['baseline'].keys()
            for doc_id,sample in paired[arm].items():
                for key in ('doc_hash','prompt_hash','target_hash'):assert sample[key]==paired['baseline'][doc_id][key]
            row[arm+'_delta_pp']=100*(row[arm]-row['baseline'])
        down.append(row)
        task_provenance[task]={k:provenance[0][k] for k in ('task_yaml_sha256','seed','num_fewshot','versions')}
    macro={arm:statistics.mean(row[arm] for row in down) for arm in ARMS}
    assert all(abs(macro[a]-summaries[a]['macro_accuracy'])<1e-12 for a in ARMS)
    recoveries=read(run/'guard-state.json').get('recoveries',[]) if (run/'guard-state.json').exists() else []
    summary={'complete':True,'architecture':'YOCO-gRet','world_size':8,'seed':42,'training':training,'macro':macro,'downstream':down,
        'sources':sources,'task_provenance':task_provenance,'baseline_automatic_recoveries':recoveries,
        'limitations':['Baseline is newly trained after B; A and B are preserved completed 400M references.',
            'A has a historical compiled/repaired prefix; baseline and B use the repaired native implementation throughout. Timing differences do not isolate the fusion method.',
            'Single seed; no claim of a stable overall accuracy gain.',
            'Successful-update timing excludes AMP failed attempts, validation, checkpoint I/O, setup and evaluation. Raw curves show the committed lineage; recovery archives are preserved.',
            '6ND and repeated-projection estimates are compute proxies, not measured device FLOPs.']}
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    write(out/'source_hashes.json',hashes);write(out/'model-preflight.json',preflight)
    csvwrite(out/'training_curves.csv',trainrows);csvwrite(out/'validation_curves.csv',valrows);csvwrite(out/'downstream.csv',down)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'baseline':'#64748b','A':'#007f86','B':'#cc5a36'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160,'svg.hashsalt':'yoco-gret-400m-baseline-ab'})
    def save(fig,name):
        for ext in ('png','pdf','svg'):
            metadata={'Date':None} if ext=='svg' else ({'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
            path=out/f'figures/{name}.{ext}';fig.savefig(path,bbox_inches='tight',metadata=metadata)
            if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for ax,field,title in zip(axes,('validation_loss','last100_training_loss','macro'),('Validation loss','Final 100-step training loss','Eight-task macro (%)')):
        values=[100*macro[a] if field=='macro' else training[a][field] for a in ARMS]
        ax.scatter(values,range(len(ARMS)),c=[colors[a] for a in ARMS],s=65)
        ax.set_yticks(range(len(ARMS)),ARMS);ax.set_ylim(len(ARMS)-.5,-.5);ax.set_title(title);ax.grid(axis='x',alpha=.2)
        for i,value in enumerate(values):ax.annotate(f'{value:.5f}',(value,i),xytext=(0,9),textcoords='offset points',ha='center')
        ax.margins(x=.35)
    save(fig,'01_quality_summary')
    for name,field,ylabel in (('02_training_loss','loss','Raw training loss'),('05_gradient_norm','grad_norm','Pre-clip gradient norm')):
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for arm in ARMS:
            rows=[r for r in trainrows if r['arm']==arm];x=np.array([r['step'] for r in rows]);y=np.array([r[field] for r in rows])
            for ax in axes:ax.plot(x,y,color=colors[arm],alpha=.55,lw=.35,label=arm)
        axes[0].set_title('All raw updates');axes[1].set_title('Late training');axes[1].set_xlim(30000,40000)
        late_values=[r[field] for r in trainrows if r['step']>=30000]
        low,high=min(late_values),max(late_values);pad=max((high-low)*.05,1e-5)
        axes[1].set_ylim(low-pad,high+pad)
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
    for i,arm in enumerate(ARMS):ax.bar(x+(i-(len(ARMS)-1)/2)*.28,[r[arm]*100 for r in down]+[macro[arm]*100],.28,color=colors[arm],label=arm)
    ax.set_xticks(x,[r['task'].replace('_','\n') for r in down]+['macro']);ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,100)
    ax.set_title('Full zero-shot evaluation; acc_norm except BoolQ / WinoGrande');ax.legend(ncol=3);ax.grid(axis='y',alpha=.2);save(fig,'04_downstream')
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,field,title in zip(axes,('tokens_per_second','active_gpu_hours','peak_rank0_memory_gib'),('Successful updates: tokens / second','GPU-hours for successful updates','Peak rank0 memory (GiB)')):
        ax.bar(ARMS,[training[a][field] for a in ARMS],color=[colors[a] for a in ARMS]);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    save(fig,'06_execution_cost')
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for arm in ARMS:
        rows=[r for r in valrows if r['arm']==arm]
        ax.plot([r['trunk_6nd_pflop_days'] for r in rows],[r['validation_loss'] for r in rows],color=colors[arm],label=arm)
    ax.set_xlabel('Trunk 6ND proxy (PFLOP-days)');ax.set_ylabel('Validation loss');ax.set_title('Compute proxy; excludes repeated K/V and attention operations')
    ax.grid(alpha=.2);ax.legend();save(fig,'07_loss_compute_proxy')

    names=['01_quality_summary','02_training_loss','03_validation_loss','04_downstream','05_gradient_norm','06_execution_cost','07_loss_compute_proxy']
    table='\n'.join(f"| {a} | {training[a]['parameters']} | {training[a]['validation_loss']:.6f} | {macro[a]*100:.4f} |" for a in ARMS)
    (out/'README.md').write_text('# YOCO-gRet 400M: baseline, method A and method B\n\nBaseline, A and B each completed 40000 updates and eight full zero-shot tasks on 8xA100. All three configurations share the original seed-42 backbone initialization, data/tokenizer, FP16 mixed / FP32 parameters, batch and 40000-step schedule. All 24 retained baseline/A/B checkpoints and all paired task documents/prompts/targets were verified. Baseline was queued after B; historical A/B results were preserved.\n\n| Arm | Parameters | Validation loss | Eight-task macro (%) |\n| --- | ---: | ---: | ---: |\n'+table+'\n\n[Numeric summary](summary.json) · [Raw training points](training_curves.csv) · [Validation points](validation_curves.csv) · [Task scores](downstream.csv) · [Source hashes](source_hashes.json)\n\n'+'\n'.join(f'![{name}](figures/{name}.png)\n\n[PDF](figures/{name}.pdf) · [SVG](figures/{name}.svg)\n' for name in names)+'\nA has a historical compiled/repaired prefix; baseline and B use the repaired native path throughout. Timings do not isolate architecture effects. Successful-update timing excludes AMP failed attempts, validation, checkpoint I/O, setup and evaluation. Raw committed-lineage points are preserved; failed/replayed suffixes remain in local archives. The 6ND figures are proxies, not measured FLOPs. This single-seed result does not establish a stable downstream gain. Generated figures require agent visual review before final publication.\n')
    write(out/'summary.json',summary)
    print(f'Verified gRet 400M baseline/A/B: 24 full tasks, 24 retained checkpoints, seven figures at {out}',flush=True)


if __name__=='__main__':main()
