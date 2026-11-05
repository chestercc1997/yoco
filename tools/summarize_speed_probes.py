"""Compare execution speed and same-batch gradients without accepting noisy gains."""
import argparse
import json
import math
from pathlib import Path
import statistics
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-case',default='reference')
    args=p.parse_args();torch.set_num_threads(4)
    reference=args.root/args.reference_case
    process_results={}
    if (args.root/'summary.json').exists():
        process_results={r['case']:r for r in json.loads((args.root/'summary.json').read_text())}
    with (reference/'config.json').open() as f:config=json.load(f)
    gradients=torch.load(reference/'first_update_gradients.pt',map_location='cpu',weights_only=True,mmap=True)
    refnorm=sum(float(g.double().square().sum()) for g in gradients.values())
    records=[]
    for folder in sorted(args.root.iterdir()):
        if not folder.is_dir() or not (folder/'config.json').exists():continue
        c=json.loads((folder/'config.json').read_text())
        for key in ('model','training','data','seed','tied_embeddings','memory_fusion'):
            assert c.get(key)==config.get(key),(folder.name,key)
        record={'case':folder.name,'training_setting_unchanged':True}
        record['process_clean_exit']=not process_results.get(folder.name,{}).get('failed',False)
        if not (folder/'benchmark_summary.json').exists():
            record.update(completed=False,accepted=False);records.append(record);continue
        record.update(json.loads((folder/'benchmark_summary.json').read_text()),completed=True)
        rows=[r for r in map(json.loads,(folder/'rank0.jsonl').read_text().splitlines()) if r['event']=='train']
        reference_rows=[r for r in map(json.loads,(reference/'rank0.jsonl').read_text().splitlines()) if r['event']=='train']
        assert [r['step'] for r in rows]==[r['step'] for r in reference_rows]
        times=[r['seconds'] for r in rows[5:]]
        record['p25_seconds']=sorted(times)[len(times)//4];record['p75_seconds']=sorted(times)[len(times)*3//4]
        record['maximum_loss_difference']=max(abs(a['loss']-b['loss']) for a,b in zip(rows,reference_rows))
        record['first_loss_difference']=abs(rows[0]['loss']-reference_rows[0]['loss'])
        other=torch.load(folder/'first_update_gradients.pt',map_location='cpu',weights_only=True,mmap=True)
        assert other.keys()==gradients.keys()
        diff=dot=norm=0.
        for name,g in other.items():
            a=g.double();b=gradients[name].double()
            assert torch.isfinite(a).all()
            diff+=float((a-b).square().sum());dot+=float((a*b).sum());norm+=float(a.square().sum())
        record['gradient_relative_l2']=math.sqrt(diff/refnorm)
        record['gradient_cosine']=dot/math.sqrt(norm*refnorm)
        record['numerical_checks_passed']=(record['gradient_relative_l2']<.005 and record['gradient_cosine']>.99999
            and record['first_loss_difference']<1e-4 and record['maximum_loss_difference']<.002)
        records.append(record)
    baseline_names=(args.reference_case,'reference_repeat') if args.reference_case=='reference' else (args.reference_case,)
    baselines=[r['median_seconds'] for r in records if r['case'] in baseline_names and r['completed']]
    reference_seconds=statistics.median(baselines)
    for r in records:
        if not r['completed']:continue
        r['speedup']=reference_seconds/r['median_seconds']
        r['accepted']=r['process_clean_exit'] and r['numerical_checks_passed'] and r['speedup']>=1.05 and r['case'] not in baseline_names
    accepted=sorted((r for r in records if r.get('accepted')),key=lambda r:r['median_seconds'])
    report={'reference_case':args.reference_case,'reference_seconds':reference_seconds,'records':records,'selected':accepted[0]['case'] if accepted else None,
        'criteria':{'minimum_speedup':1.05,'maximum_gradient_relative_l2':.005,'minimum_gradient_cosine':.99999,
            'maximum_first_loss_difference':1e-4,'maximum_trajectory_loss_difference':.002},
        'scope':'Short tests preserve the model, recipe, dtype and data order; passing numerical tolerances is not proof of bitwise identity or final convergence.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'selected':report['selected'],'reference_seconds':reference_seconds,
        'cases':[{k:r.get(k) for k in ('case','median_seconds','speedup','gradient_relative_l2','numerical_checks_passed','accepted')} for r in records]},indent=2))


if __name__=='__main__':main()
