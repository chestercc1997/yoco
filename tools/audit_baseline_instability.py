"""Rebuild a compact audit of the historical baseline's convergence failure."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/yoco-160m-baseline-8gpu-full-20261104T0114Z'
OUT = ROOT / 'docs/evidence/baseline-numerical-audit'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    events = rows(RUN/'rank0.jsonl')
    train = [r for r in events if r['event']=='train']
    valid = [r for r in events if r['event']=='validation']
    overflow = [r for r in events if r['event']=='overflow']
    by_step = {r['step']:r for r in train}
    mismatch = 0
    for rank in range(1,8):
        for row in rows(RUN/f'rank{rank}.jsonl'):
            if row['event']=='train' and row['grad_norm'] != by_step[row['step']]['grad_norm']:
                mismatch += 1
    release = json.loads((RUN/'release-manifest.json').read_text())
    frozen_verified = all(digest(RUN/'release'/name)==expected for name,expected in release['files'].items())
    assert frozen_verified and mismatch==0
    windows=[]
    for start,end in [(1001,5000),(5001,20000),(20001,20250),(21251,21500),(21501,22000),(22001,25000),(35001,40000)]:
        norms=[r['grad_norm'] for r in train if start<=r['step']<=end]
        windows.append(dict(start=start,end=end,count=len(norms),median=statistics.median(norms),
                            maximum=max(norms),clipped_steps=sum(x>1 for x in norms)))
    diagnostics={}
    for label in ['numerics-step100','numerics-validation','numerics-remedies','numerics-global40000',
                  'global-remedies','fp64-reference','early-tf32']:
        path=ROOT/f'runs/diagnostics/baseline-{label}.json'
        if not path.exists(): continue
        report=json.loads(path.read_text())
        diagnostics[label]={'source':str(path.relative_to(ROOT)),'sha256':digest(path),
            **{k:report.get(k) for k in ('checkpoint_step','batch','global_update','no_optimizer_updates')},
            'cases':[{k:v for k,v in c.items() if k!='largest_parameter_gradients'} for c in report['cases']]}
    report={
        'historical_run':str(RUN.relative_to(ROOT)),
        'verdict':'Historical FP16 run completed, but sustained convergence regression makes it an unstable baseline.',
        'established':[
            'Regression is in raw validation and training logs, not only the plot.',
            'No training restart near the regression; all eight ranks agree on gradient norms.',
            'The recorded immutable release still matches all recorded hashes.',
            'All training and validation token shard hashes were reverified successfully.',
            'Late checkpoint gradients depend strongly on arithmetic precision and FP16 loss scaling.',
            'Early checkpoint gradients agree closely across precision modes.'
        ],
        'limits':[
            'No checkpoint immediately preceding update 21354 remains, so the first trigger cannot be replayed.',
            'Late-checkpoint probes establish sensitivity after divergence, not sole causation of the first spike.',
            'No specific PyTorch version defect has been established.',
            'FP32 diagnostic rerun was stopped at user request; the active recipe must match A/B FP16 precision.',
            'A/B original checkpoints and gradient/scaler logs are unavailable locally.',
            'Do not attribute the full A/B gap against this unstable run to the method alone.'
        ],
        'integrity':{'frozen_release_verified':frozen_verified,'cross_rank_gradient_mismatches':mismatch,
            'token_verification_log':'logs/setup/baseline-audit-token-verification.log',
            'manifest_sha256':digest(ROOT/'data/fineweb_edu_10bt/manifest.json'),
            'started_events':[r for r in events if r['event']=='started']},
        'first_major_spike':{'update':21354,'pre_clip_gradient_norm':by_step[21354]['grad_norm'],
            'clip_threshold':1.0,'note':'Norm is measured before clipping; optimizer did not receive an unclipped norm of 991.',
            'overflow_events':[r for r in overflow if r['step']==21353]},
        'gradient_windows':windows,
        'validation':{'best':min(valid,key=lambda x:x['loss']),'final':valid[-1],
            'selected':[r for r in valid if r['step'] in (10000,20000,20250,21250,21500,21750,22000,40000)]},
        'overflow_count':len(overflow),'overflow_events':overflow,
        'precision_probes':diagnostics,
        'fp32_diagnostic_rerun':{'run':'runs/yoco-160m-baseline-8gpu-fp32-20261104T1450Z','state':'stopped_by_user'},
        'rerun':{'active_record':'runs/setup/baseline-matched-8gpu-active.json','precision':'fp16_mixed_fp32_params',
            'TF32':True,'world_size':8,'source_step':100,'target_step':40000,
            'checkpoint_every':5000,'keep_all_checkpoints':True,'validation_every':250,
            'source_sha256':'3560daadcff9ad6f0a4989c9fe711986ad2eaa93f426ae95e032c3ebe6930bd2',
            'preserved':['model architecture','optimizer state','RNG','data order','global batch','learning-rate schedule'],
            'evaluation_precision':'fp16_mixed_fp32_params; model and training recipe match published A/B settings',
            'execution_options':'Acceleration is measured separately; no reduced training tokens or altered optimizer hyperparameters.'}
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    axes[0].plot([r['step'] for r in valid],[r['loss'] for r in valid],label='Historical A100 FP16 baseline')
    axes[0].set_ylim(3.1,4.2);axes[0].set_ylabel('Validation loss');axes[0].legend()
    blocks=[train[i:i+100] for i in range(0,len(train),100)]
    axes[1].plot([b[-1]['step'] for b in blocks],[statistics.median(r['grad_norm'] for r in b) for b in blocks],label='Median per 100 updates')
    axes[1].scatter([21354],[by_step[21354]['grad_norm']],color='red',s=22,label='Update 21354: 991.3 (before clipping)')
    axes[1].axhline(1,color='gray',linestyle=':',label='Clipping threshold')
    axes[1].set_yscale('log');axes[1].set_ylabel('Pre-clip gradient norm');axes[1].legend(fontsize=8)
    axes[2].plot([r['step'] for r in train],[r['scale'] for r in train],linewidth=.8)
    axes[2].scatter([r['step']+1 for r in overflow],[r['scale'] for r in overflow],color='red',s=15,label='Overflow retry')
    axes[2].set_yscale('log',base=2);axes[2].set_ylabel('FP16 loss scale');axes[2].set_xlabel('Update');axes[2].legend()
    for ax in axes:
        ax.axvline(21354,color='red',linestyle='--',alpha=.5);ax.grid(alpha=.2)
    fig.suptitle('Baseline regression: loss, gradients and scaling from raw logs')
    for extension in ('png','pdf'):
        fig.savefig(OUT.with_suffix('.'+extension),dpi=180)
    plt.close(fig)
    print(json.dumps({'output':str(OUT.with_suffix('.json')),'release_verified':frozen_verified,
                      'rank_mismatches':mismatch,'overflow_count':len(overflow),'spike_norm':by_step[21354]['grad_norm']}))


if __name__=='__main__': main()
