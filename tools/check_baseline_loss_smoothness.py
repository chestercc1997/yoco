"""Compare raw published training losses; smoothing is used only for diagnostics."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'evaluation/ab_baseline_20261104/training_curves.csv'
OUT=ROOT/'docs/evidence/baseline-training-loss-smoothness'


def main():
    with SOURCE.open() as f: rows=list(csv.DictReader(f))
    series={a:np.array([float(r['loss']) for r in rows if r['arm']==a]) for a in ('baseline','A','B')}
    diagnostics={};residuals={}
    for arm,x in series.items():
        assert len(x)==40000 and np.isfinite(x).all()
        trend=np.median(np.lib.stride_tricks.sliding_window_view(np.pad(x,125,mode='edge'),251),axis=1)
        residual=x-trend;residuals[arm]=residual
        z=residual[1000:];delta=np.diff(x[1000:]);peak=int(z.argmax()+1000)
        diagnostics[arm]={'steps':40000,'analyzed_steps':[1001,40000],
            'residual_std':float(z.std()),'residual_p99':float(np.quantile(z,.99)),
            'maximum_positive_residual':float(z.max()),'peak_step':peak+1,'peak_raw_loss':float(x[peak]),
            'positive_residual_over_0_15':int(np.sum(z>.15)),
            'maximum_positive_adjacent_jump':float(delta.max()),
            'positive_adjacent_jumps_over_0_3':int(np.sum(delta>.3))}
    report={'source':str(SOURCE.relative_to(ROOT)),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'method':'Raw losses are never smoothed in the plots. Diagnostics subtract a centered 251-step median; exclude the first 1000 warmup/rapid-learning steps.',
        'threshold_note':'0.15 residual and 0.3 adjacent jump are descriptive diagnostic thresholds, not formal statistical significance tests.',
        'series':diagnostics,'scope':'Training-loss smoothness does not establish absence of gradient spikes or scaler overflows.'}
    OUT.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'baseline':'#64748b','A':'#007f86','B':'#cc5a36'}
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    peak=diagnostics['baseline']['peak_step'];steps=np.arange(1,40001)
    for arm,x in series.items():
        axes[0,0].plot(steps[1000:],x[1000:],lw=.35,alpha=.7,color=colors[arm],label=arm)
        window=(steps>=peak-80)&(steps<=peak+80)
        axes[0,1].plot(steps[window],x[window],lw=.9,color=colors[arm],label=arm)
        axes[1,0].plot(steps[1000:],residuals[arm][1000:],lw=.3,alpha=.55,color=colors[arm],label=arm)
        axes[1,1].hist(residuals[arm][1000:],bins=np.linspace(-.15,.15,100),density=True,
                       histtype='step',color=colors[arm],label=f"{arm}: std {diagnostics[arm]['residual_std']:.5f}")
    axes[0,0].set(title='Raw training losses after update 1000',xlabel='Update',ylabel='Training loss')
    axes[0,1].set(title=f'Raw detail around largest baseline residual: {peak}',xlabel='Update',ylabel='Training loss')
    axes[1,0].set(title='Deviation from local median',xlabel='Update',ylabel='Loss minus 251-step median',ylim=(-.16,.16))
    axes[1,0].axhline(.15,color='gray',ls=':',lw=.8)
    axes[1,1].set(title='Local fluctuation distributions',xlabel='Loss minus local median',ylabel='Density')
    for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Baseline / A / B: raw losses have similar local fluctuations')
    fig.savefig(OUT.with_suffix('.png'),dpi=180)
    plt.close(fig)
    print(json.dumps(diagnostics,indent=2))


if __name__=='__main__':main()
