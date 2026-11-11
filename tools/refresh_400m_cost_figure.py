"""Wrap the publication cost-figure titles without changing any experiment data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = ROOT / 'evaluation/gret_400m_3way_20261108T194918Z'
    training = json.loads((folder / 'summary.json').read_text())['training']
    arms = ('baseline','A','B');colors = ('#64748b','#007f86','#cc5a36')
    plt.rcParams.update({'font.size':10,'figure.dpi':160,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.hashsalt':'yoco-400m-cost-publication'})
    fig, axes = plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax, field, title in zip(axes, ('tokens_per_second','active_gpu_hours','peak_rank0_memory_gib'),
        ('Successful updates\ntokens / second','Successful updates\nGPU-hours','Peak rank0 memory\nGiB')):
        ax.bar(arms,[training[a][field] for a in arms],color=colors)
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    for ext in ('png','pdf','svg'):
        metadata = {'Date':None} if ext=='svg' else ({'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
        fig.savefig(folder/f'figures/06_execution_cost.{ext}',bbox_inches='tight',metadata=metadata)
        if ext=='svg':
            path=folder/'figures/06_execution_cost.svg'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':main()
