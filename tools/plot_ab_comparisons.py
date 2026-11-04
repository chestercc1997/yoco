"""Make publication figures from the released three-arm numerical evidence."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'evaluation/ab_baseline_20261104'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
S = json.loads((ROOT / 'summary.json').read_text())
with (ROOT / 'validation_curves.csv').open() as f:
    CURVES = list(csv.DictReader(f))
with (ROOT / 'loss_thresholds.csv').open() as f:
    THRESHOLDS = list(csv.DictReader(f))
ARMS = ['baseline', 'A', 'B']
COLORS = {'baseline': '#64748b', 'A': '#007f86', 'B': '#cc5a36'}
LABELS = {'baseline': 'Baseline · 8×A100', 'A': 'A · learned KV', 'B': 'B · conditional KV'}
TASK_NAMES = ['ARC Easy', 'ARC Challenge', 'HellaSwag', 'PIQA', 'WinoGrande', 'BoolQ', 'OpenBookQA', 'SciQ', 'Macro average']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
    'axes.labelcolor': '#334155', 'text.color': '#172b40', 'axes.edgecolor': '#ccd5df',
    'xtick.color': '#475569', 'ytick.color': '#475569', 'figure.dpi': 180,
    'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.hashsalt': 'yoco-ab-20261104'})


def save(fig, name, footer):
    fig.text(.07, .025, footer, fontsize=9, color='#64748b', va='bottom')
    for ext in ['png', 'pdf', 'svg']:
        metadata = {'Date': None} if ext == 'svg' else ({'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {})
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', metadata=metadata)
        if ext == 'svg':
            path = OUT / f'{name}.{ext}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def curve(arm, field):
    return np.array([float(r[field]) for r in CURVES if r['arm'] == arm])


# Dot plots make small loss differences visible without truncated bars.
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
fig.subplots_adjust(left=.17, right=.97, bottom=.23, top=.75, wspace=.52)
fig.suptitle('Final quality at 10.486B training tokens', x=.07, ha='left', fontsize=19)
metrics = [('Validation loss ↓', 'validation_loss', False),
           ('Training loss ↓', 'last100_training_loss', False),
           ('Downstream macro ↑', None, True)]
for ax, (title, key, percent) in zip(axes, metrics):
    vals = [100*S['macro'][a] if percent else S['training'][a][key] for a in ARMS]
    span = max(vals)-min(vals)
    ax.set_xlim(min(vals)-span*.3, max(vals)+span*.4)
    for i, (arm, val) in enumerate(zip(ARMS, vals)):
        ax.scatter(val, i, s=100, color=COLORS[arm], zorder=3)
        ax.annotate(f'{val:.2f}%' if percent else f'{val:.6f}', (val, i), xytext=(0, 12),
                    textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    ax.set_yticks(range(3), [LABELS[a] for a in ARMS] if ax is axes[0] else [])
    ax.set_ylim(2.6, -.65); ax.set_title(title, loc='left', pad=17)
    ax.grid(axis='x', alpha=.2); ax.tick_params(axis='y', length=0)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    ax.locator_params(axis='x', nbins=3)
    ax.set_xlabel('Accuracy (%)' if percent else 'Cross-entropy')
save(fig, '01_quality_summary', '40,000 updates · single seed. Training loss: last 100 updates. Macro: eight tasks, equal weights.\nNew baseline: 8×A100; A/B: 8×A800. Same backbone and training hyperparameters; see the comparability audit.')

fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.9), gridspec_kw={'width_ratios': [1.1, 1]})
fig.subplots_adjust(left=.14, right=.96, bottom=.15, top=.81, wspace=.25)
fig.suptitle('Eight downstream tasks: scores and differences', x=.06, ha='left', fontsize=19)
rows = S['downstream'] + [{'task': 'macro', **S['macro']}]
y = np.arange(len(rows))
for idx, arm in enumerate(ARMS):
    vals = np.array([r[arm]*100 for r in rows])
    axes[0].scatter(vals, y+(idx-1)*.19, s=45, color=COLORS[arm], label=LABELS[arm], zorder=3)
    axes[0].scatter(vals[-1], y[-1]+(idx-1)*.19, s=70, marker='D', color=COLORS[arm], zorder=3)
for idx, arm in enumerate(['A', 'B']):
    delta=np.array([(r[arm]-r['baseline'])*100 for r in rows]); pos=y+(idx-.5)*.30
    axes[1].barh(pos, delta, height=.25, color=COLORS[arm], alpha=.95)
    for yy, val in zip(pos, delta):
        axes[1].text(val+(.04 if val>=0 else -.04), yy, f'{val:+.2f}', va='center',
                     ha='left' if val>=0 else 'right', fontsize=9, color=COLORS[arm])
for ax in axes:
    ax.set_ylim(8.6, -.6); ax.axhline(7.5, color='#cbd5e1', lw=1)
    ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True); ax.tick_params(axis='y', length=0)
axes[0].set_yticks(y, TASK_NAMES); axes[1].set_yticks(y, [])
axes[0].set_xlim(20, 70); axes[0].set_xlabel('Accuracy (%)'); axes[0].set_title('Full-task zero-shot accuracy', loc='left', pad=14)
axes[1].axvline(0, color='#64748b', lw=1)
all_deltas=[100*(r[a]-r['baseline']) for r in rows for a in ('A','B')]
delta_span=max(all_deltas)-min(all_deltas)
axes[1].set_xlim(min(0,min(all_deltas))-max(.8,delta_span*.14),max(all_deltas)+max(.9,delta_span*.19))
axes[1].set_xlabel('Difference from baseline (percentage points)'); axes[1].set_title('Positive means higher accuracy', loc='left', pad=14)
handles, labels = axes[0].get_legend_handles_labels();fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.96,.94), ncol=3, frameon=False, fontsize=10)
save(fig, '02_downstream_comparison', f"21,465 examples per arm; no limit. BoolQ / WinoGrande: acc; other tasks: acc_norm.\nA − new baseline: {100*(S['macro']['A']-S['macro']['baseline']):+.2f} pp; B − new baseline: {100*(S['macro']['B']-S['macro']['baseline']):+.2f} pp. Baseline differences are descriptive.")

fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3))
fig.subplots_adjust(left=.08, right=.97, bottom=.22, top=.78, wspace=.3)
fig.suptitle('Validation convergence at the same token budget', x=.07, ha='left', fontsize=19)
for arm in ARMS:
    x=curve(arm,'tokens')/1e9;loss=curve(arm,'validation_loss')
    axes[0].plot(x,loss,color=COLORS[arm],lw=2,label=LABELS[arm])
    if arm!='baseline':
        delta=loss-curve('baseline','validation_loss')
        axes[1].plot(x,delta,color=COLORS[arm],lw=1.8,label=LABELS[arm])
        axes[1].annotate(f'{delta[-1]:+.5f}',(x[-1],delta[-1]),xytext=(-5,14 if arm=='A' else -16),textcoords='offset points',ha='right',color=COLORS[arm],fontweight='bold')
late=np.concatenate([curve(a,'validation_loss')[curve(a,'tokens')>=7e9] for a in ARMS])
axes[0].set_xlim(7,10.65);axes[0].set_ylim(late.min()-.025,late.max()+.035);axes[0].set_ylabel('Validation cross-entropy');axes[0].set_title('Late training: un-smoothed observations',loc='left',pad=15)
diffs=np.concatenate([(curve(a,'validation_loss')-curve('baseline','validation_loss'))[curve(a,'tokens')>=1e9] for a in ('A','B')])
axes[1].axhline(0,color=COLORS['baseline'],ls='--',lw=1);axes[1].set_xlim(1,10.65);axes[1].set_ylim(min(0,diffs.min())-.025,max(0,diffs.max())+.025)
axes[1].set_ylabel('Loss difference from new baseline');axes[1].set_title('Lower than zero means lower loss',loc='left',pad=15)
for ax in axes:ax.set_xlabel('Training tokens (billions)');ax.grid(alpha=.2)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(.97,.94),ncol=3,frameon=False,fontsize=10)
save(fig,'03_validation_convergence','160 validation points per arm, every 250 updates. Left panel zooms into 7–10.486B tokens; right starts at 1B.\nThe baseline curve is the completed 8×A100 run. Axes include all three curves; single-seed comparisons.')

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), gridspec_kw={'width_ratios':[1,1,1.12]})
fig.subplots_adjust(left=.06,right=.98,bottom=.24,top=.78,wspace=.36)
fig.suptitle('Compute–loss comparison: the accounting convention matters', x=.05,ha='left',fontsize=19)
for arm in ARMS:
    for ax,key in zip(axes[:2],['compute_output_inclusive_6nd_pflop_days','output_inclusive_plus_repeated_kv_proxy_pflop_days']):
        ax.plot(curve(arm,key),curve(arm,'validation_loss'),color=COLORS[arm],lw=2,label=LABELS[arm])
for ax in axes[:2]:
    ax.set_xlim(.08,.127);ax.set_ylim(late.min()-.025,late.max()+.035);ax.set_xlabel('Estimated compute (PFLOP-days)');ax.grid(alpha=.2)
axes[0].set_ylabel('Validation cross-entropy');axes[0].set_title('6ND, including the shared output head',loc='left',fontsize=11,pad=15)
axes[1].set_title('6ND + repeated K/V projections',loc='left',fontsize=11,pad=15)
# Compare accounting proxies at the same sampled validation loss threshold.
targets=[3.5,S['training']['baseline']['validation_loss'],3.2]
compute_rows=[]
plotted_values=[]
for idx,arm in enumerate(['A','B']):
    values=[]
    for target in targets:
        found={r['arm']:r for r in THRESHOLDS if float(r['target_loss'])==target}
        step=int(found[arm]['first_observed_step']) if found[arm]['first_observed_step'] else None
        bstep=int(found['baseline']['first_observed_step']) if found['baseline']['first_observed_step'] else None
        c=S['training'][arm]['output_inclusive_plus_repeated_kv_proxy_pflop_days']*step/40000 if step else None
        bc=S['training']['baseline']['compute_output_inclusive_6nd_pflop_days']*bstep/40000 if bstep else None
        change=100*(c/bc-1) if c is not None and bc is not None else None
        values.append(change if change is not None else np.nan)
        if change is not None:plotted_values.append(change)
        compute_rows.append(dict(arm=arm,target_loss=target,first_observed_step=step,baseline_first_observed_step=bstep,corrected_proxy_pflop_days=c,baseline_proxy_pflop_days=bc,relative_change_percent=change,status='compared' if change is not None else 'baseline_not_reached' if not bstep else 'arm_not_reached'))
    pos=np.arange(3)+(idx-.5)*.26
    axes[2].barh(pos,values,height=.22,color=COLORS[arm])
    for yy,val in zip(pos,values):
        if np.isfinite(val):axes[2].text(val+(.12 if val>=0 else -.12),yy,f'{val:+.2f}%',va='center',ha='left' if val>=0 else 'right',fontsize=9,color=COLORS[arm])
        else:axes[2].text(.5,yy,'baseline not reached',transform=axes[2].get_yaxis_transform(),ha='center',va='center',fontsize=9,color=COLORS[arm])
axes[2].set_yticks(range(3),[f'{x:.6f}' for x in targets]);axes[2].set_ylim(2.55,-.55);axes[2].axvline(0,color='#64748b',lw=1)
vmin=min(0,min(plotted_values));vmax=max(0,max(plotted_values));vspan=max(vmax-vmin,1)
axes[2].set_xlim(vmin-vspan*.2,vmax+vspan*.4);axes[2].set_xlabel('Corrected proxy vs baseline (%)');axes[2].set_ylabel('Validation loss threshold');axes[2].set_title('At the first observed threshold crossing',loc='left',fontsize=11,pad=15);axes[2].grid(axis='x',alpha=.2);axes[2].set_axisbelow(True)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(.98,.94),ncol=3,frameon=False,fontsize=10)
save(fig,'04_compute_tradeoff','6ND is a proxy, not measured device FLOPs. The correction adds the five extra shared K/V projections in A/B.\nMissing baseline threshold crossings are marked explicitly; they are never represented by zero compute. Resolution: 250 updates.\nBaseline: 8×A100; A/B: 8×A800. These descriptive curves do not establish measured hardware efficiency.')
with (OUT/'compute_at_loss.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(compute_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(compute_rows)
print('Generated four figures in PNG, PDF and SVG:', OUT)
