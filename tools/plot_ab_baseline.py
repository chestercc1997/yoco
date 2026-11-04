"""Rebuild figures from the public, sanitized comparison tables."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'evaluation/ab_baseline_20261104'
s=json.loads((ROOT/'summary.json').read_text())
with (ROOT/'validation_curves.csv').open() as f: curves=list(csv.DictReader(f))
colors={'baseline':'#616b75','A':'#147d92','B':'#cf623b'}
labels={'baseline':'Baseline: 8×A100','A':'A: learned KV','B':'B: conditional KV'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})


def save_formats(fig, name):
 for ext in ('png','pdf','svg'):
  path=ROOT/f'{name}.{ext}'
  fig.savefig(path)
  if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for arm in colors:
 r=[x for x in curves if x['arm']==arm];y=[float(x['validation_loss']) for x in r]
 for ax,key in [(axes[0,0],'tokens'),(axes[0,1],'compute_trunk_6nd_pflop_days'),(axes[1,0],'compute_output_inclusive_6nd_pflop_days'),(axes[1,1],'output_inclusive_plus_repeated_kv_proxy_pflop_days')]:
  x=np.array([float(v[key]) for v in r]);x=x/1e9 if key=='tokens' else x
  ax.plot(x,y,label=labels[arm],color=colors[arm],lw=1.4);ax.grid(alpha=.2);ax.set_ylabel('Validation cross-entropy')
axes[0,0].set_xlabel('Training tokens (billions)');axes[0,0].set_title('Full training trajectory');axes[0,0].legend()
late=[float(x['validation_loss']) for x in curves if float(x['tokens'])>=7e9]
for ax in [axes[0,1],axes[1,0],axes[1,1]]:ax.set_ylim(min(late)-.025,max(late)+.035)
axes[0,1].set_xlabel('Trunk 6ND (PFLOP-days)');axes[0,1].set_title('Excludes tied embedding / output matrix')
axes[1,0].set_xlabel('Output-inclusive 6ND (PFLOP-days)');axes[1,0].set_title('Counts shared output matrix once')
axes[1,1].set_xlabel('6ND + repeated KV projection proxy (PFLOP-days)');axes[1,1].set_title('Partial operation correction; not device FLOPs')
fig.suptitle('160M YOCO: 8×A100 baseline / A / B, 40,000 updates, single seed')
save_formats(fig,'loss_compute')
plt.close(fig)
fig,ax=plt.subplots(figsize=(12,4.6),constrained_layout=True);rows=s['downstream'];x=np.arange(len(rows)+1)
for i,arm in enumerate(colors):
 y=[100*r[arm] for r in rows]+[100*s['macro'][arm]]
 ax.bar(x+(i-1)*.24,y,.24,color=colors[arm],label=labels[arm])
ax.set_xticks(x,[r['task'].replace('_','\n') for r in rows]+['macro\naverage']);ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,76);ax.legend(ncol=3);ax.grid(axis='y',alpha=.2)
ax.set_title('Zero-shot full downstream evaluation; acc_norm except BoolQ / WinoGrande')
save_formats(fig,'downstream')
plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
with (ROOT/'training_curves.csv').open() as f:tr=list(csv.DictReader(f))
smoothed={}
for arm in colors:
 r=[v for v in tr if v['arm']==arm];y=np.array([float(v['loss']) for v in r]);x=np.array([int(v['step']) for v in r])*262144/1e9
 smoothed[arm]=np.convolve(y,np.ones(100)/100,mode='valid')
 for ax in axes[0]:ax.plot(x[99:],smoothed[arm],color=colors[arm],label=labels[arm],lw=1)
axes[0,0].set_title('Complete training trajectory');axes[0,0].legend()
axes[0,1].set_title('Late-training detail');axes[0,1].set_xlim(7,10.6)
late=[y[x[99:]>=7] for y in smoothed.values()]
axes[0,1].set_ylim(min(v.min() for v in late)-.025,max(v.max() for v in late)+.035)
for arm in ['A','B']:
 axes[1,0].plot(x[99:],smoothed[arm]-smoothed['baseline'],color=colors[arm],label=labels[arm],lw=1)
axes[1,0].axhline(0,color=colors['baseline'],ls='--',lw=1);axes[1,0].set_xlim(1,10.6)
axes[1,0].set_title('Difference from the new baseline');axes[1,0].set_ylabel('Training loss difference')
for i,arm in enumerate(colors):
 val=s['training'][arm]['last100_training_loss'];axes[1,1].scatter(val,i,s=85,color=colors[arm]);axes[1,1].annotate(f'{val:.6f}',(val,i),xytext=(0,10),textcoords='offset points',ha='center')
vals=[s['training'][a]['last100_training_loss'] for a in colors];span=max(vals)-min(vals)
axes[1,1].set_xlim(min(vals)-span*.25,max(vals)+span*.35);axes[1,1].set_ylim(2.5,-.7)
axes[1,1].set_yticks(range(3),['baseline','A','B']);axes[1,1].set_xlabel('Cross-entropy');axes[1,1].set_title('Final 100-update mean training loss')
for ax in axes.flat:ax.grid(alpha=.2)
for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.set_xlabel('Training tokens (billions)')
for ax in axes[0]:ax.set_ylabel('Training loss (trailing 100 updates)')
fig.suptitle('All three 40,000-update curves: baseline 8×A100; A/B 8×A800')
save_formats(fig,'training_loss')
plt.close(fig)
print('Generated three figures in PNG / PDF / SVG')
