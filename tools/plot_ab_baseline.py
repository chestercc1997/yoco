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
labels={'baseline':'YOCO baseline (historical)','A':'A: learned KV','B':'B: conditional KV'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for arm in colors:
 r=[x for x in curves if x['arm']==arm];y=[float(x['validation_loss']) for x in r]
 for ax,key in [(axes[0,0],'tokens'),(axes[0,1],'compute_trunk_6nd_pflop_days'),(axes[1,0],'compute_output_inclusive_6nd_pflop_days'),(axes[1,1],'output_inclusive_plus_repeated_kv_proxy_pflop_days')]:
  x=np.array([float(v[key]) for v in r]);x=x/1e9 if key=='tokens' else x
  ax.plot(x,y,label=labels[arm],color=colors[arm],lw=1.4);ax.grid(alpha=.2);ax.set_ylabel('Validation cross-entropy')
axes[0,0].set_xlabel('Training tokens (billions)');axes[0,0].set_title('Full training trajectory');axes[0,0].legend()
for ax in [axes[0,1],axes[1,0],axes[1,1]]:ax.set_ylim(3.15,3.40)
axes[0,1].set_xlabel('Trunk 6ND (PFLOP-days)');axes[0,1].set_title('Excludes tied embedding / output matrix')
axes[1,0].set_xlabel('Output-inclusive 6ND (PFLOP-days)');axes[1,0].set_title('Counts shared output matrix once')
axes[1,1].set_xlabel('6ND + repeated KV projection proxy (PFLOP-days)');axes[1,1].set_title('Partial operation correction; not device FLOPs')
fig.suptitle('160M YOCO: baseline / A / B, 40,000 updates, single seed')
for ext in ['png','pdf']:fig.savefig(ROOT/('loss_compute.'+ext))
plt.close(fig)
fig,ax=plt.subplots(figsize=(12,4.6),constrained_layout=True);rows=s['downstream'];x=np.arange(len(rows)+1)
for i,arm in enumerate(colors):
 y=[100*r[arm] for r in rows]+[100*s['macro'][arm]]
 ax.bar(x+(i-1)*.24,y,.24,color=colors[arm],label=labels[arm])
ax.set_xticks(x,[r['task'].replace('_','\n') for r in rows]+['macro\naverage']);ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,76);ax.legend(ncol=3);ax.grid(axis='y',alpha=.2)
ax.set_title('Zero-shot full downstream evaluation; acc_norm except BoolQ / WinoGrande')
for ext in ['png','pdf']:fig.savefig(ROOT/('downstream.'+ext))
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
with (ROOT/'ab_training_curves.csv').open() as f:tr=list(csv.DictReader(f))
for arm in ['A','B']:
 r=[v for v in tr if v['arm']==arm];y=np.array([float(v['loss']) for v in r]);x=np.array([int(v['step']) for v in r])*262144/1e9
 axes[0].plot(x[99:],np.convolve(y,np.ones(100)/100,mode='valid'),color=colors[arm],label=labels[arm],lw=1)
axes[0].set_ylim(3.1,4.2);axes[0].set_xlabel('Training tokens (billions)');axes[0].set_ylabel('Training loss (trailing 100 updates)');axes[0].legend();axes[0].grid(alpha=.2)
for i,arm in enumerate(colors):
 val=s['training'][arm]['last100_training_loss'];axes[1].bar(i,val,color=colors[arm]);axes[1].text(i,val+.00015,f'{val:.6f}',ha='center')
axes[1].set_xticks(range(3),['baseline','A','B']);axes[1].set_ylim(3.145,3.159);axes[1].set_ylabel('Final 100-update mean training loss');axes[1].grid(axis='y',alpha=.2)
fig.suptitle('Training loss: historical baseline has final summary only in this export')
for ext in ['png','pdf']:fig.savefig(ROOT/('training_loss.'+ext))
plt.close(fig)
print('Generated six figure files')
