"""Render the nine completed models with one common 6ND compute convention."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('baseline', 'A', 'B')
TASKS = {'arc_easy':2376, 'arc_challenge':1172, 'boolq':3270, 'hellaswag':10042,
         'openbookqa':500, 'piqa':1838, 'winogrande':1267, 'sciq':1000}
REPORTS = (('gRet', '160M', 'ab_baseline_20261104', 768, 6),
           ('gRet', '400M', 'gret_400m_3way_20261108T194918Z', 1024, 12),
           ('SWA', '160M', 'swa_160m_ab_20261107', 768, 6))
COLORS = {'baseline':'#64748b', 'A':'#007f86', 'B':'#cc5a36'}
MARKERS = {'baseline':'o', 'A':'^', 'B':'s'}
COEFFICIENT = 6
TOKENS = 10485760000


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def csvwrite(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def load_sources():
    models = [];curves = {};sources = {}
    for family, size, name, dim, count in REPORTS:
        folder = ROOT / 'evaluation' / name;s = read(folder/'summary.json')
        assert s.get('complete', True)
        for file in ('summary.json','training_curves.csv','validation_curves.csv','downstream.csv'):
            sources[str((folder/file).relative_to(ROOT))] = sha(folder/file)
        train = {a:{} for a in ARMS};valid = {a:{} for a in ARMS}
        for filename, target in (('training_curves.csv', train), ('validation_curves.csv', valid)):
            with (folder/filename).open() as stream:
                for row in csv.DictReader(stream):
                    arm = row['arm'];step = int(row['step'])
                    assert arm in ARMS and step not in target[arm], (name, arm, step)
                    assert int(row['tokens']) == step*262144
                    target[arm][step] = row
        tasks = {e['task']:e for e in s['downstream']};assert set(tasks) == set(TASKS)
        for task, documents in TASKS.items():
            e = tasks[task];assert e.get('documents',e.get('examples')) == documents
            assert e['metric'] == ('acc,none' if task in ('boolq','winogrande') else 'acc_norm,none')
        for arm in ARMS:
            key = f'{family} {size} {arm}';t = s['training'][arm]
            assert sorted(train[arm]) == list(range(1,40001))
            assert sorted(valid[arm]) == list(range(250,40001,250))
            raw = [float(train[arm][i]['loss']) for i in range(1,40001)]
            validation = [float(valid[arm][i]['validation_loss']) for i in range(250,40001,250)]
            assert all(math.isfinite(x) for x in raw+validation)
            assert math.isclose(statistics.mean(raw[-100:]),t['last100_training_loss'],abs_tol=1e-12)
            assert validation[-1] == t['validation_loss'] and t['tokens'] == TOKENS
            params = t['parameters'];embedding = 100288*dim
            compute = COEFFICIENT*(params-embedding)*TOKENS/8.64e19
            old_compute = t.get('trunk_6nd_pflop_days',t.get('compute_trunk_6nd_pflop_days'))
            assert math.isclose(compute,old_compute,rel_tol=1e-12)
            for step, row in valid[arm].items():
                stored = float(row.get('trunk_6nd_pflop_days',row.get('compute_trunk_6nd_pflop_days')))
                assert math.isclose(stored,compute*step/40000,rel_tol=1e-12)
            extra = 0. if arm=='baseline' else COEFFICIENT*(count-1)*2*dim**2*TOKENS/8.64e19
            old_extra = t.get('extra_repeated_kv_projection_proxy_pflop_days',t.get('extra_repeated_kv_projection_pflop_days'))
            assert math.isclose(extra,old_extra,rel_tol=1e-12,abs_tol=1e-15)
            macro = statistics.mean(tasks[k][arm] for k in TASKS)
            assert abs(macro-s['macro'][arm]) < 1e-12
            models.append(dict(model=key,family=family,size=size,arm=arm,steps=40000,tokens=TOKENS,
                parameters=params,shared_embedding_parameters=embedding,trunk_parameters=params-embedding,
                compute_coefficient=COEFFICIENT,trunk_6nd_pflop_days=compute,
                output_inclusive_6nd_pflop_days=COEFFICIENT*params*TOKENS/8.64e19,
                extra_repeated_kv_projection_proxy_pflop_days=extra,
                final_training_loss=raw[-1],last100_training_loss=statistics.mean(raw[-100:]),
                final_validation_loss=validation[-1],macro_accuracy=macro,
                relative_gain_percent=100*(macro/s['macro']['baseline']-1),
                task_scores={k:tasks[k][arm] for k in TASKS},source_report=str(folder.relative_to(ROOT))))
            curves[key] = dict(raw=raw,valid=validation)
    assert len(models) == 9
    return models, curves, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'evaluation/model_comparison_20261111')
    args = parser.parse_args();out = args.output.resolve();(out/'figures').mkdir(parents=True,exist_ok=True)
    models, curves, sources = load_sources()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'figure.dpi':160,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.hashsalt':'yoco-nine-models-20261111','pdf.fonttype':42})
    figures = []
    def save(fig,name):
        for ext in ('png','pdf','svg'):
            metadata = {'Date':None} if ext=='svg' else ({'CreationDate':None,'ModDate':None} if ext=='pdf' else {})
            fig.savefig(out/f'figures/{name}.{ext}',bbox_inches='tight',metadata=metadata)
            if ext=='svg':
                path=out/f'figures/{name}.{ext}'
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig);figures.append(name)
    for family in ('gRet','SWA'):
        group = [m for m in models if m['family']==family]
        sizes = list(dict.fromkeys(m['size'] for m in group))
        for kind in ('train','valid'):
            fig,axes = plt.subplots(len(sizes),2,figsize=(12,3.6*len(sizes)),squeeze=False,layout='constrained')
            for i,size in enumerate(sizes):
                selection = [m for m in group if m['size']==size];late = []
                for m in selection:
                    y = np.array(curves[m['model']]['raw' if kind=='train' else 'valid'])
                    x = np.arange(1,40001) if kind=='train' else np.arange(250,40001,250)
                    late.extend(y[x>=30000])
                    for ax in axes[i]:ax.plot(x,y,color=COLORS[m['arm']],label=m['arm'],
                        lw=.45 if kind=='train' else 1.25,alpha=.6 if kind=='train' else 1.)
                axes[i,0].set_title(f'{family} {size}: all updates');axes[i,1].set_title(f'{family} {size}: final 10000 updates')
                axes[i,1].set_xlim(30000,40000)
                span=max(late)-min(late);axes[i,1].set_ylim(min(late)-max(span*.06,.0002),max(late)+max(span*.06,.0002))
                for ax in axes[i]:
                    ax.set_xlabel('Training updates');ax.set_ylabel('Raw training loss' if kind=='train' else 'Validation loss')
                    ax.grid(alpha=.2);ax.legend(ncol=3)
            fig.suptitle(f'YOCO-{family}: '+('unsmoothed training loss' if kind=='train' else 'all 160 scheduled validation points'))
            index={('gRet','train'):'01',('SWA','train'):'02',('gRet','valid'):'03',('SWA','valid'):'04'}[(family,kind)]
            save(fig,f'{index}_{kind}_loss_{family.lower()}')
        # Endpoints only: precisely six gRet or three SWA data points, with no jitter.
        fig = plt.figure(figsize=(11,6.5 if family=='gRet' else 5.8),layout='constrained')
        grid = fig.add_gridspec(2,1,height_ratios=[3.5,1.55 if family=='gRet' else 1.1])
        ax = fig.add_subplot(grid[0]);table_ax = fig.add_subplot(grid[1]);table_ax.axis('off')
        labels = []
        if family=='gRet':
            positions = [(.07,.90),(.07,.66),(.07,.40),(.66,.70),(.66,.40),(.66,.12)]
        else:positions = [(.05,.90),(.05,.40),(.67,.15)]
        for m,position in zip(group,positions):
            x=m['trunk_6nd_pflop_days'];y=m['final_validation_loss']
            ax.scatter([x],[y],s=155 if m['arm']=='A' else 70,
                facecolors='none' if m['arm']=='A' else COLORS[m['arm']],
                edgecolors=COLORS[m['arm']],linewidths=1.6,marker=MARKERS[m['arm']],zorder=4)
            ax.annotate(f"{m['size']} {m['arm']}\nloss {y:.6f}",xy=(x,y),xytext=position,textcoords='axes fraction',
                color=COLORS[m['arm']],fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2),
                arrowprops=dict(arrowstyle='-',color=COLORS[m['arm']],lw=.8))
            labels.append([m['size']+' '+m['arm'],f'{x:.9f}',f'{y:.6f}'])
        ax.set_xlim(0,max(m['trunk_6nd_pflop_days'] for m in group)*1.25)
        ys=[m['final_validation_loss'] for m in group];span=max(ys)-min(ys)
        ax.set_ylim(min(ys)-max(span*.18,.0005),max(ys)+max(span*.20,.0005))
        ax.set_xlabel('Common compute proxy: 6 x N_trunk x D (PFLOP-days)')
        ax.set_ylabel('Final validation loss (lower is better)');ax.grid(alpha=.2)
        ax.set_title(f'YOCO-{family}: {len(group)} final checkpoint points; common coefficient = 6')
        tab=table_ax.table(cellText=labels,colLabels=['Model','Trunk 6ND (PFLOP-days)','Final validation loss'],loc='center',cellLoc='center')
        tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.25)
        fig.supxlabel('Shared embedding excluded once. Proxy excludes repeated K/V, attention and execution overhead; no measured-FLOP claim.',fontsize=8)
        save(fig,('05' if family=='gRet' else '06')+f'_compute_loss_{family.lower()}')
        fig,axes=plt.subplots(len(sizes),1,figsize=(14,4*len(sizes)),squeeze=False,layout='constrained')
        for i,size in enumerate(sizes):
            ax=axes[i,0];x=np.arange(9)
            for j,m in enumerate(m for m in group if m['size']==size):
                values=[100*m['task_scores'][t] for t in TASKS]+[100*m['macro_accuracy']]
                bars=ax.bar(x+(j-1)*.25,values,.25,label=m['arm'],color=COLORS[m['arm']])
                ax.text(bars[-1].get_x()+.125,values[-1]+1.3+j*3.5,f'{values[-1]:.3f}',
                    ha='center',fontsize=8,color=COLORS[m['arm']])
            ax.set_xticks(x,['ARC-e','ARC-c','BoolQ','HellaSwag','OBQA','PIQA','WinoGrande','SciQ','Macro'])
            ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);ax.legend(ncol=3)
            ax.set_title(f'{family} {size}: eight full zero-shot tasks and equal-weight macro')
        fig.supxlabel('BoolQ / WinoGrande: acc. Other tasks: acc_norm. 21465 documents per model. Higher is better.',fontsize=9)
        save(fig,('07' if family=='gRet' else '08')+f'_downstream_{family.lower()}')
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained');y=np.arange(9)
    for i,m in enumerate(models):
        axes[0].scatter(100*m['macro_accuracy'],i,color=COLORS[m['arm']],s=50)
        axes[0].annotate(f"{100*m['macro_accuracy']:.4f}%",(100*m['macro_accuracy'],i),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
        axes[1].barh(i,m['relative_gain_percent'],color=COLORS[m['arm']])
        axes[1].annotate(f"{m['relative_gain_percent']:+.4f}%",(max(m['relative_gain_percent'],0),i),
            xytext=(7,0),textcoords='offset points',va='center',ha='left',fontsize=9)
    for ax in axes:ax.set_yticks(y,[m['model'] for m in models]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    axes[0].set_xlim(45,49.8);axes[0].set_title('Overall accuracy: eight-task macro (%)')
    axes[1].set_xlim(-1.5,3.6);axes[1].axvline(0,color='#334155',lw=.7);axes[1].set_title('Relative gain over own baseline (%)')
    fig.supxlabel('Relative gain = (method macro / same-family, same-size baseline macro - 1) x 100%. Single seed.',fontsize=9)
    save(fig,'09_downstream_macro_and_relative_gain')
    write(out/'summary.json',dict(complete=True,models=models,compute_convention=dict(coefficient=6,
        N='total parameters minus one tied embedding/output matrix',D=TOKENS,unit='PFLOP-days',divisor=8.64e19,
        measured_flops=False,gret_endpoint_points=6,swa_endpoint_points=3),source_sha256=sources,figures=figures))
    csvwrite(out/'model_summary.csv',[{k:v for k,v in m.items() if k!='task_scores'} for m in models])
    csvwrite(out/'compute_loss_points.csv',[{k:m[k] for k in ('model','compute_coefficient','trunk_parameters','tokens',
        'trunk_6nd_pflop_days','output_inclusive_6nd_pflop_days','extra_repeated_kv_projection_proxy_pflop_days','final_validation_loss')} for m in models])
    csvwrite(out/'downstream.csv',[dict(model=m['model'],**m['task_scores'],macro_accuracy=m['macro_accuracy'],relative_gain_percent=m['relative_gain_percent']) for m in models])
    print(f'Validated 360000 raw training points / 1440 validation points / 72 full task scores; wrote {len(figures)} figures in three formats to {out}',flush=True)


if __name__=='__main__':main()
