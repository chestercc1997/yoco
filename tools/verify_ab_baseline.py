"""Check the public numerical evidence and its release manifest."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

root=Path(__file__).resolve().parents[1]/'evaluation/ab_baseline_20261104'
s=json.loads((root/'summary.json').read_text())
def rows(name):
    with (root/name).open() as f:return list(csv.DictReader(f))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12),(a,b)
curves=rows('validation_curves.csv');train=rows('ab_training_curves.csv');scores=rows('ab_paired_scores.csv')
assert len(scores)==21465 and len(curves)==480 and len(train)==80000
for arm,t in s['training'].items():
    assert t['final_step']==40000 and t['tokens']==40000*262144
    assert t['trunk_parameters']==t['parameters']-77021184
    close(t['compute_trunk_6nd_pflop_days'],6*t['trunk_parameters']*t['tokens']/8.64e19)
    close(t['compute_output_inclusive_6nd_pflop_days'],6*t['parameters']*t['tokens']/8.64e19)
    extra=0 if arm=='baseline' else 6*5*2*768**2*t['tokens']/8.64e19
    close(t['extra_repeated_kv_projection_pflop_days'],extra)
    close(t['output_inclusive_plus_repeated_kv_proxy_pflop_days'],t['compute_output_inclusive_6nd_pflop_days']+extra)
    close(t['active_gpu_hours'],t['active_seconds']*t['world_size']/3600)
    cr=[r for r in curves if r['arm']==arm]
    assert [int(r['step']) for r in cr]==list(range(250,40001,250))
    close(cr[-1]['validation_loss'],t['validation_loss'])
    if arm!='baseline':
        tr=[r for r in train if r['arm']==arm]
        assert [int(r['step']) for r in tr]==list(range(1,40001))
        close(statistics.mean(float(r['loss']) for r in tr[-100:]),t['last100_training_loss'])
    close(statistics.mean(r[arm] for r in s['downstream']),s['macro'][arm])
for task in s['downstream']:
    rs=[r for r in scores if r['task']==task['task']]
    assert len(rs)==task['examples'] and len({r['doc_id'] for r in rs})==len(rs)
    for arm in 'AB':close(statistics.mean(float(r[arm+'_correct']) for r in rs),task[arm])
for line in (root/'MANIFEST.sha256').read_text().splitlines():
    digest,name=line.split('  ',1);p=Path(name)
    assert not p.is_absolute() and '..' not in p.parts
    assert hashlib.sha256((root/p).read_bytes()).hexdigest()==digest,name
print('Verified: 24 scores, 480 validation points, 80,000 training points, 21,465 paired samples, 6ND and manifest')
