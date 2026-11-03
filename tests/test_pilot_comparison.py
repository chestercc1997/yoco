"""Verify acceptance handles sampled crossings, worse runs and protocol mismatches."""
import copy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compare_paired_pilots import evaluate

def fixture(mode,values):
    c=dict(model={},seed=42,training={'world_size':2},tied_embeddings=True,architecture='yoco',memory_fusion={'mode':mode})
    return dict(config={'config':c,'identity':{'data':{},'model_source_sha256':{}}},
                train=[dict(step=i,seconds=1.,loss=6.,grad_norm=.5) for i in range(1,1001)],
                validation=dict(zip([250,500,750,1000],values)),gradient={},routing=[],overflow_count=0)
b=fixture('last',[7.,6.,5.5,5.])
d=fixture('token',[7.1,6.3,5.9,5.7])
r=evaluate(b,d)
assert r['verdict']=='FASTER_CONVERGENCE_NOT_ESTABLISHED'
assert r['target_loss_observations']['6.0']['baseline']['step_bracket']==[250,500]
assert r['target_loss_observations']['5.5']['dynamic']['first_observed_step'] is None
assert evaluate(b,fixture('token',[6.9,5.9,5.4,4.9]))['verdict']=='SHORT_RUN_SUPPORT_ONLY'
bad=copy.deepcopy(d);bad['config']['config']['seed']=43
try:evaluate(b,bad)
except AssertionError:pass
else:raise AssertionError('Mismatched protocol accepted')
print('PASS: convergence decisions, threshold brackets, censored targets, protocol rejection')
