"""Check full-run review rejects missing updates and incomplete continuation."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import compare_full_training as review

identity={'seed':42,'training':{'max_updates':40000}}
config=json.dumps({'identity':identity})
pilot=[dict(event='train',step=i,loss=5.,grad_norm=.5) for i in range(1,1001)]
full=[dict(event='train',step=i,loss=4.,grad_norm=.5) for i in range(1001,40001)]
for i in range(250,40001,250):
    (pilot if i<=1000 else full).append(dict(event='validation',step=i,loss=4.))
def read(archive,run,leaf):
    if leaf=='resolved_config.json':return config
    if leaf=='full_complete.json':return json.dumps({'step':40000,'full_training':True})
    if leaf=='stage_complete.json':return json.dumps({'step':40000,'formal_complete':True})
    if leaf=='resume_origin.json':return json.dumps({'step':1000})
    if leaf=='gradient_flow_probe.json':return '{}'
    if leaf=='rank0.jsonl':return '\n'.join(json.dumps(r) for r in (full if '-full-' in run else pilot))+'\n'
    raise AssertionError(leaf)
with patch.object(review,'read',read):
    r=review.load('fixture','baseline')
    assert len(r['train'])==40000 and len(r['validation'])==160
    full.pop(0)
    try:review.load('fixture','baseline')
    except AssertionError:pass
    else:raise AssertionError('Missing continuation step accepted')
print('PASS full review: pilot/continuation merge, validation coverage, missing-step rejection')
