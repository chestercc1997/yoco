"""Verify release, paired configs and reuse the freshly SHA-verified corpus."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = root.parents[1]
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()
for name, sha in json.loads((root/'release_manifest.json').read_text()).items():
    assert digest(root/name) == sha, name
old = json.loads((base/'experiments/plan-conditional-kv-160m-v1/preflight.json').read_text())
assert old['passed']
assert digest(root/'data/fineweb_edu_10bt/manifest.json') == old['manifest_sha256']
for name, info in old['token_files'].items():
    st = Path(name).stat()
    assert st.st_size == info['size'] and st.st_mtime_ns == info['mtime_ns'], name
b = json.loads((root/'configs/yoco_160m_plan_ab_A_4gpu.json').read_text())
d = json.loads((root/'configs/yoco_160m_plan_ab_B_4gpu.json').read_text())
for key in ('model','training','data','architecture','seed','tied_embeddings'): assert b[key] == d[key], key
t=b['training']; assert t['world_size']==4 and t['max_updates']==40000
assert t['world_size']*t['microbatch']*t['gradient_accumulation']*t['sequence_length']==262144
assert t['world_size']*t['microbatch']*t['validation_batches']==2048
assert d['memory_fusion']=={'mode':'conditional_kv','source_layers':[1,2,3,4,5,6],'temperature':1.0}
assert b['memory_fusion']=={'mode':'learned_kv','source_layers':[1,2,3,4,5,6],'temperature':1.0}
print('PASS A/B full paired configuration, release hashes and verified data identity', flush=True)

(root/'preflight.json').write_text(json.dumps(dict(passed=True,arms=['A','B'],steps=40000,world_size=4,initial_weights=[1/6]*6,manifest_sha256=digest(root/'data/fineweb_edu_10bt/manifest.json')),indent=2)+'\n')
