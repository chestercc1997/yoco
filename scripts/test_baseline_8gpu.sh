#!/usr/bin/env bash
# Original YOCO baseline: real-data startup and checkpoint recovery check.
set -euo pipefail
source "$(dirname -- "$0")/activate_local.sh"
cd "$YOCO_ROOT"
run="${1:?Usage: bash scripts/test_baseline_8gpu.sh runs/unique-run-directory}"
config=configs/yoco_160m_baseline_8gpu.json
mkdir -p "$run"
exec 9>"$run/pilot.lock"
flock -n 9
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python - "$config" "$run" <<'PY'
import datetime, hashlib, json, subprocess, sys
from pathlib import Path
import torch
config,run=Path(sys.argv[1]),Path(sys.argv[2])
c=json.loads(config.read_text()); t=c['training']
base=json.loads(Path('configs/yoco_160m_plan_full_baseline_4gpu.json').read_text())
assert c['architecture']=='yoco' and 'memory_fusion' not in c
assert c['model']==base['model'] and c['seed']==base['seed']==42
for k in t:
    if k not in ('world_size','gradient_accumulation','validation_batches'):
        assert t[k]==base['training'][k],k
assert t['world_size']==8 and t['microbatch']==4 and t['gradient_accumulation']==4
assert t['world_size']*t['microbatch']*t['gradient_accumulation']*t['sequence_length']==262144
assert t['world_size']*t['microbatch']*t['validation_batches']==2048
def indices(t,step):
    return sorted(((step*t['gradient_accumulation']+j)*t['world_size']+rank)*t['microbatch']+k
                  for j in range(t['gradient_accumulation']) for rank in range(t['world_size']) for k in range(t['microbatch']))
for step in (0,49,50,99,39999):
    assert indices(t,step)==indices(base['training'],step)
manifest_path=Path(c['data']['manifest'])
manifest=json.loads(manifest_path.read_text())
verified=json.loads(Path('runs/setup/training-data-verification.json').read_text())
assert verified['passed'] and verified['data_verified']
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==verified['manifest_sha256']
assert manifest['ready'] and manifest['revision']==c['data']['revision']
for row in manifest['shards']:
    for split in ('train','validation'):
        assert Path(row[split]['path']).stat().st_size==row[split]['tokens']*4
usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
assert len(usage.splitlines())==8 and all(int(x.strip())<1000 for x in usage.splitlines()),'GPUs are busy; defer test'
assert torch.cuda.is_available() and torch.cuda.device_count()==8
devices=[torch.cuda.get_device_name(i) for i in range(8)]
assert all('A100' in name for name in devices),devices
record={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'running',
        'model':'original YOCO baseline','config':str(config),'devices':devices,'steps':100,
        'resume_at_step':50,'synthetic':False,'max_updates_schedule':40000,
        'global_batch_tokens':262144,'validation_sequences':2048,
        'manifest_sha256':verified['manifest_sha256'],'torch':torch.__version__,'cuda':torch.version.cuda}
(run/'pilot-status.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
for stop in 50 100; do
  python -m torch.distributed.run --standalone --nproc-per-node=8 -m runtime.train \
    --config "$config" --run-dir "$run" --stop-after "$stop" \
    --init-model checkpoints/initial/baseline-step00000000.pt
done
python - "$run" <<'PY'
import datetime, json, math, statistics, sys
from pathlib import Path
from runtime.train import sha256
run=Path(sys.argv[1]); all_rows=[]
for rank in range(8):
    rows=[json.loads(x) for x in (run/f'rank{rank}.jsonl').read_text().splitlines()]
    assert any(r['event']=='started' and r['step']==0 and not r['synthetic'] and r['world_size']==8 for r in rows)
    assert any(r['event']=='started' and r['step']==50 and r['resumed'] and not r['synthetic'] for r in rows)
    train=[r for r in rows if r['event']=='train']
    assert [r['step'] for r in train]==list(range(1,101)),rank
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) and r['tokens']==r['step']*262144 for r in train)
    assert all_rows or rank==0
    all_rows.append(rows)
rows=all_rows[0]; train=[r for r in rows if r['event']=='train']
valid=[r for r in rows if r['event']=='validation']
assert [r['step'] for r in valid]==[50,100] and all(math.isfinite(r['loss']) for r in valid)
stage=json.loads((run/'stage_complete.json').read_text())
assert stage=={'step':100,'formal_complete':False,'synthetic':False}
latest=json.loads((run/'latest.json').read_text())
assert latest['step']==100 and sha256(run/latest['file'])==latest['sha256']
steady=[r['seconds'] for r in train if r['step'] not in (1,2,3,4,5,51,52,53,54,55)]
result=json.loads((run/'pilot-status.json').read_text())
result.update(state='passed',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              first_loss=train[0]['loss'],last_loss=train[-1]['loss'],
              first_10_mean_loss=statistics.mean(r['loss'] for r in train[:10]),
              last_10_mean_loss=statistics.mean(r['loss'] for r in train[-10:]),
              validation=valid,total_tokens=train[-1]['tokens'],
              median_step_seconds=statistics.median(steady),
              median_tokens_per_second=262144/statistics.median(steady),
              max_peak_memory_gib=max(r['peak_memory_bytes'] for rows in all_rows for r in rows if r['event']=='train')/2**30,
              gradient_overflow_events=sum(r['event']=='overflow' for r in rows),
              resume_verified_all_ranks=True,checkpoint_sha256=latest['sha256'],checkpoint=str(run/latest['file']))
(run/'pilot-status.json').write_text(json.dumps(result,indent=2)+'\n')
(run/'PILOT_SUCCESS').touch()
print(json.dumps(result,indent=2),flush=True)
PY
