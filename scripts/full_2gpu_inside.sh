#!/usr/bin/env bash
set -euo pipefail
release=$1
config=$2
run=$3
base=/data/share/jzhou945/cchen099/yoco
py="$base/.venv/bin/python"
cd "$release"
exec 9>"$run/allocation.lock"
flock -n 9
test ! -f "$run/full_complete.json"
"$py" tools/preflight_fusion.py --config "$config" --check-only
"$py" - "$run" <<'PY'
import json,sys,torch
from pathlib import Path
from runtime.train import latest_checkpoint
run=Path(sys.argv[1]);origin=json.loads((run/'resume_origin.json').read_text())
assert torch.__version__.split('+')[0]=='2.7.1'
assert torch.cuda.device_count()==2
names=[torch.cuda.get_device_name(i) for i in range(2)]
assert all('H100' in n for n in names),names
path=latest_checkpoint(run,origin['fingerprint']);assert path
state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
assert 1000<=state['step']<40000 and state['world_size']==2
result=dict(devices=names,torch=torch.__version__,cuda=torch.version.cuda,resume_step=state['step'],release=str(Path.cwd()))
(run/'full_gpu_environment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
PY
for stop in 10000 20000 30000 40000; do
  "$py" -m torch.distributed.run --standalone --nproc-per-node=2 -m runtime.train \
    --config "$config" --run-dir "$run" --stop-after "$stop"
  "$py" tools/probe_fusion_routing.py --config "$config" --run-dir "$run"
  "$py" - "$run" "$stop" <<'PY'
import json,os,shutil,sys
from pathlib import Path
run=Path(sys.argv[1]);step=int(sys.argv[2]);meta=json.loads((run/'latest.json').read_text())
assert meta['step']>=step
if meta['step']==step:
    folder=run/'milestones'/str(step);folder.mkdir(parents=True,exist_ok=True)
    source=run/meta['file'];dest=folder/source.name
    if not dest.exists():os.link(source,dest)
    shutil.copy2(source.with_suffix('.json'),dest.with_suffix('.json'))
PY
done
"$py" - "$run" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]);stage=json.loads((run/'stage_complete.json').read_text())
assert stage['step']==40000 and stage['formal_complete'] and not stage['synthetic']
rows=[json.loads(x) for x in (run/'rank0.jsonl').read_text().splitlines()]
result=dict(step=40000,full_training=True,tokens=40000*262144,resume_step=1000,
            validation=[r for r in rows if r['event']=='validation'],
            overflows=sum(r['event']=='overflow' for r in rows))
(run/'full_complete.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
PY
