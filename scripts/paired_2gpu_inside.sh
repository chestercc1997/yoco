#!/usr/bin/env bash
set -euo pipefail
release=$1
config=$2
run=$3
base=/data/share/jzhou945/cchen099/yoco
py="$base/.venv/bin/python"
cd "$release"
mkdir -p "$run"
exec 9>"$run/allocation.lock"
flock -n 9
test ! -f "$run/pilot_complete.json"
"$py" tools/preflight_fusion.py --config "$config" --check-only
nvidia-smi > "$run/nvidia-smi.txt"
"$py" - "$run" <<'PY'
import json,sys,torch
from pathlib import Path
assert torch.__version__.split('+')[0]=='2.7.1'
assert torch.cuda.device_count()==2
names=[torch.cuda.get_device_name(i) for i in range(2)]
assert all('H100' in n for n in names),names
result=dict(torch=torch.__version__,cuda=torch.version.cuda,devices=names,release=str(Path.cwd()))
(Path(sys.argv[1])/'gpu_environment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
PY
"$py" tests/test_memory_fusion.py
for stop in 1 2; do
  "$py" -m torch.distributed.run --standalone --nproc-per-node=2 -m runtime.train \
    --config "$config" --run-dir "$run/smoke" --smoke --stop-after "$stop"
done
"$py" - "$run/smoke" <<'PY'
import json,sys
from pathlib import Path
for rank in range(2):
    rows=[json.loads(x) for x in (Path(sys.argv[1])/f'rank{rank}.jsonl').read_text().splitlines()]
    assert any(r['event']=='started' and r['resumed'] and r['step']==1 for r in rows)
    assert any(r['event']=='train' and r['step']==2 for r in rows)
print('PASS two-rank smoke and resume',flush=True)
PY
"$py" tools/probe_fusion_routing.py --config "$config" --run-dir "$run" --initial
# Identical restart boundaries for both arms. Optimizer and RNG are restored.
for stop in 250 500 1000; do
  "$py" -m torch.distributed.run --standalone --nproc-per-node=2 -m runtime.train \
    --config "$config" --run-dir "$run" --stop-after "$stop"
  "$py" tools/probe_fusion_routing.py --config "$config" --run-dir "$run"
done
"$py" - "$run" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]);rows=[json.loads(x) for x in (run/'rank0.jsonl').read_text().splitlines()]
assert json.loads((run/'stage_complete.json').read_text())['step']==1000
assert [r['step'] for r in rows if r['event']=='train']==list(range(1,1001))
result=dict(step=1000,full_training=False,validation=[r for r in rows if r['event']=='validation'],
            overflows=sum(r['event']=='overflow' for r in rows),
            routing_reports=[p.name for p in sorted(run.glob('routing_step_*.json'))])
(run/'pilot_complete.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
PY
