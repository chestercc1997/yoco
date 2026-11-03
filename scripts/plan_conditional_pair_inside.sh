#!/usr/bin/env bash
set -euo pipefail
release=/data/share/jzhou945/cchen099/yoco/experiments/plan-conditional-kv-160m-v1
base=/data/share/jzhou945/cchen099/yoco
py="$base/.venv/bin/python"
cd "$release"
"$py" tools/preflight_fusion.py --config configs/yoco_160m_conditional_kv_2gpu.json --check-only
"$py" tests/test_conditional_kv.py
"$py" -c 'import torch; assert torch.cuda.device_count() == 4; print([torch.cuda.get_device_name(i) for i in range(4)], flush=True)'
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
test "${#devices[@]}" -eq 4
run_arm() {
  local arm=$1 config=$2 visible=$3 run="$base/runs/yoco-160m-plan-conditional-kv-v1-$1"
  mkdir -p "$run"
  exec 9>"$run/allocation.lock"; flock -n 9
  test ! -f "$run/pilot_complete.json"
  export CUDA_VISIBLE_DEVICES="$visible"
  for stop in 1 2; do
    "$py" -m torch.distributed.run --standalone --nproc-per-node=2 -m runtime.train --config "$config" --run-dir "$run/smoke" --smoke --stop-after "$stop"
  done
  "$py" - "$run/smoke" <<'PY'
import json,sys
from pathlib import Path
for rank in range(2):
    rows=[json.loads(x) for x in (Path(sys.argv[1])/f'rank{rank}.jsonl').read_text().splitlines()]
    assert any(r['event']=='started' and r['resumed'] and r['step']==1 for r in rows)
    assert any(r['event']=='train' and r['step']==2 for r in rows)
PY
  for stop in 250 500 1000; do
    "$py" -m torch.distributed.run --standalone --nproc-per-node=2 -m runtime.train --config "$config" --run-dir "$run" --stop-after "$stop"
  done
  "$py" - "$run" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]); rows=[json.loads(x) for x in (run/'rank0.jsonl').read_text().splitlines()]
assert json.loads((run/'stage_complete.json').read_text())['step']==1000
assert [r['step'] for r in rows if r['event']=='train']==list(range(1,1001))
(run/'pilot_complete.json').write_text(json.dumps(dict(step=1000, validation=[r for r in rows if r['event']=='validation'], overflows=sum(r['event']=='overflow' for r in rows)),indent=2)+'\n')
PY
}
run_arm baseline configs/yoco_160m_baseline_2gpu.json "${devices[0]},${devices[1]}" > "$base/logs/plan-kv-baseline-$SLURM_JOB_ID.log" 2>&1 & p1=$!
run_arm dynamic configs/yoco_160m_conditional_kv_2gpu.json "${devices[2]},${devices[3]}" > "$base/logs/plan-kv-dynamic-$SLURM_JOB_ID.log" 2>&1 & p2=$!
s1=0; wait "$p1" || s1=$?
s2=0; wait "$p2" || s2=$?
printf 'baseline=%s dynamic=%s\n' "$s1" "$s2"
((s1 == 0 && s2 == 0))
