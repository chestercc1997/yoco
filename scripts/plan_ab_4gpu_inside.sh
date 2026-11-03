#!/bin/bash
set -euo pipefail
base=/data/share/jzhou945/cchen099/yoco
release=$base/experiments/plan-ab-full-4gpu-20261103-v2
evalroot=$base/eval-20261102
py=$base/.venv/bin/python
cd "$release"
exec 9>"$release/allocation.lock"; flock -n 9
"$py" tools/preflight_plan_ab.py
"$py" tests/test_plan_ab_kv.py
"$py" -c 'import torch; assert torch.cuda.device_count()==4; print([torch.cuda.get_device_name(i) for i in range(4)],flush=True)'
for arm in A B; do
  config="configs/yoco_160m_plan_ab_${arm}_4gpu.json"
  run="$base/runs/yoco-160m-plan-ab-4gpu-20261103-v2-$arm"
  mkdir -p "$run"
  for stop in 1 2; do
    "$py" -m torch.distributed.run --standalone --nproc-per-node=4 -m runtime.train --config "$config" --run-dir "$run/smoke" --smoke --stop-after "$stop"
  done
  "$py" - "$run/smoke" <<'PY'
import json,sys
from pathlib import Path
for rank in range(4):
    rows=[json.loads(x) for x in (Path(sys.argv[1])/f'rank{rank}.jsonl').read_text().splitlines()]
    assert any(r['event']=='started' and r['resumed'] and r['step']==1 for r in rows)
    assert any(r['event']=='train' and r['step']==2 for r in rows)
PY
done
for arm in A B; do
  config="configs/yoco_160m_plan_ab_${arm}_4gpu.json"
  run="$base/runs/yoco-160m-plan-ab-4gpu-20261103-v2-$arm"
  "$py" -m torch.distributed.run --standalone --nproc-per-node=4 -m runtime.train --config "$config" --run-dir "$run" --stop-after 40000
  "$py" - "$run" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]); stage=json.loads((run/'stage_complete.json').read_text())
assert stage['step']==40000 and stage['formal_complete'] and not stage['synthetic']
assert (run/'checkpoint_00040000.pt').exists()
(run/'full_complete.json').write_text(json.dumps(stage,indent=2)+'\n')
PY
done
export PYTHONPATH="$evalroot/site:$release" HF_HOME="$evalroot/data/cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_TRUST_REMOTE_CODE=1
export TIKTOKEN_CACHE_DIR="$evalroot/data/cache/tiktoken" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
mkdir -p runs/eval-plan-ab/smoke runs/eval-plan-ab/results
run_eval() {
  local arm=$1 config=$2 device=$3 checkpoint=$4
  "$py" evaluation/run.py --locks-dir evaluation/plan_ab_data_locks --config "$config" --checkpoint "$checkpoint" --task arc_easy --limit 4 --output "runs/eval-plan-ab/smoke/$arm" --device "$device" > "runs/eval-plan-ab/smoke-$arm.log" 2>&1
  for task in arc_easy arc_challenge boolq hellaswag openbookqa piqa winogrande sciq; do
    "$py" evaluation/run.py --locks-dir evaluation/plan_ab_data_locks --config "$config" --checkpoint "$checkpoint" --task "$task" --output "runs/eval-plan-ab/results/$arm/$task" --device "$device" > "runs/eval-plan-ab/$arm-$task.log" 2>&1
  done
}
run_eval A configs/yoco_160m_plan_ab_A_4gpu.json cuda:0 "$base/runs/yoco-160m-plan-ab-4gpu-20261103-v2-A/checkpoint_00040000.pt" & p1=$!
run_eval B configs/yoco_160m_plan_ab_B_4gpu.json cuda:1 "$base/runs/yoco-160m-plan-ab-4gpu-20261103-v2-B/checkpoint_00040000.pt" & p2=$!
s1=0; wait "$p1" || s1=$?; s2=0; wait "$p2" || s2=$?
printf 'A evaluation=%s B evaluation=%s\n' "$s1" "$s2"
((s1 == 0 && s2 == 0))
"$py" tools/summarize_plan_ab.py
touch runs/eval-plan-ab/PIPELINE_SUCCESS
