#!/usr/bin/env bash
# Synthetic preparation check only: two updates per model, including resume.
set -euo pipefail
source "$(dirname -- "$0")/activate_local.sh"
cd "$YOCO_ROOT"
export CUDA_VISIBLE_DEVICES="${YOCO_SMOKE_DEVICES:-4,5,6,7}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python - <<'PY'
import os, subprocess, torch
ids = os.environ['CUDA_VISIBLE_DEVICES']
used = subprocess.check_output(['nvidia-smi', '--id='+ids, '--query-gpu=memory.used',
                                '--format=csv,noheader,nounits'], text=True)
assert all(int(x.strip()) < 1000 for x in used.splitlines()), 'Selected GPUs are busy; smoke deferred'
assert torch.cuda.is_available() and torch.cuda.device_count() == 4
assert all(torch.cuda.get_device_capability(i) == (8, 0) for i in range(4))
print({'torch': torch.__version__, 'cuda': torch.version.cuda,
       'devices': [torch.cuda.get_device_name(i) for i in range(4)]}, flush=True)
PY
for arm in baseline A B; do
  case "$arm" in
    baseline) config=configs/yoco_160m_plan_full_baseline_4gpu.json ;;
    *) config="configs/yoco_160m_plan_ab_${arm}_4gpu.json" ;;
  esac
  run="runs/setup/a100-smoke-$arm"
  for stop in 1 2; do
    python -m torch.distributed.run --standalone --nproc-per-node=4 -m runtime.train \
      --config "$config" --run-dir "$run" --smoke --stop-after "$stop"
  done
done
python - <<'PY'
import json
from pathlib import Path
results = {}
for arm in ('baseline', 'A', 'B'):
    run = Path('runs/setup') / ('a100-smoke-'+arm)
    for rank in range(4):
        rows = [json.loads(x) for x in (run/f'rank{rank}.jsonl').read_text().splitlines()]
        assert any(x['event']=='started' and x['resumed'] and x['step']==1 for x in rows)
        assert any(x['event']=='train' and x['step']==2 for x in rows)
    results[arm] = {'world_size': 4, 'synthetic': True, 'updates': 2, 'resume_verified': True}
Path('runs/setup/a100-smoke.json').write_text(json.dumps({'passed': True, 'models': results}, indent=2)+'\n')
PY
