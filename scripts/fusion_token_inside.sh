#!/usr/bin/env bash
set -euo pipefail
base=/data/share/jzhou945/cchen099/yoco
release="$base/experiments/fusion-token-2gpu-20261101-v1"
cd "$release"
py="$base/.venv/bin/python"
run="$base/runs/yoco-160m-fusion-token-2h100-pilot-20261101-v1"
config=configs/yoco_160m_fusion_token_2gpu.json
mkdir -p "$run"
# Serializes accidental concurrent starts; each submission is also recorded remotely.
exec 9>"$run/allocation.lock"
flock -n 9
test ! -f "$run/stage_complete.json"
"$py" tools/preflight_fusion.py --check-only
nvidia-smi > "$run/nvidia-smi.txt"
"$py" - "$run" <<'PY'
import json, sys, torch
from pathlib import Path
assert torch.__version__.split('+')[0] == '2.7.1'
assert torch.cuda.device_count() == 2
names = [torch.cuda.get_device_name(i) for i in range(2)]
assert all('H100' in name for name in names), names
result = dict(torch=torch.__version__, cuda=torch.version.cuda, devices=names,
              release=str(Path.cwd()), tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
              flash_sdp=torch.backends.cuda.flash_sdp_enabled())
(Path(sys.argv[1])/'gpu_environment.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
PY
"$py" tests/test_memory_fusion.py
# Separate synthetic optimizer state from the real-data run.
"$py" -m torch.distributed.run --standalone --nproc-per-node=2 -m runtime.train \
  --config "$config" --run-dir "$run/smoke" --smoke --stop-after 1
"$py" -m torch.distributed.run --standalone --nproc-per-node=2 -m runtime.train \
  --config "$config" --run-dir "$run/smoke" --smoke --stop-after 2
"$py" - "$run/smoke" <<'PY'
import json, sys
from pathlib import Path
run=Path(sys.argv[1])
for rank in range(2):
    rows=[json.loads(line) for line in (run/f'rank{rank}.jsonl').read_text().splitlines()]
    assert any(r['event']=='started' and r['resumed'] and r['step']==1 for r in rows)
    assert any(r['event']=='train' and r['step']==2 for r in rows)
print('PASS two-rank FP16 optimizer/checkpoint resume', flush=True)
PY
"$py" -m torch.distributed.run --standalone --nproc-per-node=2 -m runtime.train \
  --config "$config" --run-dir "$run" --stop-after 1000
