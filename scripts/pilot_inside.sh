#!/usr/bin/env bash
set -euo pipefail
base=/data/share/jzhou945/cchen099/yoco
cd "$base"
run=runs/yoco-160m-4h100-pilot-20261030
py="$base/.venv/bin/python"
mkdir -p "$run"
nvidia-smi > "$run/nvidia-smi.txt"
"$py" - "$run" <<'PY'
import hashlib, json, sys, torch
from pathlib import Path
from runtime.model import build_model
root = Path.cwd()
audit = json.loads((root/'runs/setup/preflight-hpc3.json').read_text())
assert audit['passed'] and audit['data_verified']
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(root/'data/fineweb_edu_10bt/manifest.json') == audit['manifest_sha256']
for path, expected in audit['runtime_sha256'].items():
    assert digest(root/path) == expected, path
for arch in ('yoco', 'transformer'):
    assert digest(root/f'configs/{arch}_160m_4gpu.json') == audit['configs'][arch]['config_sha256']
assert torch.__version__.split('+')[0] == '2.7.1'
assert torch.cuda.device_count() == 4
names = [torch.cuda.get_device_name(i) for i in range(4)]
assert all('H100' in n for n in names), names
c = json.loads((root/'configs/yoco_160m_4gpu.json').read_text())
assert sum(p.numel() for p in build_model(c).parameters()) == 166708224
report = dict(torch=torch.__version__, cuda=torch.version.cuda, devices=names,
              tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
              flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
              reference_hardware='32 V100; numerical/backend changes expected on 4 H100')
(Path(sys.argv[1])/'gpu_environment.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
PY
"$py" tests/test_backend.py
"$py" tests/test_transformer.py
# Exercise four-rank forward/backward, optimizer, validation, checkpoint and resume.
"$py" -m torch.distributed.run --standalone --nproc-per-node=4 -m runtime.train \
  --config configs/yoco_160m_4gpu.json --run-dir "$run/smoke" --smoke --stop-after 1
"$py" -m torch.distributed.run --standalone --nproc-per-node=4 -m runtime.train \
  --config configs/yoco_160m_4gpu.json --run-dir "$run/smoke" --smoke --stop-after 2
"$py" -m torch.distributed.run --standalone --nproc-per-node=4 -m runtime.train \
  --config configs/yoco_160m_4gpu.json --run-dir "$run" --stop-after 1000
"$py" tools/compare_pilot.py "$run"
