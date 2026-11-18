#!/usr/bin/env bash
# Serial eight-GPU campaign from one immutable git revision.
set -eo pipefail
source "$(dirname -- "$0")/activate_local.sh"
set -u
cd "$YOCO_ROOT"
campaign="$YOCO_ROOT/runs/appendix-d-160m-raw-v1"
mkdir -p "$campaign"
exec 9>"$campaign/queue.lock"
flock -n 9 || { echo "Campaign controller already active"; exit 1; }
release="$campaign/release"
if [[ ! -d "$release" ]]; then
  revision=$(git rev-parse HEAD)
  staging=$(mktemp -d "$campaign/release.XXXXXXXX")
  git archive "$revision" runtime configs/appendix_d configs/160m_raw_kv tools/verify_data.py | tar -x -C "$staging"
  ln -s "$YOCO_ROOT/data" "$staging/data"
  echo "$revision" > "$staging/REVISION"
  mv "$staging" "$release"
fi
cd "$release"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHINDUCTOR_COMPILE_THREADS=1
manifest=data/fineweb_edu_10bt/yoco_samples_2048/manifest.json
echo "Waiting for complete packed dataset"
while [[ ! -s "$manifest" ]]; do
  if ! tmux has-session -t '=yoco-160m-packed-data' 2>/dev/null; then
    [[ -s "$manifest" ]] || { echo "Data preparation stopped before manifest publication"; exit 1; }
  fi
  sleep 10
done
python tools/verify_data.py --config configs/appendix_d/160m_{swa,gret}.json configs/160m_raw_kv/*.json > "$campaign/data-verification.jsonl"
for tag in swa_baseline gret_baseline swa_A gret_A swa_B gret_B swa_C gret_C; do
  arch=${tag%_*}
  arm=${tag#*_}
  config=configs/160m_raw_kv/$tag.json
  if [[ $arm == baseline ]]; then config=configs/appendix_d/160m_$arch.json; fi
  run="$campaign/$tag"
  mkdir -p "$run"
  if [[ -f "$run/TRAIN_COMPLETE" ]]; then continue; fi
  if [[ -s "$run/rank0.jsonl" ]] && ! compgen -G "$run/checkpoint_*.json" >/dev/null; then
    echo "$tag has prior events and no full checkpoint; preserving failed attempt"
    exit 1
  fi
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU allocation in use before $tag"
    exit 1
  fi
  echo "$(date -Is) starting $tag"
  torchrun --standalone --nproc-per-node=8 -m runtime.train --config "$config" --run-dir "$run" >> "$run/controller.log" 2>&1
  python - "$run" <<'PY'
import json, sys
from pathlib import Path
from runtime.train import sha256
run = Path(sys.argv[1])
stage = json.loads((run / 'stage_complete.json').read_text())
assert stage == dict(step=40000, formal_complete=True, synthetic=False), stage
latest = json.loads((run / 'latest.json').read_text())
assert latest['step'] == 40000 and sha256(run/latest['file']) == latest['sha256']
for step in range(5000, 40001, 5000):
    assert (run / f'checkpoint_{step:08d}.pt').is_file()
PY
  touch "$run/TRAIN_COMPLETE"
done
