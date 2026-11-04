#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/activate_local.sh"
cd "$YOCO_ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
mkdir -p runs/setup
python tools/download_pinned_data.py --workers "${DOWNLOAD_WORKERS:-8}" --connections "${DOWNLOAD_CONNECTIONS:-1}"
# prepare_data writes a resolved config; keep committed experiment settings intact.
python -c 'import shutil; shutil.copyfile("configs/yoco_160m_plan_ab_A_4gpu.json", "runs/setup/data-preparation-config.json")'
python tools/prepare_data.py --offline --config runs/setup/data-preparation-config.json --workers "${TOKENIZE_WORKERS:-14}"
for config in configs/yoco_160m_plan_ab_A_4gpu.json configs/yoco_160m_plan_ab_B_4gpu.json configs/yoco_160m_plan_full_baseline_4gpu.json configs/transformer_160m.json; do
  python tools/verify_data.py --config "$config"
done
python tools/prepare_downstream_data.py --output runs/setup/downstream-online.json
python tools/prepare_downstream_data.py --offline --output runs/setup/downstream-offline.json
