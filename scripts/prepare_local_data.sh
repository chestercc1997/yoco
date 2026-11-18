#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/activate_local.sh"
cd "$YOCO_ROOT"

case "${1:-}" in
  "") configs=(configs/appendix_d/*.json) ;;
  --historical-ab)
    configs=(configs/yoco_160m_plan_ab_A_4gpu.json configs/yoco_160m_plan_ab_B_4gpu.json
             configs/yoco_160m_plan_full_baseline_4gpu.json configs/transformer_160m.json)
    ;;
  *) echo "Usage: $0 [--historical-ab]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Usage: $0 [--historical-ab]" >&2; exit 2; fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
mkdir -p runs/setup
python tools/download_pinned_data.py --workers "${DOWNLOAD_WORKERS:-8}" --connections "${DOWNLOAD_CONNECTIONS:-1}"
recipe="${configs[0]}"
if [[ "${1:-}" == --historical-ab ]]; then
  # The historical stream packer writes a resolved configuration.
  recipe=runs/setup/data-preparation-config.json
  cp "${configs[0]}" "$recipe"
fi
python tools/prepare_data.py --offline --config "$recipe" --workers "${TOKENIZE_WORKERS:-14}"
python tools/verify_data.py --config "${configs[@]}"
python tools/prepare_downstream_data.py --output runs/setup/downstream-online.json
python tools/prepare_downstream_data.py --offline --output runs/setup/downstream-offline.json
