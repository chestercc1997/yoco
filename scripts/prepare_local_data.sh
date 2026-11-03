#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/activate_local.sh"
cd "$YOCO_ROOT"
# This route applies only to this process and its children.
exec env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy NO_PROXY='*' no_proxy='*' \
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  bash -c '
    set -euo pipefail
    python tools/download_pinned_data.py --workers "${DOWNLOAD_WORKERS:-6}"
    python tools/prepare_data.py --config configs/yoco_160m_4gpu.json --workers "${TOKENIZE_WORKERS:-8}"
    python tools/verify_data.py --config configs/yoco_160m_4gpu.json
    python tools/verify_data.py --config configs/transformer_160m_4gpu.json
  '
