#!/bin/bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: NODE_RANK=0 MASTER_ADDR=host $0 fixed|dynamic CONFIG CHECKPOINT OUTPUT [extra args]" >&2
  exit 2
fi
: "${NODE_RANK:?set NODE_RANK to 0, 1, 2, or 3}"
: "${MASTER_ADDR:?set MASTER_ADDR to the rank-0 host}"

MODE=$1
CONFIG=$2
CHECKPOINT=$3
OUTPUT=$4
shift 4

case "$MODE" in
  fixed) SCRIPT=benchmarks/benchmark_2k_distributed_inference.py ;;
  dynamic) SCRIPT=benchmarks/benchmark_dynamic_shape_inference.py ;;
  *) echo "mode must be fixed or dynamic" >&2; exit 2 ;;
esac

python -m torch.distributed.run \
  --nnodes="${NNODES:-4}" \
  --nproc-per-node="${GPUS_PER_NODE:-8}" \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="${MASTER_PORT:-29501}" \
  "$SCRIPT" --config "$CONFIG" --checkpoint "$CHECKPOINT" --output "$OUTPUT" "$@"
