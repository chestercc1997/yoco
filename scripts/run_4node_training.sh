#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: NODE_RANK=0 MASTER_ADDR=host $0 CONFIG RUN_DIR" >&2
  exit 2
fi
: "${NODE_RANK:?set NODE_RANK to 0, 1, 2, or 3}"
: "${MASTER_ADDR:?set MASTER_ADDR to the rank-0 host}"

CONFIG=$1
RUN_DIR=$2
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

python -m torch.distributed.run \
  --nnodes="$NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  -m runtime.train --config "$CONFIG" --run-dir "$RUN_DIR"
