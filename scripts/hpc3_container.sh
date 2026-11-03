#!/usr/bin/env bash
set -euo pipefail
base=/data/share/jzhou945/cchen099/yoco
image=/data/share/jzhou945/cchen099/vertikv-work/images/pytorch-25.06-py3.sif
flags=()
if [[ ${YOCO_GPU:-0} == 1 ]]; then
  test -n "${SLURM_JOB_ID:-}"
  flags+=(--nv)
fi
exec /bin/singularity exec --cleanenv --no-home "${flags[@]}" \
  --bind "$base:$base" --pwd="$base" \
  --env "PYTHONPATH=$base" --env "TIKTOKEN_CACHE_DIR=$base/data/cache/tiktoken" \
  --env "PIP_CACHE_DIR=$base/data/cache/pip" \
  --env OMP_NUM_THREADS=2 --env MKL_NUM_THREADS=2 \
  --env HF_HUB_OFFLINE=1 --env TOKENIZERS_PARALLELISM=false \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
  "$image" env -u PIP_CONSTRAINT -u PIP_BUILD_CONSTRAINT "$@"
