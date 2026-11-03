#!/usr/bin/env bash
#SBATCH --job-name=yoco160-fusion2
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --chdir=/data/share/jzhou945/cchen099/yoco/experiments/fusion-token-2gpu-20261101-v1
#SBATCH --output=/data/share/jzhou945/cchen099/yoco/logs/fusion-token-2gpu-%j.out
#SBATCH --error=/data/share/jzhou945/cchen099/yoco/logs/fusion-token-2gpu-%j.err
set -euo pipefail
base=/data/share/jzhou945/cchen099/yoco
release="$base/experiments/fusion-token-2gpu-20261101-v1"
image=/data/share/jzhou945/cchen099/vertikv-work/images/pytorch-25.06-py3.sif
test -f "$release/preflight.json"
exec srun --kill-on-bad-exit=1 /bin/singularity exec --cleanenv --no-home --nv \
  --bind "$base:$base" --pwd="$release" \
  --env "PYTHONPATH=$release" --env "TIKTOKEN_CACHE_DIR=$base/data/cache/tiktoken" \
  --env OMP_NUM_THREADS=2 --env MKL_NUM_THREADS=2 --env HF_HUB_OFFLINE=1 \
  --env TOKENIZERS_PARALLELISM=false --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" \
  "$image" bash "$release/scripts/fusion_token_inside.sh"
