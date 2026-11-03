#!/bin/bash
#SBATCH --job-name=yoco-plan-kv-pair
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=/data/share/jzhou945/cchen099/yoco/logs/plan-kv-pair-%j.out
#SBATCH --error=/data/share/jzhou945/cchen099/yoco/logs/plan-kv-pair-%j.err
set -euo pipefail
base=/data/share/jzhou945/cchen099/yoco
release=$base/experiments/plan-conditional-kv-160m-v1
image=/data/share/jzhou945/cchen099/vertikv-work/images/pytorch-25.06-py3.sif
exec srun --kill-on-bad-exit=1 /bin/singularity exec --cleanenv --no-home --nv \
  --bind "$base:$base" --pwd="$release" --env "PYTHONPATH=$release" \
  --env "TIKTOKEN_CACHE_DIR=$base/data/cache/tiktoken" --env OMP_NUM_THREADS=2 \
  --env MKL_NUM_THREADS=2 --env HF_HUB_OFFLINE=1 --env TOKENIZERS_PARALLELISM=false \
  --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" --env "SLURM_JOB_ID=$SLURM_JOB_ID" \
  "$image" bash -c "exec bash $release/scripts/plan_conditional_pair_inside.sh"
