#!/usr/bin/env bash
#SBATCH --job-name=yoco160m-pilot
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --chdir=/data/share/jzhou945/cchen099/yoco
#SBATCH --output=/data/share/jzhou945/cchen099/yoco/logs/pilot-%j.out
#SBATCH --error=/data/share/jzhou945/cchen099/yoco/logs/pilot-%j.err
set -euo pipefail
base=/data/share/jzhou945/cchen099/yoco
export YOCO_GPU=1
run=runs/yoco-160m-4h100-pilot-20261030
# CPU corpus hashing and configuration checks must finish before submission.
test -f runs/setup/preflight-hpc3.json
test ! -f "$run/stage_complete.json"
exec srun --kill-on-bad-exit=1 bash "$base/scripts/hpc3_container.sh" \
  bash "$base/scripts/pilot_inside.sh"
