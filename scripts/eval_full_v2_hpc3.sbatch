#!/bin/bash
#SBATCH --job-name=yoco-downstream-v2
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/share/jzhou945/cchen099/yoco/eval-20261102/slurm-%j.log
set -euo pipefail
base=/data/share/jzhou945/cchen099/yoco
export YOCO_GPU=1
exec bash "$base/scripts/hpc3_container.sh" bash "$base/eval-20261102/run-eval.sh"
