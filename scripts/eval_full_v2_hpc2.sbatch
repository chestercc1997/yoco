#!/bin/bash
#SBATCH --job-name=yoco-downstream-v2
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/eval-full-v2/slurm-%j.log
set -euo pipefail
cd /hpc2hdd/home/jzhou945/cchen099/yoco
export HF_HOME="$PWD/data/cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_TRUST_REMOTE_CODE=1
export TIKTOKEN_CACHE_DIR="$PWD/data/cache/tiktoken" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export PYTHONPATH="$PWD"
tasks=(arc_easy arc_challenge boolq hellaswag openbookqa piqa winogrande sciq)
run_arm() {
  local arm="$1" device="$2" config="$3"
  .venv/bin/python evaluation/run.py --locks-dir runs/eval-full-v2/data_locks --config "$config" --checkpoint "runs/eval-full-v2/checkpoints/$arm.pt" --task arc_easy --limit 4 --output "runs/eval-full-v2/smoke/$arm" --device "$device" > "runs/eval-full-v2/smoke-$arm.log" 2>&1
  for task in "${tasks[@]}"; do
    for attempt in {1..120}; do
      grep -q "READY $task:" runs/eval-full-v2/prepare-data-retry.log && break
      sleep 10
    done
    grep -q "READY $task:" runs/eval-full-v2/prepare-data-retry.log || { echo "Dataset not ready: $task" >&2; return 1; }
    .venv/bin/python evaluation/run.py --locks-dir runs/eval-full-v2/data_locks --config "$config" --checkpoint "runs/eval-full-v2/checkpoints/$arm.pt" --task "$task" --output "runs/eval-full-v2/results/$arm/$task" --device "$device" > "runs/eval-full-v2/$arm-$task.log" 2>&1
  done
}
run_arm baseline cuda:0 configs/yoco_160m_baseline_2gpu.json &
base_pid=$!
run_arm dynamic cuda:1 configs/yoco_160m_fusion_token_v2_2gpu.json &
dyn_pid=$!
base_status=0; wait "$base_pid" || base_status=$?
dyn_status=0; wait "$dyn_pid" || dyn_status=$?
printf 'baseline=%s dynamic=%s\n' "$base_status" "$dyn_status"
if (( base_status || dyn_status )); then exit 1; fi
.venv/bin/python tools/summarize_full_v2_downstream.py
