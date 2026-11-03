#!/bin/bash
set -euo pipefail
root=/data/share/jzhou945/cchen099/yoco/eval-20261102
cd "$root"
export PYTHONPATH="$root/site:$root" HF_HOME="$root/data/cache/huggingface"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_TRUST_REMOTE_CODE=1
export TIKTOKEN_CACHE_DIR="$root/data/cache/tiktoken" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
mkdir -p runs/eval-full-v2/smoke runs/eval-full-v2/results
run_arm() {
 local arm="$1" device="$2" config="$3"
 local python=/data/share/jzhou945/cchen099/yoco/.venv/bin/python
 "$python" evaluation/run.py --locks-dir runs/eval-full-v2/data_locks --config "$config" --checkpoint "$arm.pt" --task arc_easy --limit 4 --output "runs/eval-full-v2/smoke/$arm" --device "$device" > "runs/eval-full-v2/smoke-$arm.log" 2>&1
 for task in arc_easy arc_challenge boolq hellaswag openbookqa piqa winogrande sciq; do
   "$python" evaluation/run.py --locks-dir runs/eval-full-v2/data_locks --config "$config" --checkpoint "$arm.pt" --task "$task" --output "runs/eval-full-v2/results/$arm/$task" --device "$device" > "runs/eval-full-v2/$arm-$task.log" 2>&1
 done
}
run_arm baseline cuda:0 configs/yoco_160m_baseline_2gpu.json & p1=$!
run_arm dynamic cuda:1 configs/yoco_160m_fusion_token_v2_2gpu.json & p2=$!
s1=0; wait "$p1" || s1=$?
s2=0; wait "$p2" || s2=$?
printf 'baseline=%s dynamic=%s\n' "$s1" "$s2"
(( s1 == 0 && s2 == 0 )) || exit 1
/data/share/jzhou945/cchen099/yoco/.venv/bin/python tools/summarize_full_v2_downstream.py
