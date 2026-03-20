#!/usr/bin/env bash
set -euo pipefail
source /data1/conda/etc/profile.d/conda.sh
conda activate truthrl-verl-npu
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /root/code/SeedMitigating/verl

merge_one() {
  local local_dir="$1"
  local target_dir="$2"
  echo "[MERGE] $local_dir -> $target_dir"
  python -m verl.model_merger merge --backend fsdp --local_dir "$local_dir" --target_dir "$target_dir"
}

merge_one "/root/code/SeedMitigating/output/checkpoints/Qwen3-4B-Instruct-Verbalized-CE-20260307_235307/global_step_240/actor" "/root/code/SeedMitigating/output/merged_hf/Qwen3-4B-Instruct-Verbalized-CE-20260307_235307_global_step_240"
merge_one "/root/code/SeedMitigating/output/checkpoints/Qwen3-4B-Instruct-Verbalized-CE-20260305_200541/global_step_600/actor" "/root/code/SeedMitigating/output/merged_hf/Qwen3-4B-Instruct-Verbalized-CE-20260305_200541_global_step_600"
merge_one "/root/code/SeedMitigating/output/checkpoints/Qwen3-4B-Instruct-Binary-Reward-20260308_224555_ws4_modelresume/global_step_400/actor" "/root/code/SeedMitigating/output/merged_hf/Qwen3-4B-Instruct-Binary-Reward-20260308_224555_ws4_modelresume_global_step_400"
merge_one "/root/code/SeedMitigating/output/checkpoints/Qwen3-4B-Instruct-Verbalized-Brier-20260307_143844/global_step_400/actor" "/root/code/SeedMitigating/output/merged_hf/Qwen3-4B-Instruct-Verbalized-Brier-20260307_143844_global_step_400"
merge_one "/root/code/SeedMitigating/output/checkpoints/Qwen3-4B-Instruct-Binary-Reward-20260303_222418/global_step_520/actor" "/root/code/SeedMitigating/output/merged_hf/Qwen3-4B-Instruct-Binary-Reward-20260303_222418_global_step_520"
merge_one "/root/code/SeedMitigating/output/checkpoints/Qwen3-4B-Instruct-Verbalized-Brier-20260302_230736/global_step_480/actor" "/root/code/SeedMitigating/output/merged_hf/Qwen3-4B-Instruct-Verbalized-Brier-20260302_230736_global_step_480"

echo "[DONE] merged 6 checkpoints"
