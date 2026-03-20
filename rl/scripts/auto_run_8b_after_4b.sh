#!/usr/bin/env bash
set -euo pipefail

BASE_OUT="/data2/SeedMitigating-output/paper_reproduction/evaluation/response_level"
OUT_4B="$BASE_OUT/hotpotqa/Qwen_Qwen3-4B-Instruct-2507__thinking_off"
MERGE_SCRIPT="/root/SeedMitigating/scripts/merge_sharded_results.py"
PARALLEL_SCRIPT="/root/SeedMitigating/rl/scripts/eval_paper_plus_fact_parallel.sh"
LOG="$BASE_OUT/auto_run_8b_after_4b.log"

# 载入环境（保证后续 merge/8B 评估可用）
source /data1/conda/etc/profile.d/conda.sh
conda activate truthrl-verl-npu
# 防止 set_env.sh 内部引用未定义变量导致 -u 退出
set +u
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -u

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

stamp() { date '+%F %T'; }

# 等待 4B 分片全部完成
while true; do
  missing=$(python3 - <<'PY'
import os
base='/data2/SeedMitigating-output/paper_reproduction/evaluation/response_level/hotpotqa/Qwen_Qwen3-4B-Instruct-2507__thinking_off'
miss=[]
for i in range(32):
    if not os.path.exists(os.path.join(base, f'shard_{i}', 'results.json')):
        miss.append(i)
print(','.join(map(str, miss)))
PY
)
  if [ -z "$missing" ]; then
    echo "[$(stamp)] 4B 分片已全部完成" >> "$LOG"
    break
  fi
  echo "[$(stamp)] 仍缺分片: $missing" >> "$LOG"
  sleep 60

done

# 合并 4B 结果
if [ -f "$MERGE_SCRIPT" ]; then
  echo "[$(stamp)] 开始合并 4B 结果" >> "$LOG"
  python3 "$MERGE_SCRIPT" --input_root "$OUT_4B" >> "$LOG" 2>&1
  echo "[$(stamp)] 4B 合并完成" >> "$LOG"
else
  echo "[$(stamp)] 警告: 合并脚本不存在: $MERGE_SCRIPT" >> "$LOG"
fi

# 运行 8B 模型（串行、每卡 1 进程、用满 8 卡）
export MODELS="Qwen/Qwen3-8B,meta-llama/Meta-Llama-3.1-8B-Instruct"
export DATASETS="hotpotqa"
export THINKING_MODE="off"
export DISABLE_JUDGE=1
export USE_ALL_NPUS=1
export MODELS_SERIAL=1
export WORKERS_PER_NPU=1
export NPUS="0,1,2,3,4,5,6,7"
export HOTPOTQA_DATA_FILE="/root/Fact_Reasoning/GT_GRPO/data/nq_hotpot_searchr1/test.parquet"

# 保证离线
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [ -f "$PARALLEL_SCRIPT" ]; then
  echo "[$(stamp)] 开始运行 8B 模型 (Qwen/Llama)" >> "$LOG"
  bash "$PARALLEL_SCRIPT" >> "$LOG" 2>&1
  echo "[$(stamp)] 8B 模型完成" >> "$LOG"
else
  echo "[$(stamp)] 警告: 并行脚本不存在: $PARALLEL_SCRIPT" >> "$LOG"
fi
