#!/bin/bash
# Explicit Risk 评估脚本 (Figure 1)
# 在 AIME-2024 上评估 explicit_risk 不同训练步的表现

set -e

source /data1/conda/etc/profile.d/conda.sh
conda activate truthrl-verl-npu

if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

export HF_HUB_CACHE=/data2/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

EVAL_SCRIPT="/root/SeedMitigating/scripts/evaluate_model_fixed.py"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
CKPT_BASE="/data2/SeedMitigating-output/paper_reproduction/checkpoints/explicit_risk"
OUT_BASE="/data2/SeedMitigating-output/paper_reproduction/evaluation/explicit_risk/aime_2024"

STEPS_CSV=${EXPLICIT_RISK_STEPS:-"10,20,30,40,50,60"}
RANDOM_T=${EXPLICIT_RISK_RANDOM:-1}

mkdir -p "$OUT_BASE"

echo "=========================================="
echo "Explicit Risk 评估 (AIME-2024)"
echo "Checkpoint 基础目录: $CKPT_BASE"
echo "步数: $STEPS_CSV"
echo "随机t: $RANDOM_T"
echo "=========================================="

IFS=',' read -ra STEPS <<< "$STEPS_CSV"

for step in "${STEPS[@]}"; do
    ckpt_dir="$CKPT_BASE/global_step_${step}"
    if [ ! -d "$ckpt_dir" ]; then
        echo "警告: checkpoint不存在: $ckpt_dir"
        continue
    fi

    out_dir="$OUT_BASE/step_${step}"
    mkdir -p "$out_dir"

    echo "评估 step=${step}"
    if [ "$RANDOM_T" == "1" ]; then
        python3 "$EVAL_SCRIPT" \
            --model_path "$ckpt_dir" \
            --base_model "$BASE_MODEL" \
            --dataset aime_2024 \
            --output_dir "$out_dir" \
            --device npu:0 \
            --strategy explicit_risk \
            --explicit_risk_random
    else
        python3 "$EVAL_SCRIPT" \
            --model_path "$ckpt_dir" \
            --base_model "$BASE_MODEL" \
            --dataset aime_2024 \
            --output_dir "$out_dir" \
            --device npu:0 \
            --strategy explicit_risk
    fi

    echo "✓ step ${step} 完成"
done

# Baseline PPO for reference (可选)
BASELINE_DIR="/data2/SeedMitigating-output/paper_reproduction/checkpoints/baseline_ppo"
if [ -d "$BASELINE_DIR" ]; then
    out_dir="$OUT_BASE/baseline_ppo"
    mkdir -p "$out_dir"
    echo "评估 baseline_ppo (参考)"
    python3 "$EVAL_SCRIPT" \
        --model_path "$BASELINE_DIR" \
        --base_model "$BASE_MODEL" \
        --dataset aime_2024 \
        --output_dir "$out_dir" \
        --device npu:0 \
        --strategy verbalized_brier
fi

echo "=========================================="
echo "✓ Explicit Risk 评估完成"
echo "输出目录: $OUT_BASE"
echo "=========================================="
