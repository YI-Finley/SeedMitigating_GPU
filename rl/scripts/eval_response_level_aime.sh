#!/bin/bash
# 评估脚本: 响应级置信度校准评估 (AIME-2024 / AIME-2025)
# 对应论文 Table 2

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
OUTPUT_BASE="/data2/SeedMitigating-output/paper_reproduction/evaluation/response_level"
CHECKPOINT_BASE="/data2/SeedMitigating-output/paper_reproduction/checkpoints"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"

DATASETS=(
    "aime_2024"
    "aime_2025"
)

# Table 2 使用的模型（含原始Qwen3-4B-Instruct）
MODELS=(
    "qwen3_instruct"
    "baseline_ppo"
    "confidence_brier"
    "confidence_ce"
    "ppo_value"
)

for DATASET in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "AIME响应级评估: $DATASET"
    echo "=========================================="

    for MODEL in "${MODELS[@]}"; do
        echo "评估模型: $MODEL"

        MODEL_PATH="$CHECKPOINT_BASE/$MODEL"
        if [ "$MODEL" == "qwen3_instruct" ]; then
            MODEL_PATH="$BASE_MODEL"
        fi

        OUTPUT_DIR="$OUTPUT_BASE/$DATASET/$MODEL"
        mkdir -p "$OUTPUT_DIR"

        STRATEGY="auto"
        case "$MODEL" in
            baseline_ppo) STRATEGY="verbalized_brier" ;;
            confidence_brier) STRATEGY="verbalized_brier" ;;
            confidence_ce) STRATEGY="verbalized_ce" ;;
            ppo_value) STRATEGY="ppo_value" ;;
            qwen3_instruct) STRATEGY="verbalized_brier" ;;
        esac

        CRITIC_ARG=""
        if [ "$STRATEGY" == "ppo_value" ]; then
            if [ -d "$MODEL_PATH/critic" ]; then
                CRITIC_ARG="--critic_path $MODEL_PATH/critic"
            elif [ -d "$MODEL_PATH/../critic" ]; then
                CRITIC_ARG="--critic_path $MODEL_PATH/../critic"
            fi
        fi

        python3 "$EVAL_SCRIPT" \
            --model_path "$MODEL_PATH" \
            --base_model "$BASE_MODEL" \
            --dataset "$DATASET" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size 1 \
            --device npu:0 \
            --risk_threshold 0.5 \
            --strategy "$STRATEGY" \
            $CRITIC_ARG

        echo "✓ $MODEL on $DATASET 完成"
        echo ""
    done
done

echo "=========================================="
echo "✓ AIME响应级评估完成"
echo "结果保存在: $OUTPUT_BASE/{aime_2024,aime_2025}/"
echo "=========================================="
