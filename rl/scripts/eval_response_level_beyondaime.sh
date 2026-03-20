#!/bin/bash
# 评估脚本: 响应级置信度校准评估 (BeyondAIME)
# 对应论文 Section 4.2.2

set -e

# 激活conda环境
source /data1/conda/etc/profile.d/conda.sh
conda activate truthrl-verl-npu

# 激活CANN环境（如存在）
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# 离线模式（使用本地缓存）
export HF_HUB_CACHE=/data2/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 设置路径
EVAL_SCRIPT="/root/SeedMitigating/scripts/evaluate_model_fixed.py"
DATA_DIR="/root/SeedMitigating/data"
OUTPUT_BASE="/data2/SeedMitigating-output/paper_reproduction/evaluation"
CHECKPOINT_BASE="/data2/SeedMitigating-output/paper_reproduction/checkpoints"

echo "=========================================="
echo "响应级置信度校准评估 (BeyondAIME)"
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_BASE/response_level/beyondaime"

# 评估所有模型变体（含原始Qwen3）
MODELS=(
    "qwen3_instruct"
    "baseline_ppo"
    "confidence_brier"
    "confidence_ce"
    "ppo_value"
    "confidence_prod"
    "confidence_min"
)

for MODEL in "${MODELS[@]}"; do
    echo "评估模型: $MODEL"
    echo "----------------------------------------"

    MODEL_PATH="$CHECKPOINT_BASE/$MODEL"
    if [ "$MODEL" == "qwen3_instruct" ]; then
        MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
    fi
    OUTPUT_DIR="$OUTPUT_BASE/response_level/beyondaime/$MODEL"

    if [ "$MODEL" != "qwen3_instruct" ] && [ ! -d "$MODEL_PATH" ]; then
        echo "警告: 模型路径不存在: $MODEL_PATH"
        echo "跳过..."
        continue
    fi

    # 推断策略
    STRATEGY="auto"
    case "$MODEL" in
        qwen3_instruct) STRATEGY="verbalized_brier" ;;
        baseline_ppo) STRATEGY="verbalized_brier" ;;
        confidence_brier) STRATEGY="verbalized_brier" ;;
        confidence_ce) STRATEGY="verbalized_ce" ;;
        ppo_value) STRATEGY="ppo_value" ;;
        confidence_prod) STRATEGY="claim_product" ;;
        confidence_min) STRATEGY="claim_minimum" ;;
    esac

    CRITIC_ARG=""
    if [ "$STRATEGY" == "ppo_value" ]; then
        if [ -d "$MODEL_PATH/critic" ]; then
            CRITIC_ARG="--critic_path $MODEL_PATH/critic"
        elif [ -d "$MODEL_PATH/../critic" ]; then
            CRITIC_ARG="--critic_path $MODEL_PATH/../critic"
        fi
    fi

    # 运行评估
    python3 "$EVAL_SCRIPT" \
        --model_path "$MODEL_PATH" \
        --base_model "Qwen/Qwen3-4B-Instruct-2507" \
        --dataset beyondaime \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 1 \
        --device npu:0 \
        --risk_threshold 0.5 \
        --strategy "$STRATEGY" \
        $CRITIC_ARG

    echo "✓ $MODEL 评估完成"
    echo ""
done

echo "=========================================="
echo "✓ 所有模型评估完成"
echo "结果保存在: $OUTPUT_BASE/response_level/beyondaime/"
echo "=========================================="
echo ""
echo "下一步: 生成Table 1"
echo "  python /root/SeedMitigating\ /rl/evaluation/generate_table1.py"
