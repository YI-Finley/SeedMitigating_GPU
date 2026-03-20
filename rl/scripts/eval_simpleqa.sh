#!/bin/bash
# 评估脚本: SimpleQA 响应级评估 (Table 4)

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

# SimpleQA评分器（LLM judge）
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://noapi.ggb.today/v1}"
SIMPLEQA_GRADER_MODEL="${SIMPLEQA_GRADER_MODEL:-gpt-4o-mini}"
SIMPLEQA_GRADER_ARGS=""
if [ -n "$SIMPLEQA_GRADER_MODEL" ]; then
    SIMPLEQA_GRADER_ARGS="--simpleqa_grader $SIMPLEQA_GRADER_MODEL"
fi
if [ "$SIMPLEQA_GRADER_MODEL" != "none" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY is required for SimpleQA grader (set SIMPLEQA_GRADER_MODEL=none to disable)"
    exit 1
fi

EVAL_SCRIPT="/root/SeedMitigating/scripts/evaluate_model_fixed.py"
OUTPUT_BASE="/data2/SeedMitigating-output/paper_reproduction/evaluation/response_level/simpleqa"
CHECKPOINT_BASE="/data2/SeedMitigating-output/paper_reproduction/checkpoints"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"

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

    MODEL_PATH="$CHECKPOINT_BASE/$MODEL"
    if [ "$MODEL" == "qwen3_instruct" ]; then
        MODEL_PATH="$BASE_MODEL"
    fi

    OUTPUT_DIR="$OUTPUT_BASE/$MODEL"
    mkdir -p "$OUTPUT_DIR"

    STRATEGY="auto"
    case "$MODEL" in
        baseline_ppo) STRATEGY="verbalized_brier" ;;
        confidence_brier) STRATEGY="verbalized_brier" ;;
        confidence_ce) STRATEGY="verbalized_ce" ;;
        ppo_value) STRATEGY="ppo_value" ;;
        confidence_prod) STRATEGY="claim_product" ;;
        confidence_min) STRATEGY="claim_minimum" ;;
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
        --dataset simpleqa \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 1 \
        --device npu:0 \
        --risk_threshold 0.5 \
        --strategy "$STRATEGY" \
        $CRITIC_ARG \
        $SIMPLEQA_GRADER_ARGS

    echo "✓ $MODEL 完成"
    echo ""
done

echo "=========================================="
echo "✓ SimpleQA评估完成"
echo "结果保存在: $OUTPUT_BASE/"
echo "=========================================="
