#!/bin/bash
# 声明级评估脚本: BeyondAIME (Table 3 / Figure 5)
# 说明: 需要GPT-5等外部模型对claim进行标注。若无标注文件，仅导出待标注JSONL。

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
OUTPUT_ROOT="${OUTPUT_BASE:-/data2/SeedMitigating-output/paper_reproduction}"
if [ ! -d "$OUTPUT_ROOT" ]; then
    OUTPUT_ROOT="/root/SeedMitigating/output/paper_reproduction"
fi
CHECKPOINT_BASE="${CHECKPOINT_BASE:-$OUTPUT_ROOT/checkpoints}"
OUTPUT_BASE="$OUTPUT_ROOT/evaluation/claim_level/beyondaime"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"

# 声明级主要模型
MODELS=(
    "confidence_prod"
    "confidence_min"
    "qwen3_instruct"
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
        confidence_prod) STRATEGY="claim_product" ;;
        confidence_min) STRATEGY="claim_minimum" ;;
        qwen3_instruct) STRATEGY="claim_minimum" ;;
    esac

    python3 "$EVAL_SCRIPT" \
        --model_path "$MODEL_PATH" \
        --base_model "$BASE_MODEL" \
        --dataset beyondaime \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 1 \
        --device npu:0 \
        --risk_threshold 0.5 \
        --strategy "$STRATEGY"

    echo "✓ $MODEL 完成"
    echo ""
done

# 生成Table 3（如果提供了标注文件，会直接计算；否则导出待标注JSONL）
LABELS_FILE="${CLAIM_LABELS_FILE:-}"

# 可选：自动调用noapi标注（便宜模型）
# 使用方式：export NOAPI_LABEL_MODEL=gpt-4o-mini 或 gemini-2.5-flash
if [ -z "$LABELS_FILE" ] && [ -n "$NOAPI_LABEL_MODEL" ]; then
    # 先导出待标注文件
    python3 /root/SeedMitigating\\ /rl/evaluation/generate_table3.py
    LABELS_DIR="$OUTPUT_ROOT/evaluation/claim_level_labels_noapi"
    mkdir -p "$LABELS_DIR"
    for MODEL in "${MODELS[@]}"; do
        INPUT_DIR="$OUTPUT_ROOT/evaluation/claim_level/beyondaime"
        OUTPUT_FILE="$LABELS_DIR/${MODEL}.jsonl"
        python3 /root/SeedMitigating\\ /scripts/label_claims_noapi.py \\
            --input_dir "$INPUT_DIR" \\
            --model_dir "$MODEL" \\
            --output_file "$OUTPUT_FILE" \\
            --model "$NOAPI_LABEL_MODEL" \\
            --resume
    done
    LABELS_FILE="$LABELS_DIR"
fi

python3 /root/SeedMitigating\\ /rl/evaluation/generate_table3.py ${LABELS_FILE:+--labels_file "$LABELS_FILE"}

echo "=========================================="
echo "✓ 声明级评估完成"
echo "说明: 若无标注文件，已导出待标注JSONL"
echo "=========================================="
