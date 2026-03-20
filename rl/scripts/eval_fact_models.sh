#!/bin/bash
# 评估脚本: 事实类数据集（HotpotQA/SimpleQA等）响应级置信度评估
# 默认评估 Qwen/Llama 基础模型，不依赖训练权重

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
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}

# 数据集设置
FACT_DATASET="${FACT_DATASET:-hotpotqa}"
FACT_DATASET_FILE="${FACT_DATASET_FILE:-}"
HF_DATASET="${HF_DATASET:-}"
HF_CONFIG="${HF_CONFIG:-}"
HF_SPLIT="${HF_SPLIT:-validation}"
QUESTION_FIELD="${QUESTION_FIELD:-}"
ANSWER_FIELD="${ANSWER_FIELD:-}"

# LLM Judge（可用于 hotpotqa/simpleqa）
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://noapi.ggb.today/v1}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-gpt-4o-mini}"
LOCAL_JUDGE_MODEL="${LOCAL_JUDGE_MODEL:-}"
LOCAL_JUDGE_DEVICE="${LOCAL_JUDGE_DEVICE:-npu:0}"
LOCAL_JUDGE_ARGS=""
if [ -n "$LOCAL_JUDGE_MODEL" ]; then
    LOCAL_JUDGE_ARGS="--simpleqa_grader_local $LOCAL_JUDGE_MODEL --simpleqa_grader_device $LOCAL_JUDGE_DEVICE --simpleqa_grader_use_chat_template"
    if [ "${LOCAL_JUDGE_OFFLINE:-0}" = "1" ]; then
        LOCAL_JUDGE_ARGS="$LOCAL_JUDGE_ARGS --simpleqa_grader_local_files_only"
    fi
    if [ -n "${LOCAL_JUDGE_DTYPE:-}" ]; then
        LOCAL_JUDGE_ARGS="$LOCAL_JUDGE_ARGS --simpleqa_grader_dtype ${LOCAL_JUDGE_DTYPE}"
    fi
fi
LLM_JUDGE_ARGS=""
if [ -z "$LOCAL_JUDGE_MODEL" ] && [ -n "$LLM_JUDGE_MODEL" ]; then
    LLM_JUDGE_ARGS="--simpleqa_grader $LLM_JUDGE_MODEL"
fi

# 模型列表（逗号分隔）
# 示例：Qwen+Llama（需要本地可用或允许在线下载）
FACT_MODELS="${FACT_MODELS:-Qwen/Qwen3-4B-Instruct-2507,meta-llama/Meta-Llama-3.1-8B-Instruct}"
# 思考模式（auto/on/off）
THINKING_MODE="${THINKING_MODE:-auto}"
# 生成参数（可选）
GEN_TEMP="${GEN_TEMP:-}"
GEN_TOP_P="${GEN_TOP_P:-}"
GEN_TOP_K="${GEN_TOP_K:-}"
GEN_MIN_P="${GEN_MIN_P:-}"
GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-}"

EVAL_SCRIPT="/root/SeedMitigating/scripts/evaluate_model_fixed.py"
OUTPUT_BASE="/data2/SeedMitigating-output/paper_reproduction/evaluation/response_level/${FACT_DATASET}"

mkdir -p "$OUTPUT_BASE"

IFS=',' read -r -a MODELS <<< "$FACT_MODELS"

for MODEL_ID in "${MODELS[@]}"; do
    MODEL_ID_TRIM=$(echo "$MODEL_ID" | xargs)
    if [ -z "$MODEL_ID_TRIM" ]; then
        continue
    fi
    MODEL_TAG=$(echo "$MODEL_ID_TRIM" | tr '/:' '__')
    if [ "$THINKING_MODE" = "on" ] || [ "$THINKING_MODE" = "off" ]; then
        MODEL_TAG="${MODEL_TAG}__thinking_${THINKING_MODE}"
    fi
    OUTPUT_DIR="$OUTPUT_BASE/$MODEL_TAG"
    mkdir -p "$OUTPUT_DIR"

    echo "评估模型: $MODEL_ID_TRIM"
    echo "输出目录: $OUTPUT_DIR"

    DATA_ARGS="--dataset $FACT_DATASET"
    if [ -n "$FACT_DATASET_FILE" ]; then
        DATA_ARGS="$DATA_ARGS --dataset_file $FACT_DATASET_FILE"
    fi
    if [ -n "$HF_DATASET" ]; then
        DATA_ARGS="$DATA_ARGS --hf_dataset $HF_DATASET"
    fi
    if [ -n "$HF_CONFIG" ]; then
        DATA_ARGS="$DATA_ARGS --hf_config $HF_CONFIG"
    fi
    if [ -n "$HF_SPLIT" ]; then
        DATA_ARGS="$DATA_ARGS --hf_split $HF_SPLIT"
    fi
    if [ -n "$QUESTION_FIELD" ]; then
        DATA_ARGS="$DATA_ARGS --question_field $QUESTION_FIELD"
    fi
    if [ -n "$ANSWER_FIELD" ]; then
        DATA_ARGS="$DATA_ARGS --answer_field $ANSWER_FIELD"
    fi

    # 根据思考模式填充默认生成参数（若未显式指定）
    if [ -z "$GEN_TEMP" ] && [ "$THINKING_MODE" = "on" ]; then
        GEN_TEMP=0.6
        GEN_TOP_P=0.95
        GEN_TOP_K=20
        GEN_MIN_P=0.0
        GEN_DO_SAMPLE=true
    elif [ -z "$GEN_TEMP" ] && [ "$THINKING_MODE" = "off" ]; then
        GEN_TEMP=0.7
        GEN_TOP_P=0.8
        GEN_TOP_K=20
        GEN_MIN_P=0.0
        GEN_DO_SAMPLE=true
    fi

    GEN_ARGS=""
    if [ -n "$GEN_TEMP" ]; then GEN_ARGS="$GEN_ARGS --gen_temperature $GEN_TEMP"; fi
    if [ -n "$GEN_TOP_P" ]; then GEN_ARGS="$GEN_ARGS --gen_top_p $GEN_TOP_P"; fi
    if [ -n "$GEN_TOP_K" ]; then GEN_ARGS="$GEN_ARGS --gen_top_k $GEN_TOP_K"; fi
    if [ -n "$GEN_MIN_P" ]; then GEN_ARGS="$GEN_ARGS --gen_min_p $GEN_MIN_P"; fi
    if [ -n "$GEN_DO_SAMPLE" ]; then GEN_ARGS="$GEN_ARGS --gen_do_sample $GEN_DO_SAMPLE"; fi

    python3 "$EVAL_SCRIPT" \
        --model_path "$MODEL_ID_TRIM" \
        --base_model "$MODEL_ID_TRIM" \
        $DATA_ARGS \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 1 \
        --device npu:0 \
        --risk_threshold 0.5 \
        --strategy verbalized_brier \
        --use_chat_template \
        --thinking_mode "$THINKING_MODE" \
        $GEN_ARGS \
        --save_io \
        $LLM_JUDGE_ARGS \
        $LOCAL_JUDGE_ARGS

    echo "✓ $MODEL_ID_TRIM 完成"
    echo ""
done

echo "=========================================="
echo "✓ 事实类数据集评估完成"
echo "结果保存在: $OUTPUT_BASE/"
echo "=========================================="
