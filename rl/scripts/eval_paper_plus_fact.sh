#!/bin/bash
# 评估脚本: 论文测试集 + HotpotQA/SimpleQA 统一评估（响应级）
# 默认评估 Qwen/Llama 基础模型，不进行训练

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
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

EVAL_SCRIPT="/root/SeedMitigating/scripts/evaluate_model_fixed.py"
OUTPUT_BASE="/data2/SeedMitigating-output/paper_reproduction/evaluation/response_level"

# 模型列表（逗号分隔）
MODELS="${MODELS:-Qwen/Qwen3-4B-Instruct-2507,meta-llama/Meta-Llama-3.1-8B-Instruct}"
# 数据集列表（默认论文测试集 + hotpotqa + simpleqa）
DATASETS="${DATASETS:-beyondaime,aime_2024,aime_2025,hotpotqa,simpleqa}"
# 抽样（可选）
MAX_SAMPLES="${MAX_SAMPLES:-}"
# 思考模式（auto/on/off）
THINKING_MODE="${THINKING_MODE:-off}"
# 生成长度
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
# 生成参数（可选）
GEN_TEMP="${GEN_TEMP:-}"
GEN_TOP_P="${GEN_TOP_P:-}"
GEN_TOP_K="${GEN_TOP_K:-}"
GEN_MIN_P="${GEN_MIN_P:-}"
GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-}"

# LLM Judge（用于 simpleqa / hotpotqa）
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://noapi.ggb.today/v1}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-gpt-4o-mini}"
LOCAL_JUDGE_MODEL="${LOCAL_JUDGE_MODEL:-}"
LOCAL_JUDGE_DEVICE="${LOCAL_JUDGE_DEVICE:-npu:0}"
LOCAL_JUDGE_USE_CHAT_TEMPLATE="${LOCAL_JUDGE_USE_CHAT_TEMPLATE:-0}"
LOCAL_JUDGE_ARGS=""
if [ -n "$LOCAL_JUDGE_MODEL" ]; then
    LOCAL_JUDGE_ARGS="--simpleqa_grader_local $LOCAL_JUDGE_MODEL --simpleqa_grader_device $LOCAL_JUDGE_DEVICE"
    if [ "$LOCAL_JUDGE_USE_CHAT_TEMPLATE" = "1" ]; then
        LOCAL_JUDGE_ARGS="$LOCAL_JUDGE_ARGS --simpleqa_grader_use_chat_template"
    fi
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

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"

for MODEL_ID in "${MODEL_LIST[@]}"; do
    MODEL_ID_TRIM=$(echo "$MODEL_ID" | xargs)
        if [ -z "$MODEL_ID_TRIM" ]; then
            continue
        fi
        if echo "$MODEL_ID_TRIM" | grep -qi "thinking"; then
            echo "跳过 thinking 模型: $MODEL_ID_TRIM"
            continue
        fi
    MODEL_TAG=$(echo "$MODEL_ID_TRIM" | tr '/:' '__')
    if [ "$THINKING_MODE" = "on" ] || [ "$THINKING_MODE" = "off" ]; then
        MODEL_TAG="${MODEL_TAG}__thinking_${THINKING_MODE}"
    fi

    for DATASET in "${DATASET_LIST[@]}"; do
        DS=$(echo "$DATASET" | xargs)
        if [ -z "$DS" ]; then
            continue
        fi

        OUTPUT_DIR="$OUTPUT_BASE/$DS/$MODEL_TAG"
        mkdir -p "$OUTPUT_DIR"

        DATA_ARGS=(--dataset "$DS")
        case "$DS" in
            hotpotqa|hotpot_qa)
                DATA_ARGS+=(--hf_dataset hotpot_qa --hf_config fullwiki --hf_split validation --question_field question --answer_field answer)
                ;;
            simpleqa)
                DATA_ARGS+=(--dataset_file "/root/SeedMitigating/data/simpleqa.jsonl")
                ;;
            beyondaime)
                # 默认加载本地
                ;;
            aime_2024|aime_2025|aime)
                ;;
        esac

        # 判分器参数仅用于需要 judge 的数据集
        JUDGE_ARGS=()
        case "$DS" in
            hotpotqa|hotpot_qa|simpleqa)
                if [ -n "$LOCAL_JUDGE_MODEL" ]; then
                    JUDGE_ARGS=($LOCAL_JUDGE_ARGS)
                elif [ -n "$LLM_JUDGE_MODEL" ]; then
                    JUDGE_ARGS=($LLM_JUDGE_ARGS)
                fi
                ;;
            *)
                ;;
        esac

        SAMPLE_ARGS=""
        if [ -n "$MAX_SAMPLES" ]; then
            SAMPLE_ARGS="--max_samples $MAX_SAMPLES"
        fi

        echo "=========================================="
        echo "模型: $MODEL_ID_TRIM"
        echo "数据集: $DS"
        echo "输出: $OUTPUT_DIR"
        echo "=========================================="

        # 思考模式仅在开启时覆盖默认生成参数；关闭时沿用论文默认值（在评估脚本内）
        if [ -z "$GEN_TEMP" ] && [ "$THINKING_MODE" = "on" ]; then
            GEN_TEMP=0.6
            GEN_TOP_P=0.95
            GEN_TOP_K=20
            GEN_MIN_P=0.0
            GEN_DO_SAMPLE=true
        fi

        GEN_ARGS=""
        if [ -n "$MAX_NEW_TOKENS" ]; then GEN_ARGS="$GEN_ARGS --max_new_tokens $MAX_NEW_TOKENS"; fi
        if [ -n "$GEN_TEMP" ]; then GEN_ARGS="$GEN_ARGS --gen_temperature $GEN_TEMP"; fi
        if [ -n "$GEN_TOP_P" ]; then GEN_ARGS="$GEN_ARGS --gen_top_p $GEN_TOP_P"; fi
        if [ -n "$GEN_TOP_K" ]; then GEN_ARGS="$GEN_ARGS --gen_top_k $GEN_TOP_K"; fi
        if [ -n "$GEN_MIN_P" ]; then GEN_ARGS="$GEN_ARGS --gen_min_p $GEN_MIN_P"; fi
        if [ -n "$GEN_DO_SAMPLE" ]; then GEN_ARGS="$GEN_ARGS --gen_do_sample $GEN_DO_SAMPLE"; fi

        python3 "$EVAL_SCRIPT" \
            --model_path "$MODEL_ID_TRIM" \
            --base_model "$MODEL_ID_TRIM" \
            "${DATA_ARGS[@]}" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size 1 \
            --device npu:0 \
            --risk_threshold 0.5 \
            --strategy verbalized_brier \
            --use_chat_template \
            --thinking_mode "$THINKING_MODE" \
            --offline \
            $SAMPLE_ARGS \
            $GEN_ARGS \
            --save_io \
            "${JUDGE_ARGS[@]}"

        echo "✓ 完成: $MODEL_ID_TRIM @ $DS"
        echo ""
    done
done

echo "=========================================="
echo "✓ 所有评估完成"
echo "结果保存在: $OUTPUT_BASE"
echo "=========================================="
