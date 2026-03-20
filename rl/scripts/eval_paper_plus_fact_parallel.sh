#!/bin/bash
# 并行评估脚本: 论文测试集 + HotpotQA/SimpleQA（分片并行，尽量占满NPU）

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
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/data2/cache/huggingface/datasets}

EVAL_SCRIPT="/root/SeedMitigating/scripts/evaluate_model_fixed.py"
MERGE_SCRIPT="/root/SeedMitigating/scripts/merge_sharded_results.py"
OUTPUT_BASE="/data2/SeedMitigating-output/paper_reproduction/evaluation/response_level"

# NPU 列表（逗号分隔）
NPUS="${NPUS:-0,1,2,3,4,5,6,7}"
# 是否每个模型都使用全部 NPU（1=是，0=否）
USE_ALL_NPUS="${USE_ALL_NPUS:-0}"
# 模型是否串行执行（1=串行，0=并行）
MODELS_SERIAL="${MODELS_SERIAL:-0}"
# 模型分配到不同 NPU 组（默认占满 8 卡）
NPUS_4B="${NPUS_4B:-0,1,2,3}"
NPUS_8B_QWEN="${NPUS_8B_QWEN:-4,5}"
NPUS_8B_LLAMA="${NPUS_8B_LLAMA:-6,7}"
# 每个 NPU 上并发 worker 数
WORKERS_PER_NPU="${WORKERS_PER_NPU:-1}"
WORKERS_PER_NPU_4B="${WORKERS_PER_NPU_4B:-1}"
WORKERS_PER_NPU_8B="${WORKERS_PER_NPU_8B:-1}"
WORKERS_PER_NPU_JUDGE="${WORKERS_PER_NPU_JUDGE:-1}"
IFS=',' read -r -a NPU_LIST <<< "$NPUS"
NPU_COUNT=${#NPU_LIST[@]}
if [ "$NPU_COUNT" -le 0 ]; then
    echo "未检测到可用NPU列表"
    exit 1
fi
if [ "$WORKERS_PER_NPU" -le 0 ]; then
    echo "WORKERS_PER_NPU 必须 >= 1"
    exit 1
fi
STRIDE=$((NPU_COUNT * WORKERS_PER_NPU))

# 模型列表（逗号分隔）
MODELS="${MODELS:-Qwen/Qwen3-4B-Instruct-2507,Qwen/Qwen3-8B,meta-llama/Meta-Llama-3.1-8B-Instruct}"
# 数据集列表（默认仅跑 HotpotQA）
DATASETS="${DATASETS:-hotpotqa}"
# 抽样（可选）
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_SAMPLES_TOTAL="${MAX_SAMPLES_TOTAL:-}"
# 思考模式（auto/on/off）
THINKING_MODE="${THINKING_MODE:-off}"
# 生成长度
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
# 评估策略（默认 verbalized_brier，可设为 explicit_risk 等）
STRATEGY="${STRATEGY:-verbalized_brier}"
# 显式风险：是否随机采样 t（1=随机，0=固定 risk_threshold）
EXPLICIT_RISK_RANDOM="${EXPLICIT_RISK_RANDOM:-0}"
# risk threshold（显式风险固定 t 时使用）
RISK_THRESHOLD="${RISK_THRESHOLD:-0.5}"
# 输出目录后缀（可选，避免覆盖）
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"
# 生成参数（可选）
GEN_TEMP="${GEN_TEMP:-}"
GEN_TOP_P="${GEN_TOP_P:-}"
GEN_TOP_K="${GEN_TOP_K:-}"
GEN_MIN_P="${GEN_MIN_P:-}"
GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-}"
# vLLM 推理开关与参数
USE_VLLM="${USE_VLLM:-0}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_MEM="${VLLM_MEM:-0.9}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-0}"
VLLM_SEED="${VLLM_SEED:-42}"
VLLM_DEVICE="${VLLM_DEVICE:-npu}"

# HotpotQA（search r1）本地数据
HOTPOTQA_DATA_FILE="${HOTPOTQA_DATA_FILE:-/root/Fact_Reasoning/GT_GRPO/data/nq_hotpot_searchr1/test.parquet}"
HOTPOTQA_QUESTION_FIELD="${HOTPOTQA_QUESTION_FIELD:-question}"
HOTPOTQA_ANSWER_FIELD="${HOTPOTQA_ANSWER_FIELD:-golden_answers}"

# LLM Judge（用于 simpleqa / hotpotqa）
DISABLE_JUDGE="${DISABLE_JUDGE:-1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://noapi.ggb.today/v1}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-gpt-4o-mini}"
LOCAL_JUDGE_MODEL="${LOCAL_JUDGE_MODEL:-/data1/modelscope/Qwen/Qwen3-14B}"
LOCAL_JUDGE_DEVICE="${LOCAL_JUDGE_DEVICE:-npu:0}"
LOCAL_JUDGE_OFFLINE="${LOCAL_JUDGE_OFFLINE:-1}"
JUDGE_NPU="${JUDGE_NPU:-7}"
LOCAL_JUDGE_USE_CHAT_TEMPLATE="${LOCAL_JUDGE_USE_CHAT_TEMPLATE:-0}"
if [ "$DISABLE_JUDGE" = "1" ]; then
    LLM_JUDGE_MODEL=""
    LOCAL_JUDGE_MODEL=""
fi
LOCAL_JUDGE_ARGS_BASE=()
if [ -n "$LOCAL_JUDGE_MODEL" ]; then
    LOCAL_JUDGE_ARGS_BASE=(--simpleqa_grader_local "$LOCAL_JUDGE_MODEL")
    if [ "$LOCAL_JUDGE_USE_CHAT_TEMPLATE" = "1" ]; then
        LOCAL_JUDGE_ARGS_BASE+=(--simpleqa_grader_use_chat_template)
    fi
    if [ "${LOCAL_JUDGE_OFFLINE:-0}" = "1" ]; then
        LOCAL_JUDGE_ARGS_BASE+=(--simpleqa_grader_local_files_only)
    fi
    if [ -n "${LOCAL_JUDGE_DTYPE:-}" ]; then
        LOCAL_JUDGE_ARGS_BASE+=(--simpleqa_grader_dtype "${LOCAL_JUDGE_DTYPE}")
    fi
fi
LLM_JUDGE_ARGS=""
if [ -z "$LOCAL_JUDGE_MODEL" ] && [ -n "$LLM_JUDGE_MODEL" ]; then
    LLM_JUDGE_ARGS="--simpleqa_grader $LLM_JUDGE_MODEL"
fi

IFS=',' read -r -a MODEL_LIST <<< "$MODELS"
IFS=',' read -r -a DATASET_LIST <<< "$DATASETS"

resolve_model_path() {
    local model_id="$1"
    local cand="$model_id"
    if [ -d "$model_id" ]; then
        echo "$model_id"
        return
    fi
    case "$model_id" in
        Qwen/Qwen3-4B-Instruct-2507)
            cand="/data1/modelscope/Qwen/Qwen3-4B-Instruct-2507"
            ;;
        Qwen/Qwen3-8B)
            cand="/data1/modelscope/Qwen/Qwen3-8B"
            ;;
        meta-llama/Meta-Llama-3.1-8B-Instruct|meta-1lama/Meta-Llama-3.1-88-Instruct)
            cand="/data1/modelscope/LLM-Research/Meta-Llama-3.1-8B-Instruct"
            ;;
    esac
    if [ -d "$cand" ]; then
        echo "$cand"
    else
        echo "$model_id"
    fi
}

check_hotpotqa_offline() {
    python3 - <<'PY'
import os, sys
os.environ["HF_DATASETS_OFFLINE"] = "1"
try:
    from datasets import load_dataset
    load_dataset("hotpot_qa", "fullwiki", split="validation", download_mode="reuse_dataset_if_exists")
except Exception as exc:
    print(f"HotpotQA 离线不可用: {exc}")
    sys.exit(1)
print("HotpotQA 离线可用")
PY
}

run_sharded_eval() {
    local model_id="$1"
    local ds="$2"
    local output_dir="$3"
    local sample_args="$4"
    local gen_args="$5"
    local npu_group="$6"
    local model_workers_per_npu="$7"

    mkdir -p "$output_dir"

    echo "=========================================="
    local resolved_model_path
    local resolved_base_model
    resolved_model_path="$(resolve_model_path "$model_id")"
    resolved_base_model="$(resolve_model_path "$model_id")"
    echo "模型: $model_id"
    echo "模型路径: $resolved_model_path"
    echo "基础模型: $resolved_base_model"
    echo "数据集: $ds"
    echo "输出: $output_dir"
    local_workers_per_npu="$model_workers_per_npu"
    if [ -z "$local_workers_per_npu" ]; then
        local_workers_per_npu="$WORKERS_PER_NPU"
    fi
    if [ -n "$WORKERS_PER_NPU_4B" ] && [[ "$model_id" == *"4B"* ]]; then
        local_workers_per_npu="$WORKERS_PER_NPU_4B"
    elif [ -n "$WORKERS_PER_NPU_8B" ] && [[ "$model_id" == *"8B"* ]]; then
        local_workers_per_npu="$WORKERS_PER_NPU_8B"
    fi
    case "$ds" in
        hotpotqa|hotpot_qa|simpleqa)
            if [ -n "$WORKERS_PER_NPU_JUDGE" ] && { [ -n "$LOCAL_JUDGE_MODEL" ] || [ -n "$LLM_JUDGE_MODEL" ]; }; then
                local_workers_per_npu="$WORKERS_PER_NPU_JUDGE"
            fi
            ;;
        *)
            ;;
    esac
    if [ "$local_workers_per_npu" -le 0 ]; then
        local_workers_per_npu=1
    fi
    IFS=',' read -r -a MODEL_NPU_LIST <<< "$npu_group"
    if [ "${#MODEL_NPU_LIST[@]}" -le 0 ]; then
        MODEL_NPU_LIST=("${NPU_LIST[@]}")
    fi
    local_npu_count=${#MODEL_NPU_LIST[@]}
    local_stride=$((local_npu_count * local_workers_per_npu))

    echo "NPU: $npu_group (workers_per_npu=$local_workers_per_npu)"
    echo "=========================================="

    # 数据集参数（使用数组保证带空格路径安全）
    DATA_ARGS=(--dataset "$ds")
    case "$ds" in
        hotpotqa|hotpot_qa)
            if [ -n "$HOTPOTQA_DATA_FILE" ] && [ -f "$HOTPOTQA_DATA_FILE" ]; then
                DATA_ARGS+=(--dataset_file "$HOTPOTQA_DATA_FILE" --question_field "$HOTPOTQA_QUESTION_FIELD" --answer_field "$HOTPOTQA_ANSWER_FIELD")
            else
                DATA_ARGS+=(--hf_dataset hotpot_qa --hf_config fullwiki --hf_split validation --question_field question --answer_field answer)
            fi
            ;;
        simpleqa)
            DATA_ARGS+=(--dataset_file "/root/SeedMitigating/data/simpleqa.jsonl")
            ;;
        beyondaime)
            ;;
        aime_2024|aime_2025|aime)
            ;;
    esac

    PIDS=()
    for idx in $(seq 0 $((local_stride - 1))); do
        npu_index=$((idx % local_npu_count))
        npu="${MODEL_NPU_LIST[$npu_index]}"
        shard_dir="$output_dir/shard_${idx}"
        mkdir -p "$shard_dir"

        echo "启动分片 ${idx}/${local_stride} -> NPU ${npu}"

        # 判分器参数仅用于需要 judge 的数据集
        JUDGE_ARGS=()
        VISIBLE_DEVICES="$npu"
        case "$ds" in
            hotpotqa|hotpot_qa|simpleqa)
                if [ "$DISABLE_JUDGE" = "1" ]; then
                    JUDGE_ARGS=(--simpleqa_grader none)
                elif [ -n "$LOCAL_JUDGE_MODEL" ]; then
                    judge_device="$LOCAL_JUDGE_DEVICE"
                    if [ -n "$JUDGE_NPU" ]; then
                        if [ "$npu" = "$JUDGE_NPU" ]; then
                            VISIBLE_DEVICES="$npu"
                            judge_device="npu:0"
                        else
                            VISIBLE_DEVICES="$npu,$JUDGE_NPU"
                            judge_device="npu:1"
                        fi
                    fi
                    # 使用本地 judge
                    JUDGE_ARGS=("${LOCAL_JUDGE_ARGS_BASE[@]}" --simpleqa_grader_device "$judge_device")
                elif [ -n "$LLM_JUDGE_MODEL" ]; then
                    # 使用外部 judge
                    JUDGE_ARGS=($LLM_JUDGE_ARGS)
                else
                    JUDGE_ARGS=(--simpleqa_grader none)
                fi
                ;;
            *)
                ;;
        esac

        EXTRA_STRATEGY_ARGS=()
        if [ "$STRATEGY" = "explicit_risk" ] && [ "$EXPLICIT_RISK_RANDOM" = "1" ]; then
            EXTRA_STRATEGY_ARGS+=(--explicit_risk_random)
        fi
        VLLM_ARGS=()
        if [ "$USE_VLLM" = "1" ]; then
            VLLM_ARGS+=(--use_vllm)
            VLLM_ARGS+=(--vllm_tensor_parallel_size "$VLLM_TP")
            VLLM_ARGS+=(--vllm_gpu_memory_utilization "$VLLM_MEM")
            if [ "$VLLM_TRUST_REMOTE_CODE" = "1" ]; then
                VLLM_ARGS+=(--vllm_trust_remote_code)
            fi
            if [ -n "$VLLM_SEED" ]; then
                VLLM_ARGS+=(--vllm_seed "$VLLM_SEED")
            fi
        fi

        ASCEND_RT_VISIBLE_DEVICES=$VISIBLE_DEVICES CUDA_VISIBLE_DEVICES="" VLLM_DEVICE="$VLLM_DEVICE" python3 "$EVAL_SCRIPT" \
            --model_path "$resolved_model_path" \
            --base_model "$resolved_base_model" \
            "${DATA_ARGS[@]}" \
            --output_dir "$shard_dir" \
            --batch_size 1 \
            --device npu:0 \
            --risk_threshold "$RISK_THRESHOLD" \
            --strategy "$STRATEGY" \
            --use_chat_template \
            --thinking_mode "$THINKING_MODE" \
            --sample_offset "$idx" \
            --sample_stride "$local_stride" \
            --offline \
            $sample_args \
            $gen_args \
            --save_io \
            "${EXTRA_STRATEGY_ARGS[@]}" \
            "${VLLM_ARGS[@]}" \
            "${JUDGE_ARGS[@]}" \
            > "$shard_dir/eval.log" 2>&1 &

        PIDS+=($!)
    done

    echo "等待所有分片完成..."
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done

    if [ -f "$MERGE_SCRIPT" ]; then
        python3 "$MERGE_SCRIPT" --input_root "$output_dir"
    else
        echo "警告: 合并脚本不存在: $MERGE_SCRIPT"
    fi

    echo "✓ 完成: $model_id @ $ds"
    echo ""
}

for DATASET in "${DATASET_LIST[@]}"; do
    DS=$(echo "$DATASET" | xargs)
    if [ -z "$DS" ]; then
        continue
    fi
    case "$DS" in
        hotpotqa|hotpot_qa)
            if [ -n "$HOTPOTQA_DATA_FILE" ] && [ -f "$HOTPOTQA_DATA_FILE" ]; then
                :
            else
                if ! check_hotpotqa_offline; then
                    echo "跳过 HotpotQA（离线不可用）"
                    continue
                fi
            fi
            ;;
        simpleqa)
            if [ ! -f "/root/SeedMitigating/data/simpleqa.jsonl" ]; then
                echo "跳过 SimpleQA（缺少本地数据文件）"
                continue
            fi
            ;;
        *)
            ;;
    esac

    SAMPLE_ARGS=""
    if [ -n "$MAX_SAMPLES_TOTAL" ] && [ -z "$MAX_SAMPLES" ]; then
        per_shard=$(( (MAX_SAMPLES_TOTAL + STRIDE - 1) / STRIDE ))
        SAMPLE_ARGS="--max_samples $per_shard"
    elif [ -n "$MAX_SAMPLES" ]; then
        SAMPLE_ARGS="--max_samples $MAX_SAMPLES"
    fi

    # 生成参数（每个数据集单独拷贝，避免被循环污染）
    local_gen_temp="$GEN_TEMP"
    local_gen_top_p="$GEN_TOP_P"
    local_gen_top_k="$GEN_TOP_K"
    local_gen_min_p="$GEN_MIN_P"
    local_gen_do_sample="$GEN_DO_SAMPLE"
    if [ -z "$local_gen_temp" ] && [ "$THINKING_MODE" = "on" ]; then
        local_gen_temp=0.6
        local_gen_top_p=0.95
        local_gen_top_k=20
        local_gen_min_p=0.0
        local_gen_do_sample=true
    fi

    GEN_ARGS=""
    if [ -n "$MAX_NEW_TOKENS" ]; then GEN_ARGS="$GEN_ARGS --max_new_tokens $MAX_NEW_TOKENS"; fi
    if [ -n "$local_gen_temp" ]; then GEN_ARGS="$GEN_ARGS --gen_temperature $local_gen_temp"; fi
    if [ -n "$local_gen_top_p" ]; then GEN_ARGS="$GEN_ARGS --gen_top_p $local_gen_top_p"; fi
    if [ -n "$local_gen_top_k" ]; then GEN_ARGS="$GEN_ARGS --gen_top_k $local_gen_top_k"; fi
    if [ -n "$local_gen_min_p" ]; then GEN_ARGS="$GEN_ARGS --gen_min_p $local_gen_min_p"; fi
    if [ -n "$local_gen_do_sample" ]; then GEN_ARGS="$GEN_ARGS --gen_do_sample $local_gen_do_sample"; fi

    PIDS=()
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
        # 根据策略/输出后缀区分目录
        if [ -n "$STRATEGY" ] && [ "$STRATEGY" != "verbalized_brier" ]; then
            MODEL_TAG="${MODEL_TAG}__strategy_${STRATEGY}"
        fi
        if [ -n "$OUTPUT_SUFFIX" ]; then
            MODEL_TAG="${MODEL_TAG}__${OUTPUT_SUFFIX}"
        fi
        OUTPUT_DIR="$OUTPUT_BASE/$DS/$MODEL_TAG"

        if [ "$USE_ALL_NPUS" = "1" ]; then
            NPU_GROUP="$NPUS"
            MODEL_WPN=""
        else
            if [[ "$MODEL_ID_TRIM" == *"Qwen3-4B"* ]]; then
                NPU_GROUP="$NPUS_4B"
                MODEL_WPN=1
            elif [[ "$MODEL_ID_TRIM" == *"Qwen3-8B"* ]]; then
                NPU_GROUP="$NPUS_8B_QWEN"
                MODEL_WPN=1
            else
                NPU_GROUP="$NPUS_8B_LLAMA"
                MODEL_WPN=1
            fi
        fi

        if [ "$MODELS_SERIAL" = "1" ]; then
            run_sharded_eval "$MODEL_ID_TRIM" "$DS" "$OUTPUT_DIR" "$SAMPLE_ARGS" "$GEN_ARGS" "$NPU_GROUP" "$MODEL_WPN"
        else
            run_sharded_eval "$MODEL_ID_TRIM" "$DS" "$OUTPUT_DIR" "$SAMPLE_ARGS" "$GEN_ARGS" "$NPU_GROUP" "$MODEL_WPN" &
            PIDS+=($!)
        fi
    done

    if [ "${#PIDS[@]}" -gt 0 ]; then
        echo "等待本数据集所有模型完成..."
        for pid in "${PIDS[@]}"; do
            wait "$pid"
        done
    fi
done

echo "=========================================="
echo "✓ 所有并行评估完成"
echo "结果保存在: $OUTPUT_BASE"
echo "=========================================="
