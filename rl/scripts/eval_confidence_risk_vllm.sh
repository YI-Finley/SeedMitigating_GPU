#!/bin/bash
# All-datasets eval: confidence + explicit_risk (vLLM + save_io)
# - confidence: verbalized_brier with risk_prompt_metrics grid
# - explicit_risk: t-grid sharded eval with merge
# Usage:
#   bash rl/scripts/eval_all_datasets_confidence_risk_vllm.sh
#   MAX_SAMPLES=5 DATASETS="simpleqa_gtgrpo,flashrag_hotpotqa" bash rl/scripts/eval_all_datasets_confidence_risk_vllm.sh

set -e

# Env scripts may contain benign commands that return non-zero (e.g., mkdir on existing dirs).
# Temporarily disable errexit during environment setup, then validate activation explicitly.
set +e
source /data1/conda/etc/profile.d/conda.sh 2>/dev/null || source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate truthrl-verl-npu

if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
set -e

if [ "${CONDA_DEFAULT_ENV:-}" != "truthrl-verl-npu" ]; then
    echo "conda activate failed: expected truthrl-verl-npu, got ${CONDA_DEFAULT_ENV:-<none>}" >&2
    exit 1
fi

export PYTHONPATH="/root/vllm:/root/vllm-ascend:${PYTHONPATH:-}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"
export HF_HUB_CACHE=/data2/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/data2/cache/huggingface/datasets
export VLLM_DEVICE=npu
export GT_GRPO_ROOT=/root/code/Fact_Reasoning/GT_GRPO

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

OUTPUT_BASE="/root/code/SeedMitigating/output"


MODEL_PATH="${MODEL_PATH:-$OUTPUT_BASE/checkpoints/Qwen3-4B-Instruct-Explicit-Risk-20260204_144342/global_step_40/actor/huggingface}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$ROOT_DIR/scripts/evaluate_model_fixed.py}"
MERGE_SCRIPT="${MERGE_SCRIPT:-$ROOT_DIR/scripts/merge_sharded_results.py}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_BASE/evaluation}"
OUT_CONF_BASE="${OUT_CONF_BASE:-$OUTPUT_ROOT/response_level/all_datasets_confidence_vllm}"
OUT_RISK_BASE="${OUT_RISK_BASE:-$OUTPUT_ROOT/explicit_risk/all_datasets_explicitrisk_vllm}"

DATASETS_DEFAULT=(
    # simpleqa_gtgrpo
    # nq_hotpotqa_searchr1
    # flashrag_popqa
    flashrag_nq
    flashrag_hotpotqa
    # flashrag_musique
    # flashrag_2wikimultihopqa
    # flashrag_triviaqa
    # flashrag_bamboogle
)

if [ -n "${DATASETS:-}" ]; then
    DATASETS_CLEAN="${DATASETS//,/ }"
    read -r -a DATASET_LIST <<< "$DATASETS_CLEAN"
else
    DATASET_LIST=("${DATASETS_DEFAULT[@]}")
fi

T_VALUES_DEFAULT=(0 0.2 0.4 0.6 0.8 1.0)
if [ -n "${T_VALUES:-}" ]; then
    T_VALUES_CLEAN="${T_VALUES//,/ }"
    read -r -a T_VALUES_LIST <<< "$T_VALUES_CLEAN"
else
    T_VALUES_LIST=("${T_VALUES_DEFAULT[@]}")
fi

NPU_LIST_DEFAULT=(0 1 2 3 4 5 6 7)
if [ -n "${NPU_LIST:-}" ]; then
    NPU_LIST_CLEAN="${NPU_LIST//,/ }"
    read -r -a NPUS <<< "$NPU_LIST_CLEAN"
else
    NPUS=("${NPU_LIST_DEFAULT[@]}")
fi
NPU_COUNT="${#NPUS[@]}"

CONF_STRATEGY="${CONF_STRATEGY:-verbalized_brier}"
RISK_STRATEGY="${RISK_STRATEGY:-explicit_risk}"
RISK_PROMPT_GRID="${RISK_PROMPT_GRID:-0,0.2,0.4,0.6,0.8,1.0}"
RISK_PROMPT_T0="${RISK_PROMPT_T0:-0.0}"

SAMPLE_STRIDE="${SAMPLE_STRIDE:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_IO="${SAVE_IO:-1}"

VLLM_TP="${VLLM_TP:-$NPU_COUNT}"
VLLM_MEM_UTIL="${VLLM_MEM_UTIL:-0.8}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-32768}"
VLLM_SEED="${VLLM_SEED:-42}"
VLLM_COMPILATION_CONFIG="${VLLM_COMPILATION_CONFIG:-0}"
export VLLM_COMPILATION_CONFIG

GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.7}"
GEN_TOP_P="${GEN_TOP_P:-0.8}"
GEN_TOP_K="${GEN_TOP_K:-20}"
GEN_MIN_P="${GEN_MIN_P:-0.0}"
GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-true}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20480}"

mkdir -p "$OUT_CONF_BASE" "$OUT_RISK_BASE"

SIMPLEQA_ARGS="--simpleqa_grader none"
if [ -n "${LOCAL_JUDGE_MODEL:-}" ]; then
    SIMPLEQA_ARGS="--simpleqa_grader_local $LOCAL_JUDGE_MODEL --simpleqa_grader_device ${LOCAL_JUDGE_DEVICE:-npu:0}"
    if [ -n "${LOCAL_JUDGE_USE_CHAT_TEMPLATE:-}" ]; then
        SIMPLEQA_ARGS="$SIMPLEQA_ARGS --simpleqa_grader_use_chat_template"
    fi
    if [ -n "${LOCAL_JUDGE_FILES_ONLY:-}" ]; then
        SIMPLEQA_ARGS="$SIMPLEQA_ARGS --simpleqa_grader_local_files_only"
    fi
    if [ -n "${LOCAL_JUDGE_DTYPE:-}" ]; then
        SIMPLEQA_ARGS="$SIMPLEQA_ARGS --simpleqa_grader_dtype $LOCAL_JUDGE_DTYPE"
    fi
fi

SAVE_IO_ARG=""
if [ "$SAVE_IO" = "1" ] || [ "$SAVE_IO" = "true" ]; then
    SAVE_IO_ARG="--save_io"
fi

MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" != "0" ]; then
    MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

run_confidence() {
    local dataset="$1"
    local out_dir="$OUT_CONF_BASE/$dataset"
    mkdir -p "$out_dir"
    local npu_list
    npu_list=$(IFS=,; echo "${NPUS[*]}")
    echo "[confidence] dataset=$dataset npus=$npu_list out=$out_dir"
    ASCEND_RT_VISIBLE_DEVICES=$npu_list CUDA_VISIBLE_DEVICES="" \
    python3 "$EVAL_SCRIPT" \
        --model_path "$MODEL_PATH" \
        --base_model "$BASE_MODEL" \
        --dataset "$dataset" \
        --output_dir "$out_dir" \
        --device npu:0 \
        --strategy "$CONF_STRATEGY" \
        --risk_threshold 0.5 \
        --risk_prompt_metrics \
        --risk_prompt_grid "$RISK_PROMPT_GRID" \
        --risk_prompt_t0 "$RISK_PROMPT_T0" \
        --batch_size "$BATCH_SIZE" \
        $MAX_SAMPLES_ARG \
        --vllm_tensor_parallel_size "$VLLM_TP" \
        --vllm_gpu_memory_utilization "$VLLM_MEM_UTIL" \
        --vllm_max_model_len "$VLLM_MAX_LEN" \
        --vllm_trust_remote_code \
        --vllm_seed "$VLLM_SEED" \
        --use_chat_template \
        --thinking_mode off \
        --offline \
        $SIMPLEQA_ARGS \
        --gen_temperature "$GEN_TEMPERATURE" \
        --gen_top_p "$GEN_TOP_P" \
        --gen_top_k "$GEN_TOP_K" \
        --gen_min_p "$GEN_MIN_P" \
        --gen_do_sample "$GEN_DO_SAMPLE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        $SAVE_IO_ARG \
        2>&1 | tee "$out_dir/eval.log"
}

launch_risk_shard() {
    local dataset="$1"; local t="$2"; local npu="$3"; local offset="$4"
    local out_dir="$OUT_RISK_BASE/$dataset/t${t}/shard_${offset}"
    mkdir -p "$out_dir"
    ASCEND_RT_VISIBLE_DEVICES=$npu CUDA_VISIBLE_DEVICES="" \
    python3 "$EVAL_SCRIPT" \
        --model_path "$MODEL_PATH" \
        --base_model "$BASE_MODEL" \
        --dataset "$dataset" \
        --output_dir "$out_dir" \
        --device npu:0 \
        --strategy "$RISK_STRATEGY" \
        --risk_threshold "$t" \
        --batch_size "$BATCH_SIZE" \
        $MAX_SAMPLES_ARG \
        --vllm_tensor_parallel_size "$VLLM_TP" \
        --vllm_gpu_memory_utilization "$VLLM_MEM_UTIL" \
        --vllm_max_model_len "$VLLM_MAX_LEN" \
        --vllm_trust_remote_code \
        --vllm_seed "$VLLM_SEED" \
        --use_chat_template \
        --thinking_mode off \
        --offline \
        $SIMPLEQA_ARGS \
        --sample_stride "$SAMPLE_STRIDE" \
        --sample_offset "$offset" \
        --gen_temperature "$GEN_TEMPERATURE" \
        --gen_top_p "$GEN_TOP_P" \
        --gen_top_k "$GEN_TOP_K" \
        --gen_min_p "$GEN_MIN_P" \
        --gen_do_sample "$GEN_DO_SAMPLE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        $SAVE_IO_ARG \
        2>&1 | tee "$out_dir/eval.log" &
    local pid=$!
    echo "started dataset=${dataset} t=${t} offset=${offset} npu=${npu} pid=${pid} log=$out_dir/eval.log" >&2
    PIDS+=("$pid")
}

merge_risk_shards() {
    local dataset="$1"; local t="$2"
    local t_dir="$OUT_RISK_BASE/$dataset/t${t}"
    python3 "$MERGE_SCRIPT" \
        --input_root "$t_dir" \
        --output_file "$t_dir/merged_results.json" \
        --risk_threshold "$t"
}

for dataset in "${DATASET_LIST[@]}"; do
    echo "============================================================"
    echo "Dataset: $dataset"
    echo "============================================================"
    run_confidence "$dataset"

    for t in "${T_VALUES_LIST[@]}"; do
        PIDS=()
        idx=0
        for ((offset=0; offset< SAMPLE_STRIDE; offset++)); do
            npu="${NPUS[$((idx % NPU_COUNT))]}"
            launch_risk_shard "$dataset" "$t" "$npu" "$offset"
            idx=$((idx + 1))
            if [ "${#PIDS[@]}" -ge "$NPU_COUNT" ]; then
                for p in "${PIDS[@]}"; do
                    wait "$p"
                done
                PIDS=()
            fi
            sleep 2
        done
        for p in "${PIDS[@]}"; do
            wait "$p"
        done
        merge_risk_shards "$dataset" "$t"
    done
done

echo "All done"
echo "confidence output: $OUT_CONF_BASE"
echo "explicit_risk output: $OUT_RISK_BASE"
