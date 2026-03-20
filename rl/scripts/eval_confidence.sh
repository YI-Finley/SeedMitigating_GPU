#!/bin/bash
# Run Confidence evaluation (verbalized_brier strategy) on a single model and dataset.
# Supports multi-NPU parallel inference via vLLM tensor parallelism.
#
# Usage:
#   bash rl/scripts/eval_confidence.sh
#   NPUS=0,1 bash rl/scripts/eval_confidence.sh
#   NPUS=0,1,2,3 DATASET=flashrag_nq bash rl/scripts/eval_confidence.sh
#   MODEL_PATH=/path/to/model DATASET=flashrag_nq NPUS=0,1 bash rl/scripts/eval_confidence.sh
#   MAX_SAMPLES=500 NPUS=0,1 bash rl/scripts/eval_confidence.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL_SCRIPT="${EVAL_SCRIPT:-$ROOT_DIR/scripts/evaluate_model_fixed.py}"

# ── Core Parameters ─────────────────────────────────────────────────────────
DATASET="${DATASET:-flashrag_nq}"   # flashrag_popqa flashrag_nq flashrag_hotpotqa flashrag_musique flashrag_2wikimultihopqa flashrag_triviaqa flashrag_bamboogle
BASE_MODEL="${BASE_MODEL:-/root/code/SeedMitigating/model/Qwen3-4B-Instruct-Verbalized-Brier-20260306_223234_global_step_920}"
MODEL_PATH="${MODEL_PATH:-}"
FLASHRAG_500_DIR="${FLASHRAG_500_DIR:-/root/code/SeedMitigating/verl/data/flash_rag_500}"

# Multi-card: comma-separated NPU indices, e.g. "0,1" or "0,1,2,3"
NPUS="${NPUS:-0,1,2,3,4,5,6,7}"

# Derive card count from NPUS for vllm_tensor_parallel_size
IFS=',' read -ra _npu_arr <<< "$NPUS"
NUM_NPUS="${#_npu_arr[@]}"
unset IFS

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/code/SeedMitigating/output/evaluation}"

BASE_NAME="${BASE_NAME:-$(basename "$BASE_MODEL")}"
OUT_DIR="${OUT_DIR:-$OUTPUT_ROOT/response_level/$BASE_NAME/$DATASET}"

# ── Evaluation Parameters ────────────────────────────────────────────────────
MAX_SAMPLES="${MAX_SAMPLES:-500}"
BATCH_SIZE="${BATCH_SIZE:-128}"
# VLLM_TP can be overridden manually; defaults to number of cards
VLLM_TP="${VLLM_TP:-$NUM_NPUS}"
VLLM_MEM_UTIL="${VLLM_MEM_UTIL:-0.72}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-1024}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
# Reproducible decoding defaults: deterministic (greedy-like)
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.0}"
GEN_TOP_P="${GEN_TOP_P:-1.0}"
GEN_TOP_K="${GEN_TOP_K:--1}"
GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-false}"

# ── Environment Setup ────────────────────────────────────────────────────────
set +u
source /data1/conda/etc/profile.d/conda.sh 2>/dev/null \
    || source /workspace/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
command -v conda >/dev/null 2>&1 && conda activate truthrl-verl-npu 2>/dev/null || true
[ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
[ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]       && source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

export HF_HUB_CACHE="${HF_HUB_CACHE:-/data2/cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data2/cache/huggingface/datasets}"
export VLLM_DEVICE="${VLLM_DEVICE:-npu}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
export VLLM_VERSION="${VLLM_VERSION:-0.11.0}"

# ── Resolve Model Path ───────────────────────────────────────────────────────
if [ -z "$MODEL_PATH" ]; then
    if [[ "$BASE_MODEL" == /* ]] && [ -d "$BASE_MODEL" ]; then
        MODEL_PATH="$BASE_MODEL"
    else
        model_cache_dir="$HF_HUB_CACHE/models--${BASE_MODEL//\//--}"
        latest_snapshot="$(ls -1 "$model_cache_dir/snapshots" 2>/dev/null | sort | tail -n 1)"
        if [ -n "$latest_snapshot" ]; then
            MODEL_PATH="$model_cache_dir/snapshots/$latest_snapshot"
        else
            echo "ERROR: MODEL_PATH is not set and model not found at:" >&2
            echo "  Local path: $BASE_MODEL (does not exist or is not a directory)" >&2
            echo "  HF cache:   $model_cache_dir" >&2
            exit 1
        fi
    fi
fi

# ── Dataset Path Mapping (flash_rag_500 only) ───────────────────────────────
case "$DATASET" in
    flashrag_popqa) DATASET_FILE="$FLASHRAG_500_DIR/popqa_test_500.parquet" ;;
    flashrag_nq) DATASET_FILE="$FLASHRAG_500_DIR/nq_test_500.parquet" ;;
    flashrag_hotpotqa) DATASET_FILE="$FLASHRAG_500_DIR/hotpotqa_test_500.parquet" ;;
    flashrag_musique) DATASET_FILE="$FLASHRAG_500_DIR/musique_test_500.parquet" ;;
    flashrag_2wikimultihopqa) DATASET_FILE="$FLASHRAG_500_DIR/2wikimultihopqa_test_500.parquet" ;;
    flashrag_triviaqa) DATASET_FILE="$FLASHRAG_500_DIR/triviaqa_test_500.parquet" ;;
    flashrag_bamboogle) DATASET_FILE="$FLASHRAG_500_DIR/bamboogle_test.parquet" ;;
    *)
        echo "ERROR: Unsupported DATASET=$DATASET" >&2
        echo "Supported: flashrag_popqa flashrag_nq flashrag_hotpotqa flashrag_musique flashrag_2wikimultihopqa flashrag_triviaqa flashrag_bamboogle" >&2
        exit 1
        ;;
esac

if [ ! -f "$DATASET_FILE" ]; then
    echo "ERROR: Dataset file not found: $DATASET_FILE" >&2
    exit 1
fi

# ── Run Evaluation ───────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/eval.log"

echo "[$(date '+%F %T')] Starting Confidence evaluation (multi-card mode supported)"
echo "  dataset         : $DATASET"
echo "  dataset_file    : $DATASET_FILE"
echo "  model_path      : $MODEL_PATH"
echo "  output_dir      : $OUT_DIR"
echo "  npus            : $NPUS  (${NUM_NPUS} cards total)"
echo "  tensor_parallel : $VLLM_TP"
echo "  batch_size      : $BATCH_SIZE"
echo "  max_samples     : $MAX_SAMPLES"
echo "  max_new_tokens  : $MAX_NEW_TOKENS"
echo "  do_sample       : $GEN_DO_SAMPLE"

ASCEND_RT_VISIBLE_DEVICES="$NPUS" CUDA_VISIBLE_DEVICES="" \
python3 "$EVAL_SCRIPT" \
    --model_path      "$MODEL_PATH" \
    --base_model      "$BASE_MODEL" \
    --dataset         "$DATASET" \
    --dataset_file    "$DATASET_FILE" \
    --output_dir      "$OUT_DIR" \
    --device          npu:0 \
    --strategy        verbalized_brier \
    --risk_threshold  0.5 \
    --batch_size      "$BATCH_SIZE" \
    --max_samples     "$MAX_SAMPLES" \
    --use_vllm \
    --vllm_tensor_parallel_size     "$VLLM_TP" \
    --vllm_gpu_memory_utilization   "$VLLM_MEM_UTIL" \
    --vllm_max_model_len            "$VLLM_MAX_LEN" \
    --vllm_trust_remote_code \
    --vllm_seed 42 \
    --vllm_block_size 128 \
    --vllm_enable_prefix_caching 0 \
    --vllm_max_num_batched_tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
    --use_chat_template \
    --thinking_mode off \
    --offline \
    --simpleqa_grader none \
    --gen_temperature  "$GEN_TEMPERATURE" \
    --gen_top_p        "$GEN_TOP_P" \
    --gen_top_k        "$GEN_TOP_K" \
    --gen_min_p        0.0 \
    --gen_do_sample    "$GEN_DO_SAMPLE" \
    --max_new_tokens   "$MAX_NEW_TOKENS" \
    --save_io 2>&1 | tee "$LOG_FILE"

echo "[$(date '+%F %T')] Done. Results saved to $OUT_DIR/results.json"

python3 "$ROOT_DIR/rl/evaluation/plot_confidence_results.py" \
    --dataset_dir "$OUT_DIR" \
    --output "$OUT_DIR/confidence_calibration.png" \
    --model_name "$BASE_NAME" \
    --n_bins 10 \
    --threshold 0.5
