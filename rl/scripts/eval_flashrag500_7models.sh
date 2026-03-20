#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/code/SeedMitigating
cd "$ROOT"

source /data1/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate truthrl-verl-npu 2>/dev/null || true

export NPUS="${NPUS:-0,1,2,3,4,5,6,7}"
export VLLM_TP="${VLLM_TP:-8}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export MAX_SAMPLES="${MAX_SAMPLES:-500}"
export VLLM_MEM_UTIL="${VLLM_MEM_UTIL:-0.72}"
export VLLM_MAX_LEN="${VLLM_MAX_LEN:-4096}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-1024}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.0}"
export GEN_TOP_P="${GEN_TOP_P:-1.0}"
export GEN_TOP_K="${GEN_TOP_K:--1}"
export GEN_DO_SAMPLE="${GEN_DO_SAMPLE:-false}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/code/SeedMitigating/output/eval_flashrag500_7models}"

DATASETS=(
  flashrag_popqa
  flashrag_nq
  flashrag_hotpotqa
  flashrag_musique
  flashrag_2wikimultihopqa
  flashrag_triviaqa
  flashrag_bamboogle
)

MODELS=(
  "Binary Reward (TriviaQA)|/root/code/SeedMitigating/output/merged_hf/Qwen3-4B-Instruct-Binary-Reward-20260308_224555_ws4_modelresume_global_step_400"
  "Verbalized CE (TriviaQA)|/root/code/SeedMitigating/output/merged_hf/Qwen3-4B-Instruct-Verbalized-CE-20260307_235307_global_step_240"
  "Verbalized Brier (TriviaQA)|/root/code/SeedMitigating/output/merged_hf/Qwen3-4B-Instruct-Verbalized-Brier-20260307_143844_global_step_400"
  "Verbalized CE (NQ+HotpotQA)|/root/code/SeedMitigating/output/merged_hf/Qwen3-4B-Instruct-Verbalized-CE-20260305_200541_global_step_600"
  "Binary Reward (NQ+HotpotQA)|/root/code/SeedMitigating/output/merged_hf/Qwen3-4B-Instruct-Binary-Reward-20260303_222418_global_step_520"
  "Verbalized Brier (NQ+HotpotQA)|/root/code/SeedMitigating/output/merged_hf/Qwen3-4B-Instruct-Verbalized-Brier-20260302_230736_global_step_480"
  "Qwen3-4B-Instruct-2507 (Base)|/data2/cache/huggingface/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
)

mkdir -p "$OUTPUT_ROOT"

for item in "${MODELS[@]}"; do
  model_id="${item%%|*}"
  model_path="${item#*|}"

  if [ ! -d "$model_path" ]; then
    echo "[SKIP] missing model: $model_id -> $model_path"
    continue
  fi

  for ds in "${DATASETS[@]}"; do
    result_file="$OUTPUT_ROOT/response_level/$model_id/$ds/results.json"
    if [ -s "$result_file" ]; then
      echo "[$(date '+%F %T')] [SKIP_DONE] model=$model_id dataset=$ds result=$result_file"
      continue
    fi
    echo "[$(date '+%F %T')] [RUN] model=$model_id dataset=$ds"
    BASE_MODEL="$model_path" \
    BASE_NAME="$model_id" \
    MODEL_PATH="$model_path" \
    DATASET="$ds" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    NPUS="$NPUS" \
    VLLM_TP="$VLLM_TP" \
    BATCH_SIZE="$BATCH_SIZE" \
    MAX_SAMPLES="$MAX_SAMPLES" \
    VLLM_MEM_UTIL="$VLLM_MEM_UTIL" \
    VLLM_MAX_LEN="$VLLM_MAX_LEN" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    VLLM_MAX_NUM_BATCHED_TOKENS="$VLLM_MAX_NUM_BATCHED_TOKENS" \
    VLLM_MAX_NUM_SEQS="$VLLM_MAX_NUM_SEQS" \
    GEN_TEMPERATURE="$GEN_TEMPERATURE" \
    GEN_TOP_P="$GEN_TOP_P" \
    GEN_TOP_K="$GEN_TOP_K" \
    GEN_DO_SAMPLE="$GEN_DO_SAMPLE" \
    bash "$ROOT/rl/scripts/eval_confidence.sh"
    echo "[$(date '+%F %T')] [DONE] model=$model_id dataset=$ds"
  done
done

echo "[$(date '+%F %T')] [ALL DONE]"
