#!/usr/bin/env bash
set -e

LOG_DIR="/data2/SeedMitigating-output/paper_reproduction/evaluation/response_level"
SCRIPT="/root/SeedMitigating/rl/scripts/eval_paper_plus_fact_parallel.sh"

run_group() {
  local pids=()
  for t in "$@"; do
    suf=$(echo "$t" | sed 's/\./p/')
    log="$LOG_DIR/hotpotqa_explicit_risk_t${suf}_2k.log"
    STRATEGY=explicit_risk \
    EXPLICIT_RISK_RANDOM=0 \
    RISK_THRESHOLD="$t" \
    OUTPUT_SUFFIX="explicit_risk_t${suf}_2k" \
    MAX_SAMPLES_TOTAL=2000 \
    MODELS="Qwen/Qwen3-4B-Instruct-2507" \
    DATASETS=hotpotqa \
    USE_ALL_NPUS=1 \
    NPUS="0,1,2,3,4,5,6,7" \
    WORKERS_PER_NPU=1 \
    DISABLE_JUDGE=1 \
    THINKING_MODE=off \
    bash "$SCRIPT" > "$log" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

run_group 0.0 0.2 0.4
run_group 0.6 0.8 1.0
