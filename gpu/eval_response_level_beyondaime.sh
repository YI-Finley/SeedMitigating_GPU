#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

EVAL_SCRIPT="$SEEDMIT_PROJECT_ROOT/scripts/evaluate_model_fixed.py"
OUTPUT_BASE="$SEEDMIT_OUTPUT_BASE/evaluation/response_level/beyondaime"
CHECKPOINT_BASE="$SEEDMIT_OUTPUT_BASE/checkpoints"

MODELS=(
    qwen3_instruct
    baseline_ppo
    confidence_brier
    confidence_ce
    ppo_value
    confidence_prod
    confidence_min
)

mkdir -p "$OUTPUT_BASE"

for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="$CHECKPOINT_BASE/$MODEL"
    if [ "$MODEL" = "qwen3_instruct" ]; then
        MODEL_PATH="$SEEDMIT_BASE_MODEL"
    fi
    OUTPUT_DIR="$OUTPUT_BASE/$MODEL"
    STRATEGY="$(seedmit_model_strategy "$MODEL")"
    CRITIC_ARG=()
    if [ "$STRATEGY" = "ppo_value" ]; then
        critic_path="$(seedmit_resolve_critic_dir "$CHECKPOINT_BASE/$MODEL")"
        if [ -n "$critic_path" ]; then
            CRITIC_ARG=(--critic_path "$critic_path")
        fi
    fi
    "$SEEDMIT_PYTHON" "$EVAL_SCRIPT" \
        --model_path "$MODEL_PATH" \
        --base_model "$SEEDMIT_BASE_MODEL" \
        --dataset beyondaime \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 1 \
        --device cuda:0 \
        --risk_threshold 0.5 \
        --strategy "$STRATEGY" \
        "${CRITIC_ARG[@]}"
done
