#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

EVAL_SCRIPT="$SEEDMIT_PROJECT_ROOT/scripts/evaluate_test_time_scaling.py"
MODELS=(baseline_ppo ppo_value confidence_brier)
K_VALUES="${SEEDMIT_TTS_K_VALUES:-1,2,4,8,16,32}"
DATASET="${SEEDMIT_TTS_DATASET:-beyondaime}"

for MODEL in "${MODELS[@]}"; do
    MODEL_BASE="$SEEDMIT_OUTPUT_BASE/checkpoints/$MODEL"
    MODEL_PATH="$(seedmit_resolve_actor_dir "$MODEL_BASE")"
    OUTPUT_DIR="$SEEDMIT_OUTPUT_BASE/test_time_scaling/$MODEL"
    STRATEGY="$(seedmit_model_strategy "$MODEL")"
    CRITIC_ARG=()
    if [ "$STRATEGY" = "ppo_value" ]; then
        critic_path="$(seedmit_resolve_critic_dir "$MODEL_BASE")"
        if [ -n "$critic_path" ]; then
            CRITIC_ARG=(--critic_path "$critic_path")
        fi
    fi
    "$SEEDMIT_PYTHON" "$EVAL_SCRIPT" \
        --model_path "$MODEL_PATH" \
        --base_model "$SEEDMIT_BASE_MODEL" \
        --dataset "$DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --k_values "$K_VALUES" \
        --device cuda:0 \
        --strategy "$STRATEGY" \
        "${CRITIC_ARG[@]}"
done
