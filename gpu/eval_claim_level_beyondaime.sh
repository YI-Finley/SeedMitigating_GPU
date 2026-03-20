#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

EVAL_SCRIPT="$SEEDMIT_PROJECT_ROOT/scripts/evaluate_model_fixed.py"
OUTPUT_BASE="$SEEDMIT_OUTPUT_BASE/evaluation/claim_level/beyondaime"
CHECKPOINT_BASE="$SEEDMIT_OUTPUT_BASE/checkpoints"

MODELS=(confidence_prod confidence_min qwen3_instruct)

for MODEL in "${MODELS[@]}"; do
    MODEL_PATH="$CHECKPOINT_BASE/$MODEL"
    if [ "$MODEL" = "qwen3_instruct" ]; then
        MODEL_PATH="$SEEDMIT_BASE_MODEL"
    fi
    OUTPUT_DIR="$OUTPUT_BASE/$MODEL"
    STRATEGY="$(seedmit_model_strategy "$MODEL")"
    "$SEEDMIT_PYTHON" "$EVAL_SCRIPT" \
        --model_path "$MODEL_PATH" \
        --base_model "$SEEDMIT_BASE_MODEL" \
        --dataset beyondaime \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 1 \
        --device cuda:0 \
        --risk_threshold 0.5 \
        --strategy "$STRATEGY"
done

LABELS_FILE="${CLAIM_LABELS_FILE:-}"
if [ -n "$LABELS_FILE" ]; then
    "$SEEDMIT_PYTHON" "$SEEDMIT_PROJECT_ROOT/rl/evaluation/generate_table3.py" --labels_file "$LABELS_FILE"
else
    "$SEEDMIT_PYTHON" "$SEEDMIT_PROJECT_ROOT/rl/evaluation/generate_table3.py"
fi
