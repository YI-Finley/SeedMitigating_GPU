#!/bin/bash
# Evaluate cheap external models via noapi for response-level metrics.

set -e

export OPENAI_API_BASE="${OPENAI_API_BASE:-https://noapi.ggb.today/v1}"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY is required"
    exit 1
fi

OUTPUT_ROOT="${OUTPUT_BASE:-/data2/SeedMitigating-output/paper_reproduction}"
if [ ! -d "$OUTPUT_ROOT" ]; then
    OUTPUT_ROOT="/root/SeedMitigating/output/paper_reproduction"
fi

EVAL_SCRIPT="/root/SeedMitigating/scripts/evaluate_external_model.py"

MODELS=(
    "gpt_4o_mini:gpt-4o-mini"
)

DATASETS=(
    "beyondaime"
    "simpleqa"
)

SIMPLEQA_GRADER_MODEL="${SIMPLEQA_GRADER_MODEL:-gpt-4o-mini}"
NOAPI_CLAIM_LEVEL="${NOAPI_CLAIM_LEVEL:-0}"

MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

for entry in "${MODELS[@]}"; do
    model_dir="${entry%%:*}"
    api_model="${entry##*:}"

    for dataset in "${DATASETS[@]}"; do
        OUTPUT_DIR="$OUTPUT_ROOT/evaluation/response_level/$dataset/$model_dir"
        mkdir -p "$OUTPUT_DIR"

        SIMPLEQA_GRADER_ARGS=""
        if [ "$dataset" == "simpleqa" ] && [ -n "$SIMPLEQA_GRADER_MODEL" ]; then
            SIMPLEQA_GRADER_ARGS="--simpleqa_grader $SIMPLEQA_GRADER_MODEL"
        fi

        echo "Evaluating (response-level) $api_model on $dataset -> $OUTPUT_DIR"
        python3 "$EVAL_SCRIPT" \
            --model "$api_model" \
            --dataset "$dataset" \
            --output_dir "$OUTPUT_DIR" \
            --strategy verbalized_brier \
            --temperature 0 \
            --max_tokens 1024 \
            --resume \
            $MAX_SAMPLES_ARG \
            $SIMPLEQA_GRADER_ARGS

        if [ "$NOAPI_CLAIM_LEVEL" == "1" ] && [ "$dataset" == "beyondaime" ]; then
            CLAIM_DIR="$OUTPUT_ROOT/evaluation/claim_level/$dataset/$model_dir"
            mkdir -p "$CLAIM_DIR"
            echo "Evaluating (claim-level prompt) $api_model on $dataset -> $CLAIM_DIR"
            python3 "$EVAL_SCRIPT" \
                --model "$api_model" \
                --dataset "$dataset" \
                --output_dir "$CLAIM_DIR" \
                --strategy verbalized_brier \
                --include_claim_tags \
                --temperature 0 \
                --max_tokens 1024 \
                --resume \
                $MAX_SAMPLES_ARG
        fi
    done

    if [ -n "$NOAPI_LABEL_MODEL" ]; then
        LABELS_DIR="$OUTPUT_ROOT/evaluation/claim_level_labels_noapi"
        mkdir -p "$LABELS_DIR"
        INPUT_DIR="$OUTPUT_ROOT/evaluation/claim_level/beyondaime"
        OUTPUT_FILE="$LABELS_DIR/${model_dir}.jsonl"
        python3 "/root/SeedMitigating/scripts/label_claims_noapi.py" \
            --input_dir "$INPUT_DIR" \
            --model_dir "$model_dir" \
            --output_file "$OUTPUT_FILE" \
            --model "$NOAPI_LABEL_MODEL" \
            --resume
    fi

done

echo "Frontier noapi evaluation done."
