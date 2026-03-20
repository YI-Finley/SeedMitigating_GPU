#!/bin/bash

set -euo pipefail

GPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SEEDMIT_PROJECT_ROOT="${SEEDMIT_PROJECT_ROOT:-$(cd "$GPU_DIR/.." && pwd)}"
export SEEDMIT_BASE_MODEL="${SEEDMIT_BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
export SEEDMIT_OUTPUT_BASE="${SEEDMIT_OUTPUT_BASE:-$SEEDMIT_PROJECT_ROOT/output_gpu/paper_reproduction}"
export SEEDMIT_HF_CACHE="${SEEDMIT_HF_CACHE:-$HOME/.cache/huggingface}"
export SEEDMIT_ENV_ACTIVATE="${SEEDMIT_ENV_ACTIVATE:-$HOME/venvs/seedmitigating-gpu/bin/activate}"
export SEEDMIT_CUDA_MODULE="${SEEDMIT_CUDA_MODULE:-cuda/11.4}"
export SEEDMIT_PYTHON="${SEEDMIT_PYTHON:-python3}"

if [ -n "${SEEDMIT_CUDA_MODULE}" ] && type module >/dev/null 2>&1; then
    module load "${SEEDMIT_CUDA_MODULE}" || true
fi

if [ -f "${SEEDMIT_ENV_ACTIVATE}" ]; then
    # shellcheck disable=SC1090
    source "${SEEDMIT_ENV_ACTIVATE}"
else
    echo "[WARN] Virtualenv activate script not found: ${SEEDMIT_ENV_ACTIVATE}"
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export HF_HOME="${HF_HOME:-$SEEDMIT_HF_CACHE}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SEEDMIT_HF_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$SEEDMIT_HF_CACHE}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_DEVICE="${VLLM_DEVICE:-cuda}"
export WANDB_MODE="${WANDB_MODE:-offline}"

seedmit_count_visible_gpus() {
    if [ -n "${SEEDMIT_NUM_GPUS:-}" ]; then
        echo "${SEEDMIT_NUM_GPUS}"
        return
    fi

    if [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [[ "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
        echo "${SLURM_GPUS_ON_NODE}"
        return
    fi

    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        "${SEEDMIT_PYTHON}" - <<'PY'
import os
spec = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
print(len([x for x in spec.split(",") if x.strip()]))
PY
        return
    fi

    "${SEEDMIT_PYTHON}" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
PY
}

export SEEDMIT_NUM_GPUS="$(seedmit_count_visible_gpus)"
export SEEDMIT_TRAIN_BATCH_SIZE="${SEEDMIT_TRAIN_BATCH_SIZE:-128}"
export SEEDMIT_VAL_BATCH_SIZE="${SEEDMIT_VAL_BATCH_SIZE:-128}"
export SEEDMIT_MAX_PROMPT_LENGTH="${SEEDMIT_MAX_PROMPT_LENGTH:-512}"
export SEEDMIT_MAX_RESPONSE_LENGTH="${SEEDMIT_MAX_RESPONSE_LENGTH:-8192}"
export SEEDMIT_PPO_MINI_BATCH_SIZE="${SEEDMIT_PPO_MINI_BATCH_SIZE:-16}"
export SEEDMIT_PPO_MICRO_BATCH_SIZE="${SEEDMIT_PPO_MICRO_BATCH_SIZE:-2}"
export SEEDMIT_CRITIC_MICRO_BATCH_SIZE="${SEEDMIT_CRITIC_MICRO_BATCH_SIZE:-2}"
export SEEDMIT_ROLLOUT_N="${SEEDMIT_ROLLOUT_N:-8}"
export SEEDMIT_ROLLOUT_GPU_MEMORY_UTIL="${SEEDMIT_ROLLOUT_GPU_MEMORY_UTIL:-0.65}"
export SEEDMIT_LOGPROB_MICRO_BATCH="${SEEDMIT_LOGPROB_MICRO_BATCH:-8}"
export SEEDMIT_TRAIN_EPOCHS="${SEEDMIT_TRAIN_EPOCHS:-30}"
export SEEDMIT_SAVE_FREQ="${SEEDMIT_SAVE_FREQ:-5}"
export SEEDMIT_TEST_FREQ="${SEEDMIT_TEST_FREQ:-1}"
export SEEDMIT_CRITIC_WARMUP="${SEEDMIT_CRITIC_WARMUP:-20}"

seedmit_has_model_files() {
    local dir="$1"
    [[ -f "$dir/model.safetensors" || -f "$dir/pytorch_model.bin" || -f "$dir/model.safetensors.index.json" ]]
}

seedmit_resolve_actor_dir() {
    local base_dir="$1"
    if [ -d "$base_dir" ]; then
        local latest=""
        while IFS= read -r line; do
            latest="$line"
        done < <(ls -d "$base_dir"/global_step_* 2>/dev/null | sort -V)
        if [ -n "$latest" ] && seedmit_has_model_files "$latest/actor/huggingface"; then
            echo "$latest/actor/huggingface"
            return
        fi
        if seedmit_has_model_files "$base_dir/actor/huggingface"; then
            echo "$base_dir/actor/huggingface"
            return
        fi
    fi
    echo "$base_dir"
}

seedmit_resolve_critic_dir() {
    local base_dir="$1"
    if [ -d "$base_dir" ]; then
        local latest=""
        while IFS= read -r line; do
            latest="$line"
        done < <(ls -d "$base_dir"/global_step_* 2>/dev/null | sort -V)
        if [ -n "$latest" ] && seedmit_has_model_files "$latest/critic/huggingface"; then
            echo "$latest/critic/huggingface"
            return
        fi
        if seedmit_has_model_files "$base_dir/critic/huggingface"; then
            echo "$base_dir/critic/huggingface"
            return
        fi
    fi
    echo ""
}

seedmit_model_strategy() {
    case "$1" in
        baseline_ppo|confidence_brier|qwen3_instruct) echo "verbalized_brier" ;;
        confidence_ce) echo "verbalized_ce" ;;
        ppo_value) echo "ppo_value" ;;
        confidence_prod) echo "claim_product" ;;
        confidence_min) echo "claim_minimum" ;;
        explicit_risk) echo "explicit_risk" ;;
        *) echo "auto" ;;
    esac
}
