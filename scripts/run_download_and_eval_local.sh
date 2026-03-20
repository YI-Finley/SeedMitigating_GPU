#!/bin/bash
set -e

CACHE_DIR="/data1/modelscope"
mkdir -p "$CACHE_DIR"

# 1) 下载模型（支持断点续传）
python "/root/SeedMitigating/scripts/download_modelscope.py" \
  --models \
    Qwen/Qwen3-4B-Instruct-2507 \
    Qwen/Qwen3-4B-Thinking-2507 \
    Qwen/Qwen3-8B \
    LLM-Research/Meta-Llama-3.1-8B-Instruct \
    LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
  --cache_dir "$CACHE_DIR"

QWEN4I="$CACHE_DIR/Qwen/Qwen3-4B-Instruct-2507"
QWEN4T="$CACHE_DIR/Qwen/Qwen3-4B-Thinking-2507"
QWEN8="$CACHE_DIR/Qwen/Qwen3-8B"
LLAMA8="$CACHE_DIR/LLM-Research/Meta-Llama-3.1-8B-Instruct"
LLAMA70="$CACHE_DIR/LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

export LOCAL_JUDGE_MODEL="$LLAMA70"
export LOCAL_JUDGE_DEVICE="npu:0"
export LOCAL_JUDGE_OFFLINE=1
export LOCAL_JUDGE_DTYPE=float16
export LLM_JUDGE_MODEL=none

export DATASETS="beyondaime,aime_2024,aime_2025,hotpotqa,simpleqa"

# 2) 非思考模式（Instruct）
export MODELS="$QWEN4I,$QWEN8,$LLAMA8"
export THINKING_MODE=off
bash "/root/SeedMitigating/rl/scripts/eval_paper_plus_fact.sh"

# 3) 思考模式（Thinking）
export MODELS="$QWEN4T,$QWEN8"
export THINKING_MODE=on
bash "/root/SeedMitigating/rl/scripts/eval_paper_plus_fact.sh"
