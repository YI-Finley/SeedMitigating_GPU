#!/bin/bash
# 同时跑 thinking=off 与 thinking=on，两路并行占满 NPU
set -e

# 可通过环境变量覆盖
MODELS="${MODELS:-/data1/modelscope/Qwen/Qwen3-4B-Instruct-2507,/data1/modelscope/Qwen/Qwen3-4B-Thinking-2507,/data1/modelscope/Qwen/Qwen3-8B,/data1/modelscope/LLM-Research/Meta-Llama-3___1-8B-Instruct}"
DATASETS="${DATASETS:-beyondaime,aime_2024,aime_2025,hotpotqa,simpleqa}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-81920}"
WORKERS_PER_NPU="${WORKERS_PER_NPU:-2}"
WORKERS_PER_NPU_4B="${WORKERS_PER_NPU_4B:-4}"
WORKERS_PER_NPU_8B="${WORKERS_PER_NPU_8B:-3}"
WORKERS_PER_NPU_JUDGE="${WORKERS_PER_NPU_JUDGE:-1}"

# 本地 judge（默认 Qwen3-14B）
LOCAL_JUDGE_MODEL="${LOCAL_JUDGE_MODEL:-/data1/modelscope/Qwen/Qwen3-14B}"
LOCAL_JUDGE_DEVICE="${LOCAL_JUDGE_DEVICE:-npu:0}"
LOCAL_JUDGE_USE_CHAT_TEMPLATE="${LOCAL_JUDGE_USE_CHAT_TEMPLATE:-0}"

# 两路 NPU 切分
NPUS_OFF="${NPUS_OFF:-0,1,2,3}"
NPUS_ON="${NPUS_ON:-4,5,6,7}"

run_stream () {
    local thinking_mode="$1"
    local npus="$2"
    local tag="$3"

    export MODELS
    export DATASETS
    export THINKING_MODE="$thinking_mode"
    export MAX_NEW_TOKENS
    export NPUS="$npus"
    export WORKERS_PER_NPU
    export WORKERS_PER_NPU_4B
    export WORKERS_PER_NPU_8B
    export WORKERS_PER_NPU_JUDGE

    # 禁用外部 judge
    export LLM_JUDGE_MODEL=""
    export OPENAI_API_BASE=""

    # 启用本地 judge
    export LOCAL_JUDGE_MODEL
    export LOCAL_JUDGE_DEVICE
    export LOCAL_JUDGE_USE_CHAT_TEMPLATE

    "/root/SeedMitigating/rl/scripts/eval_paper_plus_fact_parallel.sh" \
        > "/data2/SeedMitigating-output/paper_reproduction/local_full_eval_${tag}_$(date +%Y%m%d_%H%M%S).log" 2>&1
}

# 并行启动两路
run_stream "off" "$NPUS_OFF" "thinking_off" &
PID_OFF=$!
run_stream "on" "$NPUS_ON" "thinking_on" &
PID_ON=$!

wait "$PID_OFF"
wait "$PID_ON"

echo "两路评估完成"
