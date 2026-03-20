#!/bin/bash

# 并行评估3个模型在BeyondAIME上的表现
# 使用空闲的NPU 1, 4, 5

set -e

# 激活NPU环境
source /data1/conda/etc/profile.d/conda.sh
conda activate truthrl-verl-npu

# 激活CANN环境（如存在）
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# 离线模式（使用本地缓存）
export HF_HUB_CACHE=/data2/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 配置
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_BASE="/data2/SeedMitigating-output/paper_reproduction"
EVAL_SCRIPT="/root/SeedMitigating/scripts/evaluate_model_fixed.py"
DATASET="beyondaime"

# 3个模型和对应的NPU
declare -A MODEL_NPU_MAP
MODEL_NPU_MAP["confidence_brier"]=1
MODEL_NPU_MAP["ppo_value"]=4
MODEL_NPU_MAP["confidence_min"]=5

echo "================================================================================"
echo "并行评估BeyondAIME (Table 1)"
echo "模型数量: ${#MODEL_NPU_MAP[@]}"
echo "数据集: $DATASET"
echo "================================================================================"

# 启动所有评估进程
PIDS=()
for model in "${!MODEL_NPU_MAP[@]}"; do
    npu=${MODEL_NPU_MAP[$model]}

    echo ""
    echo "启动模型: $model (NPU $npu)"

    MODEL_PATH="$OUTPUT_BASE/checkpoints/$model"
    OUTPUT_DIR="$OUTPUT_BASE/evaluation/response_level/beyondaime/$model"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "警告: 模型路径不存在: $MODEL_PATH"
        continue
    fi

    mkdir -p "$OUTPUT_DIR"

    STRATEGY="auto"
    case "$model" in
        confidence_brier) STRATEGY="verbalized_brier" ;;
        ppo_value) STRATEGY="ppo_value" ;;
        confidence_min) STRATEGY="claim_minimum" ;;
    esac

    CRITIC_ARG=""
    if [ "$STRATEGY" == "ppo_value" ]; then
        if [ -d "$MODEL_PATH/critic" ]; then
            CRITIC_ARG="--critic_path $MODEL_PATH/critic"
        elif [ -d "$MODEL_PATH/../critic" ]; then
            CRITIC_ARG="--critic_path $MODEL_PATH/../critic"
        fi
    fi

    # 在后台启动评估
    ASCEND_RT_VISIBLE_DEVICES=$npu CUDA_VISIBLE_DEVICES="" python3 "$EVAL_SCRIPT" \
        --model_path "$MODEL_PATH" \
        --base_model "$BASE_MODEL" \
        --dataset "$DATASET" \
        --output_dir "$OUTPUT_DIR" \
        --device "npu:0" \
        --strategy "$STRATEGY" \
        $CRITIC_ARG \
        > "$OUTPUT_DIR/${DATASET}_eval.log" 2>&1 &

    pid=$!
    PIDS+=($pid)
    echo "进程ID: $pid"
done

echo ""
echo "================================================================================"
echo "所有评估进程已启动"
echo "进程IDs: ${PIDS[@]}"
echo "================================================================================"

# 等待所有进程完成
echo ""
echo "等待所有评估完成..."
for pid in "${PIDS[@]}"; do
    wait $pid
    echo "进程 $pid 已完成"
done

echo ""
echo "================================================================================"
echo "所有BeyondAIME评估完成！"
echo "================================================================================"
