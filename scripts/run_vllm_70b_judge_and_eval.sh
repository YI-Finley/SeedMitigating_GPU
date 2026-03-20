#!/bin/bash
set -e

source /data1/conda/etc/profile.d/conda.sh
conda activate truthrl-verl-npu

MODEL_DIR="/data1/modelscope/LLM-Research/Meta-Llama-3___1-70B-Instruct"
SERVER_LOG="/data2/SeedMitigating-output/vllm_70b_judge.log"
EVAL_LOG="/data2/SeedMitigating-output/parallel_smoke_70b_judge.log"

# 等待权重文件下载完成
while true; do
  if [ -d "$MODEL_DIR" ]; then
    if find "$MODEL_DIR" -maxdepth 3 -type f \( -name "*.safetensors" -o -name "*.pth" -o -name "pytorch_model.bin" \) | grep -q .; then
      break
    fi
  fi
  echo "等待70B权重下载完成..."
  sleep 60
done

echo "启动 vLLM 70B Judge..."
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_DEVICE=npu nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --tensor-parallel-size 4 \
  --served-model-name llama3.1-70b \
  --dtype float16 \
  --trust-remote-code \
  --port 8000 \
  > "$SERVER_LOG" 2>&1 &

# 等待服务就绪
for i in $(seq 1 120); do
  if curl -s http://127.0.0.1:8000/v1/models | grep -q "llama3.1-70b"; then
    echo "vLLM 已就绪"
    break
  fi
  sleep 5
done

# 启动评估（NPU4-7），SimpleQA 小样本
OPENAI_API_BASE="http://127.0.0.1:8000/v1" \
OPENAI_API_KEY="local-key" \
NPUS="4,5,6,7" \
MODELS="/data1/modelscope/Qwen/Qwen3-4B-Instruct-2507" \
DATASETS="simpleqa" \
MAX_SAMPLES_TOTAL=16 \
THINKING_MODE=off \
LLM_JUDGE_MODEL="llama3.1-70b" \
LOCAL_JUDGE_MODEL="" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
bash "/root/SeedMitigating/rl/scripts/eval_paper_plus_fact_parallel.sh" > "$EVAL_LOG" 2>&1

