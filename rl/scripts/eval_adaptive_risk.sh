#!/bin/bash
# Adaptive Risk评估 (Figure 6 / 7)

set -e

source /data1/conda/etc/profile.d/conda.sh
conda activate truthrl-verl-npu

if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

export HF_HUB_CACHE=/data2/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

SCRIPT_DIR="/root/SeedMitigating/rl/scripts"
EVAL_DIR="/root/SeedMitigating/rl/evaluation"

echo "=========================================="
echo "Adaptive Risk评估 (BeyondAIME)"
echo "=========================================="

bash "$SCRIPT_DIR/eval_response_level_beyondaime.sh"
python3 "$EVAL_DIR/generate_figure6.py"
python3 "$EVAL_DIR/generate_figure7.py"

echo "=========================================="
echo "✓ Adaptive Risk评估完成"
echo "=========================================="
