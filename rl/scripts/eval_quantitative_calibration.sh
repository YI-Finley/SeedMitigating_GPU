#!/bin/bash
# 量化校准评估 (Table 1/2/4 + Figure 4)

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
echo "开始量化校准评估 (Table 1/2/4 + Figure 4)"
echo "=========================================="

bash "$SCRIPT_DIR/eval_response_level_beyondaime.sh"
bash "$SCRIPT_DIR/eval_response_level_aime.sh"
bash "$SCRIPT_DIR/eval_simpleqa.sh"

python3 "$EVAL_DIR/generate_table1.py"
python3 "$EVAL_DIR/generate_table2.py"
python3 "$EVAL_DIR/generate_table4.py"
python3 "$EVAL_DIR/generate_figure4.py"

echo "=========================================="
echo "✓ 量化校准评估完成"
echo "=========================================="
