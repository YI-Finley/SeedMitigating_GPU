#!/bin/bash
# 一键生成论文所有表格与图表（除外部API部分）

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
echo "开始生成论文全部结果（除外部API）"
echo "=========================================="

if [ "$ENABLE_NOAPI" == "1" ]; then
    echo "启用 noapi 外部模型评估..."
    bash "$SCRIPT_DIR/eval_frontier_noapi.sh"
fi

bash "$SCRIPT_DIR/eval_quantitative_calibration.sh"
bash "$SCRIPT_DIR/eval_claim_level_beyondaime.sh"
bash "$SCRIPT_DIR/eval_adaptive_risk.sh"
bash "$SCRIPT_DIR/eval_test_time_scaling.sh"
bash "$SCRIPT_DIR/eval_explicit_risk.sh"

python3 "$EVAL_DIR/generate_figure1.py"
python3 "$EVAL_DIR/generate_figure5.py" ${CLAIM_LABELS_FILE:+--labels_file "$CLAIM_LABELS_FILE"} || true

echo "=========================================="
echo "✓ 全部结果生成完成"
echo "提示: Claim-level 图表需要 CLAIM_LABELS_FILE (GPT-5 标注)"
echo "=========================================="
