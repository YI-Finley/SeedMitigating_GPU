#!/bin/bash
# NPU环境快速配置脚本
# 仅适配《Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning》baseline + 4个校准目标

set -e  # 遇到错误立即退出

echo "=========================================="
echo "NPU环境配置脚本"
echo "=========================================="

# 1. 检查Python版本
echo ""
echo "[1/7] 检查Python版本..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ ! $PYTHON_VERSION =~ ^3\.11\. ]]; then
    echo "❌ 错误：需要Python 3.11，当前版本为 $PYTHON_VERSION"
    echo "请运行: conda create -n behavioral-calibration python=3.11 -y"
    exit 1
fi
echo "✅ Python版本正确: $PYTHON_VERSION"

# 2. 检查CANN环境
echo ""
echo "[2/7] 检查CANN环境..."
if [ ! -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    echo "❌ 错误：CANN工具包未安装"
    echo "请从华为官网下载并安装CANN 8.3.RC1或更高版本"
    exit 1
fi
echo "✅ CANN工具包已安装"

# 3. 激活CANN环境
echo ""
echo "[3/7] 激活CANN环境..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
    echo "✅ CANN和NNAL环境已激活"
else
    echo "⚠️  警告：NNAL ATB库未安装（可选，用于加速）"
fi

# 4. 设置NPU环境变量
echo ""
echo "[4/7] 设置NPU环境变量..."
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE="offline"
echo "✅ 环境变量已设置"

# 5. 检查NPU设备
echo ""
echo "[5/7] 检查NPU设备..."
if command -v npu-smi &> /dev/null; then
    NPU_COUNT=$(npu-smi info | grep "NPU ID" | wc -l)
    echo "✅ 检测到 $NPU_COUNT 个NPU设备"
else
    echo "⚠️  警告：无法运行npu-smi命令"
fi

# 6. 安装依赖
echo ""
echo "[6/7] 安装依赖..."
echo "正在安装torch和torch-npu..."
pip install torch==2.7.1 torch-npu==2.7.1 -q

echo "正在安装项目依赖..."
pip install -r requirements-npu.txt -q

echo "✅ 依赖安装完成"

# 7. 验证安装
echo ""
echo "[7/7] 验证安装..."
python -c "
import torch
import torch_npu
import transformers

print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ torch-npu: {torch_npu.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ NPU可用: {torch.npu.is_available()}')
print(f'✅ NPU数量: {torch.npu.device_count()}')
"

echo ""
echo "=========================================="
echo "✅ NPU环境配置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 下载数据集: python -c \"from datasets import load_dataset; load_dataset('TIGER-Lab/DAPO-Math-17k').save_to_disk('./data/datasets/dapo-math-17k')\""
echo "2. 开始训练: python scripts/train_baseline.py --model_name Qwen/Qwen2.5-Math-7B-Instruct --dataset_name TIGER-Lab/DAPO-Math-17k --output_dir ./output/baseline"
echo ""
echo "重要提示："
echo "- 每次启动新终端都需要运行: source /usr/local/Ascend/ascend-toolkit/set_env.sh"
echo "- 建议将CANN环境变量添加到~/.bashrc以永久生效"
echo ""
