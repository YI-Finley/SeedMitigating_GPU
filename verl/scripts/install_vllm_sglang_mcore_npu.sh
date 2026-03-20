#!/bin/bash
# 昇腾 NPU 环境安装脚本
# 适用于 CANN 8.3.RC1 + PyTorch 2.7.1

set -e

echo "=========================================="
echo "昇腾 NPU 环境安装脚本"
echo "=========================================="

# 检查 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "当前 Python 版本: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" =~ ^3\.(9|10|11)$ ]]; then
    echo "错误: Python 版本必须是 3.9, 3.10 或 3.11"
    echo "当前版本: $PYTHON_VERSION"
    exit 1
fi

# 检查 CANN 环境
if [ ! -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    echo "警告: 未找到 CANN 环境，请确保已安装 CANN 8.3.RC1 或更高版本"
    echo "CANN 安装路径: /usr/local/Ascend/ascend-toolkit/"
else
    echo "正在激活 CANN 环境..."
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    if [ -f "/usr/local/Ascend/nnal/atb/set_env.sh" ]; then
        source /usr/local/Ascend/nnal/atb/set_env.sh
    fi
fi

# 检查 torch_npu
echo ""
echo "检查 torch_npu 安装状态..."
if python3 -c "import torch_npu" 2>/dev/null; then
    TORCH_NPU_VERSION=$(python3 -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null || echo "unknown")
    echo "torch_npu 已安装，版本: $TORCH_NPU_VERSION"
else
    echo "错误: 未检测到 torch_npu，请先安装 torch_npu"
    echo "安装命令示例:"
    echo "  pip3 install torch==2.7.1"
    echo "  pip3 install torch-npu==2.7.1"
    exit 1
fi

export MAX_JOBS=32

echo ""
echo "=========================================="
echo "1. 安装基础软件包"
echo "=========================================="

# 安装 torchvision (需要匹配 torch 版本)
pip install torchvision==0.22.1 --no-cache-dir

# 清理可能存在的 triton 残留
pip uninstall -y triton triton-ascend 2>/dev/null || true

# 安装 triton-ascend
pip install triton-ascend==3.2.0rc4 --no-cache-dir

echo ""
echo "=========================================="
echo "2. 安装 vllm & vllm-ascend"
echo "=========================================="

# 安装 vllm (empty backend)
if [ ! -d "vllm" ]; then
    git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm.git
fi
cd vllm && VLLM_TARGET_DEVICE=empty pip install -v -e . && cd ..

# 安装 vllm-ascend
if [ ! -d "vllm-ascend" ]; then
    git clone --depth 1 --branch v0.11.0rc1 https://github.com/vllm-project/vllm-ascend.git
fi
cd vllm-ascend && pip install -v -e . && cd ..

echo ""
echo "=========================================="
echo "3. 安装基础依赖包"
echo "=========================================="

pip install "transformers>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc wandb dill pybind11 \
    pytest py-spy pyext pre-commit ruff qwen-vl-utils mathruler \
    "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

# 安装 tensordict 和 torchdata
pip install "tensordict==0.6.2" torchdata --no-cache-dir

echo ""
echo "=========================================="
echo "4. (可选) 安装 MindSpeed"
echo "=========================================="

read -p "是否安装 MindSpeed (Megatron 后端)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 下载 MindSpeed
    if [ ! -d "MindSpeed" ]; then
        git clone https://gitcode.com/Ascend/MindSpeed.git
        cd MindSpeed && git checkout f2b0977e && cd ..
    fi

    # 下载 Megatron-LM
    if [ ! -d "Megatron-LM" ]; then
        git clone --depth 1 --branch core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git
    fi

    # 安装 MindSpeed
    pip install -e MindSpeed

    # 配置 PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:"$(pwd)/Megatron-LM"

    # 添加到 .bashrc
    if ! grep -q "MindSpeed/Megatron-LM" ~/.bashrc 2>/dev/null; then
        echo "export PYTHONPATH=\$PYTHONPATH:\"$(pwd)/Megatron-LM\"" >> ~/.bashrc
        echo "已将 Megatron-LM 路径添加到 ~/.bashrc"
    fi

    # 安装 mbridge
    pip install mbridge

    echo "MindSpeed 安装完成"
else
    echo "跳过 MindSpeed 安装"
fi

echo ""
echo "=========================================="
echo "5. 修复 opencv"
echo "=========================================="

pip install opencv-python --no-cache-dir
pip install opencv-fixer && python -c "from opencv_fixer import AutoFix; AutoFix()"

echo ""
echo "=========================================="
echo "安装完成!"
echo "=========================================="
echo ""
echo "重要提示:"
echo "1. 每次使用前请确保激活 CANN 环境:"
echo "   source /usr/local/Ascend/ascend-toolkit/set_env.sh"
echo "   source /usr/local/Ascend/nnal/atb/set_env.sh"
echo ""
echo "2. 验证安装:"
echo "   python3 -c 'import torch; import torch_npu; print(torch.npu.is_available())'"
echo ""
echo "3. 如果安装了 MindSpeed，请重新加载 shell 或运行:"
echo "   source ~/.bashrc"
echo ""
