#!/bin/bash
# 主训练脚本: 训练所有6个模型变体
# 可以选择串行或并行训练

set -e

SCRIPT_DIR="/root/SeedMitigating/rl/scripts"

echo "=========================================="
echo "论文第4节实验复现 - 训练所有模型变体"
echo "=========================================="
echo ""
echo "将训练以下6个模型变体:"
echo "  1. Baseline PPO (二元奖励)"
echo "  2. Confidence-Brier (Verbalized + Brier Score)"
echo "  3. Confidence-CE (Verbalized + Cross-Entropy, GRPO)"
echo "  4. PPO-Value (Critic Value作为置信度)"
echo "  5. Confidence-Prod (声明级 + 乘积聚合)"
echo "  6. Confidence-Min (声明级 + 最小值聚合)"
echo ""
echo "每个模型训练约需1-2天 (8×NPU)"
echo ""

# 询问训练模式
read -p "选择训练模式 [1=串行, 2=并行(需要足够NPU资源)]: " MODE

if [ "$MODE" == "2" ]; then
    echo ""
    echo "警告: 并行训练需要48个NPU (6个模型 × 8 NPU/模型)"
    read -p "确认有足够资源? [y/N]: " CONFIRM
    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        echo "取消训练"
        exit 0
    fi

    echo ""
    echo "开始并行训练所有模型..."

    # 并行启动所有训练任务
    bash "$SCRIPT_DIR/train_baseline_ppo.sh" &
    PID1=$!
    bash "$SCRIPT_DIR/train_confidence_brier.sh" &
    PID2=$!
    bash "$SCRIPT_DIR/train_confidence_ce.sh" &
    PID3=$!
    bash "$SCRIPT_DIR/train_ppo_value.sh" &
    PID4=$!
    bash "$SCRIPT_DIR/train_confidence_prod.sh" &
    PID5=$!
    bash "$SCRIPT_DIR/train_confidence_min.sh" &
    PID6=$!

    echo "所有训练任务已启动"
    echo "PIDs: $PID1 $PID2 $PID3 $PID4 $PID5 $PID6"

    # 等待所有任务完成
    wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6

else
    echo ""
    echo "开始串行训练所有模型..."
    echo ""

    # 串行训练
    echo "[1/6] 训练 Baseline PPO..."
    bash "$SCRIPT_DIR/train_baseline_ppo.sh"

    echo ""
    echo "[2/6] 训练 Confidence-Brier..."
    bash "$SCRIPT_DIR/train_confidence_brier.sh"

    echo ""
    echo "[3/6] 训练 Confidence-CE..."
    bash "$SCRIPT_DIR/train_confidence_ce.sh"

    echo ""
    echo "[4/6] 训练 PPO-Value..."
    bash "$SCRIPT_DIR/train_ppo_value.sh"

    echo ""
    echo "[5/6] 训练 Confidence-Prod..."
    bash "$SCRIPT_DIR/train_confidence_prod.sh"

    echo ""
    echo "[6/6] 训练 Confidence-Min..."
    bash "$SCRIPT_DIR/train_confidence_min.sh"
fi

echo ""
echo "=========================================="
echo "✓ 所有模型训练完成!"
echo "=========================================="
echo ""
echo "模型保存位置:"
echo "  /data2/SeedMitigating-output/paper_reproduction/checkpoints/"
echo ""
echo "下一步: 运行评估脚本"
echo "  bash /root/SeedMitigating\ /rl/scripts/eval_all_response_level.sh"
echo ""
