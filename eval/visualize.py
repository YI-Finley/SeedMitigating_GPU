"""
可视化模块
仅适配《Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning》baseline + 4个校准目标

核心约束：
- 绘制目标论文同款图表（图6、图7）
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os
import logging

logger = logging.getLogger(__name__)

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def visualize_adaptive_risk(
    thresholds: List[float],
    abstention_rates: List[float],
    hallucination_rates: List[float],
    save_path: str,
):
    """
    可视化Adaptive Risk（目标论文图6同款凹形曲线）

    图表说明：
    - X轴：风险阈值t（0-1）
    - Y轴：比率（0-1）
    - 两条曲线：
      1. 弃权率曲线（随t增加而上升）
      2. 幻觉率曲线（随t增加而下降，呈凹形）

    参数：
        thresholds: 风险阈值列表
        abstention_rates: 弃权率列表
        hallucination_rates: 幻觉率列表
        save_path: 保存路径
    """
    logger.info("Visualizing Adaptive Risk...")

    plt.figure(figsize=(10, 6))

    # 绘制弃权率曲线
    plt.plot(
        thresholds,
        abstention_rates,
        label="Abstention Rate",
        marker='o',
        linewidth=2,
        markersize=6,
        color='#2E86AB',
    )

    # 绘制幻觉率曲线
    plt.plot(
        thresholds,
        hallucination_rates,
        label="Hallucination Rate",
        marker='s',
        linewidth=2,
        markersize=6,
        color='#A23B72',
    )

    plt.xlabel("Risk Threshold (t)", fontsize=14)
    plt.ylabel("Rate", fontsize=14)
    plt.title("Adaptive Risk: Abstention vs Hallucination", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Adaptive Risk visualization saved to {save_path}")


def visualize_hallucination_reduction(
    thresholds: List[float],
    snr_values: List[float],
    snr_gains: List[float],
    save_path: str,
):
    """
    可视化Hallucination Reduction（SNR和SNR Gain曲线）

    图表说明：
    - 左图：SNR(t)曲线（信噪比随t的变化）
    - 右图：SNR Gain(t)曲线（相对于t=0的增益）

    参数：
        thresholds: 风险阈值列表
        snr_values: SNR值列表
        snr_gains: SNR增益列表
        save_path: 保存路径
    """
    logger.info("Visualizing Hallucination Reduction...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：SNR(t)
    ax1.plot(
        thresholds,
        snr_values,
        marker='o',
        linewidth=2,
        markersize=6,
        color='#F18F01',
    )
    ax1.set_xlabel("Risk Threshold (t)", fontsize=14)
    ax1.set_ylabel("SNR(t)", fontsize=14)
    ax1.set_title("Signal-to-Noise Ratio", fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # 右图：SNR Gain(t)
    ax2.plot(
        thresholds,
        snr_gains,
        marker='s',
        linewidth=2,
        markersize=6,
        color='#C73E1D',
    )
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel("Risk Threshold (t)", fontsize=14)
    ax2.set_ylabel("SNR Gain(t) = log(SNR(t)/SNR(0))", fontsize=14)
    ax2.set_title("SNR Gain (Relative to t=0)", fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Hallucination Reduction visualization saved to {save_path}")


def visualize_quantitative_calibration(
    thresholds: List[float],
    tp_values: List[float],
    fn_values: List[float],
    save_path: str,
):
    """
    可视化Quantitative Calibration（目标论文图7同款TP/FN曲线）

    图表说明：
    - X轴：风险阈值t（0-1）
    - Y轴：比例（0-1）
    - 三条曲线：
      1. TP(t)：回答中正确占比
      2. FN(t)：弃权中正确占比
      3. y=x基准线（理想校准）

    参数：
        thresholds: 风险阈值列表
        tp_values: TP值列表
        fn_values: FN值列表
        save_path: 保存路径
    """
    logger.info("Visualizing Quantitative Calibration...")

    plt.figure(figsize=(10, 6))

    # 绘制TP(t)曲线
    plt.plot(
        thresholds,
        tp_values,
        label="TP(t) - Answered Correct",
        marker='o',
        linewidth=2,
        markersize=6,
        color='#06A77D',
    )

    # 绘制FN(t)曲线
    plt.plot(
        thresholds,
        fn_values,
        label="FN(t) - Abstained Correct",
        marker='s',
        linewidth=2,
        markersize=6,
        color='#D62246',
    )

    # 绘制y=x基准线
    plt.plot(
        thresholds,
        thresholds,
        label="y=x (Perfect Calibration)",
        linestyle='--',
        linewidth=2,
        color='gray',
        alpha=0.7,
    )

    plt.xlabel("Risk Threshold (t)", fontsize=14)
    plt.ylabel("Proportion", fontsize=14)
    plt.title("Quantitative Calibration: TP vs FN", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Quantitative Calibration visualization saved to {save_path}")


def visualize_all_metrics(
    metrics: Dict,
    output_dir: str,
):
    """
    可视化所有4个校准目标

    参数：
        metrics: 包含所有指标的字典（包含behavioral_calibration和confidence_calibration）
        output_dir: 输出目录
    """
    logger.info("Visualizing all behavioral calibration metrics...")

    os.makedirs(output_dir, exist_ok=True)

    # 提取行为校准指标
    if "behavioral_calibration" in metrics:
        behavioral = metrics["behavioral_calibration"]
    else:
        # 兼容旧格式（直接包含指标）
        behavioral = metrics

    # 1. Adaptive Risk
    if "adaptive_risk" in behavioral:
        visualize_adaptive_risk(
            thresholds=behavioral["adaptive_risk"]["thresholds"],
            abstention_rates=behavioral["adaptive_risk"]["abstention_rates"],
            hallucination_rates=behavioral["adaptive_risk"]["hallucination_rates"],
            save_path=os.path.join(output_dir, "adaptive_risk.png"),
        )

    # 2. Hallucination Reduction
    if "hallucination_reduction" in behavioral:
        visualize_hallucination_reduction(
            thresholds=behavioral["hallucination_reduction"]["thresholds"],
            snr_values=behavioral["hallucination_reduction"]["snr_values"],
            snr_gains=behavioral["hallucination_reduction"]["snr_gains"],
            save_path=os.path.join(output_dir, "hallucination_reduction.png"),
        )

    # 3. Quantitative Calibration
    if "quantitative_calibration" in behavioral:
        visualize_quantitative_calibration(
            thresholds=behavioral["quantitative_calibration"]["thresholds"],
            tp_values=behavioral["quantitative_calibration"]["tp_values"],
            fn_values=behavioral["quantitative_calibration"]["fn_values"],
            save_path=os.path.join(output_dir, "quantitative_calibration.png"),
        )

    logger.info(f"All visualizations saved to {output_dir}")


def print_metrics_summary(metrics: Dict):
    """
    打印指标摘要

    参数：
        metrics: 指标字典（包含behavioral_calibration和confidence_calibration）
    """
    print("\n" + "="*80)
    print("Evaluation Metrics Summary")
    print("="*80)

    # 打印行为校准指标
    if "behavioral_calibration" in metrics:
        print("\n【Behavioral Calibration Objectives】")
        behavioral = metrics["behavioral_calibration"]

        # Accuracy Preservation
        if "accuracy_preservation" in behavioral:
            acc = behavioral["accuracy_preservation"]["accuracy_at_t0"]
            print(f"  Accuracy Preservation (t=0): {acc:.4f}")

        # Adaptive Risk (显示几个关键阈值)
        if "adaptive_risk" in behavioral:
            adaptive = behavioral["adaptive_risk"]
            print(f"\n  Adaptive Risk:")
            for i, t in enumerate([0.0, 0.5, 1.0]):
                if t in adaptive["thresholds"]:
                    idx = adaptive["thresholds"].index(t)
                    abs_rate = adaptive["abstention_rates"][idx]
                    hal_rate = adaptive["hallucination_rates"][idx]
                    print(f"    t={t:.1f}: Abstention={abs_rate:.4f}, Hallucination={hal_rate:.4f}")

        # Hallucination Reduction
        if "hallucination_reduction" in behavioral:
            reduction = behavioral["hallucination_reduction"]
            max_snr_gain = max(reduction["snr_gains"])
            print(f"\n  Hallucination Reduction:")
            print(f"    Max SNR Gain: {max_snr_gain:.4f}")

    # 打印置信度校准指标（Section 4.2）
    if "confidence_calibration" in metrics:
        print("\n【Confidence Calibration Metrics (Section 4.2)】")
        calibration = metrics["confidence_calibration"]

        print(f"  Smoothed ECE:         {calibration.get('smoothed_ece', 0):.4f} ↓")
        print(f"  Brier Score:          {calibration.get('brier_score', 0):.4f} ↓")
        print(f"  NLL:                  {calibration.get('nll', 0):.4f} ↓")
        print(f"  Confidence AUC:       {calibration.get('confidence_auc', 0):.4f} ↑")
        print(f"  SNR Gain:             {calibration.get('snr_gain', 0):.4f} ↑")
        print(f"  Abstention Accuracy:  {calibration.get('abstention_accuracy', 0):.4f} ↑")
        print(f"  Predictive Accuracy:  {calibration.get('predictive_accuracy', 0):.4f} ↑")

    print("\n" + "="*80)
    print("Note: ↑ means higher is better, ↓ means lower is better")
    print("="*80 + "\n")


# ============================================================
# 删除的TruthRL可视化逻辑说明
# ============================================================

# ❌ visualize_truthfulness_score()
#    - TruthRL的核心可视化：Truthfulness Score随训练步数的变化
#    - 目标论文使用4个独立的校准目标
#    - 删除原因：与目标论文无关

# ❌ visualize_confidence_calibration()
#    - TruthRL的置信度校准图（ECE、可靠性图）
#    - 目标论文使用Quantitative Calibration（TP/FN曲线）
#    - 删除原因：与目标论文无关

# ❌ visualize_format_compliance()
#    - TruthRL的格式遵守率可视化（<think>/<answer>标签使用率）
#    - 目标论文无此要求
#    - 删除原因：与目标论文无关

# ❌ visualize_retrieval_quality()
#    - TruthRL的检索质量可视化（检索文档相关性）
#    - 目标论文无RAG逻辑
#    - 删除原因：与目标论文无关

# ❌ visualize_training_curves()
#    - TruthRL的训练曲线（损失、奖励、KL散度等）
#    - 目标论文仅关注最终评估结果
#    - 删除原因：与目标论文无关

# ============================================================
# 保留的通用逻辑
# ============================================================

# ✅ visualize_adaptive_risk() - 目标论文图6同款
# ✅ visualize_hallucination_reduction() - SNR/SNR Gain曲线
# ✅ visualize_quantitative_calibration() - 目标论文图7同款
# ✅ print_metrics_summary() - 简洁的指标摘要
