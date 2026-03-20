"""
行为校准目标评估指标
仅适配《Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning》baseline + 4个校准目标

核心目标：
1. Adaptive Risk（自适应风险）
2. Accuracy Preservation（准确率保留）
3. Hallucination Reduction（幻觉减少）
4. Quantitative Calibration（定量校准）
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class BehavioralCalibrationMetrics:
    """
    行为校准目标评估类

    封装4个校准目标的计算逻辑
    """

    def __init__(self, thresholds: List[float]):
        """
        初始化

        参数：
            thresholds: 风险阈值列表（0-1）
        """
        self.thresholds = thresholds

    def compute_adaptive_risk(
        self,
        predictions: List[Dict],
        labels: List[int],
    ) -> Tuple[List[float], List[float]]:
        """
        计算Adaptive Risk（自适应风险）

        目标：模型应在不同风险阈值下自适应调整弃权率，同时降低幻觉率

        公式：
        - Abstention(t) = E[a(t)=ABS] = 弃权样本数 / 总样本数
        - Hallucination(t) = E[¬valid(y) ∧ a(t)=ANS] = 错误回答数 / 总样本数

        参数：
            predictions: 模型预测列表，每个元素包含：
                - "answer": 回答内容
                - "confidence": 置信度（0-1）
                - "abstained": 是否弃权（bool）
            labels: 正确性标签列表（1表示正确，0表示错误）

        返回：
            (abstention_rates, hallucination_rates)元组
        """
        logger.info("Computing Adaptive Risk...")

        abstention_rates = []
        hallucination_rates = []

        for t in self.thresholds:
            # 计算弃权率
            abstained_count = sum([1 for p in predictions if p["confidence"] < t])
            abstention_rate = abstained_count / len(predictions)

            # 计算幻觉率
            wrong_count = sum(
                1
                for i, p in enumerate(predictions)
                if p["confidence"] >= t and labels[i] == 0
            )
            hallucination_rate = wrong_count / len(predictions)

            abstention_rates.append(abstention_rate)
            hallucination_rates.append(hallucination_rate)

            logger.debug(f"t={t:.2f}: Abstention={abstention_rate:.4f}, "
                        f"Hallucination={hallucination_rate:.4f}")

        logger.info("Adaptive Risk computed successfully")
        return abstention_rates, hallucination_rates

    def compute_accuracy_preservation(
        self,
        predictions: List[Dict],
        labels: List[int],
    ) -> float:
        """
        计算Accuracy Preservation（准确率保留）

        目标：在t=0时（无弃权），模型应保持高准确率

        公式：
        - Acc(0) = 正确回答数 / 总样本数

        参数：
            predictions: 模型预测列表
            labels: 正确性标签列表

        返回：
            准确率（0-1）
        """
        logger.info("Computing Accuracy Preservation...")

        # 仅计算t=0时的准确率（所有样本都回答）
        correct_count = sum([1 for i, p in enumerate(predictions) if labels[i] == 1])
        accuracy = correct_count / len(predictions)

        logger.info(f"Accuracy Preservation (t=0): {accuracy:.4f}")
        return accuracy

    def compute_hallucination_reduction(
        self,
        predictions: List[Dict],
        labels: List[int],
    ) -> Tuple[List[float], List[float]]:
        """
        计算Hallucination Reduction（幻觉减少）

        目标：通过弃权机制提高信噪比（SNR），减少幻觉

        公式：
        - Acc(t) = E[valid(y) ∧ a(t)=ANS]
        - Hal(t) = E[¬valid(y) ∧ a(t)=ANS]
        - SNR(t) = Acc(t) / Hal(t)
        - SNR Gain = log(SNR([0,1]) / SNR(0))

        参数：
            predictions: 模型预测列表
            labels: 正确性标签列表

        返回：
            (snr_values, snr_gains)元组
        """
        logger.info("Computing Hallucination Reduction...")

        snr_values = []
        snr_gains = []
        acc_rates = []
        hal_rates = []

        for t in self.thresholds:
            # 获取回答的样本
            answered_indices = [i for i, p in enumerate(predictions) if p["confidence"] >= t]

            # 计算Acc(t)/Hal(t)：以总样本为分母，符合论文定义
            correct_count = sum([1 for i in answered_indices if labels[i] == 1])
            wrong_count = sum([1 for i in answered_indices if labels[i] == 0])
            acc_t = correct_count / len(predictions)
            hal_t = wrong_count / len(predictions)

            # 计算SNR(t)
            if hal_t > 0:
                snr_t = acc_t / hal_t
            else:
                snr_t = float('inf')  # 无幻觉时SNR为无穷大

            snr_values.append(snr_t)
            acc_rates.append(acc_t)
            hal_rates.append(hal_t)

            logger.debug(f"t={t:.2f}: SNR={snr_t:.4f}")

        # SNR Gain = log(SNR([0,1]) / SNR(0))
        snr_0 = snr_values[0] if snr_values else 0.0
        acc_int = np.trapz(acc_rates, self.thresholds) if acc_rates else 0.0
        hal_int = np.trapz(hal_rates, self.thresholds) if hal_rates else 0.0
        if hal_int > 0 and snr_0 > 0 and not np.isinf(snr_0):
            snr_all = acc_int / hal_int
            snr_gain = np.log(snr_all / snr_0)
        else:
            snr_gain = 0.0
        snr_gains = [snr_gain for _ in self.thresholds]

        logger.info("Hallucination Reduction computed successfully")
        return snr_values, snr_gains

    def compute_quantitative_calibration(
        self,
        predictions: List[Dict],
        labels: List[int],
    ) -> Tuple[List[float], List[float]]:
        """
        计算Quantitative Calibration（定量校准）

        目标：模型的置信度应与实际正确率对齐

        公式：
        - TP(t) = E[valid(y) | a(t)=ANS] ≥ t
        - FN(t) = E[valid(y) | a(t)=ABS] ≤ t

        参数：
            predictions: 模型预测列表
            labels: 正确性标签列表

        返回：
            (tp_values, fn_values)元组
        """
        logger.info("Computing Quantitative Calibration...")

        tp_values = []
        fn_values = []

        for t in self.thresholds:
            # 获取回答和弃权的样本
            answered_indices = [i for i, p in enumerate(predictions) if p["confidence"] >= t]
            abstained_indices = [i for i, p in enumerate(predictions) if p["confidence"] < t]

            # 计算TP(t)：回答中正确占比
            if len(answered_indices) > 0:
                tp = sum([1 for i in answered_indices if labels[i] == 1]) / len(answered_indices)
            else:
                tp = 0.0

            # 计算FN(t)：弃权中正确占比
            if len(abstained_indices) > 0:
                fn = sum([1 for i in abstained_indices if labels[i] == 1]) / len(abstained_indices)
            else:
                fn = 0.0

            tp_values.append(tp)
            fn_values.append(fn)

            logger.debug(f"t={t:.2f}: TP={tp:.4f}, FN={fn:.4f}")

        logger.info("Quantitative Calibration computed successfully")
        return tp_values, fn_values

    def compute_all_metrics(
        self,
        predictions: List[Dict],
        labels: List[int],
    ) -> Dict[str, any]:
        """
        计算所有4个校准目标

        参数：
            predictions: 模型预测列表
            labels: 正确性标签列表

        返回：
            包含所有指标的字典
        """
        logger.info("Computing all behavioral calibration metrics...")

        # 1. Adaptive Risk
        abstention_rates, hallucination_rates = self.compute_adaptive_risk(
            predictions, labels
        )

        # 2. Accuracy Preservation
        accuracy = self.compute_accuracy_preservation(predictions, labels)

        # 3. Hallucination Reduction
        snr_values, snr_gains = self.compute_hallucination_reduction(
            predictions, labels
        )

        # 4. Quantitative Calibration
        tp_values, fn_values = self.compute_quantitative_calibration(
            predictions, labels
        )

        metrics = {
            "adaptive_risk": {
                "thresholds": self.thresholds,
                "abstention_rates": abstention_rates,
                "hallucination_rates": hallucination_rates,
            },
            "accuracy_preservation": {
                "accuracy_at_t0": accuracy,
            },
            "hallucination_reduction": {
                "thresholds": self.thresholds,
                "snr_values": snr_values,
                "snr_gains": snr_gains,
            },
            "quantitative_calibration": {
                "thresholds": self.thresholds,
                "tp_values": tp_values,
                "fn_values": fn_values,
            },
        }

        logger.info("All metrics computed successfully")
        return metrics
