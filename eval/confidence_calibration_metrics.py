"""
置信度校准的定量评估指标
仅适配《Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning》Section 4.2

核心指标（论文Section 4.2）：
1. Smoothed ECE (smECE) - 平滑期望校准误差
2. Brier Score - 置信度与结果的均方误差
3. Negative Log-Likelihood (NLL) - 负对数似然
4. Confidence AUC - 置信度ROC曲线下面积
5. SNR Gain - 信噪比增益
6. Abstention Accuracy - 弃权准确率
7. Predictive Accuracy - 预测准确率（参考指标）
"""

import numpy as np
from typing import List, Dict, Tuple
try:
    from sklearn.metrics import roc_auc_score as _sk_roc_auc_score
except Exception:
    _sk_roc_auc_score = None
import logging

logger = logging.getLogger(__name__)


class ConfidenceCalibrationMetrics:
    """
    置信度校准的定量评估类

    实现论文Section 4.2中的所有评估指标
    """

    def __init__(self, thresholds: List[float] = None):
        """
        初始化

        参数：
            thresholds: 风险阈值列表（用于SNR Gain计算）
        """
        # 使用更细的阈值网格以近似论文中的区间积分
        self.thresholds = thresholds if thresholds else np.linspace(0, 1, 101).tolist()

    def compute_smoothed_ece(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        bandwidth: float = 0.1
    ) -> float:
        """
        计算Smoothed ECE（平滑期望校准误差）

        论文定义：
        Standard ECE is computed by comparing the model's reported confidences
        with actual accuracy in bins. However, standard ECE is biased and
        sensitive to bin count. We use smooth ECE, which utilizes kernel
        density estimation to smooth the observations and provide a
        theoretically-sound calibration measure.

        公式：
        smECE = ∫|P(Y=1|p=c) - c| * density(c) dc

        参数：
            confidences: 置信度数组 (N,)
            correctness: 正确性数组 (N,), 1表示正确，0表示错误
            bandwidth: KDE带宽

        返回：
            smECE值（越小越好，范围0-1）
        """
        logger.info("Computing Smoothed ECE...")

        if len(confidences) == 0:
            return 0.0

        # 若置信度几乎常数，退化为标准ECE
        if np.allclose(confidences, confidences[0]):
            return self._compute_standard_ece(confidences, correctness, n_bins=10)

        # 在[0,1]区间采样
        c_samples = np.linspace(0, 1, 200)
        ece_sum = 0.0
        weight_sum = 0.0

        # 核平滑：Nadaraya–Watson 估计条件准确率，并按密度加权
        for c in c_samples:
            weights = np.exp(-((confidences - c) ** 2) / (2 * bandwidth ** 2))
            density = weights.sum()
            if density <= 0:
                continue
            acc_c = (weights * correctness).sum() / density
            ece_sum += np.abs(acc_c - c) * density
            weight_sum += density

        smECE = ece_sum / (weight_sum + 1e-12)

        logger.info(f"Smoothed ECE: {smECE:.4f}")
        return float(smECE)

    def _compute_standard_ece(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        计算标准ECE（作为smECE的后备）

        公式：
        ECE = Σ (n_b/N) * |acc_b - conf_b|
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # 找到在当前bin中的样本
            # First bin must include confidence==0.0, otherwise all-zero confidence
            # degenerates to empty bins and yields an incorrect ECE=0.
            if i == 0:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            if in_bin.sum() > 0:
                bin_acc = correctness[in_bin].mean()
                bin_conf = confidences[in_bin].mean()
                bin_weight = in_bin.sum() / len(confidences)

                ece += bin_weight * np.abs(bin_acc - bin_conf)

        return float(ece)

    def compute_brier_score(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray
    ) -> float:
        """
        计算Brier Score（布赖尔分数）

        论文定义：
        The mean squared error between the model's confidence p and the
        outcome (1 for correct, 0 for incorrect).

        公式：
        Brier = (1/N) * Σ(p_i - y_i)^2

        参数：
            confidences: 置信度数组 (N,)
            correctness: 正确性数组 (N,), 1表示正确，0表示错误

        返回：
            Brier Score（越小越好，范围0-1）
        """
        logger.info("Computing Brier Score...")

        if len(confidences) == 0:
            return 0.0

        brier = np.mean((confidences - correctness) ** 2)

        logger.info(f"Brier Score: {brier:.4f}")
        return float(brier)

    def compute_nll(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        计算Negative Log-Likelihood（负对数似然）

        论文定义：
        Negative Log-Likelihood of correctness outcomes assuming the
        probability of correctness is given by the confidence p.

        公式：
        NLL = -(1/N) * Σ[y_i*log(p_i) + (1-y_i)*log(1-p_i)]

        参数：
            confidences: 置信度数组 (N,)
            correctness: 正确性数组 (N,), 1表示正确，0表示错误
            epsilon: 数值稳定性参数

        返回：
            NLL值（越小越好）
        """
        logger.info("Computing Negative Log-Likelihood...")

        if len(confidences) == 0:
            return 0.0

        # 裁剪置信度以避免log(0)
        confidences_clipped = np.clip(confidences, epsilon, 1 - epsilon)

        # 计算NLL
        nll = -np.mean(
            correctness * np.log(confidences_clipped) +
            (1 - correctness) * np.log(1 - confidences_clipped)
        )

        logger.info(f"Negative Log-Likelihood: {nll:.4f}")
        return float(nll)

    def compute_confidence_auc(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray
    ) -> float:
        """
        计算Confidence AUC（置信度ROC曲线下面积）

        论文定义：
        Area under the ROC curve by treating abstention as a classification
        task with estimated confidence. While ECE measures the alignment of
        average confidence with average accuracy, Confidence AUC is the
        critic metric for evaluating models' ability to distinguish "knowns"
        from "unknowns". It measures the probability that a correct answer
        is assigned a higher confidence score than an incorrect one.
        Crucially, AUC is independent of the risk threshold and the absolute
        problem-solving accuracy, making it fair for comparing weak and
        strong models.

        参数：
            confidences: 置信度数组 (N,)
            correctness: 正确性数组 (N,), 1表示正确，0表示错误

        返回：
            AUC值（越大越好，范围0-1，0.5表示随机）
        """
        logger.info("Computing Confidence AUC...")

        if len(confidences) == 0 or len(np.unique(correctness)) < 2:
            return 0.5  # 无法计算AUC时返回随机水平

        try:
            auc = self._roc_auc_score(correctness, confidences)
        except Exception:
            auc = 0.5

        logger.info(f"Confidence AUC: {auc:.4f}")
        return float(auc)

    @staticmethod
    def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """无sklearn时的AUC回退实现（处理并列分数）。"""
        if _sk_roc_auc_score is not None:
            return float(_sk_roc_auc_score(y_true, y_score))

        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        if y_true.size == 0 or np.unique(y_true).size < 2:
            return 0.5

        order = np.argsort(y_score)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, y_score.size + 1, dtype=float)

        sorted_scores = y_score[order]
        i = 0
        n = y_score.size
        while i < n:
            j = i
            while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
                j += 1
            if j > i:
                avg_rank = (i + 1 + j + 1) / 2.0
                ranks[order[i:j + 1]] = avg_rank
            i = j + 1

        pos = y_true == 1
        n_pos = int(pos.sum())
        n_neg = int(y_true.size - n_pos)
        if n_pos == 0 or n_neg == 0:
            return 0.5
        u = ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0
        return float(u / (n_pos * n_neg))

    def compute_snr_gain(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray
    ) -> float:
        """
        计算SNR Gain（信噪比增益）

        论文定义：
        Considering confidence estimates as the signal for rejection, we
        evaluate the policy's Signal-to-Noise Ratio (SNR), defined as the
        ratio of accuracy to hallucination. As the risk threshold t increases,
        we expect reduction in hallucination which should cause increase in
        SNR. Therefore, our metric of interest is the gain of SNR across the
        spectrum of risk thresholds t∈[0,1] relative to the baseline at t=0.

        公式：
        SNR(t) = Acc(t) / Hal(t)
        SNR([0,1]) = ∫ Acc(t) dt / ∫ Hal(t) dt
        SNR Gain = log(SNR([0,1]) / SNR(0))

        参数：
            confidences: 置信度数组 (N,)
            correctness: 正确性数组 (N,), 1表示正确，0表示错误

        返回：
            SNR Gain值（越大越好，正值表示改进）
        """
        logger.info("Computing SNR Gain...")

        if len(confidences) == 0:
            return 0.0

        thresholds = np.array(self.thresholds, dtype=float)
        thresholds = np.clip(thresholds, 0.0, 1.0)
        thresholds = np.unique(thresholds)
        thresholds.sort()
        if thresholds.size == 0 or thresholds[0] != 0.0:
            thresholds = np.insert(thresholds, 0, 0.0)

        # 统计每个阈值下的正确/错误“比例”（以总样本为分母，符合论文定义）
        acc_rates = []
        hal_rates = []
        for t in thresholds:
            mask = confidences >= t
            acc_t = float(np.sum((correctness == 1) & mask)) / len(confidences)
            hal_t = float(np.sum((correctness == 0) & mask)) / len(confidences)
            acc_rates.append(acc_t)
            hal_rates.append(hal_t)

        acc_rates = np.array(acc_rates, dtype=float)
        hal_rates = np.array(hal_rates, dtype=float)

        # t=0 的基线 SNR
        t0_idx = int(np.where(thresholds == 0.0)[0][0])
        acc_0 = acc_rates[t0_idx]
        hal_0 = hal_rates[t0_idx]
        if hal_0 <= 0:
            return 0.0
        snr_0 = acc_0 / hal_0

        # 区间积分近似 (∫Acc / ∫Hal)
        acc_int = np.trapz(acc_rates, thresholds)
        hal_int = np.trapz(hal_rates, thresholds)
        if hal_int <= 0 or snr_0 <= 0:
            return 0.0

        snr_all = acc_int / hal_int
        snr_gain = np.log(snr_all / snr_0)

        logger.info(f"SNR Gain: {snr_gain:.4f}")
        return float(snr_gain)

    def compute_abstention_accuracy(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        计算Abstention Accuracy（弃权准确率）

        论文定义：
        The proportion of responses that abstain from incorrect answers or
        answer with correct answers. Assume abstention with confidence below 0.5.

        公式：
        Abs Acc = (正确回答数 + 错误弃权数) / 总样本数

        参数：
            confidences: 置信度数组 (N,)
            correctness: 正确性数组 (N,), 1表示正确，0表示错误
            threshold: 弃权阈值（默认0.5）

        返回：
            Abstention Accuracy（越大越好，范围0-1）
        """
        logger.info("Computing Abstention Accuracy...")

        if len(confidences) == 0:
            return 0.0

        # 回答的样本（置信度>=threshold）
        answered_mask = confidences >= threshold
        # 弃权的样本（置信度<threshold）
        abstained_mask = confidences < threshold

        # 正确回答的数量
        correct_answers = (answered_mask & (correctness == 1)).sum()
        # 错误弃权的数量（弃权了错误的答案）
        correct_abstentions = (abstained_mask & (correctness == 0)).sum()

        abs_acc = (correct_answers + correct_abstentions) / len(confidences)

        logger.info(f"Abstention Accuracy (t={threshold}): {abs_acc:.4f}")
        return float(abs_acc)

    def compute_predictive_accuracy(
        self,
        correctness: np.ndarray
    ) -> float:
        """
        计算Predictive Accuracy（预测准确率）

        论文定义：
        The standard accuracy of problem solving assuming no abstention.

        这是参考指标，用于衡量模型在无弃权情况下的原始准确率。

        参数：
            correctness: 正确性数组 (N,), 1表示正确，0表示错误

        返回：
            准确率（范围0-1）
        """
        logger.info("Computing Predictive Accuracy...")

        if len(correctness) == 0:
            return 0.0

        acc = correctness.mean()

        logger.info(f"Predictive Accuracy: {acc:.4f}")
        return float(acc)

    def compute_all_metrics(
        self,
        confidences: List[float],
        correctness: List[int],
        abstention_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        计算所有定量评估指标

        参数：
            confidences: 置信度列表
            correctness: 正确性列表（1表示正确，0表示错误）
            abstention_threshold: 弃权阈值（默认0.5）

        返回：
            包含所有指标的字典
        """
        logger.info("Computing all confidence calibration metrics...")

        # 转换为numpy数组
        confidences = np.array(confidences)
        correctness = np.array(correctness)

        metrics = {
            "smoothed_ece": self.compute_smoothed_ece(confidences, correctness),
            "brier_score": self.compute_brier_score(confidences, correctness),
            "nll": self.compute_nll(confidences, correctness),
            "confidence_auc": self.compute_confidence_auc(confidences, correctness),
            "snr_gain": self.compute_snr_gain(confidences, correctness),
            "abstention_accuracy": self.compute_abstention_accuracy(
                confidences, correctness, abstention_threshold
            ),
            "predictive_accuracy": self.compute_predictive_accuracy(correctness),
        }

        logger.info("All metrics computed successfully")
        return metrics


# ============================================================
# 使用示例
# ============================================================

def example_usage():
    """
    使用示例
    """
    # 模拟数据
    np.random.seed(42)
    n_samples = 1000

    # 生成置信度和正确性
    confidences = np.random.beta(2, 2, n_samples)  # Beta分布的置信度
    correctness = (np.random.rand(n_samples) < confidences).astype(int)  # 根据置信度生成正确性

    # 计算指标
    metrics_calculator = ConfidenceCalibrationMetrics()
    metrics = metrics_calculator.compute_all_metrics(
        confidences.tolist(),
        correctness.tolist()
    )

    # 打印结果
    print("\n" + "="*60)
    print("Confidence Calibration Metrics (Section 4.2)")
    print("="*60)
    for metric_name, value in metrics.items():
        print(f"{metric_name:25s}: {value:.4f}")
    print("="*60)


if __name__ == "__main__":
    example_usage()
