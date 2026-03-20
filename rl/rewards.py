"""
奖励函数 - 修复版
实现《Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning》中的所有校准策略
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Callable


def extract_answer(text: str) -> str:
    """从模型输出中提取答案"""
    # 尝试提取\\boxed{}中的内容
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # 尝试提取"Answer:"后的内容（在|之前）
    answer_match = re.search(r'Answer:\s*([^|\n]+)', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    # 尝试提取最后一行
    lines = text.strip().split('\n')
    if lines:
        # 如果最后一行包含|，取|之前的部分
        last_line = lines[-1].strip()
        if '|' in last_line:
            return last_line.split('|')[0].strip()
        return last_line

    return text.strip()


def extract_confidence(text: str) -> Optional[float]:
    """
    从模型输出中提取置信度

    支持格式：
    - "Confidence: 0.85"
    - "| Confidence: 0.85"
    - "置信度: 0.85"
    """
    # 尝试多种模式
    patterns = [
        r'[Cc]onfidence[:\s]+([0-9]*\.?[0-9]+)',
        r'置信度[:\s：]+([0-9]*\.?[0-9]+)',
        r'\|\s*[Cc]onfidence[:\s]+([0-9]*\.?[0-9]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                conf = float(match.group(1))
                # 如果是百分比，转换为0-1
                if conf > 1:
                    conf = conf / 100.0
                # 确保在有效范围内
                return np.clip(conf, 0.0, 1.0)
            except ValueError:
                continue

    return None


def normalize_answer(answer: str) -> str:
    """标准化答案格式"""
    answer = answer.strip().lower()
    answer = re.sub(r'[,\.\!\?\;\:]', '', answer)
    answer = ' '.join(answer.split())
    return answer


def is_correct(prediction: str, ground_truth: str) -> bool:
    """判断预测答案是否正确"""
    if not prediction or not ground_truth:
        return False

    pred_answer = extract_answer(prediction)
    gt_answer = extract_answer(ground_truth)

    pred_normalized = normalize_answer(pred_answer)
    gt_normalized = normalize_answer(gt_answer)

    return pred_normalized == gt_normalized


# ============================================================================
# 奖励函数实现
# ============================================================================

def binary_reward(
    completions: List[str],
    ground_truths: List[str],
    **kwargs
) -> List[float]:
    """
    二元奖励函数（Baseline模型）

    奖励规则：
    - 回答正确：+1.0
    - 回答错误：-1.0
    """
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        if is_correct(completion, gt):
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


def verbalized_confidence_brier_reward(
    completions: List[str],
    ground_truths: List[str],
    **kwargs
) -> List[float]:
    """
    Verbalized Confidence奖励函数 - Brier Score变体

    对应论文Section 3.2.2 - Uniform Distribution

    奖励公式：R_Brier = 2p·valid(y) - p²
    等价于：R_Brier = valid(y) - (p - valid(y))²

    其中：
    - p: 模型输出的置信度 (0-1)
    - valid(y): 1 if correct, 0 if incorrect
    """
    rewards = []

    for completion, gt in zip(completions, ground_truths):
        # 提取答案和置信度
        confidence = extract_confidence(completion)

        # 如果没有提取到置信度，使用默认值0.5（但给予惩罚）
        if confidence is None:
            confidence = 0.5
            penalty = -0.5  # 惩罚没有输出置信度的情况
        else:
            penalty = 0.0

        # 判断正确性
        correct = is_correct(completion, gt)
        valid_y = 1.0 if correct else 0.0

        # 计算Brier Score奖励
        reward = 2.0 * confidence * valid_y - confidence ** 2 + penalty
        rewards.append(reward)

    return rewards


def verbalized_confidence_ce_reward(
    completions: List[str],
    ground_truths: List[str],
    epsilon: float = 0.01,
    **kwargs
) -> List[float]:
    """
    Verbalized Confidence奖励函数 - Cross-Entropy变体

    对应论文Section 3.2.2 - Beta Distribution

    论文公式（方程4）：
    R_CE = [log((1-ε)/ε)]^(-1) * [valid(y)·log(p'/ε) + (1-valid(y))·log((1-p')/(1-ε))]

    其中：
    - p' = clip(p, ε, 1-ε)
    - ε: 裁剪参数，默认0.01
    - valid(y): 1 if correct, 0 if incorrect

    注意：这是完整的公式，包含两项的和，不是if-else分支！
    """
    rewards = []
    # 论文方程4：分母为 log((1-ε)/ε)
    log_scale = np.log((1.0 - epsilon) / epsilon)

    for completion, gt in zip(completions, ground_truths):
        # 提取置信度
        confidence = extract_confidence(completion)

        if confidence is None:
            confidence = 0.5
            penalty = -0.5
        else:
            penalty = 0.0

        # 裁剪置信度到[ε, 1-ε]
        p_clipped = np.clip(confidence, epsilon, 1.0 - epsilon)

        # 判断正确性
        correct = is_correct(completion, gt)
        valid_y = 1.0 if correct else 0.0

        # 修正：使用完整的公式（两项的和）
        # R_CE = [log_scale]^(-1) * [valid(y)·log(p'/ε) + (1-valid(y))·log((1-p')/(1-ε))]
        reward = (1.0 / log_scale) * (
            valid_y * np.log(p_clipped / epsilon) +
            (1.0 - valid_y) * np.log((1.0 - p_clipped) / (1.0 - epsilon))
        )

        reward += penalty
        rewards.append(float(reward))

    return rewards


def critic_value_reward(
    completions: List[str],
    ground_truths: List[str],
    critic_values: Optional[List[float]] = None,
    **kwargs
) -> List[float]:
    """
    Critic Value奖励函数

    对应论文Section 3.2.3

    使用PPO的Critic网络输出作为置信度，然后使用Brier Score公式

    奖励公式：R = 2p·valid(y) - p²
    其中p来自Critic网络的输出
    """
    if critic_values is None:
        # 如果没有提供critic values，回退到binary reward
        return binary_reward(completions, ground_truths)

    rewards = []

    for completion, gt, critic_value in zip(completions, ground_truths, critic_values):
        # Critic value应该已经是[0,1]范围内的置信度
        confidence = np.clip(critic_value, 0.0, 1.0)

        # 判断正确性
        correct = is_correct(completion, gt)
        valid_y = 1.0 if correct else 0.0

        # 使用Brier Score公式
        reward = 2.0 * confidence * valid_y - confidence ** 2
        rewards.append(reward)

    return rewards


def claim_level_confidence_reward(
    completions: List[str],
    ground_truths: List[str],
    aggregation: str = "product",
    **kwargs
) -> List[float]:
    """
    Claim-level Confidence奖励函数

    对应论文Section 3.3 - Claim-level Calibration

    模型输出多个claim的置信度，然后聚合：
    - product: 取所有claim置信度的乘积
    - minimum: 取所有claim置信度的最小值

    然后使用Brier Score公式计算奖励
    """
    rewards = []

    for completion, gt in zip(completions, ground_truths):
        # 提取所有claim的置信度
        # 假设格式：<Confidence value=0.85>Step ...
        claim_confidences = re.findall(
            r'(?i)<\\s*Confidence\\b[^>]*\\bvalue\\s*=\\s*([0-9]+(?:\\.[0-9]+)?)',
            completion,
        )

        if not claim_confidences:
            # 如果没有claim-level置信度，尝试提取response-level置信度
            confidence = extract_confidence(completion)
            if confidence is None:
                confidence = 0.5
                penalty = -0.5
            else:
                penalty = 0.0
        else:
            # 转换为float
            claim_confs = [float(c) for c in claim_confidences]

            # 聚合
            if aggregation == "product":
                confidence = np.prod(claim_confs)
            elif aggregation == "minimum":
                confidence = np.min(claim_confs)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            penalty = 0.0

        # 判断正确性
        correct = is_correct(completion, gt)
        valid_y = 1.0 if correct else 0.0

        # 使用Brier Score公式
        reward = 2.0 * confidence * valid_y - confidence ** 2 + penalty
        rewards.append(reward)

    return rewards


# ============================================================================
# 奖励函数工厂
# ============================================================================

def get_reward_function(
    strategy: str = "baseline",
    **kwargs
) -> Callable:
    """
    获取奖励函数

    参数：
        strategy: 校准策略名称
            - "baseline": Binary Reward (PPO Only)
            - "verbalized_brier": Verbalized Confidence (Brier Score)
            - "verbalized_ce": Verbalized Confidence (Cross-Entropy)
            - "critic_value": Critic Value
            - "claim_product": Claim-level (Product aggregation)
            - "claim_minimum": Claim-level (Minimum aggregation)
        **kwargs: 策略特定的参数

    返回：
        奖励函数
    """
    if strategy == "baseline":
        return binary_reward
    elif strategy == "verbalized_brier":
        return verbalized_confidence_brier_reward
    elif strategy == "verbalized_ce":
        return lambda c, g, **kw: verbalized_confidence_ce_reward(
            c, g, epsilon=kwargs.get('epsilon', 0.01), **kw
        )
    elif strategy == "critic_value":
        return critic_value_reward
    elif strategy == "claim_product":
        return lambda c, g, **kw: claim_level_confidence_reward(
            c, g, aggregation="product", **kw
        )
    elif strategy == "claim_minimum":
        return lambda c, g, **kw: claim_level_confidence_reward(
            c, g, aggregation="minimum", **kw
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("测试奖励函数")
    print("=" * 80)

    # 测试数据
    completions = [
        "Answer: 42 | Confidence: 0.9",  # 正确，高置信度
        "Answer: 43 | Confidence: 0.3",  # 错误，低置信度
        "Answer: 42 | Confidence: 0.5",  # 正确，中等置信度
        "Answer: 43",                     # 错误，无置信度
    ]
    ground_truths = ["42", "42", "42", "42"]

    # 测试各种奖励函数
    strategies = [
        "baseline",
        "verbalized_brier",
        "verbalized_ce",
    ]

    for strategy in strategies:
        print(f"\n{strategy}:")
        print("-" * 80)
        reward_fn = get_reward_function(strategy)
        rewards = reward_fn(completions, ground_truths)
        for i, (comp, reward) in enumerate(zip(completions, rewards)):
            print(f"  {i+1}. {comp[:50]:50s} -> {reward:+.4f}")

    print("\n" + "=" * 80)
    print("✓ 所有奖励函数测试完成")
    print("=" * 80)
