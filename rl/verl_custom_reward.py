"""
VERL 自定义奖励函数（行为校准版）
基于论文作者提供的 behavioral_calibration.py，提供标准 compute_score 接口。
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

import re

import numpy as np

# 让本文件可以直接导入仓库根目录的 behavioral_calibration.py
_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from behavioral_calibration import (  # noqa: E402
    compute_score as _compute_score_explicit,
    verify as _verify_correctness,
    reward_func_confidence as _reward_confidence_brier,
    reward_func_confidence_ce as _reward_confidence_ce,
    reward_func_proof_confidence_prod as _reward_claim_prod,
    reward_func_proof_confidence_min as _reward_claim_min,
)
from verl.utils.reward_score import default_compute_score  # noqa: E402
from verl.utils.reward_score import search_r1_like_qa_em as _search_r1_qa  # noqa: E402


def _normalize_data_source(data_source: str) -> str:
    # 兼容数据源命名差异
    if data_source == "dapo_math":
        return "math_dapo"
    return data_source


def _is_search_r1_data_source(data_source: str) -> bool:
    ds = str(data_source or "").lower()
    return "searchr1" in ds


def _is_openqa_data_source(data_source: str) -> bool:
    ds = str(data_source or "").lower()
    openqa_prefixes = (
        "nq",
        "triviaqa",
        "popqa",
        "hotpotqa",
        "2wikimultihopqa",
        "musique",
        "bamboogle",
    )
    return ds.startswith(openqa_prefixes)


def _extract_search_r1_target(ground_truth: Any) -> Any:
    if isinstance(ground_truth, dict) and "target" in ground_truth:
        return ground_truth["target"]
    return ground_truth


def _normalize_ground_truth(ground_truth: Any) -> Any:
    if isinstance(ground_truth, dict):
        if "target" in ground_truth:
            return ground_truth["target"]
        if "answer" in ground_truth:
            return ground_truth["answer"]
        if "golden_answer" in ground_truth:
            return ground_truth["golden_answer"]
    if isinstance(ground_truth, (list, tuple)):
        if len(ground_truth) == 1:
            return ground_truth[0]
    return ground_truth


def _extract_qa_answer(solution_str: str | None) -> str | None:
    if not isinstance(solution_str, str) or not solution_str.strip():
        return None

    answer_match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", solution_str)
    if answer_match:
        extracted = answer_match[-1].strip()
        if extracted:
            return extracted

    final_match = re.findall(r"(?i)Final\s*Answer\s*:\s*([^\n]+)", solution_str)
    if final_match:
        extracted = final_match[-1].strip()
        if extracted:
            return extracted

    best_guess_match = re.findall(r"(?i)Best\s*Guess\s*:\s*([^\n]+)", solution_str)
    if best_guess_match:
        extracted = best_guess_match[-1].strip()
        if extracted:
            return extracted

    answer_is_match = re.findall(r"(?i)(?:My\s+answer\s+is|The\s+answer\s+is)\s*([^\n.]+)", solution_str)
    if answer_is_match:
        extracted = answer_is_match[-1].strip()
        if extracted:
            return extracted

    answer = _search_r1_qa.extract_solution(solution_str)
    if answer:
        return answer.strip()

    return None


def _is_idk_answer(answer: str | None) -> bool:
    if answer is None:
        return False
    norm = re.sub(r"\s+", " ", answer.strip().lower())
    return norm in {"<idk>", "<idk/>", "idk", "i don't know", "i do not know", "unknown"}


def _compute_qa_binary_reward(solution_str: str, ground_truth: Any) -> Dict[str, Any]:
    answer = _extract_qa_answer(solution_str)
    abstention = _is_idk_answer(answer)
    correct = False
    if answer is not None and not abstention:
        targets = _extract_search_r1_target(ground_truth)
        correct = bool(_search_r1_qa.em_check(answer, targets))

    reward = 1.0 if correct else -1.0
    pred = answer if answer is not None else "<NO_ANSWER>"
    return {
        "score": float(reward),
        "acc": bool(correct),
        "abstention": bool(abstention),
        "pred": pred,
    }


def compute_score_baseline(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dict[str, Any] | float:
    """Baseline 二元奖励 (+1/-1)，使用作者数学判题规则。"""
    data_source = _normalize_data_source(data_source)
    ground_truth = _normalize_ground_truth(ground_truth)
    if data_source == "math_dapo" or data_source.startswith("aime"):
        solution_str = solution_str[-300:]
        correct, abstention, pred = _verify_correctness(solution_str, ground_truth)
        acc = bool(correct) and not bool(abstention)
        reward = 1.0 if acc else -1.0
        return {
            "score": float(reward),
            "acc": acc,
            "abstention": bool(abstention),
            "pred": pred,
        }

    if _is_search_r1_data_source(data_source) or _is_openqa_data_source(data_source):
        return _compute_qa_binary_reward(solution_str, ground_truth)

    return default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info)


def compute_score_explicit_risk(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dict[str, Any] | float:
    """显式风险阈值奖励（Figure 1 相关）。"""
    data_source = _normalize_data_source(data_source)
    ground_truth = _normalize_ground_truth(ground_truth)
    if data_source == "math_dapo" or data_source.startswith("aime"):
        t = 0.5
        if extra_info and "confidence" in extra_info:
            try:
                t = float(extra_info["confidence"])
            except Exception:
                t = 0.5
        return _compute_score_explicit(solution_str, ground_truth, t=t)
    
    if _is_search_r1_data_source(data_source) or _is_openqa_data_source(data_source):
        t = 0.5
        if extra_info and "confidence" in extra_info:
            try:
                t = float(extra_info["confidence"])
            except Exception:
                t = 0.5
        
        t_clamped = max(min(t, 0.999), 0.0)

        answer = _search_r1_qa.extract_solution(solution_str)
        abstention = _is_idk_answer(answer)
        correct = False
        if not abstention and answer is not None:
            targets = _extract_search_r1_target(ground_truth)
            correct = bool(_search_r1_qa.em_check(answer, targets))

        if abstention:
            reward = 0.0
        else:
            reward = 1.0 if correct else -(t_clamped / (1.0 - t_clamped))

        # if abstention:
        #     reward = -1.0
        # else:
        #     reward = 1.0 if correct else - (1.0 + t_clamped) / (1.0 - t_clamped)

        pred = answer
        if pred is None:
            pred = "<IDK>" if abstention else "<NO_ANSWER>"

        return {
            "score": float(reward),
            "acc": bool(correct) and not bool(abstention),
            "abstention": bool(abstention),
            "pred": pred,
        }
    
    return default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info)


def compute_score_verbalized_brier(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Verbalized Confidence + Brier Score (Eq. 3)."""
    data_source = _normalize_data_source(data_source)
    ground_truth = _normalize_ground_truth(ground_truth)
    if data_source == "math_dapo" or data_source.startswith("aime"):
        return _reward_confidence_brier(data_source, solution_str, ground_truth, extra_info=extra_info)

    if _is_search_r1_data_source(data_source) or _is_openqa_data_source(data_source):
        extracted_answer = _extract_qa_answer(solution_str)

        conf_match = re.findall(r"(?i)Confidence\s*:\s*([^\n]+)", solution_str)
        if conf_match:
            try:
                confidence = float(conf_match[-1].strip())
            except (TypeError, ValueError):
                # If confidence is present but malformed, treat as overconfident.
                confidence = 1.0
        else:
            confidence = 0.5
        confidence = float(np.clip(confidence, 0.0, 1.0))

        correct = False
        if extracted_answer is not None and not _is_idk_answer(extracted_answer):
            targets = _extract_search_r1_target(ground_truth)
            correct = bool(_search_r1_qa.em_check(extracted_answer, targets))

        acc = 1.0 if correct else 0.0
        score = 2.0 * confidence * acc - confidence * confidence
        pred = extracted_answer if extracted_answer is not None else "[INVALID]"
        return {
            "score": float(score),
            "acc": bool(correct),
            "confidence": float(confidence),
            "pred": pred,
        }

    raise NotImplementedError(f"compute_score_verbalized_brier not implemented for data_source={data_source}")


def compute_score_verbalized_ce(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Verbalized Confidence + Cross-Entropy (Beta prior)."""
    data_source = _normalize_data_source(data_source)
    ground_truth = _normalize_ground_truth(ground_truth)

    epsilon = kwargs.get("epsilon", None)
    if epsilon is None and extra_info and "epsilon" in extra_info:
        epsilon = extra_info.get("epsilon")
    try:
        epsilon = float(epsilon) if epsilon is not None else 0.01
    except Exception:
        epsilon = 0.01
    epsilon = float(np.clip(epsilon, 1e-6, 0.49))

    if data_source == "math_dapo" or data_source.startswith("aime"):
        score_dict = _reward_confidence_ce(data_source, solution_str, ground_truth, extra_info=extra_info)
        try:
            r = 1.0 if bool(score_dict.get("acc", False)) else 0.0
            p = float(score_dict.get("confidence", 0.0))
            p = float(np.clip(p, epsilon, 1.0 - epsilon))
            score_dict["score"] = float(
                (r * np.log(p / epsilon) + (1.0 - r) * np.log((1.0 - p) / (1.0 - epsilon)))
                / np.log((1.0 - epsilon) / epsilon)
            )
        except Exception:
            pass
        return score_dict

    if _is_search_r1_data_source(data_source) or _is_openqa_data_source(data_source):
        extracted_answer = _extract_qa_answer(solution_str)

        conf_match = re.findall(r"(?i)Confidence\s*:\s*([^\n]+)", solution_str)
        if conf_match:
            try:
                confidence = float(conf_match[-1].strip())
            except (TypeError, ValueError):
                confidence = 1.0
        else:
            confidence = 0.5
        confidence = float(np.clip(confidence, 0.0, 1.0))

        correct = False
        if extracted_answer is not None and not _is_idk_answer(extracted_answer):
            targets = _extract_search_r1_target(ground_truth)
            correct = bool(_search_r1_qa.em_check(extracted_answer, targets))

        r = 1.0 if correct else 0.0
        p = float(np.clip(confidence, epsilon, 1.0 - epsilon))
        score = (r * np.log(p / epsilon) + (1.0 - r) * np.log((1.0 - p) / (1.0 - epsilon))) / np.log(
            (1.0 - epsilon) / epsilon
        )

        pred = extracted_answer if extracted_answer is not None else "[INVALID]"
        return {
            "score": float(score),
            "acc": bool(correct),
            "confidence": float(confidence),
            "pred": pred,
        }

    raise NotImplementedError(f"compute_score_verbalized_ce not implemented for data_source={data_source}")


def compute_score_critic_value(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dict[str, Any] | float:
    """
    Critic Value 作为置信度。
    若 extra_info 中未提供 value，则回退到 baseline 二元奖励。
    """
    data_source = _normalize_data_source(data_source)
    ground_truth = _normalize_ground_truth(ground_truth)

    critic_value = None
    if extra_info:
        for key in ("critic_value", "value", "vpred", "value_pred"):
            if key in extra_info:
                critic_value = extra_info[key]
                break

    if critic_value is None:
        return compute_score_baseline(data_source, solution_str, ground_truth, extra_info=extra_info)

    try:
        critic_value = float(critic_value)
    except Exception:
        return compute_score_baseline(data_source, solution_str, ground_truth, extra_info=extra_info)

    # critic 预测通常位于 [-1, 1]，映射到 [0, 1]
    if critic_value < 0.0 or critic_value > 1.0:
        confidence = (critic_value + 1.0) / 2.0
    else:
        confidence = critic_value
    confidence = float(np.clip(confidence, 0.0, 1.0))

    correct, abstention, pred = _verify_correctness(solution_str, ground_truth)
    acc = bool(correct) and not bool(abstention)

    reward = 2.0 * confidence * (1.0 if acc else 0.0) - confidence * confidence

    return {
        "score": float(reward),
        "acc": bool(acc),
        "confidence": confidence,
        "pred": pred,
    }


def compute_score_claim_product(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Claim-level Confidence + Product 聚合。"""
    data_source = _normalize_data_source(data_source)
    return _reward_claim_prod(data_source, solution_str, ground_truth, extra_info=extra_info)


def compute_score_claim_minimum(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Claim-level Confidence + Minimum 聚合。"""
    data_source = _normalize_data_source(data_source)
    return _reward_claim_min(data_source, solution_str, ground_truth, extra_info=extra_info)
