# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
import numpy as np
from typing import Optional, Any


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]



# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()



'''
Explicit Risk Thresholding
'''
def is_correct_minerva(solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)", idk_token: str = "<IDK>") -> tuple[bool, bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, is_abstention, normalized_prediction)
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    if idk_token in pred:
        return False, True, pred
    else:
        return (pred == gt), False, pred


def is_correct_strict_box(pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None) -> tuple[bool, bool, str]:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (is_correct, is_abstention, prediction)
    """
    if strict_box_verify:
        raise NotImplementedError
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, abstention, pred = is_correct_minerva(solution_str, answer)
    return correct, abstention, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    t: float,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
):
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        t: Confidence threshold
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution
    correct, abstention, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    if abstention:
        reward = 0.0
    else:
        t_clamped = max(min(t, 0.999999), 0.0)
        reward = 1.0 if correct else - (t_clamped / (1.0 - t_clamped))
    acc = correct and not abstention

    return {
        "score": reward,
        "acc": acc,
        "abstention": abstention,
        "pred": pred,
    }

def reward_func(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "math_dapo" or data_source.startswith("aime"):
        return compute_score(solution_str, ground_truth, t=extra_info['confidence'])
    else:
        raise NotImplementedError


'''
Response-Level Confidence Reward
'''
def is_correct_minerva_confidence(solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)", confidence_pattern: str = r"(?i)Confidence\s*:\s*([^\n]+)") -> tuple[bool, float, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer
        confidence_pattern: Regex pattern to extract the confidence

    Returns:
        Tuple of (is_correct, confidence, normalized_prediction)
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1].strip() if match else ""
    if not extracted_answer:
        fallback_patterns = [
            r"(?im)^Final\s*Answer\s*:\s*([^\n]+)",
            r"(?im)^Best\s*Guess\s*:\s*([^\n]+)",
            r"(?im)^(?:My\s+answer\s+is|The\s+answer\s+is)\s*([^\n.]+)",
        ]
        for pattern in fallback_patterns:
            match = re.findall(pattern, solution_str)
            if match:
                extracted_answer = str(match[-1]).strip()
                if extracted_answer:
                    break
    if not extracted_answer:
        extracted_answer = "[INVALID]"

    match = re.findall(confidence_pattern, solution_str)
    try:
        import numpy as np
        extracted_confidence = float(match[-1]) if match else 0.5
        # if not match:
            # print (1)
            # print (solution_str)
    except ValueError:
        extracted_confidence = 1.0
        # print (2)
    
    extracted_confidence = np.clip(extracted_confidence, 0.0, 1.0)
    
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), extracted_confidence, pred

def verify_confidence(solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None) -> tuple[bool, float, str]:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (is_correct, confidence, prediction)
    """
    if strict_box_verify:
        raise NotImplementedError
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, confidence, pred = is_correct_minerva_confidence(solution_str, answer)
    return correct, confidence, pred

def compute_score_confidence(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
):
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score
    """
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution
    correct, confidence, pred = verify_confidence(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    reward = None
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "confidence": confidence,
        "pred": pred,
    }

def reward_func_confidence(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "math_dapo" or data_source.startswith("aime"):
        score_dict = compute_score_confidence(solution_str, ground_truth)
        score_dict["score"] = 2 * score_dict["confidence"] * score_dict["acc"] - score_dict["confidence"] * score_dict["confidence"]
        return score_dict
    else:
        raise NotImplementedError

def reward_func_confidence_brier(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "math_dapo" or data_source.startswith("aime"):
        score_dict = compute_score_confidence(solution_str, ground_truth)
        score_dict["score"] = 2 * score_dict["confidence"] * score_dict["acc"] - score_dict["confidence"] * score_dict["confidence"]
        return score_dict
    else:
        raise NotImplementedError

def reward_func_confidence_ce(data_source, solution_str, ground_truth, extra_info=None):
    eps = 0.01
    if data_source == "math_dapo" or data_source.startswith("aime"):
        score_dict = compute_score_confidence(solution_str, ground_truth)
        r = score_dict["acc"]
        p = score_dict["confidence"]
        p = np.clip(p, eps, 1 - eps)
        score_dict["score"] = (r * np.log(p / eps) + (1 - r) * np.log((1 - p) / (1 - eps))) / np.log((1 - eps) / eps)
        return score_dict
    else:
        raise NotImplementedError

def reward_func_confidence_jeffrey(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "math_dapo" or data_source.startswith("aime"):
        score_dict = compute_score_confidence(solution_str, ground_truth)
        r = score_dict["acc"]
        p = score_dict["confidence"]
        p = np.clip(p, 0.0, 1.0)
        score_dict["score"] = (2 / np.pi) * ((2 * r - 1) * np.arcsin(np.sqrt(p)) + np.sqrt(p * (1 - p)))
        return score_dict
    else:
        raise NotImplementedError



'''
Claim-Level Confidence Reward
'''
def is_correct_minerva_prodconfidence(
    solution_str: str, 
    gt: str, 
    gt_need_extract: bool = False, 
    answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)", 
    confidence_pattern: str = r"(?i)Confidence\s*:\s*([^\n]+)",
    solution_pattern: str = r"<begin_solution>([\s\S]*?)(?=<begin_solution>|$)",
) -> tuple[bool, np.ndarray, str]:
    """Check if the solution is correct according to Minerva criteria with segment-level confidence.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        tokenizer: The tokenizer to use for tokenization
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer
        confidence_pattern: Regex pattern to extract the confidence
        solution_pattern: Regex pattern to extract the solution part

    Returns:
        Tuple of (is_correct, confidence_array, normalized_prediction)
    """
    # Extract the solution part - use the last begin_solution if multiple exist
    match = re.findall(solution_pattern, solution_str)
    if not match:
        return False, np.array([0.0]), "[INVALID]"
    solution_str = match[-1]

    # answer extraction
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    
    pred = normalize_final_answer(extracted_answer)

    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    # Extract all confidence values from the last begin_solution only
    confidence_values = []
    for match in re.findall(confidence_pattern, solution_str):
        try:
            confidence_values.append(float(match.strip()))
        except ValueError:
            confidence_values.append(1.0)
    
    # Clip confidence values to [0, 1]
    if confidence_values:
        confidence_values = np.clip(confidence_values, 0.0, 1.0)
    else:
        confidence_values = np.array([0.0])

    return (pred == gt), confidence_values, pred


def verify_proof_confidence(solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None) -> tuple[bool, np.ndarray, str]:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (is_correct, confidence, prediction)
    """
    if strict_box_verify:
        raise NotImplementedError
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, confidence_array, pred = is_correct_minerva_prodconfidence(
        solution_str, 
        answer,
        confidence_pattern=r'(?i)<\s*Confidence\b[^>]*\bvalue\s*=\s*([0-9]+(?:\.[0-9]+)?)',
        solution_pattern=r"Solution:([\s\S]*?)(?=Solution:|$)",
    )
    return correct, confidence_array, pred

def compute_score_proof_confidence(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
):
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score
    """
    # Verify the solution
    correct, confidence_array, pred = verify_proof_confidence(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    reward = None
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "confidence_array": confidence_array,
        "pred": pred,
    }

def reward_func_proof_confidence_prod(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "math_dapo" or data_source.startswith("aime"):
        score_dict = compute_score_proof_confidence(solution_str, ground_truth)
        score_dict["confidence"] = np.prod(score_dict["confidence_array"])
        del score_dict["confidence_array"]
        score_dict["score"] = 2 * score_dict["confidence"] * score_dict["acc"] - score_dict["confidence"] * score_dict["confidence"]
        return score_dict
    else:
        raise NotImplementedError

def reward_func_proof_confidence_min(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "math_dapo" or data_source.startswith("aime"):
        score_dict = compute_score_proof_confidence(solution_str, ground_truth)
        score_dict["confidence"] = np.min(score_dict["confidence_array"])
        del score_dict["confidence_array"]
        score_dict["score"] = 2 * score_dict["confidence"] * score_dict["acc"] - score_dict["confidence"] * score_dict["confidence"]
        return score_dict
    else:
        raise NotImplementedError
