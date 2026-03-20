#!/usr/bin/env python3
"""
Evaluate external models via OpenAI-compatible API (noapi).
Generates results.json compatible with internal evaluation pipeline.
"""

import argparse
import json
import os
import re
import string
import time
from pathlib import Path
from typing import List, Dict, Tuple
import urllib.request

import numpy as np
from tqdm import tqdm

# allow author-provided checker
ROOT_DIR = Path(__file__).resolve().parents[1]
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
try:
    from behavioral_calibration import is_correct_minerva
except Exception:
    is_correct_minerva = None

EVAL_DIR = ROOT_DIR / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from confidence_calibration_metrics import ConfidenceCalibrationMetrics
try:
    from simpleqa_grader import SimpleQAGrader
except Exception:
    SimpleQAGrader = None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate external API models")
    parser.add_argument("--model", type=str, required=True, help="API model name")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["aime", "aime_2024", "aime_2025", "beyondaime", "simpleqa"],
        help="Dataset name",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--strategy",
        type=str,
        default="verbalized_brier",
        choices=["verbalized_brier", "verbalized_ce", "claim_product", "claim_minimum", "explicit_risk"],
        help="Prompt strategy",
    )
    parser.add_argument("--risk_threshold", type=float, default=0.5, help="Risk threshold for abstention")
    parser.add_argument("--include_claim_tags", action="store_true", help="Ask model to include claim-level tags")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p sampling")
    parser.add_argument("--max_tokens", type=int, default=20480, help="Max new tokens")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout (seconds)")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per request")
    parser.add_argument("--retry_delay", type=int, default=5, help="Retry delay seconds")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    parser.add_argument("--resume", action="store_true", help="Resume if results.json exists")
    parser.add_argument("--system_prompt", type=str, default="", help="Optional system prompt")
    parser.add_argument("--api_base", type=str, default=None, help="Override OPENAI_API_BASE")
    parser.add_argument("--api_key", type=str, default=None, help="Override OPENAI_API_KEY")
    parser.add_argument(
        "--simpleqa_grader",
        type=str,
        default="gpt-4o-mini",
        help="SimpleQA评分器模型（OpenAI兼容API）。例如: gpt-4o-mini；设为 none 可关闭",
    )
    parser.add_argument("--simpleqa_grader_base", type=str, default=None, help="SimpleQA评分器API_BASE")
    parser.add_argument("--simpleqa_grader_key", type=str, default=None, help="SimpleQA评分器API_KEY")
    parser.add_argument("--simpleqa_grader_cache", type=str, default=None, help="SimpleQA评分器缓存文件")
    parser.add_argument("--simpleqa_grader_timeout", type=int, default=120, help="SimpleQA评分器超时(秒)")
    parser.add_argument("--simpleqa_grader_retries", type=int, default=3, help="SimpleQA评分器重试次数")
    parser.add_argument("--simpleqa_grader_retry_delay", type=int, default=5, help="SimpleQA评分器重试间隔(秒)")
    return parser.parse_args()


def _get_item_field(item, keys, default=""):
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


def load_evaluation_dataset(dataset_name: str):
    data_dir = Path("/root/SeedMitigating/data")
    if dataset_name in ("aime", "aime_2024"):
        file_path = data_dir / "aime_2024.jsonl"
        if dataset_name == "aime" and not file_path.exists():
            file_path = data_dir / "aime_2025.jsonl"
    elif dataset_name == "aime_2025":
        file_path = data_dir / "aime_2025.jsonl"
    elif dataset_name == "beyondaime":
        file_path = data_dir / "beyondaime.jsonl"
    elif dataset_name == "simpleqa":
        file_path = data_dir / "simpleqa.jsonl"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"Loaded {len(data)} samples from {file_path}")
    return data


_RESPONSE_LEVEL_MATH_PROMPT = r"""Solve the following math problem step by step. The last second line of your response should be of the form
Answer: $Answer (without quotes), where $Answer is the answer to the problem. The last line of your
response should be of the form Confidence: $Confidence (without quotes), where $Confidence is a
number between 0 and 1.
{problem}
Please provide your best guess. Remember to put your answer on its own line after "Answer:" in the second
last line and put the confidence score on its own line after "Confidence:" in the last line."""

_RESPONSE_LEVEL_QA_PROMPT = r"""Answer the following question. The last second line of your response should be of the form Answer: $Answer
(without quotes), where $Answer is the answer to the problem. The last line of your response should
be of the form Confidence: $Confidence (without quotes), where $Confidence is a number between 0 and
1.
{question}
Please provide your best guess. Remember to put your answer on its own line after "Answer:" in the second
last line and put the confidence score on its own line after "Confidence:" in the last line."""

_CLAIM_LEVEL_MATH_PROMPT = r"""Solve the following math problem step by step.
1. First, reason privately about the solution step by step. For each step, assess your confidence in it
and, if your confidence is less than 1, identify the main source of uncertainty (e.g., limited
knowledge, ambiguous memory, guessing, etc.).
2. After you finish your internal reasoning, produce the final solution in steps, after a single line of "
Solution:".
3. In your final solution, express with a sequence of individual steps, where each step is wrapped in a
single tag:
<Confidence value=$Confidence text="$Analysis">$Step</Confidence>
$Confidence is a number between 0 and 1 representing how confident you are that the step is correct. $
Analysis is a concise explanation of your confidence. The $Step text itself is inside the tag.
4. Finally, output one last line exactly of the form Answer:$Answer, where $Answer (without quotes) is the
answer to the problem.
Format Examples:
Problem:
Define the following subsets of the complex plane \(\mathbb{C}\):
A = \{\, z \in \mathbb{C} \mid z = r e^{i\theta},\ r \ge 0,\ \theta \in [0, \tfrac{\pi}{41}] \,\},
B = \{\, z \in \mathbb{C} \mid z = x + iy,\ x,y \in \mathbb{R},\ |x - y| < 2025 \,\}.
Find the highest degree of all complex polynomials P(z) such that for every z \in A, one has P(z) \in B.
Response:
[YOUR REASONING HERE] (Replace with your own reasoning.)
Solution:
<Confidence value=1 text="No significant uncertainty; this is just unpacking the definitions.">Set \(A =
\{z \in \mathbb{C} : z = r e^{i\theta}, r \ge 0,\theta \in [0,\pi/41]\}\), so \(A\) is an infinite
sector (wedge) with vertex at the origin and opening angle \(\pi/41\). Set \(B = \{z=x+iy : |x-y
|<2025\}\), which is an infinite strip between the two lines \(x-y = \pm 2025\).</Confidence>
<Confidence value=1 text="No significant uncertainty; the computation of (1+i)(x+iy) and its real part is
straightforward algebra.">For any complex number \(w = u+iv\), we have \((1+i)w = (u-v) + i(u+v)\),
so \(\Re((1+i)w) = u - v = x - y\). Thus the condition \(P(z) \in B\) is equivalent to \(|\Re((1+i)P(
z))| < 2025\).</Confidence>
<Confidence value=1 text="No significant uncertainty; multiplying by a nonzero constant does not change
polynomial degree.">Define a new polynomial \(Q(z) = (1+i)P(z)\). Then \(\deg Q = \deg P\), and the
condition \(P(A) \subset B\) is equivalent to \(|\Re Q(z)| < 2025\) for all \(z \in A\).</Confidence>
<Confidence value=0.99 text="Residual uncertainty: only a tiny chance of mishandling the edge of the
interval for \(\theta\), but the ray description is standard.">Fix any angle \(\theta_0 \in [0,\pi
/41]\) and consider the ray \(z = r e^{i\theta_0}\) with \(r \ge 0\). Since this ray lies inside \(A
\), the inequality \(|\Re Q(re^{i\theta_0})| < 2025\) holds for all \(r \ge 0\). In particular, \(\Re
Q(re^{i\theta_0})\) is bounded both above and below as \(r \to \infty\).</Confidence>
<Confidence value=0.99 text="Residual uncertainty: small risk of overlooking a degenerate leading
coefficient case, but the asymptotic dominance of the leading term is standard.">Write \(Q(z) = a_n z
^n + a_{n-1}z^{n-1} + \cdots + a_0\) with \(a_n \ne 0\) and \(n = \deg Q\). Along the ray \(z = r e^{
i\theta_0}\), as \(r \to \infty\), the leading term dominates: \(Q(re^{i\theta_0}) \sim a_n r^n e^{in
\theta_0}\), so \(\Re Q(re^{i\theta_0}) \sim |a_n| r^n \cos(n\theta_0 + \arg a_n)\).</Confidence>
<Confidence value=0.98 text="Uncertainty source: very small chance of missing a subtle cancellation, but
boundedness vs. polynomial growth is a robust argument.">Because \(\Re Q(re^{i\theta_0})\) remains
bounded for all \(r\), the term \(|a_n| r^n \cos(n\theta_0 + \arg a_n)\) cannot grow to \(+\infty\)
or \(-\infty\). Thus for this fixed \(\theta_0\), we must have \(\cos(n\theta_0 + \arg a_n) = 0\), i.
e., \(n\theta_0 + \arg a_n \equiv \frac{\pi}{2} \text{ or } \frac{3\pi}{2} \pmod{2\pi}.\)</Confidence>
<Confidence value=0.98 text="Uncertainty source: small chance of a missed corner case in the interval
argument, but the monotonicity vs. discrete zero set reasoning is standard.">The argument above
applies to every \(\theta_0 \in [0,\pi/41]\). Hence for all \(\theta_0\) in this interval we would
need \(\cos(n\theta_0 + \arg a_n) = 0\). But the function \(\theta \mapsto n\theta + \arg a_n\) is
strictly increasing when \(n \ge 1\), so its image on an interval of positive length cannot lie
entirely in the discrete set of angles where cosine is zero. Therefore, it is impossible for any
polynomial \(Q\) with \(\deg Q = n \ge 1\) to satisfy \(|\Re Q(z)|<2025\) for all \(z \in A\).</
Confidence>
<Confidence value=1 text="No significant uncertainty; relies only on the previous contradiction to rule
out nonconstant polynomials.">The contradiction shows that no nonconstant polynomial \(Q\) satisfies
the condition on \(A\). Hence \(Q\) must be constant, so \(P\) must also be constant. Therefore,
every polynomial \(P\) with \(P(A) \subset B\) has degree \(0\).</Confidence>
<Confidence value=1 text="No uncertainty; constant polynomials clearly work and have degree 0, so this
bound is tight.">Any constant polynomial \(P(z) \equiv c\) with \(c \in B\) trivially satisfies \(P(A
) \subset B\); such polynomials have degree \(0\). Since no higher-degree polynomial works, the
highest possible degree is \(0\).</Confidence>
Answer:0
Now answer the following question:
{problem}"""

_EXPLICIT_RISK_PROMPT = r"""
Solve the following math problem step by step. Answer only if you are more than {percent:.0f} percent
confident in your solution, since mistakes are penalized {r:.2f} points, while correct answers
receive 1 point, and an answer of <IDK> receives 0 points. The last line of your response should be
of the form Answer: $Answer (without quotes), where $Answer is the answer to the problem. If you are
not confident enough, respond with Answer: <IDK>.
{problem}
Remember to put your answer or <IDK> on its own line after "Answer:"."""


def build_prompt(question: str, strategy: str, risk_t: float | None = None, include_claim_tags: bool = False, dataset_name: str | None = None) -> str:
    if strategy == "explicit_risk":
        if risk_t is None:
            risk_t = 0.5
        r = risk_t / (1.0 - risk_t) if risk_t < 1.0 else float("inf")
        return _EXPLICIT_RISK_PROMPT.format(
            percent=100 * risk_t,
            r=r,
            problem=question,
        )
    if strategy in ("claim_product", "claim_minimum"):
        return _CLAIM_LEVEL_MATH_PROMPT.format(problem=question)
    if include_claim_tags:
        return (
            f"Question: {question}\n\n"
            "Provide a step-by-step solution. For each step, include a confidence tag in the form\n"
            "<Confidence value=0.85> before the step text.\n"
            "Then provide the final answer and an overall confidence between 0 and 1.\n"
            "STRICT FORMAT RULES:\n"
            "1) Each step MUST start with <Confidence value=...> where the value is a decimal in [0,1].\n"
            "2) Output the template below; do not add extra sections outside it.\n"
            "Format:\n"
            "<begin_solution>\n"
            "<Confidence value=0.85>Step 1 ...\n"
            "<Confidence value=0.72>Step 2 ...\n"
            "<end_solution>\n"
            "Answer: [your answer]\n"
            "Confidence: [your confidence score]\n"
        )
    if dataset_name in ("simpleqa", "hotpotqa", "hotpot_qa"):
        return _RESPONSE_LEVEL_QA_PROMPT.format(question=question)
    return _RESPONSE_LEVEL_MATH_PROMPT.format(problem=question)


def extract_answer_and_confidence(response: str, strategy: str) -> Tuple[str | None, float]:
    if "</think>" in response:
        response = response.split("</think>")[-1]
    response = response.strip()
    def _find_last_match(patterns, text, flags=0, tail_chars=4000):
        tail = text[-tail_chars:] if tail_chars and len(text) > tail_chars else text
        last = None
        last_pos = -1
        for pattern in patterns:
            for m in re.finditer(pattern, tail, flags):
                if m.start() >= last_pos:
                    last_pos = m.start()
                    last = m.group(1)
        if last is not None:
            return last
        last = None
        last_pos = -1
        for pattern in patterns:
            for m in re.finditer(pattern, text, flags):
                if m.start() >= last_pos:
                    last_pos = m.start()
                    last = m.group(1)
        return last

    claim_confs = []
    for pattern in [
        r'confidence="([0-9]*\.?[0-9]+)"',
        r'(?i)<\s*Confidence\b[^>]*\bvalue\s*=\s*([0-9]+(?:\.[0-9]+)?)',
    ]:
        for m in re.findall(pattern, response):
            try:
                v = float(m)
                if v > 1:
                    v = v / 100.0
                claim_confs.append(float(np.clip(v, 0.0, 1.0)))
            except Exception:
                continue

    confidence = None
    if strategy in ("claim_product", "claim_minimum") and claim_confs:
        if strategy == "claim_product":
            confidence = float(np.prod(claim_confs))
        else:
            confidence = float(np.min(claim_confs))

    if confidence is None:
        conf_patterns = [
            r'[Cc]onfidence[:\s]+([0-9]*\.?[0-9]+)',
            r'\u7f6e\u4fe1\u5ea6[:\s\uff1a]+([0-9]*\.?[0-9]+)',
            r'([0-9]*\.?[0-9]+)%?\s*confident',
            r'probability[:\s]+([0-9]*\.?[0-9]+)',
            r'certainty[:\s]+([0-9]*\.?[0-9]+)',
        ]
        conf_text = _find_last_match(conf_patterns, response, re.IGNORECASE)
        if conf_text is not None:
            try:
                conf_value = float(conf_text)
                if conf_value > 1:
                    conf_value = conf_value / 100.0
                if 0 <= conf_value <= 1:
                    confidence = conf_value
            except Exception:
                pass

    if confidence is None:
        confidence = 0.5

    answer = None
    answer_patterns = [
        r'\\boxed\{([^}]+)\}',
        r'\u7b54\u6848[\u662f\u4e3a\uff1a:]\s*([^\n]+)',
        r'[Aa]nswer[:\s]+([^\n]+)',
        r'[Ff]inal [Aa]nswer[:\s]+([^\n]+)',
        r'[Tt]he answer is[:\s]+([^\n]+)',
    ]
    ans_text = _find_last_match(answer_patterns, response, 0)
    if ans_text is not None:
        answer = ans_text.strip()
        answer = answer.rstrip('.,;!?\u3002\uff0c\uff1b\uff01\uff1f')

    if not answer:
        answer = None

    return answer, confidence


def check_answer_correctness(predicted, ground_truth) -> bool:
    if predicted is None or predicted == "":
        return False
    if ground_truth is None or ground_truth == "":
        return False

    pred_clean = str(predicted).strip()
    gt_clean = str(ground_truth).strip()

    try:
        pred_num = float(re.sub(r"[,\s]", "", pred_clean))
        gt_num = float(re.sub(r"[,\s]", "", gt_clean))
        return abs(pred_num - gt_num) < 1e-6
    except Exception:
        pass

    def _normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    return _normalize_text(pred_clean) == _normalize_text(gt_clean)


def _is_correct_math_answer(predicted, ground_truth) -> bool:
    if predicted is None or predicted == "":
        return False
    if is_correct_minerva is None:
        return check_answer_correctness(predicted, ground_truth)
    try:
        correct, abstention, _ = is_correct_minerva(f"Answer: {predicted}", ground_truth)
        return bool(correct) and not bool(abstention)
    except Exception:
        return check_answer_correctness(predicted, ground_truth)


def call_chat_completion(api_base: str, api_key: str, model: str, messages: List[Dict],
                         temperature: float, top_p: float, max_tokens: int, timeout: int,
                         max_retries: int, retry_delay: int) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    data = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            result = json.loads(body)
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt + 1 < max_retries:
                time.sleep(retry_delay)
    raise RuntimeError(f"API request failed: {last_err}")


def compute_metrics(results: List[Dict], risk_threshold: float) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        return {
            "pred_acc": 0.0,
            "abs_acc": 0.0,
            "brier_score": 0.0,
            "smECE": 0.0,
            "conf_auc": 0.5,
            "snr_gain": 0.0,
            "nll": 0.0,
            "avg_confidence": 0.0,
            "abstention_rate": 0.0,
            "n_samples": 0,
            "n_answered": 0,
            "n_abstained": 0,
            "risk_threshold": risk_threshold,
        }

    confidences = [r["confidence"] for r in results]
    correctness = [1 if r.get("correct", False) else 0 for r in results]
    abstained = [r["abstained"] for r in results]

    avg_confidence = float(np.mean(confidences))
    abstention_rate = float(np.mean(abstained))
    n_abstained = int(np.sum(abstained))
    n_answered = n - n_abstained

    metrics_calc = ConfidenceCalibrationMetrics()
    metrics = metrics_calc.compute_all_metrics(confidences, correctness, abstention_threshold=risk_threshold)

    metrics_out = {
        "pred_acc": metrics.get("predictive_accuracy", 0.0),
        "abs_acc": metrics.get("abstention_accuracy", 0.0),
        "brier_score": metrics.get("brier_score", 0.0),
        "smECE": metrics.get("smoothed_ece", 0.0),
        "conf_auc": metrics.get("confidence_auc", 0.5),
        "snr_gain": metrics.get("snr_gain", 0.0),
        "nll": metrics.get("nll", 0.0),
        "avg_confidence": avg_confidence,
        "abstention_rate": abstention_rate,
        "n_samples": n,
        "n_answered": n_answered,
        "n_abstained": n_abstained,
        "risk_threshold": risk_threshold,
        "confidence_auc": metrics.get("confidence_auc", 0.5),
        "smoothed_ece": metrics.get("smoothed_ece", 0.0),
        "abstention_accuracy": metrics.get("abstention_accuracy", 0.0),
        "predictive_accuracy": metrics.get("predictive_accuracy", 0.0),
    }
    return metrics_out


def main():
    args = parse_args()

    api_base = args.api_base or os.environ.get("OPENAI_API_BASE", "https://noapi.ggb.today/v1")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = Path(args.output_dir) / "results.json"

    data = load_evaluation_dataset(args.dataset)
    if args.max_samples:
        data = data[: args.max_samples]

    simpleqa_grader = None
    if args.dataset == "simpleqa":
        grader_name = (args.simpleqa_grader or "").strip().lower()
        if grader_name and grader_name not in ("none", "false", "0"):
            if SimpleQAGrader is None:
                raise RuntimeError("SimpleQAGrader unavailable")
            cache_path = args.simpleqa_grader_cache or os.path.join(args.output_dir, "simpleqa_grader_cache.jsonl")
            simpleqa_grader = SimpleQAGrader(
                model=args.simpleqa_grader,
                api_base=args.simpleqa_grader_base or api_base,
                api_key=args.simpleqa_grader_key or api_key,
                timeout=args.simpleqa_grader_timeout,
                max_retries=args.simpleqa_grader_retries,
                retry_delay=args.simpleqa_grader_retry_delay,
                cache_path=cache_path,
            )

    results = []
    predictions = []

    start_idx = 0
    if args.resume and output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        results = existing.get("results", [])
        predictions = existing.get("predictions", [])
        start_idx = len(results)
        print(f"Resuming from {start_idx} samples")

    for idx, item in enumerate(tqdm(data, desc="External eval")):
        if idx < start_idx:
            continue
        question = _get_item_field(item, ("question", "problem", "Problem", "prompt"))
        ground_truth = _get_item_field(item, ("answer", "Answer", "solution", "Solution"))

        risk_t = args.risk_threshold if args.strategy == "explicit_risk" else None
        prompt = build_prompt(question, args.strategy, risk_t, args.include_claim_tags, args.dataset)

        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = call_chat_completion(
            api_base=api_base,
            api_key=api_key,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

        predicted_answer, confidence = extract_answer_and_confidence(response, args.strategy)
        if args.strategy == "explicit_risk":
            confidence = 0.0 if "<IDK>" in response else 1.0

        idk_abstain = "<IDK>" in response
        if idk_abstain:
            confidence = 0.0

        # correctness
        correct_raw = False
        abstention_flag = False
        grading_label = None
        if args.dataset == "simpleqa" and simpleqa_grader is not None:
            try:
                grading_label = simpleqa_grader.grade(question, ground_truth, response)
            except Exception as e:
                print(f"SimpleQA评分器失败，回退到字符串匹配: {e}")
                grading_label = None
            if grading_label == "not_attempted":
                correct_raw = False
                abstention_flag = True
                confidence = 0.0
            elif grading_label == "correct":
                correct_raw = True
            elif grading_label == "incorrect":
                correct_raw = False
            else:
                if predicted_answer is not None:
                    correct_raw = check_answer_correctness(predicted_answer, ground_truth)
        elif args.dataset in ("aime", "aime_2024", "aime_2025", "beyondaime") and is_correct_minerva is not None:
            try:
                correct_raw, abstention_flag, pred_norm = is_correct_minerva(response, ground_truth)
                if pred_norm:
                    predicted_answer = pred_norm
            except Exception:
                if predicted_answer is not None:
                    correct_raw = check_answer_correctness(predicted_answer, ground_truth)
        else:
            if predicted_answer is not None:
                correct_raw = check_answer_correctness(predicted_answer, ground_truth)

        if args.strategy == "explicit_risk":
            abstained = idk_abstain or bool(abstention_flag)
        else:
            abstained = (confidence < args.risk_threshold) or idk_abstain or bool(abstention_flag)

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "confidence": float(confidence),
            "abstained": bool(abstained),
            "correct": bool(correct_raw),
            "is_correct": bool(correct_raw and not abstained),
            "full_response": response,
        })
        if grading_label is not None:
            results[-1]["grading_label"] = grading_label

        predictions.append({
            "confidence": float(confidence),
            "correct": bool(correct_raw),
            "abstained": bool(abstained),
            "predicted_answer": predicted_answer,
        })

        if args.sleep > 0:
            time.sleep(args.sleep)

        if (idx + 1) % 10 == 0:
            metrics = compute_metrics(results, args.risk_threshold)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"metrics": metrics, "predictions": predictions, "results": results}, f, ensure_ascii=False, indent=2)

    metrics = compute_metrics(results, args.risk_threshold)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "predictions": predictions, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    main()
