"""
自定义数据集封装：支持字符串 prompt，并按策略注入输出格式指令。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np

from verl.utils.dataset.rl_dataset import RLHFDataset


class BehavioralCalibrationDataset(RLHFDataset):
    """为行为校准训练定制的 RLHFDataset。"""

    _RESPONSE_LEVEL_MATH_PROMPT = r"""Solve the following math problem step by step. The last second line of your response should be of the form
Answer: $Answer (without quotes), where $Answer is the answer to the problem. The last line of your
response should be of the form Confidence: $Confidence (without quotes), where $Confidence is a
number between 0 and 1.
{problem}
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
receive 1 point, and an answer of <IDK> receives -1 points. The last line of your response should be
of the form Answer: $Answer (without quotes), where $Answer is the answer to the problem. If you are
not confident enough, respond with Answer: <IDK>.
{problem}
Remember to put your answer or <IDK> on its own line after "Answer:"."""

    def _format_prompt(self, question: str) -> str:
        strategy = self.config.get("strategy", "baseline")
        question = question.strip()

        if strategy in ("verbalized_brier", "verbalized_ce"):
            return self._RESPONSE_LEVEL_MATH_PROMPT.format(problem=question)

        if strategy in ("claim_product", "claim_minimum"):
            return self._CLAIM_LEVEL_MATH_PROMPT.format(problem=question)

        # baseline / ppo_value / fallback
        return "Question: " + question + "\n\nAnswer:"

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = example.pop(self.prompt_key, None)
        strategy = self.config.get("strategy", "baseline")

        if strategy == "explicit_risk":
            # 显式风险策略：无论 prompt 是否为 list，都强制注入 t prompt
            prompt_text = None
            if isinstance(prompt, list) and len(prompt) > 0:
                for msg in reversed(prompt):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content")
                        if isinstance(content, str):
                            prompt_text = content
                            break
            if prompt_text is None:
                prompt_text = "" if prompt is None else str(prompt)

            t = round(float(np.random.rand()), 3)
            t = max(0.0, min(t, 0.999))
            r = t / (1.0 - t)

            extra_info = example.get("extra_info") or {}
            extra_info["confidence"] = t
            example["extra_info"] = extra_info
            messages = [
                {
                    "role": "user",
                    "content": (
                        self._EXPLICIT_RISK_PROMPT.format(
                            percent=100 * t,
                            r=r,
                            problem=str(prompt_text).strip(),
                        )
                    ),
                }
            ]
            # print("[debug] prompt:", prompt_text)
            # print("[debug] messages:", messages)
            # raise RuntimeError("debug stop: printed prompt and messages")
        else:
            if isinstance(prompt, list) and len(prompt) > 0:
                messages = prompt
            else:
                if prompt is None:
                    prompt = ""
                messages = [{"role": "user", "content": self._format_prompt(str(prompt))}]
            # print("[debug] prompt:", prompt)
            # print("[debug] messages:", messages)
            # raise RuntimeError("debug stop: printed prompt and messages")

        # 处理多模态占位符（保持与 RLHFDataset 一致）
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages
