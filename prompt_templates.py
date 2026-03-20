from __future__ import annotations

import math
import textwrap


RESPONSE_LEVEL_MATH_PROMPT = r"""Solve the following math problem step by step. The last second line of your response should be of the form
Answer: $Answer (without quotes), where $Answer is the answer to the problem. The last line of your
response should be of the form Confidence: $Confidence (without quotes), where $Confidence is a
number between 0 and 1.
{problem}
Please provide your best guess. Remember to put your answer on its own line after "Answer:" in the second
last line and put the confidence score on its own line after "Confidence:" in the last line."""

# RESPONSE_LEVEL_QA_PROMPT = r"""Answer the following question. The last second line of your response should be of the form Answer: $Answer
# (without quotes), where $Answer is the answer to the question. The last line of your response should
# be of the form Confidence: $Confidence (without quotes), where $Confidence is a number between 0 and
# 1.
# {question}
# Please provide your best guess. Remember to put your answer on its own line after "Answer:" in the second
# last line and put the confidence score on its own line after "Confidence:" in the last line."""

RESPONSE_LEVEL_QA_PROMPT = r"""Answer the following question.

Your final output must end with exactly these two lines:

Answer: <your short answer>
Confidence: <a number between 0 and 1>

Rules:
- The second last line must start with `Answer:`
- The last line must start with `Confidence:`
- Do not output anything after the `Confidence:` line
- If unsure, still provide your best guess and a low confidence

Example:
Answer: Paris
Confidence: 0.62

Question:
{question}
"""

CLAIM_LEVEL_MATH_PROMPT = r"""Solve the following math problem step by step.
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

EXPLICIT_RISK_MATH_PROMPT = r"""Solve the following math problem step by step. Answer only if you are more than {percent:.0f} percent
confident in your solution, since mistakes are penalized {r:.2f} points, while correct answers
receive 1 point, and an answer of <IDK> receives 0 points. The last line of your response should be
of the form Answer: $Answer (without quotes), where $Answer is the answer to the problem. If you are
not confident enough, respond with Answer: <IDK>.
{problem}
Remember to put your answer or <IDK> on its own line after "Answer:"."""


def response_level_math_prompt(problem: str) -> str:
    return RESPONSE_LEVEL_MATH_PROMPT.format(problem=problem).strip()


def response_level_qa_prompt(question: str) -> str:
    return RESPONSE_LEVEL_QA_PROMPT.format(question=question).strip()


def claim_level_math_prompt(problem: str) -> str:
    return CLAIM_LEVEL_MATH_PROMPT.format(problem=problem).strip()


def explicit_risk_math_prompt(problem: str, risk_t: float) -> str:
    r = _risk_t_to_r(risk_t)
    return EXPLICIT_RISK_MATH_PROMPT.format(
        percent=100 * risk_t,
        r=r,
        problem=problem,
    ).strip()


def explicit_risk_qa_prompt(question: str, risk_t: float, r: float | None = None) -> str:
    percent = 100 * risk_t
    # prompt = textwrap.dedent(
    #     f"""\
    #     Answer the given question. Answer only if you are more than {percent:.0f} percent confident in your answer,
    #     since mistakes are penalized {r:.2f} points, while correct answers receive 1 point, and an answer of
    #     <answer><IDK></answer> receives -1 points.
    #     The reasoning process is enclosed within <think> and </think>. After your thinking, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. If you are not confident enough, respond with <answer> <IDK> </answer>.
    #     Question: {question}\n"""
    # )
    # prompt = textwrap.dedent(
    #     f"""\
    #     Answer the given question. Answer only if you are more than {percent:.0f} percent confident in your answer,
    #     since mistakes are penalized {r:.2f} points, while correct answers receive 1 point, and an answer of
    #     <answer> <IDK> </answer> receives 0 points.
    #     The reasoning process is enclosed within <think> and </think>. After your thinking, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. If you are not confident enough, respond with <answer> <IDK> </answer>.
    #     Question: {question}\n"""
    # )
    # prompt = textwrap.dedent(
    #     f"""\
    #     Answer the given question.
    #     The reasoning process is enclosed within <think> and </think>. After your thinking, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
    #     Question: {question}\n"""
    # )
    prompt = textwrap.dedent(
        f"""\
        Answer the given question. Answer only if you are more than {percent:.0f} percent confident in your answer,
        since mistakes are penalized {r:.2f} points, while correct answers receive 1 point, and an answer of
        <answer> <IDK> </answer> receives 0 points.
        The reasoning process is enclosed within <think> and </think>. During your thinking, you must estimate your confidence level and provide it within <confidence> and </confidence> tags. Use your confidence to decide whether to answer or ABSTAIN. After your thinking, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. If you decide to ABSTAIN, respond with <answer> <IDK> </answer>.
        Question: {question}\n"""
    )
    prompt = prompt.strip()
    return " ".join(line.strip() for line in prompt.splitlines() if line.strip())
