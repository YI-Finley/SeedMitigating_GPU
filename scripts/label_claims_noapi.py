#!/usr/bin/env python3
"""
Label claim-level correctness using OpenAI-compatible API (noapi).
Produces JSONL with fields: {"index": int, "claim_labels": [0/1,...]}.
Inputs are expected to include full_response and final_answer_correct when available.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser(description="Label claim correctness via noapi")
    parser.add_argument("--input_file", type=str, default=None, help="claims_for_labeling.jsonl")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing claims_for_labeling.jsonl")
    parser.add_argument("--model_dir", type=str, default=None, help="Model dir name (used with input_dir)")
    parser.add_argument("--output_file", type=str, required=True, help="Output labels jsonl")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="API model name")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries per request")
    parser.add_argument("--retry_delay", type=int, default=5, help="Retry delay")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    parser.add_argument("--resume", action="store_true", help="Resume if output exists")
    parser.add_argument("--api_base", type=str, default=None, help="Override OPENAI_API_BASE")
    parser.add_argument("--api_key", type=str, default=None, help="Override OPENAI_API_KEY")
    return parser.parse_args()


def _resolve_input_file(input_file: str | None, input_dir: str | None, model_dir: str | None) -> Path:
    if input_file:
        return Path(input_file)
    if input_dir and model_dir:
        base = Path(input_dir) / model_dir
        cand = base / "claims_for_labeling.jsonl"
        if cand.exists():
            return cand
        fallback = base / "results.json"
        return fallback
    raise ValueError("input_file or (input_dir + model_dir) required")


def extract_claim_confidences(response: str) -> list[float]:
    confs = []
    for match in re.findall(r'(?i)<\\s*Confidence\\b[^>]*\\bvalue\\s*=\\s*([0-9]+(?:\\.[0-9]+)?)', response):
        try:
            v = float(match)
            if v > 1:
                v = v / 100.0
            confs.append(float(max(0.0, min(1.0, v))))
        except Exception:
            continue
    return confs


def call_chat_completion(api_base: str, api_key: str, model: str, messages, temperature: float,
                         max_tokens: int, timeout: int, max_retries: int, retry_delay: int) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
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


def count_claims(response: str) -> int:
    return len(re.findall(r"(?i)<\s*Confidence\b[^>]*\bvalue\s*=", response or ""))


def parse_label_list(text: str, expected_len: int) -> list[int] | None:
    # Prefer JSON object per paper prompt, fallback to legacy JSON array.
    obj = None
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = None
    labels: list[int] = []
    if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
        for step in obj["steps"]:
            acc = 0
            if isinstance(step, dict):
                val = step.get("acc", 0)
                if isinstance(val, bool):
                    acc = 1 if val else 0
                elif isinstance(val, (int, float)):
                    acc = 1 if val > 0 else 0
            labels.append(acc)
    else:
        match = re.search(r"\[[\s\S]*?\]", text)
        if not match:
            return None
        try:
            arr = json.loads(match.group(0))
        except Exception:
            return None
        if not isinstance(arr, list):
            return None
        for x in arr:
            if isinstance(x, bool):
                labels.append(1 if x else 0)
            elif isinstance(x, (int, float)):
                labels.append(1 if x > 0 else 0)
            else:
                labels.append(0)
    if len(labels) < expected_len:
        labels.extend([0] * (expected_len - len(labels)))
    if len(labels) > expected_len:
        labels = labels[:expected_len]
    return labels


def build_prompt(question: str, solution: str, final_label: str, claim_count: int) -> str:
    template = r"""Your job is to look at a problem, a solution including the predicted answer, and a label for the
correctness of the predicted answer, and then check the correctness of the solution per step. Your
input is generated by a solver that follows the instructions:
Solver Instructions:
Solve the following math problem step by step.
1. Output the solution step by step. For each step, assess your confidence in it and, if your confidence
is less than 1, identify the main source of uncertainty (e.g., limited knowledge, ambiguous memory,
guessing, etc.).
2. In your final solution, express with a sequence of individual steps, where each step is wrapped in a
single tag:
<Confidence value=$Confidence text="$Analysis">$Step</Confidence>
$Confidence is a number between 0 and 1 representing how confident you are that the step is correct. $
Analysis is a concise explanation of your confidence. The $Step text itself is inside the tag.
3. Finally, output one last line exactly of the form Answer:$Answer, where $Answer (without quotes) is the
answer to the problem.
### Instructions
- You should check the correctness of the solution step by step.
- The correctness label is for the final answer. If the predicted answer is correct, the solution may also
contain incorrect steps. If the predicted answer is incorrect, the solution may also contain correct
steps.
- If the reasoning of the step is correct, then the step is correct, even when the conclusion of the step
is wrong because of dependence on a false previous step.
- Each step in the solution is wrapped in a tag <Confidence value=$Confidence text="$Analysis">$Step</Confidence>.
- You should also extract the $Confidence value for each step according to the confidence tag. The $
Confidence value is supposed to be in [0,1], but it can accidentally fall out of this range. If it
does, you should clip it to the nearest endpoint.
- Ignore all texts that are not wrapped in the confidence tag, including the final answer.
- It is possible that no steps are included in the solution.
- For each step, provide a concise explanation for your judgement.
- Your response should be in the JSON format specified below.
### Formatting
Your response should be in the following JSON format (no comments):
```json
{
"steps": [
{
"acc": $Correctness,
"confidence": $Confidence,
"explanation": "$Explanation"
},
...
/* one object per step */
]
}
```
- $Correctness is 1 if the step is correct otherwise 0.
- $Confidence is a number between 0 and 1 representing the confidence of the step. It's not your
confidence in the judgement.
- $Explanation is a concise explanation for the judgement of the step.
### Task
Question: {problem}
Solution: {solution}
Correctness Label: {correctness}""".format(
    )
    return (
        template
        .replace("{problem}", question)
        .replace("{solution}", solution)
        .replace("{correctness}", final_label)
    )


def main():
    args = parse_args()

    api_base = args.api_base or os.environ.get("OPENAI_API_BASE", "https://noapi.ggb.today/v1")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")

    input_path = _resolve_input_file(args.input_file, args.input_dir, args.model_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labeled = {}
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                idx = obj.get("index")
                if idx is not None:
                    labeled[int(idx)] = obj.get("claim_labels")
        print(f"Resuming, loaded {len(labeled)} labeled items")

    count = 0
    def iter_items(path: Path):
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f_json:
                data = json.load(f_json)
            for idx, item in enumerate(data.get("results", [])):
                response = item.get("full_response", "")
                confs = extract_claim_confidences(response)
                if not confs:
                    continue
                yield {
                    "index": idx,
                    "question": item.get("question", ""),
                    "final_answer_correct": item.get("correct", item.get("is_correct")),
                    "full_response": response,
                    "claim_confidences": confs,
                }
        else:
            with open(path, "r", encoding="utf-8") as f_jsonl:
                for line in f_jsonl:
                    obj = json.loads(line)
                    yield obj

    with open(output_path, "a", encoding="utf-8") as f_out:
        for obj in iter_items(input_path):
            idx = obj.get("index")
            if idx is None:
                continue
            if idx in labeled:
                continue
            question = obj.get("question", "")
            response = obj.get("full_response", "")
            claim_confidences = obj.get("claim_confidences") or []
            expected_len = len(claim_confidences)
            if expected_len == 0:
                expected_len = count_claims(response)
            if expected_len == 0:
                continue

            final_label = "UNKNOWN"
            if "final_answer_correct" in obj:
                final_label = "CORRECT" if obj.get("final_answer_correct") else "INCORRECT"

            prompt = build_prompt(question, response, final_label, expected_len)
            messages = [{"role": "user", "content": prompt}]

            resp = call_chat_completion(
                api_base=api_base,
                api_key=api_key,
                model=args.model,
                messages=messages,
                temperature=0.0,
                max_tokens=256,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
            )

            labels = parse_label_list(resp, expected_len)
            if labels is None:
                labels = [0] * expected_len

            out_obj = {"index": int(idx), "claim_labels": labels}
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            f_out.flush()

            count += 1
            if args.max_samples and count >= args.max_samples:
                break
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Saved labels to: {output_path}")


if __name__ == "__main__":
    main()
