#!/usr/bin/env python3
"""合并并行分片的评估结果并重算指标"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from scripts.evaluate_model_fixed import compute_metrics
except Exception:
    # 兼容直接从仓库根运行
    from evaluate_model_fixed import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="合并并行分片评估结果")
    parser.add_argument("--input_root", type=str, required=True, help="包含 shard_* 子目录的根目录")
    parser.add_argument("--pattern", type=str, default="shard_*", help="分片目录模式")
    parser.add_argument("--output_file", type=str, default=None, help="输出结果文件（默认: <input_root>/merged_results.json）")
    parser.add_argument("--risk_threshold", type=float, default=None, help="可选：覆盖风险阈值")
    return parser.parse_args()


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"input_root 不存在: {input_root}")

    shard_dirs = sorted(input_root.glob(args.pattern))
    if not shard_dirs:
        raise FileNotFoundError(f"未找到分片目录: {input_root}/{args.pattern}")

    all_results = []
    risk_threshold = args.risk_threshold

    for shard_dir in shard_dirs:
        results_path = shard_dir / "results.json"
        if not results_path.exists():
            continue
        obj = _load_json(results_path)
        results = obj.get("results") or []
        all_results.extend(results)
        if risk_threshold is None:
            metrics = obj.get("metrics") or {}
            if "risk_threshold" in metrics:
                risk_threshold = metrics["risk_threshold"]

    if not all_results:
        raise RuntimeError("未找到任何可合并的 results.json")

    if risk_threshold is None:
        risk_threshold = 0.5

    metrics = compute_metrics(all_results, risk_threshold)
    predictions = [
        {
            "confidence": r.get("confidence"),
            "correct": r.get("correct", False),
            "abstained": r.get("abstained", False),
            "predicted_answer": r.get("predicted_answer"),
        }
        for r in all_results
    ]

    output_file = Path(args.output_file) if args.output_file else input_root / "merged_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "predictions": predictions,
            "results": all_results,
            "n_shards": len(shard_dirs),
        }, f, ensure_ascii=False, indent=2)

    print(f"合并完成: {output_file}")
    print(f"样本数: {metrics.get('n_samples', 0)}")


if __name__ == "__main__":
    main()
