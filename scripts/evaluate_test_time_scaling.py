#!/usr/bin/env python3
"""
Test-Time Scaling评估 (Section 4.5)
为每个问题生成k个响应，使用置信度进行选择
"""
import os
import sys

import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
from collections import Counter

# 复用评估工具
SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import evaluate_model_fixed as eval_utils

# 论文判题逻辑（Minerva）
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
try:
    from behavioral_calibration import is_correct_minerva
except Exception:
    is_correct_minerva = None

def parse_args():
    parser = argparse.ArgumentParser(description="Test-Time Scaling评估")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="基础模型")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["aime", "aime_2024", "aime_2025", "beyondaime"],
        help="评估数据集",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--k_values", type=str, default="1,2,4,8,16,32", help="k值列表，逗号分隔")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "baseline", "verbalized_brier", "verbalized_ce", "ppo_value", "claim_product", "claim_minimum"],
        help="评估策略（自动或显式指定）",
    )
    parser.add_argument("--critic_path", type=str, default=None, help="PPO-Value的critic模型路径（可选）")
    parser.add_argument("--offline", action="store_true", help="强制离线加载HF模型")
    return parser.parse_args()

def load_model_and_tokenizer(base_model, model_path, device, offline=False):
    return eval_utils.load_model_and_tokenizer(base_model, model_path, device, offline=offline)

def load_evaluation_dataset(dataset_name):
    """加载评估数据集"""
    data_dir = Path("/root/SeedMitigating/data")

    if dataset_name in ("aime", "aime_2024"):
        file_path = data_dir / "aime_2024.jsonl"
        if dataset_name == "aime" and not file_path.exists():
            file_path = data_dir / "aime_2025.jsonl"
    elif dataset_name == "aime_2025":
        file_path = data_dir / "aime_2025.jsonl"
    elif dataset_name == "beyondaime":
        file_path = data_dir / "beyondaime.jsonl"
    else:
        raise ValueError(f"未知数据集: {dataset_name}")

    if not file_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"加载了 {len(data)} 条数据从 {file_path}")
    return data


def _get_item_field(item, keys, default=""):
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default

def extract_answer_and_confidence(response, strategy="auto"):
    """从模型响应中提取答案和置信度（复用主评估脚本逻辑）"""
    return eval_utils.extract_answer_and_confidence(response, strategy=strategy)

def check_answer_correctness(predicted, ground_truth):
    """检查答案是否正确"""
    return eval_utils.check_answer_correctness(predicted, ground_truth)

def _is_correct_math_answer(predicted, ground_truth):
    if predicted is None or predicted == "":
        return False
    if is_correct_minerva is None:
        return check_answer_correctness(predicted, ground_truth)
    try:
        correct, abstention, _ = is_correct_minerva(f"Answer: {predicted}", ground_truth)
        return bool(correct) and not bool(abstention)
    except Exception:
        return check_answer_correctness(predicted, ground_truth)

def check_response_correctness(response_text, predicted_answer, ground_truth, dataset_name):
    """优先使用Minerva判题逻辑（AIME/BeyondAIME）"""
    if dataset_name in ("aime", "aime_2024", "aime_2025", "beyondaime") and is_correct_minerva is not None:
        try:
            correct, abstention_flag, pred_norm = is_correct_minerva(response_text, ground_truth)
            if pred_norm:
                predicted_answer = pred_norm
            return bool(correct) and not bool(abstention_flag), predicted_answer
        except Exception:
            pass
    return check_answer_correctness(predicted_answer, ground_truth), predicted_answer

def generate_k_responses(model, tokenizer, question, k, device, strategy, dataset_name, critic_model=None):
    """为一个问题生成k个响应"""
    prompt = eval_utils.build_prompt(question, strategy, None, dataset_name)
    
    responses = []
    for i in range(k):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20480,
                temperature=1.0,
                top_p=0.7,
                top_k=-1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        answer, confidence = extract_answer_and_confidence(response, strategy=strategy)

        if "<IDK>" in response:
            confidence = 0.0

        if strategy == "ppo_value" and critic_model is not None:
            try:
                critic_inputs = tokenizer(prompt + response, return_tensors="pt").to(device)
                with torch.no_grad():
                    critic_out = critic_model(**critic_inputs)
                value = critic_out.logits[0, -1, 0].item()
                confidence = float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))
            except Exception:
                pass
        
        responses.append({
            'answer': answer,
            'confidence': confidence,
            'full_response': response
        })
    
    return responses

def max_confidence_selection(responses):
    """Max Confidence策略：选择置信度最高的响应"""
    if not responses:
        return "", 0.0
    
    max_resp = max(responses, key=lambda x: x['confidence'])
    return max_resp.get('final_answer', max_resp.get('answer')), max_resp['confidence']

def majority_voting(responses):
    """Majority Voting策略（不加权）"""
    if not responses:
        return "", 0.0

    answer_counts = {}
    for resp in responses:
        answer = resp.get('final_answer', resp.get('answer'))
        if answer:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1

    if not answer_counts:
        return "", 0.0

    best_answer, best_count = max(answer_counts.items(), key=lambda x: x[1])
    return best_answer, best_count / len(responses)

def confidence_weighted_majority_voting(responses):
    """Confidence Weighted Majority Voting策略"""
    if not responses:
        return "", 0.0
    
    # 统计每个答案的加权投票
    answer_weights = {}
    for resp in responses:
        answer = resp.get('final_answer', resp.get('answer'))
        confidence = resp['confidence']
        if answer:
            answer_weights[answer] = answer_weights.get(answer, 0) + confidence
    
    if not answer_weights:
        return "", 0.0
    
    # 选择权重最高的答案
    best_answer = max(answer_weights.items(), key=lambda x: x[1])
    return best_answer[0], best_answer[1] / len(responses)

def evaluate_test_time_scaling(model, tokenizer, dataset, device, k_values, strategy, dataset_name, max_samples=None, critic_model=None):
    """评估Test-Time Scaling"""
    if max_samples:
        dataset = dataset[:max_samples]
    
    results = {k: {'max_conf': [], 'weighted_voting': [], 'mean': [], 'best': [], 'majority': []} for k in k_values}
    
    for item in tqdm(dataset, desc="评估Test-Time Scaling"):
        question = _get_item_field(item, ("question", "problem", "Problem", "prompt"))
        ground_truth = _get_item_field(item, ("answer", "Answer", "solution", "Solution"))
        
        # 生成最大k值的响应
        max_k = max(k_values)
        all_responses = generate_k_responses(model, tokenizer, question, max_k, device, strategy, dataset_name, critic_model)
        
        # 预先计算每个响应的正确性与归一化答案
        for resp in all_responses:
            is_correct, final_answer = check_response_correctness(
                resp['full_response'],
                resp.get('answer'),
                ground_truth,
                dataset_name,
            )
            resp['is_correct'] = is_correct
            resp['final_answer'] = final_answer

        # 对每个k值进行评估
        for k in k_values:
            k_responses = all_responses[:k]
            
            # Max Confidence策略
            max_conf_answer, max_conf_score = max_confidence_selection(k_responses)
            max_conf_correct = False
            for resp in k_responses:
                if resp.get('final_answer') == max_conf_answer:
                    max_conf_correct = resp.get('is_correct', False)
                    break
            
            # Weighted Voting策略
            weighted_answer, weighted_score = confidence_weighted_majority_voting(k_responses)
            if dataset_name in ("aime", "aime_2024", "aime_2025", "beyondaime"):
                is_correct_weighted = _is_correct_math_answer(weighted_answer, ground_truth)
            else:
                is_correct_weighted = check_answer_correctness(weighted_answer, ground_truth)

            # Mean@k
            mean_acc = float(np.mean([1.0 if r.get('is_correct') else 0.0 for r in k_responses]))

            # Best@k
            best_acc = 1.0 if any(r.get('is_correct') for r in k_responses) else 0.0

            # Majority@k
            majority_answer, majority_score = majority_voting(k_responses)
            if dataset_name in ("aime", "aime_2024", "aime_2025", "beyondaime"):
                is_correct_majority = _is_correct_math_answer(majority_answer, ground_truth)
            else:
                is_correct_majority = check_answer_correctness(majority_answer, ground_truth)
            
            results[k]['max_conf'].append({
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': max_conf_answer,
                'confidence': max_conf_score,
                'is_correct': max_conf_correct
            })
            
            results[k]['weighted_voting'].append({
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': weighted_answer,
                'confidence': weighted_score,
                'is_correct': is_correct_weighted
            })

            results[k]['mean'].append({
                'question': question,
                'ground_truth': ground_truth,
                'accuracy': mean_acc,
            })

            results[k]['best'].append({
                'question': question,
                'ground_truth': ground_truth,
                'accuracy': best_acc,
            })

            results[k]['majority'].append({
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': majority_answer,
                'confidence': majority_score,
                'is_correct': is_correct_majority,
            })
    
    return results

def compute_metrics(results_list):
    """计算评估指标"""
    n = len(results_list)
    if n == 0:
        return {'accuracy': 0.0, 'avg_confidence': 0.0}
    
    accuracy = sum(r.get('is_correct', False) for r in results_list) / n
    avg_confidence = np.mean([r.get('confidence', 0.0) for r in results_list])
    
    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'n_samples': n
    }

def compute_scalar_accuracy(results_list, key='accuracy'):
    n = len(results_list)
    if n == 0:
        return {'accuracy': 0.0, 'n_samples': 0}
    acc = float(np.mean([r.get(key, 0.0) for r in results_list]))
    return {'accuracy': acc, 'n_samples': n}

def main():
    args = parse_args()

    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    k_values = [int(k) for k in args.k_values.split(',')]
    
    print("=" * 80)
    print(f"Test-Time Scaling评估")
    print(f"模型: {args.model_path}")
    print(f"数据集: {args.dataset}")
    print(f"k值: {k_values}")
    print("=" * 80)
    
    strategy = eval_utils.infer_strategy(args.model_path, args.strategy)

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.model_path, args.device, offline=args.offline)
    critic_model = None
    if strategy == "ppo_value":
        critic_model = eval_utils.load_critic_model(args.critic_path, args.device, args.offline)
    
    # 加载数据集
    dataset = load_evaluation_dataset(args.dataset)
    
    # 评估
    results = evaluate_test_time_scaling(
        model,
        tokenizer,
        dataset,
        args.device,
        k_values,
        strategy,
        args.dataset,
        args.max_samples,
        critic_model,
    )
    
    # 计算并打印结果
    print("\n" + "=" * 80)
    print("Test-Time Scaling结果:")
    print("=" * 80)
    
    summary = {}
    for k in k_values:
        print(f"\nk = {k}:")
        
        max_conf_metrics = compute_metrics(results[k]['max_conf'])
        weighted_metrics = compute_metrics(results[k]['weighted_voting'])
        mean_metrics = compute_scalar_accuracy(results[k]['mean'])
        best_metrics = compute_scalar_accuracy(results[k]['best'])
        majority_metrics = compute_metrics(results[k]['majority'])
        
        print(f"  Max Confidence:")
        print(f"    准确率: {max_conf_metrics['accuracy']:.4f}")
        print(f"    平均置信度: {max_conf_metrics['avg_confidence']:.4f}")
        
        print(f"  Weighted Voting:")
        print(f"    准确率: {weighted_metrics['accuracy']:.4f}")
        print(f"    平均置信度: {weighted_metrics['avg_confidence']:.4f}")

        print(f"  Mean@k:")
        print(f"    准确率: {mean_metrics['accuracy']:.4f}")

        print(f"  Best@k:")
        print(f"    准确率: {best_metrics['accuracy']:.4f}")

        print(f"  Majority@k:")
        print(f"    准确率: {majority_metrics['accuracy']:.4f}")
        print(f"    平均置信度: {majority_metrics['avg_confidence']:.4f}")
        
        summary[f'k{k}'] = {
            'max_confidence': max_conf_metrics,
            'weighted_voting': weighted_metrics,
            'mean': mean_metrics,
            'best': best_metrics,
            'majority': majority_metrics,
        }
    
    print("=" * 80)
    
    # 保存结果
    model_name = Path(args.model_path).name
    output_file = Path(args.output_dir) / f"{args.dataset}_test_time_scaling.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model': model_name,
            'dataset': args.dataset,
            'k_values': k_values,
            'summary': summary,
            'detailed_results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
