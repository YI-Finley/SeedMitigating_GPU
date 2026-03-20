#!/usr/bin/env python3
"""
实现Behavioral Calibration评估
包括：
1. Abstention机制
2. Smoothed ECE (smECE)
3. 正确的置信度提取
4. Abstention Accuracy和Predictive Accuracy
"""
import os
import sys

import json
import argparse
import torch
import numpy as np
import inspect
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel
import re
import string
import glob
from prompt_templates import (
    claim_level_math_prompt,
    explicit_risk_math_prompt,
    explicit_risk_qa_prompt,
    response_level_math_prompt,
    response_level_qa_prompt,
)
import search_r1_like_qa_em as _search_r1_qa

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

# 允许导入论文作者提供的校验逻辑
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
GT_GRPO_ROOT = os.environ.get("GT_GRPO_ROOT", os.path.join(ROOT_DIR, "external", "GT_GRPO"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _explicit_risk_debug_enabled() -> bool:
    flag = os.getenv("EXPLICIT_RISK_DEBUG", "").strip().lower()
    return flag in ("1", "true", "yes", "y", "on")


def _er_debug(msg: str) -> None:
    if _explicit_risk_debug_enabled():
        print(f"[explicit_risk][evaluate_model_fixed] {msg}")

def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    val = value.strip().lower()
    if val in ("1", "true", "yes", "y", "on"):
        return True
    if val in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"无效布尔值: {value}")


def _normalize_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)) and torch.npu.is_available():
        return "npu:0"
    return "cpu"


def _move_model_to_device(model, device: str):
    device = _normalize_device(device)
    if device.startswith("npu"):
        if not hasattr(torch, "npu"):
            raise RuntimeError("当前torch环境不支持NPU")
        npu_id = int(device.split(":")[1]) if ":" in device else 0
        torch.npu.set_device(npu_id)
        return model.to(f"npu:{npu_id}")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("当前torch环境不支持CUDA")
        cuda_id = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(cuda_id)
        return model.to(f"cuda:{cuda_id}")
    return model.to(device)

try:
    from behavioral_calibration import is_correct_minerva
except Exception:
    is_correct_minerva = None

# 论文指标实现（KDE smECE / SNR Gain / NLL 等）
EVAL_DIR = os.path.join(ROOT_DIR, "eval")
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)
try:
    from confidence_calibration_metrics import ConfidenceCalibrationMetrics
except Exception:
    ConfidenceCalibrationMetrics = None

FLASHRAG_DATASETS = {
    "flashrag_popqa": "popqa_test_500.parquet",
    "flashrag_nq": "nq_test_500.parquet",
    "flashrag_hotpotqa": "hotpotqa_test_500.parquet",
    "flashrag_musique": "musique_test_500.parquet",
    "flashrag_2wikimultihopqa": "2wikimultihopqa_test_500.parquet",
    "flashrag_triviaqa": "triviaqa_test_500.parquet",
    "flashrag_bamboogle": "bamboogle_test.parquet",
}

FLASHRAG_DATASETS_TRUE = {
    "flashrag_popqa_true": "popqa_test.parquet",
    "flashrag_nq_true": "nq_test.parquet",
    "flashrag_hotpotqa_true": "hotpotqa_test.parquet",
    "flashrag_musique_true": "musique_test.parquet",
    "flashrag_2wikimultihopqa_true": "2wikimultihopqa_test.parquet",
    "flashrag_triviaqa_true": "triviaqa_test.parquet",
    "flashrag_bamboogle_true": "bamboogle_test.parquet",
}

QA_DATASETS = {
    "simpleqa",
    "hotpotqa",
    "hotpot_qa",
    "simpleqa_gtgrpo",
    "nq_hotpotqa_searchr1",
    "flashrag_all",
    *FLASHRAG_DATASETS.keys(),
    *FLASHRAG_DATASETS_TRUE.keys(),
}

def parse_args():
    parser = argparse.ArgumentParser(description="评估训练好的模型（修复版）")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="基础模型")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "aime",
            "aime_2024",
            "aime_2025",
            "beyondaime",
            "simpleqa",
            "hotpotqa",
            "hotpot_qa",
            "simpleqa_gtgrpo",
            "nq_hotpotqa_searchr1",
            "flashrag_all",
            *FLASHRAG_DATASETS.keys(),
            *FLASHRAG_DATASETS_TRUE.keys(),
        ],
        help="评估数据集",
    )
    parser.add_argument("--dataset_file", type=str, default=None, help="可选：本地数据文件(.jsonl/.json)")
    parser.add_argument("--hf_dataset", type=str, default=None, help="可选：HuggingFace datasets 名称")
    parser.add_argument("--hf_config", type=str, default=None, help="可选：HF 数据集配置名")
    parser.add_argument("--hf_split", type=str, default="validation", help="HF 数据集切分")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--risk_threshold", type=float, default=0.5, help="风险阈值t，用于abstention")
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=[
            "auto",
            "baseline",
            "verbalized_brier",
            "verbalized_ce",
            "ppo_value",
            "claim_product",
            "claim_minimum",
            "explicit_risk",
        ],
        help="评估策略（自动或显式指定）",
    )
    parser.add_argument("--critic_path", type=str, default=None, help="PPO-Value的critic模型路径（可选）")
    parser.add_argument("--offline", action="store_true", help="强制离线加载HF模型")
    parser.add_argument(
        "--risk_prompt_metrics",
        action="store_true",
        help="为SNR/AbsAcc/PredAcc等t相关指标额外使用显式风险prompt(Answer only if...)",
    )
    parser.add_argument(
        "--risk_prompt_t0",
        type=float,
        default=0.0,
        help="显式风险评估的t基线（用于PredAcc/SNR基准，默认0.0）",
    )
    parser.add_argument(
        "--risk_prompt_grid",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="显式风险评估遍历的t网格（逗号分隔）",
    )
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
    parser.add_argument("--simpleqa_grader_local", type=str, default=None, help="可选：本地LLM判分模型路径/ID")
    parser.add_argument("--simpleqa_grader_device", type=str, default="cuda:0", help="本地判分模型设备")
    parser.add_argument("--simpleqa_grader_use_chat_template", action="store_true", help="本地判分使用聊天模板")
    parser.add_argument("--simpleqa_grader_local_files_only", action="store_true", help="本地判分仅离线加载")
    parser.add_argument("--simpleqa_grader_dtype", type=str, default="float16", help="本地判分模型dtype")
    parser.add_argument("--use_chat_template", action="store_true", help="使用tokenizer聊天模板包装prompt")
    parser.add_argument("--thinking_mode", type=str, default="auto", choices=["auto", "on", "off"], help="思考模式: auto/on/off")
    parser.add_argument("--gen_temperature", type=float, default=None, help="生成温度(可选)")
    parser.add_argument("--gen_top_p", type=float, default=None, help="Top-p(可选)")
    parser.add_argument("--gen_top_k", type=int, default=None, help="Top-k(可选)")
    parser.add_argument("--gen_min_p", type=float, default=None, help="Min-p(可选)")
    parser.add_argument("--gen_do_sample", type=str, default=None, help="是否采样(true/false)，默认自动")
    parser.add_argument("--max_new_tokens", type=int, default=20480, help="最大生成长度")
    parser.add_argument("--save_io", action="store_true", help="保存每条样本的输入prompt与生成参数")
    parser.add_argument("--sample_offset", type=int, default=0, help="样本起始偏移（并行切分）")
    parser.add_argument("--sample_stride", type=int, default=1, help="样本步长（并行切分）")
    parser.add_argument("--use_vllm", action="store_true", help="使用vLLM进行推理加速")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9, help="vLLM显存占用比例")
    parser.add_argument("--vllm_max_model_len", type=int, default=None, help="vLLM最大序列长度(可选)")
    parser.add_argument("--vllm_trust_remote_code", action="store_true", help="vLLM信任远程代码")
    parser.add_argument("--vllm_seed", type=int, default=42, help="vLLM随机种子")
    parser.add_argument("--vllm_block_size", type=int, default=None, help="vLLM KV cache block size (1/8/16/32/64/128)")
    parser.add_argument("--vllm_enable_prefix_caching", type=_parse_optional_bool, default=None, help="vLLM prefix caching (true/false)")
    parser.add_argument("--vllm_max_num_batched_tokens", type=int, default=None, help="vLLM max_num_batched_tokens")
    parser.add_argument("--vllm_max_num_seqs", type=int, default=None, help="vLLM max_num_seqs")
    return parser.parse_args()


def _load_json_or_jsonl(path: Path):
    if path.suffix.lower() == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return data if isinstance(data, list) else [data]
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except Exception as e:
            raise ImportError("需要安装 pandas 才能读取 parquet") from e
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    raise ValueError(f"不支持的数据格式: {path}")


def _resolve_model_dir(path: str) -> str:
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
        return path
    if os.path.isdir(path):
        # 优先查找最新的 global_step_* 目录（训练输出常见结构）
        step_dirs = sorted(glob.glob(os.path.join(path, "global_step_*")))
        for step in reversed(step_dirs):
            for sub in ("actor/huggingface", "actor", "policy", "model", "hf_model"):
                cand = os.path.join(step, sub)
                if os.path.exists(os.path.join(cand, "config.json")):
                    return cand
        for sub in ("actor", "policy", "model", "hf_model"):
            cand = os.path.join(path, sub)
            if os.path.exists(os.path.join(cand, "config.json")):
                return cand
    return path


def load_model_and_tokenizer(base_model, model_path, device, offline=False):
    """加载模型和tokenizer（支持全量模型或LoRA）"""
    print(f"加载基础模型: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=offline)

    model_path = _resolve_model_dir(model_path)
    # 仅在显式本地路径时检查存在性；repo id 允许交给 HF 缓存/离线加载处理
    is_local_path = os.path.isabs(model_path) or model_path.startswith(".") or model_path.startswith("~")
    if is_local_path and not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    def _load_causal(path, local_only):
        return AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
            local_files_only=local_only,
        )

    model = None
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        print(f"检测到LoRA适配器: {model_path}")
        base = _load_causal(base_model, offline)
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        print(f"加载全量模型: {model_path}")
        try:
            model = _load_causal(model_path, offline)
        except Exception:
            if offline:
                raise
            # 允许在线下载
            model = _load_causal(model_path, False)

    model = _move_model_to_device(model, device)
    model.eval()

    print(f"模型已加载到 {device}")
    return model, tokenizer


def load_vllm_engine(
    model_path,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    trust_remote_code=False,
    seed=42,
    max_model_len=None,
    block_size=None,
    enable_prefix_caching=None,
    max_num_batched_tokens=None,
    max_num_seqs=None,
):
    """使用vLLM加载模型（仅推理）"""
    try:
        from vllm import LLM
    except Exception as e:
        raise ImportError("需要安装vLLM才能使用--use_vllm") from e
    model_path = _resolve_model_dir(model_path)
    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
        "seed": seed,
    }
    enforce_eager_env = os.getenv("VLLM_ENFORCE_EAGER")
    if enforce_eager_env is not None:
        llm_kwargs["enforce_eager"] = str(enforce_eager_env).lower() in ("1", "true", "yes")
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = int(max_model_len)
    if block_size is not None:
        block_size = int(block_size)
        if block_size not in (1, 8, 16, 32, 64, 128):
            raise ValueError(f"vllm_block_size 无效: {block_size}")
        llm_kwargs["block_size"] = block_size
    if enable_prefix_caching is not None:
        llm_kwargs["enable_prefix_caching"] = bool(enable_prefix_caching)
    if max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
    if max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = int(max_num_seqs)
    llm = LLM(**llm_kwargs)
    tokenizer = None
    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        pass
    return llm, tokenizer

def load_evaluation_dataset(
    dataset_name,
    dataset_file: str | None = None,
    hf_dataset: str | None = None,
    hf_config: str | None = None,
    hf_split: str = "validation",
):
    """加载评估数据集（支持本地文件或HF datasets）"""
    data_dir = Path(ROOT_DIR) / "data"

    file_path = None
    if dataset_file:
        file_path = Path(dataset_file)
    else:
        if dataset_name in ("aime", "aime_2024"):
            file_path = data_dir / "aime_2024.jsonl"
            if dataset_name == "aime" and not file_path.exists():
                print(f"警告: {file_path} 不存在，尝试加载AIME 2025")
                file_path = data_dir / "aime_2025.jsonl"
        elif dataset_name == "aime_2025":
            file_path = data_dir / "aime_2025.jsonl"
        elif dataset_name == "beyondaime":
            file_path = data_dir / "beyondaime.jsonl"
        elif dataset_name == "simpleqa":
            file_path = data_dir / "simpleqa.jsonl"
        elif dataset_name == "simpleqa_gtgrpo":
            file_path = Path(GT_GRPO_ROOT) / "data/simpleqa/simpleqa.parquet"
        elif dataset_name in ("hotpotqa", "hotpot_qa"):
            for cand in ("hotpotqa.jsonl", "hotpotqa_dev.jsonl", "hotpot_qa.jsonl"):
                cand_path = data_dir / cand
                if cand_path.exists():
                    file_path = cand_path
                    break
        elif dataset_name == "nq_hotpotqa_searchr1":
            file_path = Path(GT_GRPO_ROOT) / "data/nq_hotpot_searchr1/test.parquet"
        elif dataset_name in FLASHRAG_DATASETS:
            file_path = Path(GT_GRPO_ROOT) / "data/flash_rag" / FLASHRAG_DATASETS[dataset_name]
        elif dataset_name in FLASHRAG_DATASETS_TRUE:
            file_path = Path(GT_GRPO_ROOT) / "data/flash_rag" / FLASHRAG_DATASETS_TRUE[dataset_name]    
        elif dataset_name == "flashrag_all":
            flash_rag_dir = Path(GT_GRPO_ROOT) / "data/flash_rag"
            all_data = []
            for sub_name, filename in FLASHRAG_DATASETS.items():
                sub_path = flash_rag_dir / filename
                if not sub_path.exists():
                    print(f"警告: {sub_path} 不存在，跳过")
                    continue
                sub_data = _load_json_or_jsonl(sub_path)
                for item in sub_data:
                    item["_source_dataset"] = sub_name
                all_data.extend(sub_data)
                print(f"  加载 {sub_name}: {len(sub_data)} 条")
            print(f"flashrag_all 共加载 {len(all_data)} 条数据")
            return all_data
    data = None
    if file_path:
        if not file_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")
        data = _load_json_or_jsonl(file_path)
        print(f"加载了 {len(data)} 条数据从 {file_path}")
    else:
        hf_name = hf_dataset
        if not hf_name and dataset_name in ("hotpotqa", "hotpot_qa"):
            hf_name = "hotpot_qa"
            hf_config = hf_config or "fullwiki"
            hf_split = hf_split or "validation"
        if not hf_name:
            raise ValueError(f"未找到数据集文件，且未指定hf_dataset: {dataset_name}")
        try:
            from datasets import load_dataset
        except Exception as e:
            raise ImportError("需要安装 datasets 才能加载HF数据集") from e
        ds = load_dataset(hf_name, hf_config, split=hf_split)
        data = list(ds)
        print(f"从HF加载了 {len(data)} 条数据: {hf_name} [{hf_config or '-'}]/{hf_split}")

    return data


def _get_item_field(item, keys, default=""):
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default

def _extract_prompt_content(prompt):
    if hasattr(prompt, "tolist") and not isinstance(prompt, (str, bytes, dict, list)):
        try:
            prompt = prompt.tolist()
        except Exception:
            pass
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        content = prompt.get("content")
        if content:
            return content
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict):
                content = msg.get("content")
                if content:
                    return content
            elif isinstance(msg, str) and msg.strip():
                return msg
    return ""

def _strip_prompt_to_question(text: str) -> str:
    if not isinstance(text, str):
        return text
    cleaned = text.strip()
    for marker in ("Question:", "Question："):
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1].strip()
            if "\n" in cleaned:
                cleaned = cleaned.split("\n")[0].strip()
            break
    return cleaned

def _extract_question(item, dataset_name: str | None = None):
    question = _get_item_field(item, ("question", "problem", "Problem"))
    if question is not None and not (isinstance(question, str) and question == ""):
        return question.strip() if isinstance(question, str) else question
    prompt_text = _extract_prompt_content(item.get("prompt"))
    if not prompt_text:
        return ""
    if dataset_name in QA_DATASETS:
        return _strip_prompt_to_question(prompt_text)
    return prompt_text

def _extract_ground_truth(item):
    ground_truth = _get_item_field(item, ("answer", "Answer", "solution", "Solution", "golden_answers", "ground_truth"))
    if ground_truth is not None and not (isinstance(ground_truth, str) and ground_truth == ""):
        return ground_truth
    reward_model = item.get("reward_model")
    if isinstance(reward_model, dict):
        gt = reward_model.get("ground_truth")
        if isinstance(gt, dict):
            if "target" in gt:
                return gt["target"]
            for key in ("answer", "text", "value"):
                if key in gt:
                    return gt[key]
    return None

def extract_answer_and_confidence(response, strategy="auto"):
    """
    从模型响应中提取答案和置信度
    支持声明级置信度标签（<Confidence value=...>）的聚合
    """
    solution_str=response.strip()
    answer_patterns = [
        r"(?im)^Answer\s*:\s*([^\n]+)",
        r"(?im)^Final\s*Answer\s*:\s*([^\n]+)",
        r"(?im)^Best\s*Guess\s*:\s*([^\n]+)",
        r"(?im)^(?:My\s+answer\s+is|The\s+answer\s+is)\s*([^\n.]+)",
    ]
    extracted_answer = None
    for pattern in answer_patterns:
        answer_match = re.findall(pattern, solution_str)
        if answer_match:
            extracted_answer = answer_match[-1].strip()
            if extracted_answer:
                break
    if not extracted_answer:
        extracted_answer = _search_r1_qa.extract_solution(solution_str)

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
    if not extracted_answer:
        extracted_answer = None
    return extracted_answer, confidence
    # if "</think>" in response:
    #     response = response.split("</think>")[-1]
    # response = response.strip()
    # def _find_last_match(patterns, text, flags=0, tail_chars=4000):
    #     tail = text[-tail_chars:] if tail_chars and len(text) > tail_chars else text
    #     last = None
    #     last_pos = -1
    #     for pattern in patterns:
    #         for m in re.finditer(pattern, tail, flags):
    #             if m.start() >= last_pos:
    #                 last_pos = m.start()
    #                 last = m.group(1)
    #     if last is not None:
    #         return last
    #     last = None
    #     last_pos = -1
    #     for pattern in patterns:
    #         for m in re.finditer(pattern, text, flags):
    #             if m.start() >= last_pos:
    #                 last_pos = m.start()
    #                 last = m.group(1)
    #     return last

    # # 1) 解析声明级置信度（如果有）
    # claim_confs = []
    # for pattern in [
    #     r'confidence="([0-9]*\.?[0-9]+)"',
    #     r'(?i)<\s*Confidence\b[^>]*\bvalue\s*=\s*([0-9]+(?:\.[0-9]+)?)',
    # ]:
    #     for m in re.findall(pattern, response):
    #         try:
    #             v = float(m)
    #             if v > 1:
    #                 v = v / 100.0
    #             claim_confs.append(np.clip(v, 0.0, 1.0))
    #         except Exception:
    #             continue

    # confidence = None
    # if strategy in ("claim_product", "claim_minimum") and claim_confs:
    #     if strategy == "claim_product":
    #         confidence = float(np.prod(claim_confs))
    #     else:
    #         confidence = float(np.min(claim_confs))

    # # 2) 解析响应级置信度
    # if confidence is None:
    #     conf_patterns = [
    #         r'[Cc]onfidence[:\s]+([0-9]*\.?[0-9]+)',
    #         r'置信度[:\s：]+([0-9]*\.?[0-9]+)',
    #     ]
    #     conf_text = _find_last_match(conf_patterns, response, re.IGNORECASE)
    #     if conf_text is not None:
    #         try:
    #             conf_value = float(conf_text)
    #             if conf_value > 1:
    #                 conf_value = conf_value / 100.0
    #             if 0 <= conf_value <= 1:
    #                 confidence = conf_value
    #         except Exception:
    #             pass

    # # 3) 默认置信度
    # if confidence is None:
    #     confidence = 0.5

    # # 4) 提取答案（严格模式）
    # # 只接受训练格式：<answer>...</answer>
    # answer = None
    # tag_answer_pattern = r'(?is)<\s*answer\b[^>]*>\s*(.*?)\s*<\s*/\s*answer\s*>'
    # ans_text = _find_last_match([tag_answer_pattern], response, re.IGNORECASE | re.DOTALL)

    # if ans_text is not None:
    #     answer = ans_text.strip()
    #     answer = answer.rstrip('.,;!?。，；！？')

    # if not answer:
    #     answer = None

    #return answer, confidence

def check_answer_correctness(predicted, ground_truth):
    """
    检查答案是否正确
    修复版：正确处理None和空字符串
    """
    if not isinstance(ground_truth, (str, bytes)) and hasattr(ground_truth, "tolist"):
        try:
            return check_answer_correctness(predicted, ground_truth.tolist())
        except Exception:
            pass
    if isinstance(ground_truth, (list, tuple)):
        for gt in ground_truth:
            if check_answer_correctness(predicted, gt):
                return True
        return False

    # 如果预测答案为None或空，直接返回False
    if predicted is None or predicted == "":
        return False

    if ground_truth is None or ground_truth == "":
        return False

    pred_clean = str(predicted).strip()
    gt_clean = str(ground_truth).strip()

    # 尝试数值比较
    try:
        pred_num = float(re.sub(r'[,\s]', '', pred_clean))
        gt_num = float(re.sub(r'[,\s]', '', gt_clean))
        return abs(pred_num - gt_num) < 1e-6
    except Exception:
        pass

    def _normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    pred_norm = _normalize_text(pred_clean)
    gt_norm = _normalize_text(gt_clean)
    return pred_norm == gt_norm


def _normalize_ground_truth(value):
    if value is None:
        return None
    if not isinstance(value, (str, bytes)) and hasattr(value, "tolist"):
        try:
            return _normalize_ground_truth(value.tolist())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        normalized = []
        for item in value:
            norm_item = _normalize_ground_truth(item)
            if norm_item not in (None, ""):
                normalized.append(norm_item)
        return normalized if normalized else None
    if isinstance(value, dict):
        for key in ("answer", "text", "value"):
            if key in value:
                return _normalize_ground_truth(value[key])
        return None
    return value


def infer_strategy(model_path: str, strategy: str) -> str:
    if strategy and strategy != "auto":
        return strategy
    name = model_path.lower()
    if "confidence_brier" in name:
        return "verbalized_brier"
    if "confidence_ce" in name:
        return "verbalized_ce"
    if "confidence_prod" in name:
        return "claim_product"
    if "confidence_min" in name:
        return "claim_minimum"
    if "ppo_value" in name:
        return "ppo_value"
    if "baseline" in name:
        return "baseline"
    if "explicit_risk" in name:
        return "explicit_risk"
    return "verbalized_brier"


def build_prompt(question: str, strategy: str, risk_t: float | None = None, dataset_name: str | None = None) -> str:
    if strategy == "explicit_risk":
        if risk_t is None:
            risk_t = 0.5
        r = risk_t / (1.0 - risk_t) if risk_t < 1.0 else float("inf")
        _er_debug(f"build_prompt explicit_risk: t={risk_t:.3f} r={r:.3f} dataset={dataset_name}")
        if dataset_name in QA_DATASETS:
            return explicit_risk_qa_prompt(question, risk_t)
        return explicit_risk_math_prompt(question, risk_t)
    if strategy in ("claim_product", "claim_minimum"):
        return claim_level_math_prompt(question)
    if strategy == "ppo_value":
        return f"Question: {question}\n\nAnswer:"
    # baseline / verbalized_brier / verbalized_ce
    if dataset_name in QA_DATASETS:
        return response_level_qa_prompt(question)
    return response_level_math_prompt(question)


def load_critic_model(critic_path: str | None, device: str, offline: bool):
    if not critic_path:
        return None
    if not os.path.exists(critic_path):
        print(f"警告: critic路径不存在: {critic_path}")
        return None
    critic_path = _resolve_model_dir(critic_path)
    try:
        critic = AutoModelForTokenClassification.from_pretrained(
            critic_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=offline,
        )
    except Exception:
        if offline:
            raise
        critic = AutoModelForTokenClassification.from_pretrained(
            critic_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=False,
        )

    critic = _move_model_to_device(critic, device)
    critic.eval()
    return critic

def compute_smoothed_ece(confidences, correctness, num_bins=15):
    """
    计算Smoothed ECE (smECE)
    使用kernel density estimation
    """
    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=float)

    if len(confidences) == 0:
        return 0.0

    # 使用Gaussian kernel进行平滑（密度加权）
    bandwidth = 0.1
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ece = 0.0
    total_weight = 0.0

    for center in bin_centers:
        weights = np.exp(-((confidences - center) ** 2) / (2 * bandwidth ** 2))
        density = np.sum(weights)
        if density <= 0:
            continue

        avg_accuracy = np.sum(weights * correctness) / density
        ece += density * abs(avg_accuracy - center)
        total_weight += density

    return ece / (total_weight + 1e-10)



def evaluate_model(
    model,
    tokenizer,
    dataset,
    device,
    strategy,
    dataset_name,
    risk_threshold=0.5,
    max_samples=None,
    batch_size=1,
    sample_offset=0,
    sample_stride=1,
    critic_model=None,
    simpleqa_grader=None,
    use_chat_template=False,
    thinking_mode="auto",
    max_new_tokens=2048,
    gen_temperature=None,
    gen_top_p=None,
    gen_top_k=None,
    gen_min_p=None,
    gen_do_sample=None,
    save_io=False,
    use_vllm=False,
):
    """评估模型 - 实现Behavioral Calibration（支持批量推理）"""
    results = []

    if strategy == "explicit_risk":
        _er_debug(f"evaluate_model explicit_risk start: mode=fixed_t t={risk_threshold} dataset={dataset_name}")

    if sample_stride < 1:
        raise ValueError("sample_stride 必须 >= 1")
    if sample_offset < 0:
        raise ValueError("sample_offset 必须 >= 0")
    if sample_stride != 1 or sample_offset != 0:
        dataset = dataset[sample_offset::sample_stride]

    if max_samples:
        dataset = dataset[:max_samples]

    if batch_size is None or int(batch_size) < 1:
        raise ValueError("batch_size 必须 >= 1")
    batch_size = int(batch_size)

    if tokenizer is None:
        raise ValueError("tokenizer 为空，无法推理")
    if not use_vllm:
        # 批量推理需要 pad token
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif getattr(tokenizer, "unk_token", None) is not None:
                tokenizer.pad_token = tokenizer.unk_token
        try:
            tokenizer.padding_side = "left"
        except Exception:
            pass

    # 生成参数（根据思考模式调整默认值）
    temp = 1.0
    top_p = 0.7
    top_k = -1
    min_p = None
    do_sample = True
    if thinking_mode == "on":
        temp = 0.6
        top_p = 0.95
        top_k = 20
        min_p = 0.0
    elif thinking_mode == "off":
        temp = 0.7
        top_p = 0.8
        top_k = 20
        min_p = 0.0

    if gen_temperature is not None:
        temp = gen_temperature
    if gen_top_p is not None:
        top_p = gen_top_p
    if gen_top_k is not None:
        top_k = gen_top_k
    if gen_min_p is not None:
        min_p = gen_min_p
    if gen_do_sample is not None:
        do_sample = gen_do_sample

    gen_kwargs = None
    sampling_params = None
    if use_vllm:
        try:
            from vllm import SamplingParams
        except Exception as e:
            raise ImportError("需要安装vLLM才能使用--use_vllm") from e
        sampling_kwargs = {
            "n": 1,
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_new_tokens,
            "stop": None,
        }
        if min_p is not None:
            sampling_kwargs["min_p"] = min_p
        if do_sample is False:
            sampling_kwargs["temperature"] = 0.0
            sampling_kwargs["top_p"] = 1.0
            sampling_kwargs["top_k"] = -1
            sampling_kwargs.pop("min_p", None)
        try:
            sampling_params = SamplingParams(**sampling_kwargs)
        except TypeError:
            sampling_kwargs.pop("min_p", None)
            sampling_params = SamplingParams(**sampling_kwargs)
    else:
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temp,
            "top_p": top_p,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if min_p is not None and hasattr(getattr(model, "generation_config", object()), "min_p"):
            gen_kwargs["min_p"] = min_p

    total = len(dataset)
    if total == 0:
        return results

    import time as _time

    n_batches = (total + batch_size - 1) // batch_size

    # 终端写到 /dev/tty（动态进度条），log 写到 stdout（静态行）
    try:
        _tty = open("/dev/tty", "w")
        _tty_ok = True
    except OSError:
        _tty = sys.stderr
        _tty_ok = False

    _pbar = tqdm(range(0, total, batch_size), desc="评估中",
                 total=n_batches, file=_tty, dynamic_ncols=True)
    _batch_start_times = []

    for start in _pbar:
        _batch_idx = start // batch_size + 1
        _batch_start_times.append(_time.time())

        # 估算剩余时间（基于最近批次的平均耗时）
        if len(_batch_start_times) >= 2:
            _avg_sec = (_batch_start_times[-1] - _batch_start_times[0]) / (len(_batch_start_times) - 1)
            _remaining = _avg_sec * (n_batches - _batch_idx + 1)
            _eta_str = f"{int(_remaining // 3600):02d}:{int(_remaining % 3600 // 60):02d}:{int(_remaining % 60):02d}"
        else:
            _eta_str = "--:--:--"

        print(f"[进度] batch {_batch_idx}/{n_batches} | 样本 {start+1}~{min(start+batch_size, total)}/{total} | ETA {_eta_str}",
              flush=True)

        batch_items = dataset[start:start + batch_size]

        prompt_texts = []
        meta = []
        for item in batch_items:
            question = _extract_question(item, dataset_name)
            ground_truth = _extract_ground_truth(item)
            ground_truth = _normalize_ground_truth(ground_truth)

            # 构建prompt（与训练策略保持一致）
            risk_t = None
            if strategy == "explicit_risk":
                risk_t = float(risk_threshold)
            prompt = build_prompt(question, strategy, risk_t, dataset_name)

            prompt_text = prompt
            if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    kwargs = {"tokenize": False, "add_generation_prompt": True}
                    try:
                        sig = inspect.signature(tokenizer.apply_chat_template)
                        if "enable_thinking" in sig.parameters:
                            if thinking_mode == "on":
                                kwargs["enable_thinking"] = True
                            elif thinking_mode == "off":
                                kwargs["enable_thinking"] = False
                    except Exception:
                        pass
                    prompt_text = tokenizer.apply_chat_template(messages, **kwargs)
                except Exception:
                    prompt_text = prompt

            prompt_texts.append(prompt_text)
            meta.append({
                "question": question,
                "ground_truth": ground_truth,
                "prompt": prompt,
                "prompt_text": prompt_text,
                "risk_t": risk_t,
            })

        if use_vllm:
            outputs = model.generate(prompt_texts, sampling_params)
            response_texts = []
            for out in outputs:
                if not out.outputs:
                    response_texts.append("")
                else:
                    response_texts.append(out.outputs[0].text)
        else:
            # 批量生成
            inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            if "attention_mask" in inputs:
                input_lens = inputs["attention_mask"].sum(dim=1).tolist()
            else:
                input_lens = [inputs["input_ids"].shape[1]] * len(batch_items)
            response_texts = []
            for i in range(len(meta)):
                input_len = int(input_lens[i])
                response_texts.append(tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True))

        # 逐条解析
        for i, info in enumerate(meta):
            response = response_texts[i]

            predicted_answer, confidence = extract_answer_and_confidence(response, strategy=strategy)
            if strategy == "explicit_risk":
                confidence = 0.0 if "<IDK>" in response else 1.0

            # PPO-Value：使用critic输出作为置信度
            if strategy == "ppo_value" and critic_model is not None:
                try:
                    critic_inputs = tokenizer(info["prompt"] + response, return_tensors="pt").to(device)
                    with torch.no_grad():
                        critic_out = critic_model(**critic_inputs)
                    value = critic_out.logits[0, -1, 0].item()
                    confidence = float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))
                except Exception:
                    pass

            # 若模型显式输出IDK，视为弃权
            idk_abstain = "<IDK>" in response
            if idk_abstain:
                confidence = 0.0

            # 判断答案是否正确
            correct_raw = False
            abstention_flag = False
            grading_label = None
            if dataset_name in QA_DATASETS and simpleqa_grader is not None:
                try:
                    grading_label = simpleqa_grader.grade(info["question"], info["ground_truth"], response)
                except Exception as e:
                    raise RuntimeError(f"SimpleQA评分器失败: {e}") from e
                if grading_label == "not_attempted":
                    correct_raw = False
                    abstention_flag = True
                    confidence = 0.0
                elif grading_label == "correct":
                    correct_raw = True
                elif grading_label == "incorrect":
                    correct_raw = False
                else:
                    raise ValueError(f"SimpleQA评分器返回未知标签: {grading_label}")
            elif dataset_name in ("aime", "aime_2024", "aime_2025", "beyondaime") and is_correct_minerva is not None:
                try:
                    correct_raw, abstention_flag, pred_norm = is_correct_minerva(response, info["ground_truth"])
                    if pred_norm:
                        predicted_answer = pred_norm
                except Exception:
                    if predicted_answer is not None:
                        correct_raw = check_answer_correctness(predicted_answer, info["ground_truth"])
            else:
                if predicted_answer is not None:
                    # search_r1 类 QA 数据集使用 em_check，与训练时奖励计算口径一致
                    if dataset_name in ("nq_hotpotqa_searchr1", *FLASHRAG_DATASETS.keys(), *FLASHRAG_DATASETS_TRUE.keys(),"hotpotqa", "hotpot_qa"):
                        targets = info["ground_truth"]
                        correct_raw = bool(_search_r1_qa.em_check(predicted_answer, targets))
                    else:
                        correct_raw = check_answer_correctness(predicted_answer, info["ground_truth"])

            # Abstention决策：如果置信度低于阈值或显式IDK，拒绝回答
            if strategy == "explicit_risk":
                abstained = idk_abstain or bool(abstention_flag)
            else:
                abstained = (confidence < risk_threshold) or idk_abstain or bool(abstention_flag)

            item_out = {
                'question': info["question"],
                'ground_truth': info["ground_truth"],
                'predicted_answer': predicted_answer,
                'confidence': float(confidence),
                'abstained': bool(abstained),
                'correct': bool(correct_raw),
                'is_correct': bool(correct_raw and not abstained),
                'full_response': response
            }
            if info.get("risk_t") is not None:
                item_out["risk_t"] = float(info["risk_t"])
            if save_io:
                item_out["prompt"] = info["prompt"]
                item_out["prompt_text"] = info["prompt_text"]
                item_out["generation_params"] = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temp,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": min_p,
                    "do_sample": do_sample,
                    "thinking_mode": thinking_mode,
                }
            results.append(item_out)
            if grading_label is not None:
                results[-1]["grading_label"] = grading_label

    return results
def compute_metrics(results, risk_threshold=0.5):
    """
    计算评估指标（论文 Section 4.2）
    """
    n = len(results)
    if n == 0:
        return {
            'predictive_accuracy': 0.0,
            'abstention_accuracy': 0.0,
            'brier_score': 0.0,
            'smoothed_ece': 0.0,
            'confidence_auc': 0.5,
            'snr_gain': 0.0,
            'nll': 0.0,
            'avg_confidence': 0.0,
            'abstention_rate': 0.0,
            'n_samples': 0,
            'n_answered': 0,
            'n_abstained': 0,
            'risk_threshold': risk_threshold,
        }

    confidences = [r['confidence'] for r in results]
    correctness = [1 if r.get('correct', False) else 0 for r in results]
    abstained = [r['abstained'] for r in results]

    avg_confidence = float(np.mean(confidences))
    abstention_rate = float(np.mean(abstained))
    n_abstained = int(np.sum(abstained))
    n_answered = n - n_abstained

    metrics = {}
    if ConfidenceCalibrationMetrics is not None:
        metrics_calc = ConfidenceCalibrationMetrics()
        metrics = metrics_calc.compute_all_metrics(confidences, correctness, abstention_threshold=risk_threshold)
    else:
        # 回退到简化指标（理论上不应触发）
        smece = compute_smoothed_ece(confidences, correctness)
        try:
            from sklearn.metrics import roc_auc_score
            conf_auc = roc_auc_score(correctness, confidences)
        except Exception:
            conf_auc = 0.5
        metrics = {
            "smoothed_ece": float(smece),
            "brier_score": float(np.mean((np.array(confidences) - np.array(correctness)) ** 2)),
            "nll": 0.0,
            "confidence_auc": float(conf_auc),
            "snr_gain": 0.0,
            "abstention_accuracy": 0.0,
            "predictive_accuracy": float(np.mean(correctness)),
        }

    metrics_out = {
        'predictive_accuracy': metrics.get('predictive_accuracy', 0.0),
        'abstention_accuracy': metrics.get('abstention_accuracy', 0.0),
        'brier_score': metrics.get('brier_score', 0.0),
        'smoothed_ece': metrics.get('smoothed_ece', 0.0),
        'confidence_auc': metrics.get('confidence_auc', 0.5),
        'snr_gain': metrics.get('snr_gain', 0.0),
        'nll': metrics.get('nll', 0.0),
        'avg_confidence': avg_confidence,
        'abstention_rate': abstention_rate,
        'n_samples': n,
        'n_answered': n_answered,
        'n_abstained': n_abstained,
        'risk_threshold': risk_threshold,
    }

    return metrics_out


def compute_risk_metrics(results_random, results_t0=None, thresholds=None, abstention_threshold=0.5):
    """
    使用显式风险(prompt含t)的结果计算 SNR Gain / Abstention Accuracy / Predictive Accuracy。
    - results_random: 显式风险评估结果（必须包含 risk_t）
    - results_t0: t基准点（默认0.0）的评估结果（用于 SNR 基准与 PredAcc）
    """
    def _count_stats(rows):
        total = len(rows)
        if total == 0:
            return {
                "total": 0,
                "answered": 0,
                "abstained": 0,
                "correct_answered": 0,
                "wrong_answered": 0,
                "incorrect_abstained": 0,
            }
        answered = 0
        correct_answered = 0
        wrong_answered = 0
        incorrect_abstained = 0
        abstained = 0
        for r in rows:
            is_abstained = bool(r.get("abstained", False))
            is_correct = bool(r.get("correct", False))
            if is_abstained:
                abstained += 1
                if not is_correct:
                    incorrect_abstained += 1
            else:
                answered += 1
                if is_correct:
                    correct_answered += 1
                else:
                    wrong_answered += 1
        return {
            "total": total,
            "answered": answered,
            "abstained": abstained,
            "correct_answered": correct_answered,
            "wrong_answered": wrong_answered,
            "incorrect_abstained": incorrect_abstained,
        }

    results_random = results_random or []
    if not results_random:
        return {
            "snr_gain": 0.0,
            "abstention_accuracy": 0.0,
            "predictive_accuracy": 0.0,
            "abstention_rate": 0.0,
            "n_samples": 0,
            "n_answered": 0,
            "n_abstained": 0,
            "risk_thresholds": [],
        }

    # 必须包含 risk_t，并按 t 分组（用于评估网格）
    has_risk_t = any(r.get("risk_t") is not None for r in results_random)
    if not has_risk_t:
        raise ValueError("results_random 缺少 risk_t，已移除旧的随机t近似路径。")

    grouped = {}
    for r in results_random:
        t_val = r.get("risk_t")
        if t_val is None:
            continue
        key = round(float(t_val), 6)
        grouped.setdefault(key, []).append(r)

    # Ensure t=0 baseline is available for Eq.(1) style comparison when caller
    # provides an explicit t0 run separately from the sweep grid.
    if results_t0:
        grouped.setdefault(0.0, list(results_t0))

    if thresholds is None:
        thresholds = sorted(grouped.keys())
    else:
        thresholds = [round(float(t), 6) for t in thresholds]
        thresholds = [t for t in thresholds if t in grouped]
    # Prefer using [0,1] endpoints when available to better match paper's
    # interval integration definition of SNR([0,1]).
    if 0.0 in grouped and 0.0 not in thresholds:
        thresholds = [0.0] + thresholds
    if 1.0 in grouped and 1.0 not in thresholds:
        thresholds = thresholds + [1.0]
    thresholds = sorted(set(thresholds))
    if not thresholds:
        return {
            "snr_gain": 0.0,
            "abstention_accuracy": 0.0,
            "predictive_accuracy": 0.0,
            "abstention_rate": 0.0,
            "n_samples": 0,
            "n_answered": 0,
            "n_abstained": 0,
            "risk_thresholds": [],
        }

    acc_rates = []
    hal_rates = []
    abst_rates = []
    for t in thresholds:
        stats = _count_stats(grouped.get(t, []))
        total = stats["total"]
        if total == 0:
            acc_rates.append(0.0)
            hal_rates.append(0.0)
            abst_rates.append(0.0)
        else:
            acc_rates.append(stats["correct_answered"] / total)
            hal_rates.append(stats["wrong_answered"] / total)
            abst_rates.append(stats["abstained"] / total)

    acc_rates = np.array(acc_rates, dtype=float)
    hal_rates = np.array(hal_rates, dtype=float)

    # SNR(0)
    snr_gain = 0.0
    pred_acc = 0.0
    if results_t0:
        t0_stats = _count_stats(results_t0)
        total_t0 = t0_stats["total"]
        if total_t0 > 0:
            acc_0 = t0_stats["correct_answered"] / total_t0
            hal_0 = t0_stats["wrong_answered"] / total_t0
            pred_acc = acc_0
            if hal_0 > 0:
                snr_0 = acc_0 / hal_0
                acc_int = np.trapz(acc_rates, thresholds)
                hal_int = np.trapz(hal_rates, thresholds)
                if hal_int > 0:
                    snr_gain = float(np.log((acc_int / hal_int) / snr_0))

    # Abstention Accuracy 用指定阈值
    abst_t = round(float(abstention_threshold), 6)
    if abst_t not in grouped:
        # 取最近的 t
        abst_t = min(thresholds, key=lambda x: abs(x - abst_t))
    abst_stats = _count_stats(grouped.get(abst_t, []))
    abst_total = abst_stats["total"]
    abst_acc = 0.0
    abst_rate = 0.0
    if abst_total > 0:
        abst_acc = (abst_stats["correct_answered"] + abst_stats["incorrect_abstained"]) / abst_total
        abst_rate = abst_stats["abstained"] / abst_total

    return {
        "snr_gain": float(snr_gain),
        "abstention_accuracy": float(abst_acc),
        "predictive_accuracy": float(pred_acc),
        "abstention_rate": float(abst_rate),
        "n_samples": int(abst_stats.get("total", 0)),
        "n_answered": int(abst_stats.get("answered", 0)),
        "n_abstained": int(abst_stats.get("abstained", 0)),
        "risk_thresholds": thresholds,
    }


def compute_policy_metrics_at_t(results):
    """在单一风险阈值 t 下计算策略层指标（用于 explicit_risk 日志）。"""
    rows = results or []
    total = len(rows)
    if total == 0:
        return {
            "predictive_accuracy": 0.0,
            "abstention_accuracy": 0.0,
            "abstention_rate": 0.0,
            "hallucination_rate": 0.0,
            "snr": 0.0,
            "n_samples": 0,
            "n_answered": 0,
            "n_abstained": 0,
        }

    answered_rows = [r for r in rows if not bool(r.get("abstained", False))]
    abstained_rows = [r for r in rows if bool(r.get("abstained", False))]

    n_answered = len(answered_rows)
    n_abstained = len(abstained_rows)
    correct_answered = sum(1 for r in answered_rows if bool(r.get("correct", False)))
    wrong_answered = n_answered - correct_answered
    incorrect_abstained = sum(1 for r in abstained_rows if not bool(r.get("correct", False)))

    pred_acc = correct_answered / total
    abst_acc = (correct_answered + incorrect_abstained) / total
    hall_rate = wrong_answered / total
    abst_rate = n_abstained / total

    if wrong_answered > 0:
        snr = correct_answered / wrong_answered
    elif correct_answered > 0:
        snr = float("inf")
    else:
        snr = 0.0

    return {
        "predictive_accuracy": float(pred_acc),
        "abstention_accuracy": float(abst_acc),
        "abstention_rate": float(abst_rate),
        "hallucination_rate": float(hall_rate),
        "snr": float(snr),
        "n_samples": int(total),
        "n_answered": int(n_answered),
        "n_abstained": int(n_abstained),
    }

def main():
    args = parse_args()

    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'

    os.makedirs(args.output_dir, exist_ok=True)

    strategy = infer_strategy(args.model_path, args.strategy)
    _er_debug(f"main: inferred strategy={strategy} args.strategy={args.strategy} model_path={args.model_path}")

    print("=" * 80)
    print(f"评估模型: {args.model_path}")
    print(f"数据集: {args.dataset}")
    print(f"策略: {strategy}")
    print(f"风险阈值: {args.risk_threshold}")
    if args.risk_prompt_metrics:
        print(f"t相关指标: 显式风险prompt (grid={args.risk_prompt_grid}, t0={args.risk_prompt_t0})")
    print("=" * 80)

    if args.use_vllm:
        print("推理引擎: vLLM")
        model, tokenizer = load_vllm_engine(
            args.model_path,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            trust_remote_code=args.vllm_trust_remote_code,
            seed=args.vllm_seed,
            max_model_len=args.vllm_max_model_len,
            block_size=args.vllm_block_size,
            enable_prefix_caching=args.vllm_enable_prefix_caching,
            max_num_batched_tokens=args.vllm_max_num_batched_tokens,
            max_num_seqs=args.vllm_max_num_seqs,
        )
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                args.base_model,
                trust_remote_code=True,
                local_files_only=args.offline,
            )
    else:
        model, tokenizer = load_model_and_tokenizer(args.base_model, args.model_path, args.device, offline=args.offline)
    critic_model = None
    if strategy == "ppo_value":
        critic_model = load_critic_model(args.critic_path, args.device, args.offline)

    simpleqa_grader = None
    if args.dataset in QA_DATASETS:
        if args.simpleqa_grader_local:
            try:
                from simpleqa_grader import LocalSimpleQAGrader
            except Exception:
                SIMPLEQA_DIR = os.path.join(ROOT_DIR, "eval")
                if SIMPLEQA_DIR not in sys.path:
                    sys.path.insert(0, SIMPLEQA_DIR)
                from simpleqa_grader import LocalSimpleQAGrader
            cache_path = args.simpleqa_grader_cache or os.path.join(args.output_dir, "simpleqa_grader_cache.jsonl")
            simpleqa_grader = LocalSimpleQAGrader(
                model_path=args.simpleqa_grader_local,
                device=args.simpleqa_grader_device,
                use_chat_template=args.simpleqa_grader_use_chat_template,
                local_files_only=args.simpleqa_grader_local_files_only,
                dtype=args.simpleqa_grader_dtype,
                cache_path=cache_path,
            )
        else:
            grader_name = (args.simpleqa_grader or "").strip().lower()
            if grader_name and grader_name not in ("none", "false", "0"):
                try:
                    from simpleqa_grader import SimpleQAGrader
                except Exception:
                    SIMPLEQA_DIR = os.path.join(ROOT_DIR, "eval")
                    if SIMPLEQA_DIR not in sys.path:
                        sys.path.insert(0, SIMPLEQA_DIR)
                    from simpleqa_grader import SimpleQAGrader
                cache_path = args.simpleqa_grader_cache or os.path.join(args.output_dir, "simpleqa_grader_cache.jsonl")
                simpleqa_grader = SimpleQAGrader(
                    model=args.simpleqa_grader,
                    api_base=args.simpleqa_grader_base,
                    api_key=args.simpleqa_grader_key,
                    timeout=args.simpleqa_grader_timeout,
                    max_retries=args.simpleqa_grader_retries,
                    retry_delay=args.simpleqa_grader_retry_delay,
                    cache_path=cache_path,
                )

    dataset = load_evaluation_dataset(
        args.dataset,
        dataset_file=args.dataset_file,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
    )
    results_conf = evaluate_model(
        model,
        tokenizer,
        dataset,
        args.device,
        strategy,
        args.dataset,
        args.risk_threshold,
        args.max_samples,
        args.batch_size,
        args.sample_offset,
        args.sample_stride,
        critic_model,
        simpleqa_grader,
        args.use_chat_template,
        args.thinking_mode,
        args.max_new_tokens,
        args.gen_temperature,
        args.gen_top_p,
        args.gen_top_k,
        args.gen_min_p,
        None if args.gen_do_sample is None else (str(args.gen_do_sample).lower() in ("1", "true", "yes", "y")),
        args.save_io,
        args.use_vllm,
    )
    metrics_conf = compute_metrics(results_conf, args.risk_threshold)

    risk_results_random = None
    risk_results_t0 = None
    metrics_risk = None
    if args.risk_prompt_metrics:
        # 固定t网格评估（非随机）
        try:
            grid = [float(x) for x in args.risk_prompt_grid.split(",") if x.strip() != ""]
        except Exception:
            grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        if not grid:
            grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        grid = sorted(set([float(max(0.0, min(1.0, t))) for t in grid]))

        risk_results_random = []
        t0_cached = None
        for t_val in grid:
            res_t = evaluate_model(
                model,
                tokenizer,
                dataset,
                args.device,
                "explicit_risk",
                args.dataset,
                t_val,
                args.max_samples,
                args.batch_size,
                args.sample_offset,
                args.sample_stride,
                None,
                simpleqa_grader,
                args.use_chat_template,
                args.thinking_mode,
                args.max_new_tokens,
                args.gen_temperature,
                args.gen_top_p,
                args.gen_top_k,
                args.gen_min_p,
                None if args.gen_do_sample is None else (str(args.gen_do_sample).lower() in ("1", "true", "yes", "y")),
                args.save_io,
                args.use_vllm,
            )
            risk_results_random.extend(res_t)
            if float(t_val) == float(args.risk_prompt_t0):
                t0_cached = res_t

        if t0_cached is None:
            risk_results_t0 = evaluate_model(
                model,
                tokenizer,
                dataset,
                args.device,
                "explicit_risk",
                args.dataset,
                args.risk_prompt_t0,
                args.max_samples,
                args.batch_size,
                args.sample_offset,
                args.sample_stride,
                None,
                simpleqa_grader,
                args.use_chat_template,
                args.thinking_mode,
                args.max_new_tokens,
                args.gen_temperature,
                args.gen_top_p,
                args.gen_top_k,
                args.gen_min_p,
                None if args.gen_do_sample is None else (str(args.gen_do_sample).lower() in ("1", "true", "yes", "y")),
                args.save_io,
                args.use_vllm,
            )
        else:
            risk_results_t0 = t0_cached

        metrics_risk = compute_risk_metrics(
            risk_results_random,
            risk_results_t0,
            thresholds=grid,
            abstention_threshold=args.risk_threshold,
        )

    # 合并：默认沿用置信度指标；t相关指标使用显式风险评估结果
    metrics = dict(metrics_conf)
    if metrics_risk:
        metrics.update({
            "snr_gain": metrics_risk.get("snr_gain", metrics.get("snr_gain", 0.0)),
            "abstention_accuracy": metrics_risk.get("abstention_accuracy", metrics.get("abstention_accuracy", 0.0)),
            "predictive_accuracy": metrics_risk.get("predictive_accuracy", metrics.get("predictive_accuracy", 0.0)),
            "abstention_rate": metrics_risk.get("abstention_rate", metrics.get("abstention_rate", 0.0)),
            "n_samples": metrics_risk.get("n_samples", metrics.get("n_samples", 0)),
            "n_answered": metrics_risk.get("n_answered", metrics.get("n_answered", 0)),
            "n_abstained": metrics_risk.get("n_abstained", metrics.get("n_abstained", 0)),
        })

    # 打印结果（按口径分离日志）
    print("\n" + "=" * 80)
    print("评估结果:")
    print("=" * 80)
    if strategy == "explicit_risk":
        policy_metrics = compute_policy_metrics_at_t(results_conf)
        print(f"Policy Mode: explicit_risk (t={args.risk_threshold})")
        print(f"Predictive Accuracy: {policy_metrics['predictive_accuracy']:.4f}")
        print(f"Abstention Accuracy: {policy_metrics['abstention_accuracy']:.4f}")
        print(f"Abstention Rate: {policy_metrics['abstention_rate']:.4f}")
        print(f"Hallucination Rate: {policy_metrics['hallucination_rate']:.4f}")
        print(f"SNR: {policy_metrics['snr']:.4f}")
        print(f"样本数: {policy_metrics['n_samples']} (回答: {policy_metrics['n_answered']}, 拒绝: {policy_metrics['n_abstained']})")
    else:
        print(f"Predictive Accuracy: {metrics['predictive_accuracy']:.4f}")
        print(f"Abstention Accuracy: {metrics['abstention_accuracy']:.4f}")
        print(f"Brier Score: {metrics['brier_score']:.4f}")
        print(f"Negative Log-Likelihood: {metrics['nll']:.4f}")
        print(f"Smoothed ECE: {metrics['smoothed_ece']:.4f}")
        print(f"Confidence AUC: {metrics['confidence_auc']:.4f}")
        print(f"SNR Gain: {metrics['snr_gain']:.4f}")
        print(f"平均置信度: {metrics['avg_confidence']:.4f}")
        print(f"Abstention Rate: {metrics['abstention_rate']:.4f}")
        print(f"样本数: {metrics['n_samples']} (回答: {metrics['n_answered']}, 拒绝: {metrics['n_abstained']})")
        if metrics_risk:
            print("-" * 80)
            print("t相关指标（显式风险prompt）:")
            print(f"Predictive Accuracy(t={args.risk_prompt_t0}): {metrics_risk.get('predictive_accuracy', 0.0):.4f}")
            print(f"Abstention Accuracy: {metrics_risk.get('abstention_accuracy', 0.0):.4f}")
            print(f"SNR Gain: {metrics_risk.get('snr_gain', 0.0):.4f}")
    print("=" * 80)

    # 保存结果
    predictions = [
        {
            "confidence": r["confidence"],
            "correct": r.get("correct", False),
            "abstained": r.get("abstained", False),
            "predicted_answer": r.get("predicted_answer"),
        }
        for r in results_conf
    ]

    risk_predictions = None
    if risk_results_random:
        risk_predictions = [
            {
                "confidence": r.get("confidence", 0.0),
                "correct": r.get("correct", False),
                "abstained": r.get("abstained", False),
                "predicted_answer": r.get("predicted_answer"),
                "risk_t": r.get("risk_t"),
            }
            for r in risk_results_random
        ]

    output_file = Path(args.output_dir) / "results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'metrics_confidence': metrics_conf,
            'metrics_risk': metrics_risk,
            'predictions': predictions,
            'predictions_risk': risk_predictions,
            'results': results_conf,
            'results_risk': risk_results_random,
            'results_risk_t0': risk_results_t0,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
