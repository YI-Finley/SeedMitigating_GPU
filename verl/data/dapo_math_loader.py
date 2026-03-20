"""
DAPO-Math-17k数据集加载器
仅适配《Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning》baseline + 4个校准目标

核心约束：
- 仅加载DAPO-Math-17k数据集
"""

from datasets import load_dataset, Dataset, DatasetDict
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_dapo_math_dataset(
    dataset_name: str = "TIGER-Lab/DAPO-Math-17k",
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    加载DAPO-Math-17k数据集

    数据集格式：
    {
        "question": str,  # 数学问题
        "answer": str,    # 标准答案
        "solution": str,  # 解题步骤（可选）
    }

    参数：
        dataset_name: 数据集名称（默认为DAPO-Math-17k）
        split: 数据集分割（train/test/validation）
        max_samples: 最大样本数（None表示使用全部数据）

    返回：
        Dataset对象
    """
    logger.info(f"Loading dataset: {dataset_name}")

    try:
        # 加载数据集
        if split:
            dataset = load_dataset(dataset_name, split=split)
        else:
            dataset = load_dataset(dataset_name)
            # 如果返回DatasetDict，取train分割
            if isinstance(dataset, DatasetDict):
                dataset = dataset["train"]

        logger.info(f"Dataset loaded: {len(dataset)} samples")

        # 限制样本数
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))
            logger.info(f"Limited to {max_samples} samples")

        # 验证数据格式
        required_fields = ["question", "answer"]
        for field in required_fields:
            if field not in dataset.column_names:
                raise ValueError(f"Missing required field: {field}")

        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def preprocess_for_training(
    dataset: Dataset,
    tokenizer: Any,
    max_length: int = 512,
) -> Dataset:
    """
    预处理数据集用于PPO训练

    处理步骤：
    1. 格式化问题为prompt
    2. 不进行tokenization（由PPOTrainer处理）
    3. 保留原始answer用于奖励计算

    参数：
        dataset: 原始数据集
        tokenizer: 分词器
        max_length: 最大长度（用于截断）

    返回：
        预处理后的Dataset
    """
    def format_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化问题为prompt

        格式：
        Question: {question}
        Answer:
        """
        question = example["question"]
        prompt = f"Question: {question}\nAnswer:"

        return {
            "query": prompt,  # PPOTrainer需要的字段
            "answer": example["answer"],  # 保留标准答案用于奖励计算
        }

    logger.info("Preprocessing dataset for training...")
    dataset = dataset.map(
        format_prompt,
        remove_columns=dataset.column_names,
        desc="Formatting prompts"
    )

    logger.info(f"Preprocessing complete: {len(dataset)} samples")
    return dataset


def preprocess_for_evaluation(
    dataset: Dataset,
    tokenizer: Any,
    max_length: int = 512,
) -> Dataset:
    """
    预处理数据集用于评估

    处理步骤：
    1. 格式化问题为prompt
    2. Tokenize输入
    3. 保留原始answer用于评估

    参数：
        dataset: 原始数据集
        tokenizer: 分词器
        max_length: 最大长度

    返回：
        预处理后的Dataset
    """
    def format_and_tokenize(example: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化并tokenize
        """
        question = example["question"]
        prompt = f"Question: {question}\nAnswer:"

        # Tokenize
        tokenized = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "question": question,
            "answer": example["answer"],
        }

    logger.info("Preprocessing dataset for evaluation...")
    dataset = dataset.map(
        format_and_tokenize,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    logger.info(f"Preprocessing complete: {len(dataset)} samples")
    return dataset


def create_train_test_split(
    dataset: Dataset,
    test_size: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    创建训练/测试分割

    参数：
        dataset: 原始数据集
        test_size: 测试集比例
        seed: 随机种子

    返回：
        DatasetDict包含train和test分割
    """
    logger.info(f"Creating train/test split (test_size={test_size})...")

    split_dataset = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
    )

    logger.info(f"Train: {len(split_dataset['train'])} samples")
    logger.info(f"Test: {len(split_dataset['test'])} samples")

    return split_dataset