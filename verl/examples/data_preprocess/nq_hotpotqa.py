
import datasets
import argparse
import os
from huggingface_hub import snapshot_download

# 只在需要时才 import pyarrow，避免环境没装时报错不清晰
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    pa = None
    pq = None


def make_prefix(dp):
    question = dp['question']

    prefix = f"""Answer the given question. \
You first think about the reasoning process in the mind and then provides the user with the answer. \
The reasoning process is enclosed within <think> and </think>. After your thinking, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    return prefix


def load_local_parquet_safely(data_dir: str):
    """
    从本地 parquet 目录安全读取 train/test，只读最需要的列，避免 schema cast 报错。
    默认期望存在：
      - {data_dir}/train.parquet
      - {data_dir}/test.parquet
    """
    if pq is None:
        raise RuntimeError(
            "pyarrow is not available in your env. Please install it: pip install pyarrow"
        )

    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    # 只读取必要列，直接绕开复杂 struct 字段
    needed_cols = ["question", "golden_answers"]

    train_table = pq.read_table(train_path, columns=needed_cols)
    test_table = pq.read_table(test_path, columns=needed_cols)

    train_ds = datasets.Dataset(train_table)
    test_ds = datasets.Dataset(test_table)

    return datasets.DatasetDict({"train": train_ds, "test": test_ds})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_hotpot_searchr1')
    parser.add_argument('--cache_dir', default="~/.cache")
    parser.add_argument("--hf_name", default="PeterJinGo/nq_hotpotqa_train", help="HF dataset name used when cache_dir is empty or parquet files not found")
    parser.add_argument('--data_source', default='nq_hotpotqa_searchr1')
    parser.add_argument('--num_sample', default=0, type=int, help="Sample num_sample QA pairs for training set if num_samples > 0")
    parser.add_argument('--num_sample_test', default=0, type=int, help="Sample num_sample_test QA pairs for training set if num_samples > 0")
    parser.add_argument("--seed", default=54, type=int)

    args = parser.parse_args()

    print("[INFO] Loading dataset")
    # 1) 优先尝试从本地 parquet 读取（安全模式：只读必要列）
    dataset = None
    cache_dir = os.path.expanduser(args.cache_dir) if args.cache_dir else ""
    if cache_dir:
        train_path = os.path.join(cache_dir, "train.parquet")
        test_path = os.path.join(cache_dir, "test.parquet")
        if os.path.exists(train_path) and os.path.exists(test_path):
            print(f"[INFO] Detected local parquet in: {cache_dir}")
            dataset = load_local_parquet_safely(cache_dir)
        else:
            print(f"[WARN] cache_dir provided but train/test parquet not found in: {cache_dir}")

    # 2) 如果本地不可用，就从 HF 加载
    if dataset is None:
        print(f"[INFO] Loading from HuggingFace: {args.hf_name}")
        # Try to download parquet files directly and load them safely to avoid schema issues in other columns
        try:
            download_path = snapshot_download(repo_id=args.hf_name, repo_type='dataset', allow_patterns=["*.parquet"])
            print(f"[INFO] Downloaded to: {download_path}")
            dataset = load_local_parquet_safely(download_path)
        except Exception as e:
            print(f"[WARN] Failed to load via snapshot_download + safe load: {e}")
            print("[INFO] Fallback to standard datasets.load_dataset")
            dataset = datasets.load_dataset(args.hf_name, cache_dir=cache_dir or None)

            # 保险：如果 HF 数据集也包含复杂列，你又只需要两列，就删掉其他列
            keep_cols = {"question", "golden_answers"}
            for split in dataset.keys():
                rm_cols = [c for c in dataset[split].column_names if c not in keep_cols]
                if rm_cols:
                    dataset[split] = dataset[split].remove_columns(rm_cols)
    
    data_source = args.data_source if args.data_source else "nq_hotpotqa_searchr1"
    
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # ===================== 新增：随机抽样 =====================
    if args.num_sample > 0 and args.num_sample < len(train_dataset):
        train_dataset = train_dataset.shuffle(seed=args.seed).select(
            range(args.num_sample)
        )
        print(f"[INFO] Randomly sampled {args.num_sample} training examples.")
        train_file = f"train_{args.num_sample}.parquet"
    else:
        train_file = f"train.parquet"
    # ========================================================
    
    # ===================== 新增：随机抽样 =====================
    if args.num_sample_test > 0 and args.num_sample_test < len(test_dataset):
        test_dataset = test_dataset.shuffle(seed=args.seed).select(
            range(args.num_sample_test)
        )
        print(f"[INFO] Randomly sampled {args.num_sample_test} test examples.")
        test_file = f"test_{args.num_sample_test}.parquet"
    else:
        test_file = f"test.parquet"
    # ========================================================

    def make_map_fn(split):

        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example)
            solution = {
                "target": example['golden_answers'],
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    print(f"[INFO] Number of training data: {len(train_dataset)}")
    print(f"[INFO] Number of test data: {len(test_dataset)}")
    print(train_dataset[0])

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, train_file))
    test_dataset.to_parquet(os.path.join(local_dir, test_file))    



