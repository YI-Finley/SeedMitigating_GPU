# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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
"""
Preprocess the CRAG dataset to parquet format
"""

import re
import os
import datasets
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from transformers import AutoTokenizer

from verl.utils.hdfs_io import copy, makedirs
import argparse
from tqdm import tqdm


# add a row to each data item that represents a unique id
def make_map_fn(split, tokenizer, ook_ids, max_seq_length, max_new_tokens, extract_answer_box=False):

    def process_fn(example, idx):

        is_ook = example['interaction_id'] in ook_ids

        example['query'] = example['query'].strip()
        if example['query'][-1] != '?':
            example['query'] += '?'
        
        # if extract_answer_box:
        #     # extract the answer box \\boxed{...} from the prompt
        #     completion = example['completion'][0]['content']
        #     answer_box = re.search(r'\\boxed{(.*?)}', completion)
        #     if answer_box:
        #         answer = answer_box.group(1)
        #     else:
        #         raise ValueError(f"No answer box found in the completion: {completion}")

        #     solution = {
        #         "target": [answer],
        #     }
        # else:
        #     solution = {
        #         "target": [example['answer']] + example['alt_ans'],
        #     }
        
        solution = {
            "problem": example["query"],
            "target": [example['answer']] + example['alt_ans'],
            "out_of_knowledge": is_ook,
        }

        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": format_prompt(example['prompt'][0]['content'], tokenizer, max_seq_length, max_new_tokens),
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

        example.pop('completion')

        return data

    return process_fn

# Function to analyze tokenization lengths
def analyze_tokenization_lengths(dataset, split_name):
    print(f"\n=== Tokenization Analysis for {split_name} ===")
    
    prompt_lengths = []
    
    for i, example in tqdm(enumerate(dataset)):
        # Get the prompt content
        prompt_content = example['prompt'][0]['content']

        # Tokenize the prompt
        tokens = tokenizer.encode(prompt_content, add_special_tokens=False)
        prompt_lengths.append(len(tokens))
        
        # Print first few examples for debugging
        if i < 3:
            print(f"\n\n>>>>>>>>>>>>>>> Example {i}:")
            print(f"  Prompt: {prompt_content[:100]}...")
            print(f"  Token length: {len(tokens)}")
            print(f"  Tokens: {tokens[:10]}...")
            print()

    # Calculate statistics
    prompt_lengths = np.array(prompt_lengths)
    avg_length = np.mean(prompt_lengths)
    max_length = np.max(prompt_lengths)
    min_length = np.min(prompt_lengths)
    median_length = np.median(prompt_lengths)
    std_length = np.std(prompt_lengths)
    
    print(f"Prompt Length Statistics for {split_name}:")
    print(f"  Average length: {avg_length:.2f} tokens")
    print(f"  Maximum length: {max_length} tokens")
    print(f"  Minimum length: {min_length} tokens")
    print(f"  Median length: {median_length:.2f} tokens")
    print(f"  Standard deviation: {std_length:.2f} tokens")
    print(f"  Total examples: {len(prompt_lengths)}")
    
    # Length distribution analysis
    print(f"\nLength Distribution for {split_name}:")
    length_counts = Counter(prompt_lengths)
    sorted_lengths = sorted(length_counts.items())
    
    # Print distribution in ranges
    ranges = [(0, 100), (101, 200), (201, 300), (301, 400), (401, 500), (501, 1000), (1001, float('inf'))]
    for start, end in ranges:
        count = sum(1 for length in prompt_lengths if start <= length <= end)
        percentage = (count / len(prompt_lengths)) * 100
        print(f"  {start}-{end if end != float('inf') else '∞'} tokens: {count} examples ({percentage:.1f}%)")
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    plt.hist(prompt_lengths, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(avg_length, color='red', linestyle='--', label=f'Mean: {avg_length:.1f}')
    plt.axvline(median_length, color='green', linestyle='--', label=f'Median: {median_length:.1f}')
    plt.xlabel('Prompt Length (tokens)')
    plt.ylabel('Frequency')
    plt.title(f'Prompt Length Distribution - {split_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(args.local_dir, f'{split_name}_prompt_length_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Length distribution plot saved to: {plot_path}")
    
    return {
        'avg_length': float(avg_length),
        'max_length': int(max_length),
        'min_length': int(min_length),
        'median_length': int(median_length),
        'std_length': float(std_length),
        'lengths': prompt_lengths.tolist()
    }
    

def format_prompt(prompt, tokenizer, max_seq_length=16384, max_new_tokens=2048):


    # Use tokenizer for accurate token-based truncation (consistent with batch_generate_answer)
    # Tokenize the final message to check length
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    max_prompt_tokens = max_seq_length - max_new_tokens

    final_message = prompt
    original_tokens = len(prompt_tokens)
    
    if len(prompt_tokens) > max_prompt_tokens:
        # Truncate the prompt to fit within the context window (consistent with batch_generate_answer)
        # Preserve the last 9 tokens like in batch_generate_answer
        # print(f"Prompt tokens:\n{tokenizer.convert_ids_to_tokens(prompt_tokens)}")
        prompt_tokens = prompt_tokens[:(max_prompt_tokens-4)] + prompt_tokens[-4:]
        # print(f"Prompt tokens after truncation:\n{tokenizer.convert_ids_to_tokens(prompt_tokens)}")
        final_message = tokenizer.decode(prompt_tokens, skip_special_tokens=True)


    # Return token count for accurate length tracking
    # final_tokens = tokenizer.encode(final_message, add_special_tokens=False)

    # print(f"Prompt truncated from {original_tokens} tokens to {len(final_tokens)} tokens")
    # print(f">>>>> Prompt (before truncation):\n\n{prompt}")
    # print(f">>>>> Prompt (after truncation):\n\n{final_message}")

    return final_message


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/fsx_0/user/zhepei/projects/verl/data/crag')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'searchR1_crag' # check verl/verl/utils/reward_score/__init__.py for predefined reward function

    ook_questions_json = "/fsx_0/user/zhepei/projects/RAG-Online-DPO/Datasets/CRAG/Uncertain/Llama-3.1-8B-Instruct/Naive-SFT/train_uncertain_questions.json"

    ook_ids = []
    if os.path.exists(ook_questions_json):
        with open(ook_questions_json, "r") as f:
            data = json.load(f)
            for item in data:
                ook_ids.append(item['interaction_id'])
    
    print(f"Found {len(ook_ids)} out-of-knowledge questions")


    dataset_vanilla = datasets.load_dataset('weizhepei/CRAG')

    # dataset_uncertain = datasets.load_dataset('weizhepei/CRAG-Uncertain')

    train_dataset = dataset_vanilla['train']
    test_dataset = dataset_vanilla['test']

    # Load Llama 3.1-8B-Instruct tokenizer
    print("Loading Llama 3.1-8B-Instruct tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = train_dataset.map(function=make_map_fn('train', tokenizer, ook_ids, max_seq_length=16384, max_new_tokens=2048, extract_answer_box=True), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test', tokenizer, ook_ids, max_seq_length=16384, max_new_tokens=2048, extract_answer_box=False), with_indices=True)

    # Analyze both datasets
    train_stats = analyze_tokenization_lengths(train_dataset, 'train')
    test_stats = analyze_tokenization_lengths(test_dataset, 'test')
    
    # Save statistics to JSON
    stats_data = {
        'train': train_stats,
        'test': test_stats,
        'tokenizer_info': {
            'model': 'meta-llama/Llama-3.1-8B-Instruct',
            'vocab_size': tokenizer.vocab_size,
            'model_max_length': tokenizer.model_max_length
        }
    }
    
    stats_path = os.path.join(args.local_dir, 'tokenization_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_data, f, indent=2)
    print(f"\nTokenization statistics saved to: {stats_path}")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # Save one example as JSON for reference
    example = train_dataset[0]
    # Convert any remaining Timestamp objects to strings for JSON serialization
    def convert_timestamps(obj):
        if hasattr(obj, 'isoformat'):  # Check if it's a Timestamp/datetime object
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_timestamps(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_timestamps(item) for item in obj]
        else:
            return obj
    
    example = convert_timestamps(example)
    all_examples_train = [convert_timestamps(example) for example in train_dataset]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=4)

    with open(os.path.join(local_dir, "train_all_examples.json"), "w") as f:
        json.dump(all_examples_train, f, indent=4)

    
    example = test_dataset[0]
    example = convert_timestamps(example)
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=4)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)