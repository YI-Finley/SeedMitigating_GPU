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

from collections import defaultdict
from typing import Any

import torch
import re
from typing import Optional

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


def is_valid_sequence(text: str):
    # We validate only <think>...</think><answer>...</answer>
    content = text  # no need to locate <|im_start|>assistant

    # 1) Balanced tags check (only think/answer)
    for tag in ["think", "answer"]:
        opening_count = len(re.findall(fr"<{tag}>", content))
        closing_count = len(re.findall(fr"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    # 2) Split by think/answer tags (keep tags in parts)
    split_pattern = r"(</?(?:think|answer)>)"
    parts = re.split(split_pattern, content)

    state = "start"  # start -> in_think -> after_think -> in_answer -> end

    for part in parts:
        if not part.strip():
            continue

        # Tag token?
        if re.fullmatch(r"</?(?:think|answer)>", part):
            if part == "<think>" and state == "start":
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # Plain text
            if state in ["in_think", "in_answer"]:
                pass  # content allowed inside tags
            elif state in ["start", "after_think"]:
                # Only whitespace allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"



def extract_solution(solution_str: str):
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if len(matches) == 0:
        return None
    return matches[-1].group(1).strip()


def _answers_equal(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    sa, sb = str(a).strip(), str(b).strip()
    if sa == '' or sb == '':
        return False
    return sa == sb


@register("fact_naive")
class FactNaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        num_valid_response = 0
        
        N = len(data)
        total_format_reward, total_ans_reward = 0.0, 0.0
        total_reward = 0.0
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            is_valid_str, _ = is_valid_sequence(response_str)
            format_reward = 0.0
            if is_valid_str:
                format_reward = 1.0
                num_valid_response += 1

            ground_truth_list = data_item.non_tensor_batch["reward_model"]["ground_truth"]['target']
            predict_ans = extract_solution(response_str)
            
            answer_reward = 0.0
            if predict_ans is not None:
                for gt in ground_truth_list:
                    if _answers_equal(predict_ans, gt):
                        answer_reward = 1.0
                        break 
            
            reward = format_reward + answer_reward

            reward_tensor[i, valid_response_length - 1] = reward
            
            total_format_reward += format_reward
            total_ans_reward += answer_reward
            total_reward += reward
        
        avg_format_reward = total_format_reward / N
        avg_ans_reward = total_ans_reward / N
        avg_reward = total_reward / N
        
        metrics = {
            "reward/avg_format_reward": avg_format_reward,
            "reward/avg_ans_reward": avg_ans_reward,
            "reward/avg_reward": avg_reward,
            "reward/num_valid_response": num_valid_response
        }

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {"metrics": metrics},
            }
        else:
            return reward_tensor
