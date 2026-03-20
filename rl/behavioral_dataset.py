"""
自定义数据集封装:支持字符串 prompt，并按策略注入输出格式指令。
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List

import numpy as np

try:
    from prompt_templates import (
        explicit_risk_qa_prompt,
        response_level_qa_prompt,
    )
except ModuleNotFoundError:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from prompt_templates import (
        explicit_risk_qa_prompt,
        response_level_qa_prompt,
    )
from verl.utils.dataset.rl_dataset import RLHFDataset


class BehavioralCalibrationDataset(RLHFDataset):
    """为行为校准训练定制的 RLHFDataset。"""

    def _get_math_data_sources(self) -> set[str]:
        default_sources = {
            "dapo_math",
            "math_dapo",
            "aime",
            "aime_2024",
            "aime_2025",
            "beyondaime",
            "gsm8k",
            "gsm8k_train",
            "gsm8k_test",
        }
        extra = self.config.get("math_data_sources", None)
        if extra:
            if isinstance(extra, (list, tuple, set)):
                default_sources.update([str(x).lower() for x in extra])
            else:
                default_sources.add(str(extra).lower())
        return {s.lower() for s in default_sources}

    @staticmethod
    def _is_math_data_source(data_source: str | None, math_sources: set[str]) -> bool:
        ds = str(data_source or "").strip().lower()
        if not ds:
            return False
        if ds in math_sources:
            return True
        if ds.startswith("aime"):
            return True
        if ds.startswith("gsm8k"):
            return True
        return False

    def _filter_out_math_datasets(self) -> None:
        if not self.config.get("remove_math_datasets", True):
            return
        if not hasattr(self, "dataframe"):
            return
        if "data_source" not in self.dataframe.column_names:
            return
        math_sources = self._get_math_data_sources()

        def keep_fn(doc, math_sources):
            return not BehavioralCalibrationDataset._is_math_data_source(doc.get("data_source"), math_sources)

        before = len(self.dataframe)
        self.dataframe = self.dataframe.filter(
            keep_fn,
            fn_kwargs={"math_sources": math_sources},
            num_proc=self.num_workers,
            desc="Filtering math datasets",
        )
        after = len(self.dataframe)

    def _read_files_and_tokenize(self):
        super()._read_files_and_tokenize()
        self._filter_out_math_datasets()

    @staticmethod
    def _is_qa_example(example: Dict[str, Any], prompt_text: str) -> bool:
        data_source = str(example.get("data_source") or "").lower()
        if data_source:
            qa_sources = {
                "simpleqa",
                "simpleqa_gtgrpo",
                "hotpotqa",
                "hotpot_qa",
                "nq_hotpotqa_searchr1",
                "flashrag_popqa",
                "flashrag_nq",
                "flashrag_hotpotqa",
                "flashrag_musique",
                "flashrag_2wikimultihopqa",
                "flashrag_triviaqa",
                "flashrag_bamboogle",
                "popqa",
                "nq",
                "musique",
                "2wikimultihopqa",
                "triviaqa",
                "bamboogle",
            }
            if data_source in qa_sources:
                return True
        text = (prompt_text or "").lower()
        if "question:" in text or "question:" in text:
            return True
        if text.strip().endswith("?"):
            return True
        return False

    @staticmethod
    def _strip_prompt_to_question(text: str) -> str:
        if not isinstance(text, str):
            return text
        cleaned = text.strip()
        for marker in ("Question:", "Question:", "Problem:", "Problem:"):
            if marker in cleaned:
                cleaned = cleaned.split(marker)[-1].strip()
                if "\n" in cleaned:
                    cleaned = cleaned.split("\n")[0].strip()
                break
        return cleaned

    def _format_prompt(self, question: str) -> str:
        strategy = self.config.get("strategy", "baseline")
        question = question.strip()

        if strategy in ("verbalized_brier", "verbalized_ce"):
            return response_level_qa_prompt(question)

        # baseline / ppo_value / fallback
        return (
            "Question: "
            + question
            + "\n\nPlease put your final short answer inside <answer>...</answer>.\nAnswer:"
        )

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        reward_model = example.get("reward_model")
        if not isinstance(reward_model, dict):
            reward_model = {}

        if reward_model.get("ground_truth") is None:
            gt = None
            golden_answers = example.get("golden_answers")
            if isinstance(golden_answers, (list, tuple)) and len(golden_answers) > 0:
                first = golden_answers[0]
                if isinstance(first, str) and first.strip():
                    gt = first.strip()
                elif isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], str):
                    gt = first[0].strip()
            answer_text = example.get("answer")
            if gt is None and isinstance(answer_text, str) and answer_text.strip():
                gt = answer_text.strip()
            reward_model["ground_truth"] = gt if gt is not None else ""

        example["reward_model"] = reward_model

        prompt = example.pop(self.prompt_key, None)
        strategy = self.config.get("strategy", "baseline")

        prompt_text = None
        if isinstance(prompt, list) and len(prompt) > 0:
            for msg in reversed(prompt):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content")
                    if isinstance(content, str):
                        prompt_text = content
                        break
        question_text = example.get("question")
        if isinstance(question_text, str) and question_text.strip():
            prompt_text = question_text.strip()
        
        if prompt_text is None:
            prompt_text = "" if prompt is None else str(prompt)
        
        prompt_text = self._strip_prompt_to_question(str(prompt_text))

        if strategy == "explicit_risk":
            t_override = self.config.get("explicit_risk_t", None)
            if t_override is None and isinstance(example.get("extra_info"), dict):
                t_override = example.get("extra_info", {}).get("explicit_risk_t", None)
            
            if t_override is not None:
                try:
                    t = float(t_override)
                except Exception:
                    t = 0.5
                t = max(0.0, min(1.0, t))
            
            else:
                t_raw = float(np.random.uniform(0.0, 1.0))
                t = float(f"{t_raw:.3f}")
            
            # r = (1.0 + t) / (1.0 - t) if t < 1.0 else float("inf")
            r = t / (1.0 - t) if t < 1.0 else float("inf")

            extra_info = example.get("extra_info") or {}
            extra_info["confidence"] = t
            example["extra_info"] = extra_info

            content = explicit_risk_qa_prompt(str(prompt_text).strip(), t, r)
            
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": self._format_prompt(str(prompt_text))}]


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
