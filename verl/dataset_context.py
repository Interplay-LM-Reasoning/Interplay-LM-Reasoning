import copy
import json
import logging
import math
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
from datasets import concatenate_datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def extract_answer_flexible(text: str) -> str:
    """Extract answer from generated text accommodating multiple formats."""
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    pos = text.lower().rfind("<answer>")
    if pos != -1:
        tail = text[pos + len("<answer>"):]
        m2 = re.search(r"(.*?)(<|\n|$)", tail, flags=re.DOTALL)
        if m2:
            return m2.group(1).strip()
    m3 = re.search(r"Answer:\s*(.*)", text, flags=re.IGNORECASE)
    if m3:
        answer_segment = m3.group(1).strip()
        answer_segment = re.split(r"(?:\n|<)", answer_segment, maxsplit=1)[0].strip()
        return answer_segment
    return ""



def compose_prompt(obj: dict) -> str:
    """Compose a training text string from a raw example.

    When add_special_tokens=True, format as:
      <question> <problem+question> </question> <solution> <solution_body> </solution> <answer> <answer> </answer>
    falling back to plain fields if required keys are missing.
    """
    problem = (obj.get("problem") or "").strip()
    question = (obj.get("question") or "").strip()

    pq = (problem + " " + question).strip()
    parts = []
    if pq:
        parts.extend(["<question>", pq, "</question>"])
    text = " ".join([p for p in parts if p]).strip() + " <solution>"
    solution = (obj.get("solution") or "").strip()
    answer = extract_answer_flexible(solution).rstrip(".")
    if not answer:
        answer = (obj.get("answer") or "").strip().rstrip(".")
    return text, answer



def _format_context_summary(info: dict) -> str:
    extras = {op: extra for op, extra in info.get("extras", {}).items() if extra}
    context = info["context"]
    num_ops = info["num_ops"]
    per_op = info["per_op"]
    if extras:
        extra_ops = ", ".join(str(op) for op in sorted(extras))
        return f"{context}: {per_op} per op x {num_ops} ops (+1 ops: [{extra_ops}])"
    return f"{context}: {per_op} per op x {num_ops} ops"



# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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





def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}



class CustomRLHFDataset(Dataset):
    """
    Load and preprocess data for Physics of RL.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        is_train: bool = True,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        # Resolve relative paths to absolute paths relative to project root
        resolved_data_files = []
        for fp in data_files:
            if not os.path.isabs(fp):
                # Get the project root directory (assuming this file is in verl/)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                fp = os.path.join(project_root, fp)
            resolved_data_files.append(fp)

        self.data_files = copy.deepcopy(resolved_data_files)
        self.original_data_files = copy.deepcopy(resolved_data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "problem")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.preset_path = config.get("preset_path")
        self.task_name = config.get("task_name")
        self.task_setting = config.get("task_setting")
        self.preset_config = json.load(open(self.preset_path))
        self.active_contexts: list[str] | None = None
        self.active_context_probs: list[float] | None = None
        self.is_train = is_train

        self._read_files_and_tokenize()
        self._filter_by_preset()
        self._enforce_equal_op_sampling()

    def _read_files_and_tokenize(self):
        dataframe = datasets.load_dataset(
            path="json",
            data_files=f"{self.data_files[0]}/*.jsonl",
            num_proc=16,
            )['train']
        self.dataframe: datasets.Dataset = dataframe
        before_len = len(self.dataframe)
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
        print(f"dataset len filtered: {len(self.dataframe)}, removed {before_len - len(self.dataframe)}")
        
    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor

            if processor is not None:

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    )
                    return len(processor(text=[raw_prompt])["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    
                    return len(
                        tokenizer(compose_prompt(doc), add_special_tokens=True)["input_ids"][0]
                    )
            
            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe
    def _filter_by_preset(self):
        if self.dataframe is None:
            return
        if len(self.dataframe) == 0:
            return

        if not self.task_name or not self.task_setting:
            logger.warning("Preset filtering skipped because task configuration is missing.")
            return

        try:
            dataset_name, task_config_key = self.task_name.split("/")
        except ValueError:
            logger.warning("Task name '%s' is invalid; expected '<dataset>/<config>'.", self.task_name)
            return

        try:
            setting_config = self.preset_config[dataset_name][task_config_key][self.task_setting]
        except KeyError as exc:
            logger.warning(
                "Preset configuration not found for dataset '%s', config '%s', setting '%s': %s",
                dataset_name,
                task_config_key,
                self.task_setting,
                exc,
            )
            return

        preset_seed = int(self.config.get("preset_seed", 42))

        context_ids = setting_config.get("id_context")
        context_probs = setting_config.get("context_prob")
        self.active_contexts = None
        self.active_context_probs = None
        if context_ids:
            base_dataset = self.dataframe
            context_ids = [str(context_id) for context_id in context_ids]
            if context_probs is None:
                context_probs = [1.0] * len(context_ids)
            if len(context_ids) != len(context_probs):
                raise ValueError("Lengths of 'id_context' and 'context_prob' must match in preset configuration.")

            context_probs = [float(probability) for probability in context_probs]
            total_prob = float(sum(context_probs))
            if total_prob <= 0.0:
                raise ValueError("Sum of context probabilities must be positive.")
            if not math.isclose(total_prob, 1.0, rel_tol=1e-3):
                logger.warning(
                    "Context probabilities sum to %.4f (expected 1.0). Continuing with provided ratios.",
                    total_prob,
                )

            sampled_contexts = []
            retained_contexts: list[str] = []
            retained_probs: list[float] = []

            for context_id, probability in zip(context_ids, context_probs):
                prob = float(probability)
                if prob <= 0.0:
                    logger.info("Skipping context '%s' due to non-positive probability %.4f.", context_id, prob)
                    continue
                if prob > 1.0:
                    raise ValueError(
                        f"Context probability for '{context_id}' must be in [0, 1], got {prob}."
                    )
                subset = base_dataset.filter(
                    lambda example, target=context_id: example.get("template") == target,
                    num_proc=8,
                )
                subset_len = len(subset)
                if subset_len == 0:
                    logger.warning("No samples found for context '%s'; skipping.", context_id)
                    continue
                if 0.0 < prob < 1.0:
                    split = subset.train_test_split(test_size=1 - prob, seed=preset_seed)
                    subset = split["train"]
                sampled_contexts.append(subset)
                retained_contexts.append(context_id)
                retained_probs.append(prob)
                logger.info(
                    "Retained %d samples for context '%s' with sampling ratio %.4f.",
                    len(subset),
                    context_id,
                    prob,
                )

            if not sampled_contexts:
                raise RuntimeError("Context preset filtering removed all samples.")

            self.dataframe = concatenate_datasets(sampled_contexts).shuffle(seed=preset_seed)
            prob_sum = float(sum(retained_probs))
            self.active_contexts = retained_contexts
            self.active_context_probs = [prob / prob_sum for prob in retained_probs]
            logger.info(
                "Preset filtering retained %d samples across %d contexts for setting '%s'.",
                len(self.dataframe),
                len(sampled_contexts),
                self.task_setting,
            )

    def _enforce_equal_op_sampling(self) -> None:
        if not self.is_train:
            return

        if self.dataframe is None or len(self.dataframe) == 0:
            return

        raw_task_sample = self.config.get("task_sample")
        if raw_task_sample is None:
            return

        try:
            task_sample = int(raw_task_sample)
        except (TypeError, ValueError) as exc:
            raise ValueError("task_sample must be an integer") from exc

        if task_sample <= 0:
            raise ValueError(f"task_sample must be positive, got {task_sample}")

        column_names = set(self.dataframe.column_names)
        if "op" not in column_names:
            raise ValueError("Dataset does not contain an 'op' column required for balanced sampling.")

        op_values = self.dataframe["op"]
        template_values = (
            [str(template) for template in self.dataframe["template"]]
            if "template" in column_names
            else ["__ALL__"] * len(self.dataframe)
        )

        if self.active_contexts and self.active_context_probs:
            context_keys = list(self.active_contexts)
            context_probs = list(self.active_context_probs)
        else:
            if "template" in column_names:
                context_keys = sorted({template for template in template_values})
            else:
                context_keys = ["__ALL__"]
            context_probs = [1.0 / len(context_keys)] * len(context_keys)

        if not context_keys:
            raise RuntimeError("No contexts available for balanced sampling.")

        prob_sum = float(sum(context_probs))
        if prob_sum <= 0.0:
            raise ValueError("Context probabilities must sum to a positive value.")
        context_probs = [float(prob) / prob_sum for prob in context_probs]

        context_to_ops: dict[str, dict[int, list[int]]] = {
            ctx: defaultdict(list) for ctx in context_keys
        }
        unexpected_contexts: set[str] = set()
        ignored_samples = 0

        for idx, (context_label, raw_op) in enumerate(zip(template_values, op_values)):
            if raw_op is None:
                continue
            try:
                op_int = int(raw_op)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Encountered non-integer op value: {raw_op!r}") from exc

            if context_label not in context_to_ops:
                unexpected_contexts.add(context_label)
                ignored_samples += 1
                continue

            context_to_ops[context_label][op_int].append(idx)

        if unexpected_contexts:
            logger.warning(
                "Ignored %d samples from unexpected contexts: %s",
                ignored_samples,
                sorted(unexpected_contexts),
            )

        context_info: list[dict] = []
        for ctx, prob in zip(context_keys, context_probs):
            op_index_map = context_to_ops.get(ctx, defaultdict(list))
            available_ops = {op: indices for op, indices in op_index_map.items() if indices}
            if not available_ops:
                raise RuntimeError(
                    f"Context '{ctx}' has no samples with allowed ops after filtering."
                )

            num_ops = len(available_ops)
            context_info.append(
                {
                    "context": ctx,
                    "prob": prob,
                    "ops": available_ops,
                    "num_ops": num_ops,
                    "extras": defaultdict(int),
                }
            )

        rng = np.random.default_rng(int(self.config.get("preset_seed", 42)))

        for info in context_info:
            raw_target = info["prob"] * task_sample
            per_op_float = raw_target / info["num_ops"]
            per_op_base = math.floor(per_op_float)
            info["per_op"] = per_op_base
            info["fractional"] = per_op_float - per_op_base

        allocated_total = sum(info["per_op"] * info["num_ops"] for info in context_info)
        if allocated_total > task_sample:
            raise RuntimeError(
                "Allocated more samples than requested during balancing: %d > %d"
                % (allocated_total, task_sample)
            )

        remainder = task_sample - allocated_total
        soft_remainder_used = False
        sorted_indices = []
        if remainder > 0:
            sorted_indices = sorted(
                range(len(context_info)),
                key=lambda idx: context_info[idx]["fractional"],
                reverse=True,
            )

            while remainder > 0:
                progress = False
                for idx_info in sorted_indices:
                    info = context_info[idx_info]
                    if remainder >= info["num_ops"]:
                        info["per_op"] += 1
                        remainder -= info["num_ops"]
                        progress = True
                        if remainder == 0:
                            break
                if not progress:
                    break

            if remainder > 0:
                soft_remainder_used = True
                while remainder > 0:
                    progress = False
                    for idx_info in sorted_indices:
                        info = context_info[idx_info]
                        # prefer ops with more available samples
                        ops_sorted = sorted(
                            info["ops"].items(),
                            key=lambda item: len(item[1]),
                            reverse=True,
                        )
                        for op, _ in ops_sorted:
                            if info["extras"].get(op, 0) >= 1:
                                continue
                            info["extras"][op] = 1
                            remainder -= 1
                            progress = True
                            break
                        if remainder == 0:
                            break
                    if not progress:
                        break

            if remainder > 0:
                raise ValueError(
                    "task_sample=%d cannot be satisfied even with soft balancing;"
                    " not enough distinct (context, op) pairs."
                    % task_sample
                )

        for info in context_info:
            if info["per_op"] <= 0:
                raise ValueError(
                    f"Context '{info['context']}' receives zero samples per op; increase task_sample or adjust probabilities."
                )

        selected_indices = []
        oversampled_ops: list[tuple[str, int]] = []

        for info in context_info:
            samples_per_op = info["per_op"]
            for op, indices in info["ops"].items():
                total_samples = samples_per_op + info["extras"].get(op, 0)
                indices_array = np.asarray(indices, dtype=np.int64)
                replace = indices_array.size < total_samples
                if replace:
                    oversampled_ops.append((info["context"], op))
                sampled = rng.choice(indices_array, size=total_samples, replace=replace)
                selected_indices.append(sampled)

        if not selected_indices:
            raise RuntimeError("No samples were selected during op balancing.")

        stacked_indices = np.concatenate(selected_indices)
        rng.shuffle(stacked_indices)

        final_indices = stacked_indices.tolist()
        if len(final_indices) != task_sample:
            raise RuntimeError(
                f"Balanced sampling expected {task_sample} samples but produced {len(final_indices)}."
            )

        self.dataframe = self.dataframe.select(final_indices)

        self.active_contexts = [info["context"] for info in context_info]
        self.active_context_probs = [info["prob"] for info in context_info]

        context_summary = ", ".join(
            _format_context_summary(info)
            for info in context_info
        )

        logger.info(
            "Balanced ops across %d contexts to reach %d samples. Breakdown: %s. Oversampled (context, op): %s.",
            len(context_info),
            task_sample,
            context_summary,
            oversampled_ops if oversampled_ops else "None",
        )

        if soft_remainder_used:
            adjustments = ", ".join(
                f"{info['context']}: {sorted(op for op, extra in info['extras'].items() if extra)}"
                for info in context_info
                if any(info["extras"].values())
            )
            if not adjustments:
                adjustments = "None"
            logger.warning(
                "Applied soft balancing adjustments: some contexts received +1 samples for selected ops to satisfy task_sample. Details: %s",
                adjustments,
            )

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        return example.pop(self.prompt_key)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        model_inputs = {}
        raw_prompt, answer = compose_prompt(row_dict)
        gold_solution = row_dict.get("solution")
        reward_ground_truth = {"raw_prompt": raw_prompt, "answer": answer}
        if gold_solution:
            reward_ground_truth["solution"] = gold_solution
        row_dict["reward_model"] = {"style": "rule", "ground_truth": reward_ground_truth}
        row_dict['data_source'] = f"context-10B/{row_dict['op']}-{row_dict['template']}"
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = raw_prompt

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        extra_info = row_dict.get("extra_info") or {}
        if not isinstance(extra_info, dict):
            extra_info = dict(extra_info)
        if gold_solution:
            extra_info.setdefault("gold_solution", gold_solution)
        if answer:
            extra_info.setdefault("gold_answer", answer)
        if row_dict.get("op") is not None:
            extra_info.setdefault("op", row_dict.get("op"))
        if row_dict.get("id") is not None:
            extra_info.setdefault("example_id", row_dict.get("id"))
        if row_dict.get("template"):
            extra_info.setdefault("template", row_dict.get("template"))
        row_dict["extra_info"] = extra_info
        index = extra_info.get("index", 0)
        tools_kwargs = extra_info.get("tools_kwargs", {})
        interaction_kwargs = extra_info.get("interaction_kwargs", {})
        need_tools_kwargs = extra_info.get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()


if __name__ == "__main__":
    ds = CustomRLHFDataset(
        data_files=["data/context/heldout"],
        tokenizer=AutoTokenizer.from_pretrained("/home/zhangchenlong/reasoning_exp/physics_of_rl/LLaMA-Factory/saves/difficulty/diff2_14-tok5B-lr2e4-bs250k-schedcos-minlr3e-5-250910"),
        config=DictConfig(
            {
                "cache_dir": "/home/zhangchenlong/reasoning_exp/physics_of_rl/LLaMA-Factory/saves/difficulty/diff2_14-tok5B-lr2e4-bs250k-schedcos-minlr3e-5-250910",
                "prompt_key": "problem",
                "max_prompt_length": 2048,
                "truncation": "right",
                "filter_overlong_prompts": True,
                "filter_prompts": True,
                "use_shm": False,
                "need_tools_kwargs": True,
                "return_raw_chat": True,
                "return_full_prompt": True,
                "preset_path": "data/PRESET.json",
                "task_name": "context-10B/context_level_rl",
                "task_setting": "contextzoo_0.5zoo_0.5teacher",
                "task_sample": 2000000

            }
        ),
    )
    print(ds[0])
    print(ds[0]["reward_model"])
