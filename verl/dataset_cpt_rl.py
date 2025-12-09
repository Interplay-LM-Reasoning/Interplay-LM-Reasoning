import copy
import json
import logging
import os
import re
from collections import Counter, defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from datasets import concatenate_datasets

import math

logger = logging.getLogger(__name__)



def extract_answer_flexible(text: str) -> str:
    """Extract answer from generated text.
    Priority:
      1) Between <answer>...</answer>
      2) If <answer> exists but no closing tag, take text after it up to a tag start '<' or newline.
    """
    # 1) Proper tags
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 2) Open tag only
    pos = text.lower().rfind("<answer>")
    if pos != -1:
        tail = text[pos + len("<answer>"):]
        # find the numeric text up to the next tag or at the end, if not found return ""
        m2 = re.search(r"(.*?)(<|\n|$)", tail, flags=re.DOTALL)
        if m2:
            return m2.group(1).strip()
    # 3) "Answer:" pattern as a fallback
    m3 = re.search(r"Answer:\s*(.*)", text, flags=re.IGNORECASE)
    if m3:
        answer_segment = m3.group(1).strip()
        # stop at newline or start of another tag if present
        answer_segment = re.split(r"(?:\n|<)", answer_segment, maxsplit=1)[0].strip()
        return answer_segment
    return ""



def compute_score(solution_str: str, ground_truth: str, data_source=None, extra_info=None) -> bool:
    answer = extract_answer_flexible(solution_str)
    ret_score = answer.strip().rstrip(".") == ground_truth.strip().rstrip(".")  

    return ret_score



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


def _examples_from_batch(batch: dict) -> list[dict]:
    """
    Convert a batched dict from datasets.map into a list of example dictionaries.
    """
    if not batch:
        return []
    keys = list(batch.keys())
    if not keys:
        return []
    batch_size = len(batch[keys[0]])
    examples: list[dict] = []
    for row_idx in range(batch_size):
        row = {key: batch[key][row_idx] for key in keys}
        examples.append(row)
    return examples


def _compute_prompt_lengths_batch(batch: dict, tokenizer: PreTrainedTokenizer) -> dict:
    """
    Batched helper that builds prompts, tokenizes them, and returns their lengths.
    Defined at module scope so it can be pickled for multiprocessing.
    """
    examples = _examples_from_batch(batch)
    if not examples:
        return {"_prompt_len": []}

    prompts: list[str] = []
    for example in examples:
        prompt_text, _ = compose_prompt(example)
        prompts.append(prompt_text)

    encodings = tokenizer(
        prompts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    lengths = [len(ids) for ids in encodings["input_ids"]]
    return {"_prompt_len": lengths}



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
        self.cpt_epoch = self._coerce_float(config.get("cpt-epoch", 1.0), "cpt-epoch")
        self.is_train = is_train
        total_budget_value = config.get("total_budget", 122000)
        self.total_budget = int(self._coerce_float(total_budget_value, "total_budget"))

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "problem")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = max(1, min(self.num_workers, os.cpu_count()))
        batch_size_value = config.get("filter_prompt_batch_size", 64)
        try:
            self.filter_prompt_batch_size = max(1, int(batch_size_value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Configuration value 'filter_prompt_batch_size' must be an integer, got {batch_size_value!r}"
            ) from exc
        self.use_shm = config.get("use_shm", False)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.preset_path = config.get("preset_path")
        self.task_name = config.get("task_name")
        self.task_setting = config.get("task_setting")
        preset_config_override = config.get("preset_config")
        self.preset_config = None
        if preset_config_override is not None:
            self.preset_config = copy.deepcopy(preset_config_override)
        elif self.preset_path:
            try:
                with open(self.preset_path, "r", encoding="utf-8") as preset_fp:
                    self.preset_config = json.load(preset_fp)
            except FileNotFoundError:
                logger.warning("Preset path '%s' not found; preset filtering disabled.", self.preset_path)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse preset file '%s': %s", self.preset_path, exc)
        self.active_contexts: list[str] | None = None
        self.active_context_probs: list[float] | None = None
        self.id_max_op = config.get("id_max_op")
        self._read_files_and_tokenize()
        self._filter_by_preset()

    def _log_per_op_counts(self, dataframe: datasets.Dataset | None, note: str | None = None):
        if dataframe is None:
            return
        if "op" not in dataframe.column_names:
            return
        ops = dataframe["op"]
        if ops is None:
            return
        ops_list = list(ops)
        if not ops_list:
            return
        counter = Counter(str(op) for op in ops_list)
        total = sum(counter.values())
        if total == 0:
            return
        sorted_items = sorted(counter.items(), key=lambda item: item[0])
        preview = ", ".join(f"{op}: {count}" for op, count in sorted_items)
        suffix = f" ({note})" if note else ""
        logger.info("Per-op counts%s [total=%d]: %s", suffix, total, preview)

    @staticmethod
    def _coerce_float(value, name: str) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.split("#", 1)[0].strip()
            if not cleaned:
                raise ValueError(f"Configuration value '{name}' is empty after stripping comments.")
            try:
                return float(cleaned)
            except ValueError as exc:
                raise ValueError(f"Configuration value '{name}' must be numeric, got {value!r}") from exc
        raise TypeError(f"Configuration value '{name}' must be numeric, got {type(value).__name__}.")

    def _read_files_and_tokenize(self):
        dataframes = []
        for fp in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("json", data_files=fp)['train']
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        before_len = len(self.dataframe)
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
        self._log_per_op_counts(self.dataframe, note="post-filtering")
        print(f"dataset len filtered: {len(self.dataframe)}, removed {before_len - len(self.dataframe)}")
    
    def _filter_by_preset(self):
        raw_rest_budget = self.config.get("rest_budget")
        if raw_rest_budget is None:
            rest_budget_float = (1 - self.cpt_epoch) * self.total_budget
        else:
            rest_budget_float = self._coerce_float(raw_rest_budget, "rest_budget")
        self.rest_budget = max(0, int(math.floor(rest_budget_float)))
        if self.dataframe is None or self.rest_budget <= 0:
            return
        if len(self.dataframe) == 0:
            return

        if not self.preset_config:
            logger.warning("Preset filtering skipped because preset configuration is missing.")
            return

        if not self.task_name or not self.task_setting:
            logger.warning("Preset filtering skipped because task configuration is missing.")
            return

        before_filter_len = len(self.dataframe)
        logger.info(
            "Preset filtering starting with %d samples (rest_budget=%d).",
            before_filter_len,
            self.rest_budget,
        )

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

        id_ops = setting_config.get("id_op")
        op_probs = setting_config.get("op_prob")
        self.active_contexts = None
        self.active_context_probs = None
        if id_ops:
            base_dataset = self.dataframe
            id_ops = [str(id_op) for id_op in id_ops]
            if op_probs is None:
                op_probs = [1.0] * len(id_ops)
            if len(id_ops) != len(op_probs):
                raise ValueError("Lengths of 'id_op' and 'op_prob' must match in preset configuration.")

            op_probs = [float(probability) for probability in op_probs]
            total_prob = float(sum(op_probs))
            if total_prob <= 0.0:
                raise ValueError("Sum of operation probabilities must be positive.")
            if not math.isclose(total_prob, 1.0, rel_tol=1e-3):
                logger.warning(
                    " operation probabilities sum to %.4f (expected 1.0). Continuing with provided ratios.",
                    total_prob,
                )

            op_entries: list[dict] = []
            retained_contexts: list[str] = []
            retained_probs: list[float] = []

            for op_id, probability in zip(id_ops, op_probs):
                prob = float(probability)
                if prob <= 0.0:
                    logger.info("Skipping context '%s' due to non-positive probability %.4f.", op_id, prob)
                    continue
                if prob > 1.0:
                    raise ValueError(
                        f"Operation probability for '{op_id}' must be in [0, 1], got {prob}."
                    )
                subset = base_dataset.filter(
                    # Cast dataset value to str to handle numeric `op` columns.
                    lambda example, target=op_id: str(example.get("op")) == target,
                    num_proc=8,
                )
                subset_len = len(subset)
                if subset_len == 0:
                    logger.warning("No samples found for context '%s'; skipping.", op_id)
                    continue
                op_entries.append(
                    {
                        "op_id": op_id,
                        "prob": prob,
                        "dataset": subset,
                        "length": subset_len,
                    }
                )
                retained_contexts.append(op_id)
                retained_probs.append(prob)
                logger.info(
                    "Cached %d candidates for context '%s' with requested probability %.4f.",
                    subset_len,
                    op_id,
                    prob,
                )

            if not op_entries:
                raise RuntimeError("Context preset filtering removed all samples.")

            prob_sum = float(sum(entry["prob"] for entry in op_entries))
            if prob_sum <= 0.0:
                raise ValueError("Sum of retained operation probabilities must be positive.")
            normalized_probs = [entry["prob"] / prob_sum for entry in op_entries]

            base_counts = []
            fractional_parts = []
            for norm_prob in normalized_probs:
                raw_target = norm_prob * self.rest_budget
                base = math.floor(raw_target)
                base_counts.append(base)
                fractional_parts.append(raw_target - base)

            allocated = sum(base_counts)
            remainder = self.rest_budget - allocated
            if remainder > 0:
                order = sorted(
                    range(len(fractional_parts)),
                    key=lambda idx: (-fractional_parts[idx], idx),
                )
                while remainder > 0 and order:
                    for idx in order:
                        if remainder == 0:
                            break
                        base_counts[idx] += 1
                        remainder -= 1

            rng = np.random.default_rng(preset_seed)
            sampled_contexts = []
            oversampled_ops: list[str] = []

            for entry, target_count in zip(op_entries, base_counts):
                if target_count <= 0:
                    continue

                subset = entry["dataset"]
                subset_len = entry["length"]
                replace = self.is_train and target_count > subset_len
                if not self.is_train and target_count > subset_len:
                    logger.info(
                        "Validation preset requested %d samples for context '%s' but only %d available; skipping replication.",
                        target_count,
                        entry["op_id"],
                        subset_len,
                    )
                effective_count = target_count if self.is_train else min(target_count, subset_len)

                if subset_len == 0:
                    continue
                if effective_count <= 0:
                    continue

                indices = rng.choice(subset_len, size=effective_count, replace=replace)
                if not replace:
                    indices = np.sort(indices)
                selected_subset = subset.select(indices.tolist())
                sampled_contexts.append(selected_subset)
                if replace:
                    oversampled_ops.append(entry["op_id"])

            if not sampled_contexts:
                raise RuntimeError("Context preset filtering removed all samples.")

            self.dataframe = concatenate_datasets(sampled_contexts).shuffle(seed=preset_seed)
            self.active_contexts = retained_contexts
            self.active_context_probs = [prob / prob_sum for prob in retained_probs]
            after_filter_len = len(self.dataframe)
            logger.info(
                "Preset filtering sampled %d examples (rest_budget=%d) across %d contexts for setting '%s'. Oversampled contexts: %s.",
                after_filter_len,
                self.rest_budget,
                len(sampled_contexts),
                self.task_setting,
                oversampled_ops if oversampled_ops else "None",
            )
            logger.info(
                "Preset filtering finished with %d samples (removed %d).",
                after_filter_len,
                before_filter_len - after_filter_len,
            )
            self._log_per_op_counts(self.dataframe, note=f"preset setting {self.task_setting}")


    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if not self.filter_overlong_prompts or not self.filter_prompts or dataframe is None:
            return dataframe

        processor = self.processor
        if processor is None:
            dataframe = dataframe.map(
                _compute_prompt_lengths_batch,
                batched=True,
                batch_size=self.filter_prompt_batch_size,
                num_proc=self.num_workers,
                load_from_cache_file=False,
                fn_kwargs={"tokenizer": self.tokenizer},
                desc="Computing prompt lengths for filtering",
            )
            dataframe = dataframe.filter(
                lambda prompt_len: prompt_len <= self.max_prompt_length,
                input_columns="_prompt_len",
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )
            dataframe = dataframe.remove_columns("_prompt_len")
        else:

            def doc2len(doc) -> int:
                messages = self._build_messages(doc)
                raw_prompt = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )
                return len(processor(text=[raw_prompt])["input_ids"][0])

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=1,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

        print(f"filter dataset len: {len(dataframe)}")
        return dataframe

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
        row_dict['reward_model'] = {"style": "rule", "ground_truth": answer}
        row_dict['data_source'] = f"op-difficulty-cpt-rl/{row_dict['op']}"
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
        gold_solution = row_dict.get("solution")
        if gold_solution:
            extra_info.setdefault("gold_solution", gold_solution)
        if answer:
            extra_info.setdefault("gold_answer", answer)
        if row_dict.get("op") is not None:
            extra_info.setdefault("op", row_dict.get("op"))
        if row_dict.get("id") is not None:
            extra_info.setdefault("example_id", row_dict.get("id"))
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
        data_files=["data/composition/heldout/op2-50k.jsonl", "data/composition/heldout/op3-50k.jsonl"],
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
            }
        ),
    )
    print(ds[0])
    print(ds[0]["reward_model"])
