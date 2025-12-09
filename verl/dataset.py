import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

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
        self.id_max_op = config.get("id_max_op")
        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        dataframes = []
        for fp in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("json", data_files=fp)['train']
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
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
        row_dict['data_source'] = f"difficulty-5B/{row_dict['op']}"
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
