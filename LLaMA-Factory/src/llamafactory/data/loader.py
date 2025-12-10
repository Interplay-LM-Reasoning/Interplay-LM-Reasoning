# Copyright 2025 the LlamaFactory team.
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

import json
import math
import os
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from datasets import Features, Value

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import check_version, has_tokenized_data
from .converter import align_dataset
from .data_utils import get_dataset_module, merge_dataset, read_cloud_json, split_dataset
from .parser import get_dataset_list
from .processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
)


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)
def readable2int(size_str: str) -> int:
    size_str = size_str.strip().upper()
    if size_str.endswith("B"):
        return int(float(size_str[:-1]) * 1e9)
    elif size_str.endswith("M"):
        return int(float(size_str[:-1]) * 1e6)
    elif size_str.endswith("K"):
        return int(float(size_str[:-1]) * 1e3)
    else:
        return int(size_str)




def load_compose_data(local_path, data_args: "DataArguments") -> list[str]:
    preset_path = data_args.preset_path
    total_tokens = readable2int(data_args.task_token_count)
    data_args.task_token_count
    with open(preset_path, "r", encoding="utf-8") as f:
        preset_config = json.load(f)
        if "train" in os.listdir(local_path):
            local_path = os.path.join(local_path, "train")
        try:
            dataset_name, task_config = data_args.task_name.split("/")
            task_config = preset_config[dataset_name][task_config][data_args.task_setting]
            composed_files, remainders = {}, {}
            # assert max(task_config["id_op"]) == data_args.id_max_op, "id_max_op should be equal to the max id_op in the task config"
            for op, dist in zip(task_config["id_op"], task_config["op_prob"]):
                # given dist e.g. 0.165 and total token count e.g. 10B, we can get the approximate number of samples for this op
                op_dir = os.path.join(local_path, str(op))
                if not os.path.exists(op_dir):
                    raise ValueError(f"Directory {op_dir} does not exist.")
                max_op_tokens = (dist * int(total_tokens))
                # get the file counts according to the max_op_tokens
                # get the filename and split the last part as the file size
                file_size = os.listdir(op_dir)[0].split(".")[0].split("_")[-1]
                file_size = readable2int(file_size)
                max_files = int(max_op_tokens // file_size)
                remainder = max_op_tokens % file_size
                files = [os.path.join(op_dir, f) for f in os.listdir(op_dir) if f.endswith(".jsonl")]
                files = sorted(files, key=str.lower)[:max_files + 1]
                remainders[op] = remainder / file_size if remainder > 0 else 0
                composed_files[op] = files
        except Exception as e:
            raise ValueError(f"Error in loading preset config from {preset_path}: {e}")
    return composed_files, remainders

def load_context_data(
    local_path, data_args: "DataArguments"
) -> tuple[dict[str, list[str]], dict[str, float]]:
    preset_path = data_args.preset_path
    total_tokens = readable2int(data_args.task_token_count)
    with open(preset_path, "r", encoding="utf-8") as f:
        preset_config = json.load(f)

    if "train" in os.listdir(local_path):
        local_path = os.path.join(local_path, "train")

    try:
        dataset_name, task_config = data_args.task_name.split("/")
        setting_config = preset_config[dataset_name][task_config][data_args.task_setting]
    except Exception as exc:
        raise ValueError(f"Error in loading preset config from {preset_path}: {exc}") from exc

    context_ids = setting_config.get("id_context")
    context_probs = setting_config.get("context_prob")
    if not context_ids or not context_probs:
        raise ValueError(
            f"Context configuration for {data_args.task_setting} must provide 'id_context' and 'context_prob'."
        )
    if len(context_ids) != len(context_probs):
        raise ValueError(
            "The lengths of 'id_context' and 'context_prob' must be identical in the preset configuration."
        )

    context_files: dict[str, list[str]] = {}
    context_prob_map: dict[str, float] = {}
    total_tokens = int(total_tokens)

    for context_id, probability in zip(context_ids, context_probs):
        context_prob_map[context_id] = float(probability)
        context_dir = os.path.join(local_path, context_id)
        available_files = [f for f in os.listdir(os.path.join(context_dir, "train")) if f.endswith(".jsonl")]
        if not available_files:
            raise ValueError(f"No .jsonl files found in context directory {context_dir}.")
        available_files.sort(key=str.lower)
        selected_files = [os.path.join(context_dir, "train", fname) for fname in available_files]
        context_files[context_id] = selected_files
    return context_files, context_prob_map


def load_mix_data(local_path, data_args: "DataArguments") -> list[dict[str, Any]]:
    preset_path = data_args.preset_path
    total_tokens = readable2int(data_args.task_token_count)

    with open(preset_path, "r", encoding="utf-8") as f:
        preset_config = json.load(f)

    try:
        dataset_name, task_config = data_args.task_name.split("/")
        setting_config = preset_config[dataset_name][task_config][data_args.task_setting]
    except Exception as exc:
        raise ValueError(f"Error in loading preset config from {preset_path}: {exc}") from exc

    if isinstance(setting_config, list):
        components = setting_config
    elif isinstance(setting_config, dict) and "components" in setting_config:
        components = setting_config["components"]
    else:
        raise ValueError(
            f"Mix configuration for {data_args.task_setting} must be a list of components or contain a 'components' key."
        )

    mix_plan: list[dict[str, Any]] = []
    total_tokens = int(total_tokens)

    for component in components:
        context_id = component.get("id_context")
        context_prob = component.get("context_prob")
        op_ids = component.get("id_op")
        op_probs = component.get("op_prob")

        if context_id is None or context_prob is None:
            raise ValueError(
                f"Mix component {component} must provide 'id_context' and 'context_prob'."
            )
        if not op_ids or not op_probs:
            raise ValueError(
                f"Mix component for context {context_id} must provide 'id_op' and 'op_prob'."
            )
        if len(op_ids) != len(op_probs):
            raise ValueError(
                f"Context {context_id}: the lengths of 'id_op' and 'op_prob' must match."
            )

        context_dir = os.path.join(local_path, context_id)
        train_dir = os.path.join(context_dir, "train")
        if not os.path.isdir(train_dir):
            raise ValueError(f"Context directory {train_dir} does not exist.")

        op_plan: dict[int, dict[str, Any]] = {}
        for op_id, op_prob in zip(op_ids, op_probs):
            pattern = f"op{op_id}_"
            available_files = [
                fname for fname in os.listdir(train_dir) if fname.startswith(pattern) and fname.endswith(".jsonl")
            ]
            if not available_files:
                raise ValueError(f"No .jsonl files found for op {op_id} in context directory {train_dir}.")
            available_files.sort(key=str.lower)
            selected_file = available_files[0]
            file_path = os.path.join(train_dir, selected_file)
            file_size_str = selected_file.split(".")[0].split("_")[-1]
            file_size = readable2int(file_size_str)
            # total_tokens: 10B
            # context_prob: e.g. for teacher=0.3
            # op_prob: e.g. for op2=1
            # requested_tokens = 10B * 0.99 * 0.0526316 = 521M
            requested_tokens = float(total_tokens) * float(context_prob) * float(op_prob)
            sample_ratio = requested_tokens / file_size if file_size > 0 else 0.0
            full_repeats = int(math.floor(sample_ratio)) if sample_ratio > 0 else 0
            residual_ratio = sample_ratio - full_repeats
            if residual_ratio < 1e-6:
                residual_ratio = 0.0
            if sample_ratio > 1.0:
                logger.warning(
                    (
                        "Requested tokens %.2f for context %s op %s exceed available %.2f; "
                        "repeating shard %d time(s) with residual ratio %.4f."
                    ),
                    requested_tokens,
                    context_id,
                    op_id,
                    float(file_size),
                    full_repeats,
                    residual_ratio,
                )
            op_plan[int(op_id)] = {
                "file": file_path,
                "ratio": float(sample_ratio),
                "repeat": full_repeats,
                "residual_ratio": float(residual_ratio),
                "requested_tokens": requested_tokens,
                "available_tokens": float(file_size),
            }

        mix_plan.append(
            {
                "context_id": context_id,
                "prob": float(context_prob),
                "ops": op_plan,
            }
        )

    return mix_plan


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Load a single dataset and aligns it to the standard format."""
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_dict = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "cloud_file":
        data_path = dataset_attr.dataset_name

    elif dataset_attr.load_from == "file":
        data_dict = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_dict.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_dict.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_dict[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))
        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_dict):
            raise ValueError("File types should be identical.")
    elif dataset_attr.load_from == "compose":
        data_dict, data_dist = load_compose_data(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name), data_args)
    elif dataset_attr.load_from == "context":
        data_dict, data_dist = load_context_data(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name), data_args)
    elif dataset_attr.load_from == "mix":
        # mix dataset will be handled later
        data_path = "json"
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        check_version("modelscope>=1.14.0", mandatory=True)
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_dict,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_dict,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    elif dataset_attr.load_from == "cloud_file":
        dataset = Dataset.from_list(read_cloud_json(data_path), split=dataset_attr.split)
    elif dataset_attr.load_from == "compose":
        dataset = []
        features = Features({
            "problem":  Value("string"),
            "question": Value("string"),
            "solution": Value("string"),
            "op":       Value("int64"),
            "id":       Value("string"),   # ← fix the mixed types here
            "template": Value("string"),
            "mode":     Value("string"),
            "length":   Value("string"),
            "d":        Value("int64"),
        })
        
        for op in data_dict:
            remain_ratio = data_dist[op]
            last_file = data_dict[op].pop(-1) #note: it won't cause bug here because previously we add 1 more file, so the pop operation is necessary.

            # use remain_ratio to sample from remain_dataset
            if  len(data_dict[op]) > 0:
                op_dataset = load_dataset(
                    path="json",
                    data_files=data_dict[op],
                    split=dataset_attr.split,
                    features=features,
                    cache_dir=model_args.cache_dir,
                    token=model_args.hf_hub_token,
                    num_proc=data_args.preprocessing_num_workers,
                    trust_remote_code=model_args.trust_remote_code,
                    streaming=data_args.streaming and dataset_attr.load_from != "file",
                    )
                dataset.append(op_dataset)
                logger.info_rank0(f"Composed dataset for op {op} with {len(op_dataset)} samples")
            if remain_ratio > 0:
                remain_dataset = load_dataset(
                    path="json",
                    data_files=[last_file],
                    split=dataset_attr.split,
                    cache_dir=model_args.cache_dir,
                    features=features,
                    token=model_args.hf_hub_token,
                    num_proc=data_args.preprocessing_num_workers,
                    trust_remote_code=model_args.trust_remote_code,
                    streaming=data_args.streaming and dataset_attr.load_from != "file",
                )
                n = max(1, math.ceil(remain_ratio * len(remain_dataset)))
                remain_dataset = remain_dataset.shuffle(seed=training_args.seed).select(range(n))
                dataset.append(remain_dataset)
                logger.info_rank0(f"Including {len(remain_dataset)} samples from the last file for op {op} with sampling ratio {remain_ratio:.4f}")
        dataset = concatenate_datasets(dataset)
        logger.info_rank0(f"Composed dataset with {len(dataset)} samples from {len(data_dict)} ops.")
    elif dataset_attr.load_from == "context":
        dataset = load_dataset(
            path="json",
            data_files=data_dict,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            num_proc=data_args.preprocessing_num_workers,
            trust_remote_code=model_args.trust_remote_code,
            streaming=data_args.streaming and dataset_attr.load_from != "file",
            )
        sampled_dataset = []
        for context_id in data_dict:
            logger.info_rank0(f"Context id {context_id} includes {len(data_dict[context_id])} files.")
            prob = data_dist[context_id]
            context_dataset = dataset[context_id]
            if 1-prob > 0:
                context_dataset = context_dataset.train_test_split(test_size=1-prob, seed=42)['train']
            sampled_dataset.append(context_dataset)
            logger.info_rank0(f"Sampling {len(context_dataset)} samples from context id {context_id} with sampling ratio {prob:.4f}")
        dataset = concatenate_datasets(sampled_dataset)
        logger.info_rank0(f"Composed dataset with {len(dataset)} samples from {len(data_dict)} context ids.")
    elif dataset_attr.load_from == "mix":
        mix_plan = load_mix_data(os.path.join(data_args.dataset_dir, dataset_attr.dataset_name), data_args)
        features = Features({
            "problem":  Value("string"),
            "question": Value("string"),
            "solution": Value("string"),
            "op":       Value("int64"),
            "id":       Value("string"),
            "template": Value("string"),
            "mode":     Value("string"),
            "length":   Value("string"),
            "d":        Value("int64"),
        })

        dataset_slices = []
        for entry in mix_plan:
            context_id = entry["context_id"]
            op_plan = entry["ops"]
            for op_id, op_info in op_plan.items():
                ratio = op_info["ratio"]
                repeats = op_info.get("repeat", 0)
                residual_ratio = max(0.0, op_info.get("residual_ratio", ratio - repeats))
                if ratio <= 0 and repeats <= 0 and residual_ratio <= 0:
                    continue
                file_path = op_info["file"]
                op_dataset = load_dataset(
                    path="json",
                    data_files=[file_path],
                    split=dataset_attr.split,
                    features=features,
                    cache_dir=model_args.cache_dir,
                    token=model_args.hf_hub_token,
                    num_proc=data_args.preprocessing_num_workers,
                    trust_remote_code=model_args.trust_remote_code,
                    streaming=data_args.streaming and dataset_attr.load_from != "file",
                )
                op_dataset_len = len(op_dataset)
                if op_dataset_len == 0:
                    logger.warning(
                        "Mix sampling context %s op %s: shard %s is empty; skipping.",
                        context_id,
                        op_id,
                        file_path,
                    )
                    continue

                total_selected = 0
                op_slices: list[Dataset] = []
                if repeats > 0:
                    op_slices.extend([op_dataset] * repeats)
                    total_selected += repeats * op_dataset_len

                if residual_ratio > 0:
                    base_seed = getattr(training_args, "seed", None)
                    shuffle_seed = base_seed + repeats if base_seed is not None else None
                    shuffled_dataset = (
                        op_dataset.shuffle(seed=shuffle_seed) if shuffle_seed is not None else op_dataset.shuffle()
                    )
                    if residual_ratio >= 0.999999:
                        op_slices.append(shuffled_dataset)
                        total_selected += op_dataset_len
                    else:
                        sample_size = max(1, math.ceil(residual_ratio * op_dataset_len))
                        partial_dataset = shuffled_dataset.select(range(sample_size))
                        op_slices.append(partial_dataset)
                        total_selected += len(partial_dataset)

                dataset_slices.extend(op_slices)
                logger.info_rank0(
                    "Mix sampling context %s op %s: ratio %.4f -> repeat %d time(s) + residual %.4f, "
                    "using %d sample(s) from shard size %d.",
                    context_id,
                    op_id,
                    ratio,
                    repeats,
                    residual_ratio,
                    total_selected,
                    len(op_dataset),
                )

        if not dataset_slices:
            raise ValueError("Mix dataset sampling produced no data slices; please check preset configuration.")

        dataset = concatenate_datasets(dataset_slices)
        logger.info_rank0(
            "Composed mixed dataset with %d samples from %d context-op shards.",
            len(dataset),
            len(dataset_slices),
        )
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_dict,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            num_proc=data_args.preprocessing_num_workers,
            trust_remote_code=model_args.trust_remote_code,
            streaming=data_args.streaming and dataset_attr.load_from != "file",
        )
        if data_args.streaming and dataset_attr.load_from == "file":
            dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: Optional[list[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    return_dict: bool = False,
) -> Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]:
    r"""Return the merged datasets in the standard format."""
    if dataset_names is None:
        return None

    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, data_args, training_args)

    if return_dict:
        return datasets
    else:
        return merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    r"""Return the corresponding dataset processor."""
    if stage == "pt":
        dataset_processor_class = PretrainDatasetProcessor
        
    elif stage == "sft" and not do_generate:
        if data_args.packing:
            if data_args.neat_packing:  # hack datasets to have int32 attention mask
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            dataset_processor_class = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_class = SupervisedDatasetProcessor

    elif stage == "rm":
        dataset_processor_class = PairwiseDatasetProcessor
    elif stage == "kto":
        dataset_processor_class = FeedbackDatasetProcessor
    else:
        dataset_processor_class = UnsupervisedDatasetProcessor

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""Preprocesses the dataset, including format checking and tokenization."""
    if dataset is None:
        return None

    dataset_processor = _get_dataset_processor(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
        )

    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        if isinstance(eval_dataset, dict):
            for eval_name, eval_data in eval_dataset.items():
                eval_dataset[eval_name] = _get_preprocessed_dataset(
                    eval_data, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )
        else:
            eval_dataset = _get_preprocessed_dataset(
                eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        dataset_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
        if data_args.tokenized_path is not None:  # save tokenized dataset to disk
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`.")

        return get_dataset_module(dataset_dict)
