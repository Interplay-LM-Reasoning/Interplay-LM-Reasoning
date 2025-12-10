# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

from dataclasses import dataclass
from itertools import chain
from typing import Any

from .processor_utils import DatasetProcessor


@dataclass
class PretrainDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
        eos_token = "<|end_of_text|>" if self.data_args.template == "llama3" else self.tokenizer.eos_token
        if isinstance(examples["_prompt"][0], str):
            text_examples = [example + eos_token for example in examples["_prompt"]]
        else:
            text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]

        if not self.data_args.packing:
            if getattr(self.tokenizer, "add_bos_token", False):
                text_examples = [self.tokenizer.bos_token + example for example in text_examples]

            result = self.tokenizer(
                text_examples, add_special_tokens=False, truncation=True, max_length=self.data_args.cutoff_len
            )
        else:
            # Fast path: batch-tokenize once, then pack into fixed-length blocks
            tokenized = self.tokenizer(
                text_examples,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]

            # Use smart FFD packing by default to reduce padding waste
            result = self.pack_and_pad_to_cutoff(tokenized)

        return result

    def pack_and_pad_to_cutoff(self, text_or_tokens: list[Any], smart_packing: bool = True) -> dict[str, list[list[int]]]:
        """Pack samples into fixed-length blocks.

        Behavior:
        - Tokenize each text without adding special tokens (texts should already include EOS if desired),
          or accept pre-tokenized lists of token ids.
        - Fill a block up to `cutoff_len`; if the next sample would overflow the block, pad the
          current block to length `cutoff_len` with `pad_token_id` and start a fresh block.
        - If a single sample is longer than `cutoff_len`, it will be chunked across multiple blocks.
        - If the tokenizer has `add_bos_token=True`, the first token of each block is replaced with BOS.

        When `smart_packing=True`, apply a First-Fit Decreasing (FFD) bin-packing heuristic to mix
        shorter and longer samples within a block to reduce padding waste (after emitting any full
        cutoff-sized chunks). This reorders samples within a batch only for packing efficiency.

        Returns a dict with keys `input_ids` and `attention_mask`, each mapped to a list of blocks
        (each of length `cutoff_len`).
        """
        cutoff_len = self.data_args.cutoff_len
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            # Fallbacks if pad token isn't set
            pad_id = getattr(self.tokenizer, "eos_token_id", None) or 0
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        add_bos = getattr(self.tokenizer, "add_bos_token", False)

        blocks: list[list[int]] = []
        masks: list[list[int]] = []
        cur: list[int] = []
        cur_mask: list[int] = []

        # Normalize input: list of strings or list of token-id lists
        if not text_or_tokens:
            return {"input_ids": [], "attention_mask": []}

        if isinstance(text_or_tokens[0], str):
            # Tokenize on the fly (legacy path)
            tokens_list = [self.tokenizer.encode(t, add_special_tokens=False) for t in text_or_tokens]
        else:
            tokens_list = text_or_tokens  # already tokenized

        if not smart_packing:
            # Streamed sample-aware packing (original logic)
            for tokens in tokens_list:
                # If a single sample is extremely long, emit as full blocks first
                while len(tokens) >= cutoff_len:
                    block = tokens[:cutoff_len]
                    if add_bos and bos_id is not None and block:
                        block[0] = bos_id
                    blocks.append(block)
                    masks.append([1] * cutoff_len)
                    tokens = tokens[cutoff_len:]

                # Now tokens length < cutoff_len
                if not tokens:
                    continue

                # If next sample doesn't fit in current block, finalize current with padding
                if cur and len(cur) + len(tokens) > cutoff_len:
                    if add_bos and bos_id is not None and cur:
                        cur[0] = bos_id
                    pad_len = cutoff_len - len(cur)
                    cur = cur + [pad_id] * pad_len
                    cur_mask = cur_mask + [0] * pad_len
                    blocks.append(cur)
                    masks.append(cur_mask)
                    cur = []
                    cur_mask = []

                # Start fresh or continue filling the current block
                cur.extend(tokens)
                cur_mask.extend([1] * len(tokens))

                # If exactly filled, flush the block immediately
                if len(cur) == cutoff_len:
                    if add_bos and bos_id is not None and cur:
                        cur[0] = bos_id
                    blocks.append(cur)
                    masks.append(cur_mask)
                    cur = []
                    cur_mask = []
        else:
            # Smart packing via First-Fit Decreasing (FFD)
            # 1) Emit full blocks from very long samples; collect remainders (< cutoff_len)
            remainders: list[list[int]] = []
            for tokens in tokens_list:
                t = tokens
                # Full blocks
                while len(t) >= cutoff_len:
                    block = t[:cutoff_len]
                    if add_bos and bos_id is not None and block:
                        block[0] = bos_id
                    blocks.append(block)
                    masks.append([1] * cutoff_len)
                    t = t[cutoff_len:]
                if t:
                    remainders.append(t)

            if remainders:
                # 2) Sort by length descending
                remainders.sort(key=len, reverse=True)

                # 3) Bin packing
                bin_tokens: list[list[int]] = []
                bin_masks: list[list[int]] = []
                bin_remaining: list[int] = []

                for r in remainders:
                    placed = False
                    r_len = len(r)
                    # Try to place into the first bin that fits
                    for i in range(len(bin_tokens)):
                        if bin_remaining[i] >= r_len:
                            bin_tokens[i].extend(r)
                            bin_masks[i].extend([1] * r_len)
                            bin_remaining[i] -= r_len
                            placed = True
                            break
                    if not placed:
                        # Open a new bin
                        bin_tokens.append(r.copy())
                        bin_masks.append([1] * r_len)
                        bin_remaining.append(cutoff_len - r_len)

                # 4) Finalize bins with padding and BOS handling
                for i in range(len(bin_tokens)):
                    block = bin_tokens[i]
                    mask = bin_masks[i]
                    if add_bos and bos_id is not None and block:
                        block[0] = bos_id
                    pad_len = cutoff_len - len(block)
                    if pad_len > 0:
                        block.extend([pad_id] * pad_len)
                        mask.extend([0] * pad_len)
                    blocks.append(block)
                    masks.append(mask)

        # Flush remaining tokens with padding
        if not smart_packing and cur:
            if add_bos and bos_id is not None and cur:
                cur[0] = bos_id
            pad_len = cutoff_len - len(cur)
            cur = cur + [pad_id] * pad_len
            cur_mask = cur_mask + [0] * pad_len
            blocks.append(cur)
            masks.append(cur_mask)

        return {"input_ids": blocks, "attention_mask": masks}

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
