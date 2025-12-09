import logging
import math
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
from datasets import concatenate_datasets

from .dataset_context import (
    CustomRLHFDataset as ContextDataset,
    collate_fn,
    compute_score,
    compose_prompt,
    extract_answer_flexible,
)

logger = logging.getLogger(__name__)


def _normalize_prob_map(prob_map: Dict[int, float]) -> Dict[int, float]:
    total = sum(prob for prob in prob_map.values() if prob > 0.0)
    if total <= 0.0:
        return {}
    return {int(op): float(prob) / total for op, prob in prob_map.items() if prob > 0.0}


def _format_mix_summary(allocation: Dict[str, Dict[int, int]]) -> str:
    if not allocation:
        return ""
    parts = []
    for context in sorted(allocation.keys()):
        op_alloc = allocation[context]
        if not op_alloc:
            parts.append(f"{context}: 0")
            continue
        inner = ", ".join(f"{op}:{count}" for op, count in sorted(op_alloc.items()))
        parts.append(f"{context}: {inner}")
    return "; ".join(parts)


class CustomMixedRLHFDataset(ContextDataset):
    """Dataset class that samples per-context and per-op mixes defined in presets."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer,
        config,
        processor: Optional[object] = None,
        is_train: bool = True,
    ):
        self.context_op_probs: Dict[str, Dict[int, float]] = {}
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            is_train=is_train,
        )

    def _filter_by_preset(self) -> None:  # type: ignore[override]
        if self.dataframe is None or len(self.dataframe) == 0:
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

        if isinstance(setting_config, list):
            components = setting_config
        elif isinstance(setting_config, dict) and "components" in setting_config:
            components = setting_config["components"]
        else:
            # fall back to the vanilla context loader behaviour
            return ContextDataset._filter_by_preset(self)

        preset_seed = int(self.config.get("preset_seed", 42))

        parsed_components = []
        for component in components:
            context_id = component.get("id_context")
            context_prob = component.get("context_prob")
            if context_id is None or context_prob is None:
                logger.warning("Skipping mix component without context id/probability: %s", component)
                continue
            try:
                context_prob = float(context_prob)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Context probability for component {component} must be numeric"
                ) from exc

            op_ids = component.get("id_op") or []
            op_probs = component.get("op_prob") or []
            if op_ids and op_probs and len(op_ids) != len(op_probs):
                raise ValueError(
                    f"Context '{context_id}' must provide matching lengths for 'id_op' and 'op_prob'."
                )

            op_map = {}
            for op_id, op_prob in zip(op_ids, op_probs):
                try:
                    op_map[int(op_id)] = float(op_prob)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid op specification ({op_id}, {op_prob}) for context '{context_id}'."
                    ) from exc

            parsed_components.append(
                {
                    "context": str(context_id),
                    "prob": context_prob,
                    "op_map": op_map,
                }
            )

        if not parsed_components:
            raise RuntimeError(
                f"No valid mix components found for preset '{self.task_setting}'."
            )

        base_dataset = self.dataframe
        sampled_contexts = []
        retained_components = []

        for component in parsed_components:
            context_id = component["context"]
            prob = float(component["prob"])
            if prob <= 0.0:
                logger.info(
                    "Skipping context '%s' due to non-positive probability %.4f.",
                    context_id,
                    prob,
                )
                continue
            if prob > 1.0:
                raise ValueError(
                    f"Context probability for '{context_id}' must be within [0, 1]; got {prob}."
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
            retained_components.append(component)
            logger.info(
                "Retained %d samples for context '%s' with sampling ratio %.4f.",
                len(subset),
                context_id,
                prob,
            )

        if not sampled_contexts:
            raise RuntimeError("Context preset filtering removed all samples for the mixed dataset.")

        self.dataframe = concatenate_datasets(sampled_contexts).shuffle(seed=preset_seed)

        prob_sum = sum(component["prob"] for component in retained_components)
        if prob_sum <= 0.0:
            raise ValueError("Sum of context probabilities must be positive after filtering.")

        self.active_contexts = [component["context"] for component in retained_components]
        self.active_context_probs = [component["prob"] / prob_sum for component in retained_components]

        self.context_op_probs = {}
        for component in retained_components:
            op_map = _normalize_prob_map(component["op_map"])
            if op_map:
                self.context_op_probs[component["context"]] = op_map

        logger.info(
            "Preset filtering retained %d samples across %d contexts for mix setting '%s'.",
            len(self.dataframe),
            len(retained_components),
            self.task_setting,
        )

    def _enforce_equal_op_sampling(self) -> None:  # type: ignore[override]
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
            raise ValueError("task_sample must be positive for mixed dataset sampling.")

        column_names = self.dataframe.column_names
        op_values = (
            self.dataframe["op"] if "op" in column_names else [None] * len(self.dataframe)
        )
        template_values = (
            self.dataframe["template"]
            if "template" in column_names
            else ["__ALL__"] * len(self.dataframe)
        )

        if not self.active_contexts or not self.active_context_probs:
            logger.warning(
                "Active context probabilities missing; falling back to uniform balancing,"
                " mix preset will be ignored.")
            return ContextDataset._enforce_equal_op_sampling(self)

        context_prob_map = {
            context: float(prob)
            for context, prob in zip(self.active_contexts, self.active_context_probs)
        }

        context_to_ops: Dict[str, Dict[int, list[int]]] = {
            ctx: defaultdict(list) for ctx in context_prob_map
        }

        for idx, (context_label, raw_op) in enumerate(zip(template_values, op_values)):
            if context_label not in context_to_ops or raw_op is None:
                continue
            try:
                op_int = int(raw_op)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Encountered non-integer op value: {raw_op!r}") from exc
            context_to_ops[context_label][op_int].append(idx)

        combos = []
        for context, ctx_prob in context_prob_map.items():
            op_index_map = context_to_ops.get(context, {})
            available_ops = {op: indices for op, indices in op_index_map.items() if indices}
            if not available_ops:
                logger.warning(
                    "Context '%s' has no samples with allowed ops after filtering; it will be ignored.",
                    context,
                )
                continue

            desired_op_dist = self.context_op_probs.get(context)
            if desired_op_dist:
                filtered_dist = {
                    op: prob for op, prob in desired_op_dist.items() if op in available_ops and prob > 0.0
                }
                missing_ops = sorted(set(desired_op_dist) - set(filtered_dist))
                if missing_ops:
                    logger.warning(
                        "Context '%s' is missing ops %s in available data; redistributing remaining probability.",
                        context,
                        missing_ops,
                    )
                desired_op_dist = filtered_dist
            else:
                desired_op_dist = {}

            if not desired_op_dist:
                uniform_prob = 1.0 / len(available_ops)
                desired_op_dist = {op: uniform_prob for op in available_ops}
            else:
                desired_op_dist = _normalize_prob_map(desired_op_dist)
                if not desired_op_dist:
                    uniform_prob = 1.0 / len(available_ops)
                    desired_op_dist = {op: uniform_prob for op in available_ops}

            for op, op_prob in desired_op_dist.items():
                raw_target = ctx_prob * op_prob * task_sample
                base = math.floor(raw_target)
                fraction = raw_target - base
                combos.append(
                    {
                        "context": context,
                        "op": op,
                        "indices": available_ops[op],
                        "raw": raw_target,
                        "base": base,
                        "fraction": fraction,
                    }
                )

        if not combos:
            raise RuntimeError("No context/op combinations available for mixed sampling.")

        allocations: Dict[tuple[str, int], int] = {
            (combo["context"], combo["op"]): combo["base"] for combo in combos
        }

        remainder = task_sample - sum(allocations.values())
        if remainder > 0:
            combos_sorted = sorted(combos, key=lambda combo: combo["fraction"], reverse=True)
            for combo in combos_sorted:
                if remainder == 0:
                    break
                key = (combo["context"], combo["op"])
                allocations[key] += 1
                remainder -= 1
            if remainder > 0:
                # distribute any leftover evenly starting from the largest pools
                combos_sorted = sorted(
                    combos,
                    key=lambda combo: len(combo["indices"]),
                    reverse=True,
                )
                for combo in combos_sorted:
                    if remainder == 0:
                        break
                    key = (combo["context"], combo["op"])
                    allocations[key] += 1
                    remainder -= 1
        elif remainder < 0:
            combos_sorted = sorted(combos, key=lambda combo: combo["fraction"])
            for combo in combos_sorted:
                if remainder == 0:
                    break
                key = (combo["context"], combo["op"])
                if allocations[key] > 0:
                    allocations[key] -= 1
                    remainder += 1

        final_total = sum(allocations.values())
        if final_total != task_sample:
            raise RuntimeError(
                f"Failed to allocate exactly {task_sample} samples; got {final_total}."
            )

        rng = np.random.default_rng(int(self.config.get("preset_seed", 42)))
        selected_indices = []
        oversampled_ops: list[tuple[str, int]] = []
        allocation_summary: Dict[str, Dict[int, int]] = defaultdict(dict)

        for combo in combos:
            key = (combo["context"], combo["op"])
            target = allocations.get(key, 0)
            if target <= 0:
                continue

            indices = combo["indices"]
            if not indices:
                logger.warning(
                    "No samples available for context '%s' op %d despite allocation; skipping.",
                    combo["context"],
                    combo["op"],
                )
                continue

            replace = len(indices) < target
            if replace:
                oversampled_ops.append((combo["context"], combo["op"]))

            sampled = rng.choice(np.asarray(indices, dtype=np.int64), size=target, replace=replace)
            selected_indices.append(sampled)
            allocation_summary[combo["context"]][combo["op"]] = allocation_summary[combo["context"]].get(combo["op"], 0) + target

        if not selected_indices:
            raise RuntimeError("No samples were selected during mixed op balancing.")

        stacked_indices = np.concatenate(selected_indices)
        rng.shuffle(stacked_indices)

        if len(stacked_indices) != task_sample:
            raise RuntimeError(
                f"Balanced sampling expected {task_sample} samples but produced {len(stacked_indices)}."
            )

        self.dataframe = self.dataframe.select(stacked_indices.tolist())

        effective_contexts = [ctx for ctx in self.active_contexts if ctx in allocation_summary]
        effective_probs = [context_prob_map[ctx] for ctx in effective_contexts]
        prob_total = sum(effective_probs)
        if prob_total > 0:
            self.active_contexts = effective_contexts
            self.active_context_probs = [prob / prob_total for prob in effective_probs]

        logger.info(
            "Balanced mixed dataset to %d samples. Breakdown: %s. Oversampled (context, op): %s.",
            task_sample,
            _format_mix_summary(allocation_summary),
            oversampled_ops if oversampled_ops else "None",
        )


__all__ = [
    "CustomMixedRLHFDataset",
    "collate_fn",
    "compute_score",
    "compose_prompt",
    "extract_answer_flexible",
]

