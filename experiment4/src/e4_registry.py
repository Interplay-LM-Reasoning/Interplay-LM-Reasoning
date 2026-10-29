"""Checkpoint registry and shared constants for Experiment 4.

Keep in sync with config/e4_env.sh (bash side of the same registry).
"""

from __future__ import annotations

from dataclasses import dataclass

CKPT_PARENT = "idzoo_0.999zoo_0.001teacher"


@dataclass(frozen=True)
class Checkpoint:
    label: str          # filesystem-safe label used in our output tree
    display: str        # name used in tables / plot legends (paper Fig. 7 wording)
    subfolder: str      # path inside the HF repo Interplay-LM-Reasoning/context_pretrain
    reward: str         # reward used during RL training
    color: str          # plot color (matches paper Fig. 7 palette)


CHECKPOINTS: list[Checkpoint] = [
    Checkpoint("base", "Base (no RL)", f"{CKPT_PARENT}/base",
               "none (pre-training only)", "#555555"),
    Checkpoint("outcome", "1.0 Outcome", f"{CKPT_PARENT}/rl/contextzoo_0.9zoo_0.1teacher",
               "presumed answer-only (no config in repo)", "#8c8c8c"),
    Checkpoint("mix_0.5_0.5", "0.5 Outcome + 0.5 Process",
               f"{CKPT_PARENT}/rl/contextzoo_0.9zoo_0.1teacher_process0.5",
               "0.5 answer + 0.5 step score", "#e6a817"),
    Checkpoint("mix_0.2_0.8", "0.2 Outcome + 0.8 Process",
               f"{CKPT_PARENT}/rl/contextzoo_0.9zoo_0.1teacher_process",
               "0.2 answer + 0.8 step score", "#2e7bb5"),
    Checkpoint("strict", "1.0 Outcome if Process Correct",
               f"{CKPT_PARENT}/rl/contextzoo_0.9zoo_0.1teacher_process_strict",
               "answer reward only if every step is right", "#2f9e5f"),
]

BY_LABEL = {c.label: c for c in CHECKPOINTS}
LABELS = [c.label for c in CHECKPOINTS]

# The ckpt folder name is what eval_checkpoints.py prefixes output files with
# (e.g. contextzoo_0.9zoo_0.1teacher_ood_generations.jsonl).
CKPT_FOLDER_NAME = {c.label: c.subfolder.rsplit("/", 1)[-1] for c in CHECKPOINTS}

CONTEXTS = {"zoo": "crazy_zootopia", "teachers": "teachers_in_school"}
OPS = [12, 14, 16, 18, 20]

# Difficulty bands used for reporting (same convention as the experiment guide:
# edge = just beyond pre-training reach for RL data op ranges, hard = far out).
BANDS = {"edge (op 12-14)": [12, 14], "hard (op 16-20)": [16, 18, 20]}

KS = [1, 2, 4, 8, 16, 32, 64]
N_PROBLEMS_PER_OP = 200
