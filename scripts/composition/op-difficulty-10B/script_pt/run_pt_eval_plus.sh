#!/usr/bin/env bash

# Evaluate the cpt0.2-uniform_0.8-11-14_plus checkpoints via meta_run.sh.
# The script processes checkpoints in fixed-size chunks (default: 3) so that
# multiple evaluation jobs can be launched in parallel by varying CKPT_GROUP_INDEX.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

derive_results_dir_from_config() {
    local config_path="$1"
    local stage_override="${2:-pt}"
    python3 - "$config_path" "$REPO_ROOT" "$stage_override" <<'PY'
import os
import sys

cfg_path = os.path.abspath(os.path.expanduser(sys.argv[1]))
repo_root = os.path.abspath(os.path.expanduser(sys.argv[2]))
stage_override = sys.argv[3] if len(sys.argv) > 3 else "pt"

config_stem = os.path.splitext(os.path.basename(cfg_path))[0]

try:
    rel = os.path.relpath(cfg_path, repo_root)
except ValueError:
    print("")
    sys.exit(0)

if rel.startswith(".."):
    print("")
    sys.exit(0)

rel_target = rel
if rel_target.startswith("scripts/"):
    rel_target = rel_target[len("scripts/"):]

rel_no_ext, _ = os.path.splitext(rel_target)
parts = [segment for segment in rel_no_ext.split('/') if segment]

drop_suffixes = {"llamafactory-config", "verl-config", "config"}
if parts:
    tail = parts[-1]
    if tail in drop_suffixes or tail.endswith("-config"):
        parts = parts[:-1]

config_segment = None
if parts:
    config_segment = parts[-1]
    parts = parts[:-1]

include_config_segment = config_segment is not None
if config_segment == config_stem and stage_override in {"pt"}:
    include_config_segment = False

result_parts = list(parts)
if stage_override:
    if not result_parts or result_parts[-1] != stage_override:
        result_parts.append(stage_override)

if include_config_segment and config_segment:
    result_parts.append(config_segment)

if result_parts:
    out_path = os.path.join(repo_root, "results", *result_parts)
    print(out_path)
    sys.exit(0)

print("")
PY
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi

CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-saves/composition-10B/op_level/id2-10_0.2easy_0.3medium_0.5hard/cpt0.2-uniform_0.8-11-14_plus}"
CONFIG_ROOT_DEFAULT="scripts/composition/op-difficulty-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5"
CONFIG_ROOT="${CONFIG_ROOT:-${CONFIG_ROOT_DEFAULT}}"
CONFIG_NAME="${CONFIG_NAME:-id2-10_0.2easy_0.3medium_0.5hard_cpt11-14_plus.yaml}"
LLAMA_CONFIG="${LLAMA_CONFIG:-${CONFIG_ROOT}/${CONFIG_NAME}}"
CKPT_GROUP_SIZE="${CKPT_GROUP_SIZE:-0}"
CKPT_GROUP_INDEX="${CKPT_GROUP_INDEX:-0}"
CKPT_STRIDE="${CKPT_STRIDE:-1}"
CKPT_STRIDE_OFFSET="${CKPT_STRIDE_OFFSET:-0}"
SCORE_GRAPH_SOLUTIONS="${SCORE_GRAPH_SOLUTIONS:-1}"
GRAPH_GOLD_DIR="${GRAPH_GOLD_DIR:-data/composition/val_graphs}"
GRAPH_SCORE_PYTHON="${GRAPH_SCORE_PYTHON:-python3}"
GRAPH_SCORE_SCRIPT="${GRAPH_SCORE_SCRIPT:-scripts/score_dependency_graphs.py}"
GRAPH_SCORE_WORKERS="${GRAPH_SCORE_WORKERS:-48}"
GRAPH_SCORE_EXTRA_ARGS="${GRAPH_SCORE_EXTRA_ARGS:-}"
GRAPH_SCORING_SPLITS="${GRAPH_SCORING_SPLITS:-id,ood}"
EVAL_SUMMARY_PER_CKPT="${EVAL_SUMMARY_PER_CKPT:-1}"
EVAL_SUMMARY_BASENAME="${EVAL_SUMMARY_BASENAME:-summary}"

if [[ ! -d "${CHECKPOINTS_ROOT}" ]]; then
    echo "Checkpoints root not found: ${CHECKPOINTS_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${LLAMA_CONFIG}" ]]; then
    echo "LLaMA config not found: ${LLAMA_CONFIG}" >&2
    exit 1
fi

CONFIG_FILENAME="$(basename "${LLAMA_CONFIG}")"
CONFIG_STEM="${CONFIG_FILENAME%.*}"
DEFAULT_RUN_NAME="${CONFIG_STEM}"
RESOLVED_RUN_NAME="${LLAMA_RUN_NAME:-${DEFAULT_RUN_NAME}}"

EVAL_OUTPUT_DIR_DEFAULT="$(derive_results_dir_from_config "${LLAMA_CONFIG}" "pt")"
if [[ -z "${EVAL_OUTPUT_DIR_DEFAULT}" ]]; then
    EVAL_OUTPUT_DIR_DEFAULT="${REPO_ROOT}/results/${CONFIG_STEM}"
fi

if [[ -n "${EVAL_OUTPUT_DIR_DEFAULT}" ]]; then
    base_segment="$(basename "${EVAL_OUTPUT_DIR_DEFAULT}")"
    if [[ -n "${RESOLVED_RUN_NAME}" && "${RESOLVED_RUN_NAME}" != "${base_segment}" ]]; then
        EVAL_OUTPUT_DIR_DEFAULT="${EVAL_OUTPUT_DIR_DEFAULT}/${RESOLVED_RUN_NAME}"
    fi
fi

RESULTS_OUTPUT_DIR="${RESULTS_OUTPUT_DIR:-${EVAL_OUTPUT_DIR_DEFAULT}}"
GRAPH_RESULTS_DIR_DEFAULT=""
if [[ -n "${RESULTS_OUTPUT_DIR}" ]]; then
    GRAPH_RESULTS_DIR_DEFAULT="${RESULTS_OUTPUT_DIR/\/results\//\/graph_results\/}"
    if [[ "${GRAPH_RESULTS_DIR_DEFAULT}" == "${RESULTS_OUTPUT_DIR}" ]]; then
        GRAPH_RESULTS_DIR_DEFAULT="${REPO_ROOT}/graph_results/${CONFIG_STEM}"
    fi
else
    GRAPH_RESULTS_DIR_DEFAULT="${REPO_ROOT}/graph_results/${CONFIG_STEM}"
fi

GRAPH_RESULTS_DIR="${GRAPH_RESULTS_DIR:-${GRAPH_RESULTS_DIR_DEFAULT}}"

IFS=',' read -r -a _RAW_GRAPH_SPLITS <<< "${GRAPH_SCORING_SPLITS}"
GRAPH_SCORING_SPLIT_LIST=()
for raw_split in "${_RAW_GRAPH_SPLITS[@]}"; do
    raw_split="${raw_split//[[:space:]]/}"
    if [[ -n "${raw_split}" ]]; then
        GRAPH_SCORING_SPLIT_LIST+=("${raw_split}")
    fi
done
if [[ ${#GRAPH_SCORING_SPLIT_LIST[@]} -eq 0 ]]; then
    GRAPH_SCORING_SPLIT_LIST=(id ood)
fi
GRAPH_GOLD_AVAILABLE=0
if [[ -d "${GRAPH_GOLD_DIR}" ]]; then
    GRAPH_GOLD_AVAILABLE=1
elif [[ "${SCORE_GRAPH_SOLUTIONS}" == "1" ]]; then
    echo "Warning: gold graphs directory not found (${GRAPH_GOLD_DIR}); disabling dependency graph scoring" >&2
fi

if ! [[ "${CKPT_GROUP_SIZE}" =~ ^[0-9]+$ ]] || (( CKPT_GROUP_SIZE < 0 )); then
    echo "CKPT_GROUP_SIZE must be a non-negative integer (got '${CKPT_GROUP_SIZE}')" >&2
    exit 1
fi

if ! [[ "${CKPT_GROUP_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "CKPT_GROUP_INDEX must be a non-negative integer (got '${CKPT_GROUP_INDEX}')" >&2
    exit 1
fi

if ! [[ "${CKPT_STRIDE}" =~ ^[0-9]+$ ]] || (( CKPT_STRIDE <= 0 )); then
    echo "CKPT_STRIDE must be a positive integer (got '${CKPT_STRIDE}')" >&2
    exit 1
fi

if ! [[ "${CKPT_STRIDE_OFFSET}" =~ ^[0-9]+$ ]]; then
    echo "CKPT_STRIDE_OFFSET must be a non-negative integer (got '${CKPT_STRIDE_OFFSET}')" >&2
    exit 1
fi

if (( CKPT_STRIDE_OFFSET >= CKPT_STRIDE )); then
    echo "CKPT_STRIDE_OFFSET (${CKPT_STRIDE_OFFSET}) must be less than CKPT_STRIDE (${CKPT_STRIDE})" >&2
    exit 1
fi

mapfile -t ALL_CHECKPOINTS < <(
    CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT}" python3 - <<'PY'
import os
import re

root = os.environ["CHECKPOINTS_ROOT"]
pattern = re.compile(r"^checkpoint-(\d+)$")
pairs = []
for name in os.listdir(root):
    match = pattern.match(name)
    if match:
        pairs.append((int(match.group(1)), name))

for _, ckpt in sorted(pairs):
    print(ckpt)
PY
)

TOTAL="${#ALL_CHECKPOINTS[@]}"
if (( TOTAL == 0 )); then
    echo "No checkpoints found under ${CHECKPOINTS_ROOT}" >&2
    exit 1
fi

STRIDED_CHECKPOINTS=()
for idx in "${!ALL_CHECKPOINTS[@]}"; do
    remainder=$(( idx % CKPT_STRIDE ))
    if (( remainder == CKPT_STRIDE_OFFSET )); then
        STRIDED_CHECKPOINTS+=("${ALL_CHECKPOINTS[$idx]}")
    fi
done

TOTAL_STRIDED="${#STRIDED_CHECKPOINTS[@]}"
if (( TOTAL_STRIDED == 0 )); then
    echo "No checkpoints selected after applying stride=${CKPT_STRIDE} offset=${CKPT_STRIDE_OFFSET}" >&2
    exit 1
fi

if (( CKPT_GROUP_SIZE == 0 )); then
    if (( CKPT_GROUP_INDEX != 0 )); then
        echo "CKPT_GROUP_INDEX must be 0 when CKPT_GROUP_SIZE is 0 (got index ${CKPT_GROUP_INDEX})" >&2
        exit 1
    fi
    START=0
    END="${TOTAL_STRIDED}"
else
    START=$(( CKPT_GROUP_INDEX * CKPT_GROUP_SIZE ))
    if (( START >= TOTAL_STRIDED )); then
        echo "Group index ${CKPT_GROUP_INDEX} (size ${CKPT_GROUP_SIZE}) exceeds strided checkpoint count (${TOTAL_STRIDED})" >&2
        exit 0
    fi
    END=$(( START + CKPT_GROUP_SIZE ))
    if (( END > TOTAL_STRIDED )); then
        END="${TOTAL_STRIDED}"
    fi
fi
LEN=$(( END - START ))

SELECTED=("${STRIDED_CHECKPOINTS[@]:${START}:${LEN}}")

echo "Evaluating strided checkpoints ${START}..$((END-1)) of ${TOTAL_STRIDED} (stride ${CKPT_STRIDE}, offset ${CKPT_STRIDE_OFFSET}; original total ${TOTAL})"
printf '  - %s\n' "${SELECTED[@]}"

run_graph_scoring_for_checkpoint() {
    local ckpt="$1"
    local ckpt_results_dir="${2:-${RESULTS_OUTPUT_DIR}}"
    if [[ "${SCORE_GRAPH_SOLUTIONS}" != "1" ]]; then
        return 0
    fi
    if [[ "${GRAPH_GOLD_AVAILABLE}" != "1" ]]; then
        return 0
    fi
    if [[ -z "${ckpt_results_dir}" ]]; then
        echo "Graph scoring skipped for ${ckpt}; evaluation output directory is unknown" >&2
        return 0
    fi
    if [[ -z "${GRAPH_RESULTS_DIR}" ]]; then
        echo "Graph scoring skipped for ${ckpt}; GRAPH_RESULTS_DIR is unset" >&2
        return 0
    fi
    for split in "${GRAPH_SCORING_SPLIT_LIST[@]}"; do
        local gen_file="${ckpt_results_dir}/${ckpt}_${split}_generations.jsonl"
        if [[ ! -f "${gen_file}" ]]; then
            echo "Graph scoring: ${gen_file} not found; skipping" >&2
            continue
        fi
        mkdir -p -- "${GRAPH_RESULTS_DIR}"
        local cmd=( "${GRAPH_SCORE_PYTHON}" "${GRAPH_SCORE_SCRIPT}" "${GRAPH_GOLD_DIR}" "${gen_file}" "--results-dir" "${GRAPH_RESULTS_DIR}" "--workers" "${GRAPH_SCORE_WORKERS}" )
        local extra_args=()
        if [[ -n "${GRAPH_SCORE_EXTRA_ARGS}" ]]; then
            # shellcheck disable=SC2206
            extra_args=( ${GRAPH_SCORE_EXTRA_ARGS} )
            cmd+=("${extra_args[@]}")
        fi
        echo "Scoring dependency graphs for ${ckpt} (${split})"
        "${cmd[@]}"
    done
}

for ckpt in "${SELECTED[@]}"; do
    echo
    echo "==> Evaluating ${ckpt}"
    ckpt_results_dir="${RESULTS_OUTPUT_DIR}"
    if [[ -n "${ckpt_results_dir}" ]]; then
        ckpt_results_dir="${ckpt_results_dir}/${ckpt}"
    fi
    summary_name="${EVAL_SUMMARY_BASENAME}"
    if [[ -z "${summary_name}" ]]; then
        summary_name="summary"
    fi
    if [[ "${EVAL_SUMMARY_PER_CKPT}" == "1" ]]; then
        summary_name="${summary_name}_${ckpt}"
    fi
    EVAL_DATA_ROOT="data/composition/val" \
    EVAL_SUMMARY_FILENAME="${summary_name}" \
    EVAL_CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT}" \
    EVAL_CHECKPOINTS_PATTERN="${ckpt}" \
    EVAL_OUTPUT_TEMPLATE="${ckpt_results_dir}" \
    LLAMA_CONFIG="${LLAMA_CONFIG}" \
        ./scripts/meta_run.sh --skip-rl --do-eval "$@"
    run_graph_scoring_for_checkpoint "${ckpt}" "${ckpt_results_dir}"
done
