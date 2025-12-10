#!/usr/bin/env bash
# Run LLaMA-Factory pretraining and verl RL pipelines for paired configs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_ROOT=""
RUN_PRETRAIN=1
RUN_RL=1
RUN_EVAL=1
DO_EVAL_ONLY=0
DRY_RUN="${DRY_RUN:-0}"
LLAMA_BIN="${LLAMA_BIN:-llamafactory-cli}"
VERL_PYTHON="${VERL_PYTHON:-python3}"
VERL_MODULE="${VERL_MODULE:-verl.trainer.main_ppo}"
LLAMA_RUN_NAME="${LLAMA_RUN_NAME:-}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
VERL_EXTRA_ARGS="${VERL_EXTRA_ARGS:-}"
LLAMA_WANDB_PROJECT="${LLAMA_WANDB_PROJECT:-}"
LLAMA_CONFIG="${LLAMA_CONFIG:-}"
VERL_CONFIG="${VERL_CONFIG:-}"
EVAL_PYTHON="${EVAL_PYTHON:-python3}"
EVAL_SCRIPT="${EVAL_SCRIPT:-scripts/eval_checkpoints.py}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-}"
EVAL_CHECKPOINTS_ROOT="${EVAL_CHECKPOINTS_ROOT:-}"
EVAL_CHECKPOINTS_PATTERN="${EVAL_CHECKPOINTS_PATTERN:-}"
EVAL_OUTPUT_TEMPLATE="${EVAL_OUTPUT_TEMPLATE:-}"
EVAL_GEN_BACKEND="${EVAL_GEN_BACKEND:-vllm}"
EVAL_GEN_BATCH_SIZE="${EVAL_GEN_BATCH_SIZE:-64}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1024}"
EVAL_VLLM_TP_SIZE="${EVAL_VLLM_TP_SIZE:-}"
EVAL_VLLM_DP_SIZE="${EVAL_VLLM_DP_SIZE:-}"
EVAL_VLLM_DTYPE="${EVAL_VLLM_DTYPE:-}"
EVAL_SAMPLE_K="${EVAL_SAMPLE_K:-}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-}"
EVAL_TOP_P="${EVAL_TOP_P:-}"
EVAL_TOP_K="${EVAL_TOP_K:-}"
EVAL_DEVICE="${EVAL_DEVICE:-}"
EVAL_SKIP_LOSS="${EVAL_SKIP_LOSS:-1}"
EVAL_SKIP_VALIDATION="${EVAL_SKIP_VALIDATION:-1}"
EVAL_RESUME="${EVAL_RESUME:-0}"
EVAL_SUMMARY_FILENAME="${EVAL_SUMMARY_FILENAME:-summary}"
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS:-}"

usage() {
  cat <<'USAGE'
Usage: meta_run.sh [options]

Options:
  --config-root DIR   Root directory that holds config pairs (default: unset)
  --llama-config FILE Explicit LLaMA-Factory config file (overrides --config-root discovery)
  --verl-config  FILE Explicit verl config file (overrides --config-root discovery)
  --skip-pretrain     Skip the LLaMA-Factory stage
  --skip-rl           Skip the verl stage
  --skip-eval         Skip the evaluation stage
  --do-eval           Run evaluation only (skips pretrain and RL)
  --dry-run           Print the commands without executing them
  -h, --help          Show this message

Environment overrides:
  LLAMA_BIN           Command used to launch LLaMA-Factory (default: llamafactory-cli)
  VERL_PYTHON         Python executable for verl (default: python3)
  VERL_MODULE         Python module passed to -m for verl (default: verl.trainer.main_ppo)
  LLAMA_RUN_NAME      Optional explicit run_name override for LLaMA-Factory
  LLAMA_EXTRA_ARGS    Extra space-separated overrides appended to the LLaMA-Factory CLI call
  LLAMA_WANDB_PROJECT W&B project for the LLaMA-Factory stage (if unset, current env is used)
  LLAMA_CONFIG        Explicit LLaMA-Factory config file (same as --llama-config)
  VERL_CONFIG         Explicit verl config file (same as --verl-config)
  VERL_EXTRA_ARGS     Extra space-separated overrides appended to the verl call
  RUN_EVAL            Set to 0 to disable the evaluation stage (default: 1)
  EVAL_PYTHON         Python executable for evaluation (default: python3)
  EVAL_SCRIPT         Path to the evaluation runner (default: scripts/eval_checkpoints.py)
  EVAL_OUTPUT_TEMPLATE Template for the evaluation output dir; {run_name}, {config_name}, and {stage} are replaced automatically
  EVAL_CHECKPOINTS_ROOT Optional checkpoints root override (defaults to LLaMA output_dir); {run_name}, {config_name}, and {stage} are replaced automatically
  EVAL_EXTRA_ARGS     Extra space-separated overrides appended to the evaluation call
USAGE
}

log() {
  printf '[meta-run] %s\n' "$*"
}

extract_llama_output_dir() {
  local config_path="$1"
  python3 - "$config_path" <<'PY'
import sys

cfg_path = sys.argv[1]
value = None
with open(cfg_path, 'r', encoding='utf-8') as f:
    for raw_line in f:
        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('output_dir:'):
            value = line.split(':', 1)[1].strip()
            break

if value is None:
    sys.exit(1)

if value and value[0] in {'"', "'"} and len(value) > 1 and value[-1] == value[0]:
    value = value[1:-1]

print(value)
PY
}

abspath() {
  python3 - "$1" <<'PY'
import os
import sys

print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

find_latest_llama_model() {
  python3 - "$1" <<'PY'
import os
import re
import sys

base = os.path.abspath(os.path.expanduser(sys.argv[1]))
if not os.path.isdir(base):
    sys.exit(1)

pattern = re.compile(r"^checkpoint-(\d+)$")
best_path = None
best_step = -1

for entry in os.scandir(base):
    if not entry.is_dir(follow_symlinks=False):
        continue
    match = pattern.match(entry.name)
    if not match:
        continue
    step = int(match.group(1))
    if step > best_step:
        best_step = step
        best_path = entry.path

if best_path is not None:
    print(best_path)
else:
    print(base)
PY
}

find_latest_verl_checkpoint() {
  python3 - "$1" <<'PY'
import os
import re
import sys

base = os.path.abspath(os.path.expanduser(sys.argv[1]))
if not os.path.isdir(base):
    sys.exit(1)

latest_path = None
latest_step = -1

tracker = os.path.join(base, "latest_checkpointed_iteration.txt")
if os.path.isfile(tracker):
    try:
        with open(tracker, "r", encoding="utf-8") as f:
            step = int(f.read().strip())
    except Exception:
        step = None
    if step is not None:
        candidate = os.path.join(base, f"global_step_{step}")
        if os.path.isdir(candidate):
            latest_step = step
            latest_path = candidate

if latest_path is None:
    pattern = re.compile(r"global_step_(\d+)$")
    for entry in os.scandir(base):
        if entry.is_dir(follow_symlinks=False):
            match = pattern.match(entry.name)
            if match:
                try:
                    step = int(match.group(1))
                except ValueError:
                    continue
                if step > latest_step:
                    latest_step = step
                    latest_path = entry.path

if latest_path is None:
    sys.exit(1)

print(latest_path)
PY
}

detect_verl_backend() {
  python3 - "$1" <<'PY'
import os
import sys

from omegaconf import OmegaConf

cfg_path = sys.argv[1]
try:
    cfg = OmegaConf.load(cfg_path)
except Exception:
    sys.exit(1)

backend = None
strategy = OmegaConf.select(cfg, "actor_rollout_ref.actor.strategy", default=None)
if strategy is not None:
    backend = str(strategy).strip().lower()

if backend in {"fsdp", "fsdp2"}:
    print("fsdp")
    sys.exit(0)
if backend in {"megatron", "megatron_lm"}:
    print("megatron")
    sys.exit(0)

engine_name = OmegaConf.select(cfg, "trainer.engine.name", default=None)
if engine_name is not None:
    backend = str(engine_name).strip().lower()
    if backend in {"fsdp", "fsdp2"}:
        print("fsdp")
        sys.exit(0)
    if backend in {"megatron", "megatron_lm"}:
        print("megatron")
        sys.exit(0)

sys.exit(1)
PY
}

read_verl_gpu_requests() {
  python3 - "$1" <<'PY'
import sys

from omegaconf import OmegaConf

cfg_path = sys.argv[1]
cfg = OmegaConf.load(cfg_path)

def format_val(value):
    if value is None:
        return ""
    return str(value)

trainer_gpus = OmegaConf.select(cfg, "trainer.n_gpus_per_node", default=None)
rollout_gpus = OmegaConf.select(cfg, "actor_rollout_ref.rollout.n", default=None)

print(format_val(trainer_gpus))
print(format_val(rollout_gpus))
PY
}

read_verl_experiment_name() {
  python3 - "$1" <<'PY'
import sys

from omegaconf import OmegaConf

cfg_path = sys.argv[1]
cfg = OmegaConf.load(cfg_path)

value = OmegaConf.select(cfg, "trainer.experiment_name", default="")
if value is None:
    value = ""

print(value)
PY
}

read_verl_actor_model_path() {
  python3 - "$1" <<'PY'
import sys

from omegaconf import OmegaConf

cfg_path = sys.argv[1]
try:
    cfg = OmegaConf.load(cfg_path)
except Exception:
    sys.exit(1)

value = OmegaConf.select(cfg, "actor_rollout_ref.model.path", default="")
if value is None:
    value = ""

print(value)
PY
}

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

get_visible_device_count() {
  python3 - "$CUDA_VISIBLE_DEVICES" <<'PY'
import os
import sys

env = sys.argv[1]
count = 0
if env:
    devices = [d.strip() for d in env.split(',') if d.strip()]
    count = len(devices)
if count <= 0:
    try:
        import torch
        count = torch.cuda.device_count()
    except Exception:
        count = 0
if count <= 0:
    count = 1
print(count)
PY
}

resolve_vllm_dp_size() {
  local requested="$1"
  local visible_count
  visible_count="$(get_visible_device_count)"

  if [[ -z "${visible_count}" ]]; then
    visible_count=1
  fi

  local final_size="${visible_count}"
  if [[ -n "${requested}" ]]; then
    if [[ "${requested}" =~ ^[0-9]+$ && "${requested}" -gt 0 ]]; then
      final_size="${requested}"
      if (( final_size > visible_count )); then
        log "Requested vLLM data parallel size ${final_size} exceeds visible device count ${visible_count}; clipping"
        final_size="${visible_count}"
      fi
    else
      log "Invalid EVAL_VLLM_DP_SIZE='${requested}', defaulting to visible device count ${visible_count}"
      final_size="${visible_count}"
    fi
  fi

  if (( final_size < 1 )); then
    final_size=1
  fi

  echo "${final_size}"
}

maybe_run_eval() {
  local stage_label="$1"
  local checkpoints_root_ref="$2"
  local config_path="$3"
  local stage_override="$4"
  local run_name_ref="$5"
  local config_name_ref="$6"
  local explicit_ckpt_pattern="${7:-}"
  local fallback_output_dir="${8:-}"

  if [[ ${RUN_EVAL} -ne 1 ]]; then
    return 0
  fi

  if [[ -n "${EVAL_CHECKPOINTS_ROOT}" ]]; then
    local override_root="${EVAL_CHECKPOINTS_ROOT}"
    override_root="${override_root//\{run_name\}/${run_name_ref}}"
    override_root="${override_root//\{config_name\}/${config_name_ref}}"
    override_root="${override_root//\{stage\}/${stage_label}}"
    checkpoints_root_ref="${override_root}"
  fi

  if [[ -z "${checkpoints_root_ref}" ]]; then
    log "Skipping ${stage_label} evaluation; checkpoints root is empty"
    return 0
  fi

  local checkpoints_root
  if ! checkpoints_root="$(abspath "${checkpoints_root_ref}")"; then
    log "Failed to resolve evaluation checkpoints root for ${stage_label}: ${checkpoints_root_ref}"
    exit 1
  fi

  if [[ ${DRY_RUN} -ne 1 && ! -d "${checkpoints_root}" ]]; then
    log "Evaluation checkpoints root not found for ${stage_label}: ${checkpoints_root}"
    exit 1
  fi

  local default_output_dir=""
  if [[ -n "${config_path}" ]]; then
    default_output_dir="$(derive_results_dir_from_config "${config_path}" "${stage_override}")"
  fi
  if [[ -z "${default_output_dir}" ]]; then
    default_output_dir="${fallback_output_dir}"
  fi
  if [[ -n "${default_output_dir}" ]]; then
    local base_segment
    base_segment="$(basename "${default_output_dir}")"
    if [[ -n "${run_name_ref}" && "${run_name_ref}" != "${base_segment}" ]]; then
      default_output_dir="${default_output_dir}/${run_name_ref}"
    fi
  fi

  local eval_output_dir=""
  if [[ -n "${EVAL_OUTPUT_TEMPLATE}" ]]; then
    local templ="${EVAL_OUTPUT_TEMPLATE}"
    templ="${templ//\{run_name\}/${run_name_ref}}"
    templ="${templ//\{config_name\}/${config_name_ref}}"
    templ="${templ//\{stage\}/${stage_label}}"
    if ! eval_output_dir="$(abspath "${templ}")"; then
      log "Failed to resolve evaluation output dir for ${stage_label}: ${templ}"
      exit 1
    fi
  else
    eval_output_dir="${default_output_dir}"
  fi

  local ckpt_pattern="${explicit_ckpt_pattern}"
  if [[ -z "${ckpt_pattern}" ]]; then
    if latest_checkpoint_path="$(find_latest_llama_model "${checkpoints_root}" 2>/dev/null)"; then
      if [[ -n "${latest_checkpoint_path}" && "${latest_checkpoint_path}" != "${checkpoints_root}" ]]; then
        local latest_checkpoint_name
        latest_checkpoint_name="$(basename "${latest_checkpoint_path}")"
        if [[ "${latest_checkpoint_name}" == checkpoint-* ]]; then
          ckpt_pattern="${latest_checkpoint_name}"
          log "Auto-selecting latest checkpoint ${latest_checkpoint_name} for ${stage_label} evaluation"
        fi
      fi
    fi
  fi

  if [[ -n "${eval_output_dir}" ]]; then
    if [[ -n "${EVAL_SUMMARY_FILENAME}" ]]; then
      local summary_base="${eval_output_dir}/${EVAL_SUMMARY_FILENAME}"
      if [[ -e "${summary_base}" || -e "${summary_base}.json" || -e "${summary_base}.csv" ]]; then
        log "Skipping ${stage_label} evaluation; results already exist at ${eval_output_dir}"
        return 0
      fi
    fi
    if [[ ${DRY_RUN} -ne 1 ]]; then
      mkdir -p -- "${eval_output_dir}"
    fi
  fi

  local resolved_vllm_dp_size
  resolved_vllm_dp_size="$(resolve_vllm_dp_size "${EVAL_VLLM_DP_SIZE}")"

  local eval_cmd=("${EVAL_PYTHON}" -u "${EVAL_SCRIPT}" --checkpoints-root "${checkpoints_root}")
  if [[ -n "${eval_output_dir}" ]]; then
    eval_cmd+=("--output-dir" "${eval_output_dir}")
  fi
  if [[ -n "${EVAL_DATA_ROOT}" ]]; then
    eval_cmd+=("--data-root" "${EVAL_DATA_ROOT}")
  fi
  if [[ -n "${EVAL_GEN_BACKEND}" ]]; then
    eval_cmd+=("--gen-backend" "${EVAL_GEN_BACKEND}")
  fi
  if [[ -n "${EVAL_GEN_BATCH_SIZE}" ]]; then
    eval_cmd+=("--gen-batch-size" "${EVAL_GEN_BATCH_SIZE}")
  fi
  if [[ -n "${EVAL_MAX_NEW_TOKENS}" ]]; then
    eval_cmd+=("--max-new-tokens" "${EVAL_MAX_NEW_TOKENS}")
  fi
  if [[ -n "${EVAL_CHECKPOINTS_PATTERN}" ]]; then
    eval_cmd+=("--checkpoints-pattern" "${EVAL_CHECKPOINTS_PATTERN}")
  elif [[ -n "${ckpt_pattern}" ]]; then
    eval_cmd+=("--checkpoints-pattern" "${ckpt_pattern}")
  fi
  if [[ -n "${EVAL_VLLM_TP_SIZE}" ]]; then
    eval_cmd+=("--vllm-tensor-parallel-size" "${EVAL_VLLM_TP_SIZE}")
  fi
  if [[ -n "${resolved_vllm_dp_size}" ]]; then
    eval_cmd+=("--vllm-data-parallel-size" "${resolved_vllm_dp_size}")
  fi
  if [[ -n "${EVAL_VLLM_DTYPE}" ]]; then
    eval_cmd+=("--vllm-dtype" "${EVAL_VLLM_DTYPE}")
  fi
  if [[ -n "${EVAL_SAMPLE_K}" ]]; then
    eval_cmd+=("--sample-k" "${EVAL_SAMPLE_K}")
  fi
  if [[ -n "${EVAL_TEMPERATURE}" ]]; then
    eval_cmd+=("--temperature" "${EVAL_TEMPERATURE}")
  fi
  if [[ -n "${EVAL_TOP_P}" ]]; then
    eval_cmd+=("--top-p" "${EVAL_TOP_P}")
  fi
  if [[ -n "${EVAL_TOP_K}" ]]; then
    eval_cmd+=("--top-k" "${EVAL_TOP_K}")
  fi
  if [[ -n "${EVAL_DEVICE}" ]]; then
    eval_cmd+=("--device" "${EVAL_DEVICE}")
  fi
  if [[ -n "${EVAL_SUMMARY_FILENAME}" ]]; then
    eval_cmd+=("--summary-filename" "${EVAL_SUMMARY_FILENAME}")
  fi
  if [[ "${EVAL_RESUME}" == "1" ]]; then
    eval_cmd+=("--resume")
  fi
  if [[ -n "${EVAL_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    local eval_extra=( ${EVAL_EXTRA_ARGS} )
    eval_cmd+=("${eval_extra[@]}")
  fi

  run_cmd "evaluation-${stage_label}" "${eval_cmd[@]}"
}

print_cmd() {
  printf '        '
  printf '%q ' "$@"
  printf '\n'
}

run_cmd() {
  local phase="${1}"; shift
  if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '[dry-run] %s\n' "${phase}"
    print_cmd "$@"
    return 0
  fi
  log "starting ${phase}"
  print_cmd "$@"
  "$@"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config-root)
        [[ $# -lt 2 ]] && { log "--config-root expects a value"; usage; exit 1; }
        CONFIG_ROOT="$2"
        shift 2
        ;;
      --llama-config)
        [[ $# -lt 2 ]] && { log "--llama-config expects a value"; usage; exit 1; }
        LLAMA_CONFIG="$2"
        shift 2
        ;;
      --verl-config)
        [[ $# -lt 2 ]] && { log "--verl-config expects a value"; usage; exit 1; }
        VERL_CONFIG="$2"
        shift 2
        ;;
      --skip-pretrain)
        RUN_PRETRAIN=0
        shift
        ;;
      --skip-rl)
        RUN_RL=0
        shift
        ;;
      --skip-eval)
        RUN_EVAL=0
        shift
        ;;
      --do-eval)
        RUN_PRETRAIN=0
        RUN_RL=0
        RUN_EVAL=1
        DO_EVAL_ONLY=1
        shift
        ;;
      --llama-wandb-project)
        [[ $# -lt 2 ]] && { log "--llama-wandb-project expects a value"; usage; exit 1; }
        LLAMA_WANDB_PROJECT="$2"
        shift 2
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        log "Unknown argument: $1"
        usage
        exit 1
        ;;
    esac
  done
}

parse_args "$@"

llama_configs=()
verl_configs=()
default_run_names=()

if [[ -n "${LLAMA_CONFIG}" || -n "${VERL_CONFIG}" ]]; then
  if [[ -z "${LLAMA_CONFIG}" ]]; then
    log "LLAMA_CONFIG must be provided when using explicit config mode"
    exit 1
  fi
  if [[ ${RUN_RL} -eq 1 && -z "${VERL_CONFIG}" ]]; then
    log "VERL_CONFIG must be provided unless --skip-rl is set"
    exit 1
  fi
  llama_configs+=("${LLAMA_CONFIG}")
  if [[ -n "${VERL_CONFIG}" ]]; then
    verl_configs+=("${VERL_CONFIG}")
  else
    verl_configs+=("")
  fi
  base_name="$(basename "${LLAMA_CONFIG}")"
  base_name="${base_name%.*}"
  default_run_names+=("${base_name}")
else
  if [[ -z "${CONFIG_ROOT}" ]]; then
    log "CONFIG_ROOT is unset; provide --llama-config/--verl-config or pass --config-root"
    exit 1
  fi
  if [[ ! -d "${CONFIG_ROOT}" ]]; then
    log "Config root not found: ${CONFIG_ROOT}"
    exit 1
  fi

  if [[ -f "${CONFIG_ROOT}/llamafactory-config.yaml" ]]; then
    llama_configs+=("${CONFIG_ROOT}/llamafactory-config.yaml")
    if [[ -f "${CONFIG_ROOT}/verl-config.yaml" ]]; then
      verl_configs+=("${CONFIG_ROOT}/verl-config.yaml")
    else
      if [[ ${RUN_RL} -eq 1 ]]; then
        log "Missing verl-config.yaml under ${CONFIG_ROOT}"
        exit 1
      fi
      verl_configs+=("")
    fi
    default_run_names+=("$(basename "${CONFIG_ROOT}")")
  else
    shopt -s nullglob
    for dir in "${CONFIG_ROOT}"/*/; do
      dir="${dir%/}"
      if [[ -f "${dir}/llamafactory-config.yaml" ]]; then
        if [[ ${RUN_RL} -eq 1 && ! -f "${dir}/verl-config.yaml" ]]; then
          log "Missing verl-config.yaml under ${dir}" 
          exit 1
        fi
        llama_configs+=("${dir}/llamafactory-config.yaml")
        if [[ -f "${dir}/verl-config.yaml" ]]; then
          verl_configs+=("${dir}/verl-config.yaml")
        else
          verl_configs+=("")
        fi
        default_run_names+=("$(basename "${dir}")")
      fi
    done
    shopt -u nullglob
  fi
fi

if [[ ${#llama_configs[@]} -eq 0 ]]; then
  if [[ -n "${LLAMA_CONFIG}" || -n "${VERL_CONFIG}" ]]; then
    log "No runnable config pair detected from LLAMA_CONFIG/VERL_CONFIG"
  else
    log "No config pairs found under ${CONFIG_ROOT}"
  fi
  exit 1
fi

for idx in "${!llama_configs[@]}"; do
  llama_cfg="${llama_configs[$idx]}"
  verl_cfg="${verl_configs[$idx]}"
  default_run_name="${default_run_names[$idx]}"

  log "Processing config pair"
  log "  LLaMA-Factory config: ${llama_cfg}"
  if [[ -n "${verl_cfg}" ]]; then
    log "  verl config: ${verl_cfg}"
  else
    log "  verl config: (skipped)"
  fi

  if [[ ! -f "${llama_cfg}" ]]; then
    log "Missing LLaMA-Factory config: ${llama_cfg}"
    exit 1
  fi
  if [[ ${RUN_RL} -eq 1 ]]; then
    if [[ -z "${verl_cfg}" || ! -f "${verl_cfg}" ]]; then
      log "Missing verl config for RL stage: ${verl_cfg:-<unset>}"
      exit 1
    fi
  elif [[ -n "${verl_cfg}" && ! -f "${verl_cfg}" ]]; then
    log "Missing verl config: ${verl_cfg}"
    exit 1
  fi

  if ! llama_output_dir=$(extract_llama_output_dir "${llama_cfg}"); then
    log "Failed to read output_dir from ${llama_cfg}"
    exit 1
  fi

  if [[ -n "${LLAMA_EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    override_args=( ${LLAMA_EXTRA_ARGS} )
    for arg in "${override_args[@]}"; do
      if [[ "${arg}" == output_dir=* ]]; then
        llama_output_dir="${arg#output_dir=}"
        break
      fi
    done
  fi

  if ! llama_output_dir=$(abspath "${llama_output_dir}"); then
    log "Failed to resolve output_dir path for ${llama_cfg}"
    exit 1
  fi

  resolved_run_name="${LLAMA_RUN_NAME:-${default_run_name}}"
  pretrain_eval_attempted=0
  rl_eval_attempted=0

  llama_cfg_base="$(basename "${llama_cfg}")"
  llama_cfg_base="${llama_cfg_base%.*}"
  if [[ "${llama_cfg_base}" == "llamafactory-config" ]]; then
    llama_cfg_base="$(basename "$(dirname "${llama_cfg}")")"
  fi

  verl_context_available=0
  verl_cfg_dir=""
  verl_cfg_name=""
  verl_cfg_base=""
  suffix=""
  base_experiment_name=""
  experiment_name_override=""
  resolved_experiment_name=""
  storage_slug=""
  rl_stage_root=""
  rl_checkpoint_dir=""
  has_default_local_dir_override=0
  custom_default_local_dir=""
  rl_config_name_for_template=""
  rl_eval_run_name=""
  rl_latest_checkpoint_dir=""
  rl_merged_model_dir=""
  verl_extra_args_list=()
  actor_model_path_from_config=""
  actor_model_path_override=""

  if [[ -n "${verl_cfg}" ]]; then
    verl_context_available=1
    verl_cfg_dir="$(dirname "${verl_cfg}")"
    if ! verl_cfg_dir="$(abspath "${verl_cfg_dir}")"; then
      log "Failed to resolve verl config directory: ${verl_cfg_dir}"
      exit 1
    fi
    verl_cfg_name="$(basename "${verl_cfg}")"
    verl_cfg_name="${verl_cfg_name%.*}"

    verl_cfg_base="${verl_cfg_name}"
    if [[ "${verl_cfg_base}" == "verl-config" ]]; then
      verl_cfg_base="$(basename "$(dirname "${verl_cfg}")")"
    fi

    if [[ -n "${VERL_EXTRA_ARGS}" ]]; then
      # shellcheck disable=SC2206
      verl_extra_args_list=( ${VERL_EXTRA_ARGS} )
    fi

    if [[ -n "${llama_cfg_base}" && -n "${verl_cfg_base}" ]]; then
      suffix="${llama_cfg_base}--${verl_cfg_base}"
    elif [[ -n "${llama_cfg_base}" ]]; then
      suffix="${llama_cfg_base}"
    elif [[ -n "${verl_cfg_base}" ]]; then
      suffix="${verl_cfg_base}"
    fi

    if base_experiment_tmp="$(read_verl_experiment_name "${verl_cfg}" 2>/dev/null)"; then
      base_experiment_name="${base_experiment_tmp}"
    fi

    if actor_model_path_tmp="$(read_verl_actor_model_path "${verl_cfg}" 2>/dev/null)"; then
      actor_model_path_from_config="${actor_model_path_tmp}"
    fi

    has_extra_experiment_override=0
    for extra_arg in "${verl_extra_args_list[@]}"; do
      if [[ "${extra_arg}" == trainer.experiment_name=* ]]; then
        has_extra_experiment_override=1
        resolved_experiment_name="${extra_arg#trainer.experiment_name=}"
        break
      fi
    done

    if (( has_extra_experiment_override == 0 )); then
      if [[ -n "${suffix}" ]]; then
        experiment_name_override="${suffix}"
      elif [[ -n "${base_experiment_name}" ]]; then
        experiment_name_override="${base_experiment_name}"
      fi
    fi

    if [[ -n "${experiment_name_override}" ]]; then
      resolved_experiment_name="${experiment_name_override}"
    fi
    if [[ -z "${resolved_experiment_name}" && -n "${suffix}" ]]; then
      resolved_experiment_name="${suffix}"
    fi
    if [[ -z "${resolved_experiment_name}" && -n "${base_experiment_name}" ]]; then
      resolved_experiment_name="${base_experiment_name}"
    fi
    if [[ -z "${resolved_experiment_name}" ]]; then
      resolved_experiment_name="${default_run_name}"
    fi

    rl_stage_root="$(dirname "${llama_output_dir}")/rl"

    storage_slug="${resolved_experiment_name}"
    if [[ -z "${storage_slug}" ]]; then
      storage_slug="${verl_cfg_base:-${default_run_name}}"
    fi
    if [[ -z "${storage_slug}" ]]; then
      storage_slug="${resolved_run_name:-${default_run_name}}"
    fi

    rl_checkpoint_dir="${rl_stage_root}/${storage_slug}"

    has_default_local_dir_override=0
    custom_default_local_dir=""
    for extra_arg in "${verl_extra_args_list[@]}"; do
      if [[ "${extra_arg}" == trainer.default_local_dir=* ]]; then
        has_default_local_dir_override=1
        custom_default_local_dir="${extra_arg#trainer.default_local_dir=}"
        rl_checkpoint_dir="${custom_default_local_dir}"
        break
      fi
    done

    rl_config_name_for_template="${verl_cfg_base:-${default_run_name}}"
    rl_eval_run_name="${resolved_experiment_name:-${resolved_run_name}}"

    for extra_arg in "${verl_extra_args_list[@]}"; do
      if [[ "${extra_arg}" == actor_rollout_ref.model.path=* ]]; then
        actor_model_path_override="${extra_arg#actor_rollout_ref.model.path=}"
        break
      fi
    done
  fi

  if [[ ${RUN_PRETRAIN} -eq 1 ]]; then
    run_name="${resolved_run_name}"
    llama_cmd=("${LLAMA_BIN}" train "${llama_cfg}" "run_name=${run_name}")
    if [[ -n "${LLAMA_EXTRA_ARGS}" ]]; then
      # shellcheck disable=SC2206
      llama_extra=( ${LLAMA_EXTRA_ARGS} )
      llama_cmd+=("${llama_extra[@]}")
    fi
    if [[ -n "${LLAMA_WANDB_PROJECT}" ]]; then
      llama_cmd=(env "WANDB_PROJECT=${LLAMA_WANDB_PROJECT}" "${llama_cmd[@]}")
    fi
    run_cmd "LLaMA-Factory" "${llama_cmd[@]}"
    if [[ ${RUN_EVAL} -eq 1 ]]; then
      pretrain_eval_attempted=1
      maybe_run_eval "pretrain" "${llama_output_dir}" "${llama_cfg}" "pt" "${resolved_run_name}" "${default_run_name}" "" ""
    fi
  else
    log "Skipping LLaMA-Factory stage"
  fi

  if [[ ${RUN_RL} -eq 1 ]]; then
    if (( verl_context_available == 0 )); then
      log "RL stage requested but no verl config is available"
      exit 1
    fi
    llama_model_path=""
    if [[ -n "${actor_model_path_override}" ]]; then
      llama_model_path="${actor_model_path_override}"
      log "Using actor model path from VERL_EXTRA_ARGS: ${llama_model_path}"
    elif [[ -n "${actor_model_path_from_config}" ]]; then
      llama_model_path="${actor_model_path_from_config}"
      log "Using actor model path from verl config: ${llama_model_path}"
    else
      if ! llama_model_path=$(find_latest_llama_model "${llama_output_dir}"); then
        log "Unable to locate LLaMA-Factory outputs under ${llama_output_dir}"
        exit 1
      fi
    fi
    verl_cmd=("${VERL_PYTHON}" -m "${VERL_MODULE}" --config-path "${verl_cfg_dir}" --config-name "${verl_cfg_name}")

    auto_adjust_flag="${META_RUN_AUTO_GPU_ADJUST:-1}"
    if [[ "${auto_adjust_flag}" == "1" ]]; then
      visible_gpus="$(get_visible_device_count)"
      trainer_requested=""
      rollout_requested=""
      if read_gpu_info="$(read_verl_gpu_requests "${verl_cfg}" 2>/dev/null)"; then
        IFS=$'\n' read -r trainer_requested rollout_requested <<< "${read_gpu_info}" || true
      fi

      auto_overrides=()
      if [[ -n "${visible_gpus}" && "${visible_gpus}" =~ ^[0-9]+$ ]]; then
        if [[ -n "${trainer_requested}" && "${trainer_requested}" =~ ^[0-9]+$ ]]; then
          if (( trainer_requested > visible_gpus )); then
            log "Adjusting trainer.n_gpus_per_node from ${trainer_requested} to ${visible_gpus}"
            auto_overrides+=("trainer.n_gpus_per_node=${visible_gpus}")
          fi
        fi
        if [[ -n "${rollout_requested}" && "${rollout_requested}" =~ ^[0-9]+$ ]]; then
          if (( rollout_requested > visible_gpus )); then
            log "Adjusting actor_rollout_ref.rollout.n from ${rollout_requested} to ${visible_gpus}"
            auto_overrides+=("actor_rollout_ref.rollout.n=${visible_gpus}")
          fi
        fi
      fi

      if [[ ${#auto_overrides[@]} -gt 0 ]]; then
        verl_cmd+=("${auto_overrides[@]}")
      fi
    else
      log "META_RUN_AUTO_GPU_ADJUST disabled; relying on provided config values"
    fi
    if (( has_default_local_dir_override == 0 )); then
      log "Using RL checkpoint directory ${rl_checkpoint_dir}"
      verl_cmd+=("trainer.default_local_dir=${rl_checkpoint_dir}")
    else
      log "Using RL checkpoint directory override from VERL_EXTRA_ARGS: ${custom_default_local_dir}"
      rl_checkpoint_dir="${custom_default_local_dir}"
    fi

    if [[ -n "${experiment_name_override}" ]]; then
      verl_cmd+=("trainer.experiment_name=${experiment_name_override}")
    fi
    if [[ -z "${actor_model_path_override}" && -z "${actor_model_path_from_config}" ]]; then
      verl_cmd+=("actor_rollout_ref.model.path=${llama_model_path}")
    fi
    if (( ${#verl_extra_args_list[@]} > 0 )); then
      verl_cmd+=("${verl_extra_args_list[@]}")
    fi
    run_cmd "verl" "${verl_cmd[@]}"

    rl_latest_checkpoint_dir=""
    if rl_latest_checkpoint_dir=$(find_latest_verl_checkpoint "${rl_checkpoint_dir}" 2>/dev/null); then
      merge_backend="${META_RUN_RL_MERGE_BACKEND:-}"
      if [[ -z "${merge_backend}" ]]; then
        if detected_backend=$(detect_verl_backend "${verl_cfg}" 2>/dev/null); then
          merge_backend="${detected_backend}"
        fi
      fi
      if [[ -z "${merge_backend}" ]]; then
        merge_backend="fsdp"
      fi

      rl_actor_dir="${rl_latest_checkpoint_dir}/actor"
      if [[ ! -d "${rl_actor_dir}" ]]; then
        log "RL checkpoint actor directory missing at ${rl_actor_dir}; skipping merge"
      else
        rl_merged_model_dir="${rl_actor_dir}/huggingface"
        if [[ -d "${rl_merged_model_dir}" ]]; then
          log "Reusing existing merged directory ${rl_merged_model_dir}"
        else
          if [[ ${DRY_RUN} -eq 1 ]]; then
            log "[dry-run] Would create merged directory ${rl_merged_model_dir}"
          else
            mkdir -p -- "${rl_merged_model_dir}"
          fi
        fi
        log "Merging RL checkpoint ${rl_actor_dir} using backend ${merge_backend}"
        merge_cmd=("${VERL_PYTHON}" -m verl.model_merger merge --backend "${merge_backend}" --local_dir "${rl_actor_dir}" --target_dir "${rl_merged_model_dir}")
        if [[ -n "${META_RUN_RL_MERGE_EXTRA_ARGS:-}" ]]; then
          # shellcheck disable=SC2206
          merge_extra=( ${META_RUN_RL_MERGE_EXTRA_ARGS} )
          merge_cmd+=("${merge_extra[@]}")
        fi
        run_cmd "merge" "${merge_cmd[@]}"
      fi
    else
      log "Unable to locate RL checkpoints under ${rl_checkpoint_dir}; skipping merge"
    fi

  else
    log "Skipping verl stage"
  fi

  if [[ ${RUN_EVAL} -eq 1 && ${verl_context_available} -eq 1 && ( ${RUN_RL} -eq 1 || ${DO_EVAL_ONLY} -eq 1 ) ]]; then
    if [[ -z "${rl_checkpoint_dir}" ]]; then
      log "Skipping RL evaluation; checkpoints root is empty"
    else
      if [[ -z "${rl_latest_checkpoint_dir}" ]]; then
        if rl_latest_checkpoint_dir=$(find_latest_verl_checkpoint "${rl_checkpoint_dir}" 2>/dev/null); then
          :
        else
          rl_latest_checkpoint_dir=""
        fi
      fi

      if [[ -z "${rl_merged_model_dir}" && -n "${rl_latest_checkpoint_dir}" ]]; then
        candidate="${rl_latest_checkpoint_dir}/actor/huggingface"
        if [[ -d "${candidate}" ]]; then
          rl_merged_model_dir="${candidate}"
        fi
      fi

      rl_eval_root="${rl_checkpoint_dir}"
      if [[ -n "${rl_merged_model_dir}" && -d "${rl_merged_model_dir}" ]]; then
        rl_eval_root="${rl_merged_model_dir}"
      elif [[ -n "${rl_latest_checkpoint_dir}" && -d "${rl_latest_checkpoint_dir}" ]]; then
        rl_eval_root="${rl_latest_checkpoint_dir}"
      fi

      if [[ -z "${rl_eval_run_name}" ]]; then
        rl_eval_run_name="${resolved_run_name}"
      fi
      if [[ -z "${rl_eval_run_name}" ]]; then
        rl_eval_run_name="${default_run_name}"
      fi
      if [[ -z "${rl_config_name_for_template}" ]]; then
        rl_config_name_for_template="${default_run_name}"
      fi

      rl_eval_attempted=1
      maybe_run_eval "rl" "${rl_eval_root}" "${verl_cfg}" "rl" "${rl_eval_run_name}" "${rl_config_name_for_template}" "" "${rl_checkpoint_dir}"
    fi
  fi

  if [[ ${RUN_EVAL} -eq 1 && ${pretrain_eval_attempted} -eq 0 && ${rl_eval_attempted} -eq 0 ]]; then
    fallback_run_name="${resolved_run_name}"
    if [[ -z "${fallback_run_name}" ]]; then
      fallback_run_name="${default_run_name}"
    fi
    maybe_run_eval "manual" "${llama_output_dir}" "${llama_cfg}" "pt" "${fallback_run_name}" "${default_run_name}" "" ""
  elif [[ ${RUN_EVAL} -ne 1 ]]; then
    log "Skipping evaluation stage"
  fi

done

log "All tasks completed"
