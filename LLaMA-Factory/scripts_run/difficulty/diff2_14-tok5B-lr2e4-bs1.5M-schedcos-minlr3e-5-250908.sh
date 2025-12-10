
CONFIG="./examples/difficulty/diff2_14-tok5B-lr2e4-bs1.5M-schedcos-minlr3e-5-250908.yaml"
RUN_NAME="$(basename "${CONFIG}" .yaml)"
export WANDB_PROJECT="pr-difficulty" 

echo "[info] Using wandb project: ${WANDB_PROJECT}" >&2

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi

# Fixed defaults (6 GPUs single node)
GPU_LIST="4,5,6,7,8,9"
MASTER_PORT="29500"

# Derive NPROC from GPU_LIST if not explicitly set.
if [[ -z "${NPROC:-}" ]]; then
  IFS=, read -ra _gpus <<<"${GPU_LIST}"
  NPROC="${#_gpus[@]}"
fi

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT}"






if [[ "${NPROC}" -ne 6 ]]; then
  echo "[warn] Requested GPUs: ${NPROC} (not 6). Proceeding with ${NPROC}." >&2
fi

echo "[info] Launching LLaMA-Factory training with ${NPROC} GPU(s): ${GPU_LIST}" >&2

# Explicit overrides to lock config from the shell (no external env needed)
llamafactory-cli train ${CONFIG} \
  run_name=${RUN_NAME} \
  # overwrite_cache \



