if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES must be set before running this script" >&2
    exit 1
fi


BASE_MODEL=id2-10_0.2easy_0.3medium_0.5hard
EVAL_DATA_ROOT=data/composition/test \
LLAMA_CONFIG=scripts/composition/op-difficulty-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/${BASE_MODEL}.yaml \
VERL_CONFIG=scripts/composition/op-difficulty-10B/rl-200steps/op11-14_uniform_process0.5.yaml \
./scripts/meta_run.sh \
 --skip-pretrain
