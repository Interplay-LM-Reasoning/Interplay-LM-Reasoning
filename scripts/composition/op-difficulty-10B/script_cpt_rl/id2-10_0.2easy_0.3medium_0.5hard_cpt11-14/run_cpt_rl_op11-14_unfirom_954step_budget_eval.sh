CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}

PT_CONFIG="scripts/composition/op-difficulty-10B/pt-diff2_10-tok10B-lr1e-4-bs512k-schedcos-minlr3e-5/id2-10_0.2easy_0.3medium_0.5hard_cpt11-14_plus.yaml"
CONFIG_NAME="scripts/composition/op-difficulty-10B/cpt-rl-200steps/cpt-rl-op11-14_uniform-954step.yaml" \


# # 80% CPT + 20% RL
# EVAL_DATA_ROOT="data/composition/val" \
# EVAL_DATA_DIR="${EVAL_DATA_ROOT}" \
# CPT_CHECKPOINT_PATH="saves/composition-10B/op_level/id2-10_0.2easy_0.3medium_0.5hard/cpt0.2-uniform_0.8-11-14_plus/checkpoint-30573" \
# VERL_EXTRA_ARGS="actor_rollout_ref.model.path=${CPT_CHECKPOINT_PATH} data.cpt-epoch=0.8 trainer.experiment_name=cpt0.8-rl-op11-14_uniform-954step-0.2RL" \
# LLAMA_CONFIG="${PT_CONFIG}" \
# VERL_CONFIG="${CONFIG_NAME}" \
# ./scripts/meta_run.sh \
#   --skip-pretrain

# 50% CPT + 50% RL
EVAL_DATA_ROOT="data/composition/val" \
EVAL_DATA_DIR="${EVAL_DATA_ROOT}" \
CPT_CHECKPOINT_PATH="saves/composition-10B/op_level/id2-10_0.2easy_0.3medium_0.5hard/cpt0.2-uniform_0.8-11-14_plus/checkpoint-18963" \
VERL_EXTRA_ARGS="actor_rollout_ref.model.path=${CPT_CHECKPOINT_PATH} data.cpt-epoch=0.5 trainer.experiment_name=cpt0.5-rl-op11-14_uniform-954step-0.5RL" \
LLAMA_CONFIG="${PT_CONFIG}" \
VERL_CONFIG="${CONFIG_NAME}" \
./scripts/meta_run.sh \
  --skip-pretrain --do-eval


# 20% CPT + 80% RL
EVAL_DATA_ROOT="data/composition/val" \
EVAL_DATA_DIR="${EVAL_DATA_ROOT}" \
CPT_CHECKPOINT_PATH="saves/composition-10B/op_level/id2-10_0.2easy_0.3medium_0.5hard/cpt0.2-uniform_0.8-11-14_plus/checkpoint-7740" \
VERL_EXTRA_ARGS="actor_rollout_ref.model.path=${CPT_CHECKPOINT_PATH} data.cpt-epoch=0.2 trainer.experiment_name=cpt0.2-rl-op11-14_uniform-954step-0.8RL" \
LLAMA_CONFIG="${PT_CONFIG}" \
VERL_CONFIG="${CONFIG_NAME}" \
./scripts/meta_run.sh \
  --skip-pretrain --do-eval