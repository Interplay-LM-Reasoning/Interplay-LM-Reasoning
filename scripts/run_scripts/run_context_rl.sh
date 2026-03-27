# This scripts corresponds to the RL training in Section 3.
# It runs RL training for composition tasks with different operation sets.

# For pre-training:
bash scripts/composition/op-difficulty-10B/script_pt/run_pretrain_id2-10_0.2easy_0.3medium_0.5hard.sh

# For RL post-training with different variants of the composition tasks:
bash scripts/composition/op-difficulty-10B/script_rl/run_rl_id2-10_0.2easy_0.3medium_0.5hard.sh


