# This scripts corresponds to the contextual pre-training in Section 4.
# We run pre-training for contextual tasks with different contextual templates and fix the post-train RL data distribution.
# The pre-training consists of zoo with op=2-20 tasks and teacher with op=2 tasks 


# For pre-training:

## Zoo with op=2-20 tasks (90%) + Teacher with op=2 tasks (10%)
bash scripts/mixed/mix-10B/script_pt/0.9zoo_op2-20+0.1teacher_1.0op2.sh
## Zoo with op=2-20 tasks (99%) + Teacher with op=2 tasks (1%)
bash scripts/mixed/mix-10B/script_pt/0.99zoo_op2-20+0.01teacher_1.0op2.sh
## Zoo with op=2-20 tasks (99.9%) + Teacher with op=2 tasks (0.1%)
bash scripts/mixed/mix-10B/script_pt/0.999zoo_op2-20+0.001teacher_1.0op2.sh


# For post-training RL with 50% zoo and 50% teacher data (the setting used in the paper):

bash scripts/mixed/mix-10B/script_rl/process/0.9zoo_op2-20+0.1teacher_1.0op2/run_rl_0.5_strict.sh

bash scripts/mixed/mix-10B/script_rl/process/0.99zoo_op2-20+0.01teacher_1.0op2/run_rl_0.5_strict.sh

bash scripts/mixed/mix-10B/script_rl/process/0.999zoo_op2-20+0.001teacher_1.0op2/run_rl_0.5_strict.sh