# This scripts corresponds to the mid / post-training data mixing ratio mentioned in Section 5.
# We explore on the extrapolation setting:
# For pre-training: (the same as Section 3) we use the data mixture of 20% easy, 30% medium, and 50% hard tasks.
bash scripts/composition/op-difficulty-10B/script_pt/run_pretrain_id2-10_0.2easy_0.3medium_0.5hard.sh

# In Section 5, we demonstrate the results of totally 1B tokens (50 steps for RL) for mid- / post-training with different mixing ratios.
# For mid-training: 


