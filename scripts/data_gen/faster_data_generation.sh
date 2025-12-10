# Ultra-fast generation with all optimizations
python scripts/data_gen/generate_zero_context_medium.py \
    --samples_per_op 12000000 \
    --numprocs 32 \
    --batch_generate 100 \
    --aggressive_opmax \
    --dedicated_ops \
    --resume \
    --max_attempts 500000000