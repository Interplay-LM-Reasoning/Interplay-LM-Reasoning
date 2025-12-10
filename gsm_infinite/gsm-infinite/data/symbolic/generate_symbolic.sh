#!/bin/bash

dataset_base_name="Clockz/gsm_infinite_symbolic"


# feel free to adjust the parameters.
python3 generate_symbolic.py --dataset-name $dataset_base_name --length 0 --stride 1 --min-op 1 --max-op 120 --examples-per-op 100

python3 generate_symbolic.py --dataset-name $dataset_base_name --length 8k --stride 1 --min-op 1 --max-op 30 --examples-per-op 100

python3 generate_symbolic.py --dataset-name $dataset_base_name --length 16k --stride 1 --min-op 1 --max-op 30 --examples-per-op 100

python3 generate_symbolic.py --dataset-name $dataset_base_name --length 32k --stride 1 --min-op 1 --max-op 30 --examples-per-op 100