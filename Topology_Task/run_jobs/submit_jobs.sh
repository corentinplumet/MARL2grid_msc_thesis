#!/bin/bash

# path="train"
# scripts=(
#     "PI_gradient_M_steps_last_error.py"
#     "PI_gradient_M_steps.py"
#     "PI_gradient.py"
#     "PI_lookback_gradient.py"
#     "PI_lookback.py"
#     "PI_M_steps_last_error.py"
#     "PI_M_steps.py"
#     "PI.py"
# )

path="train_real"
scripts=(
    "PI_gradient_M_steps.py"
    "PI_lookback_gradient.py"
)

for script in "${scripts[@]}"; do
    path_complete="${path}/${script}"
    sbatch train.sh "${path_complete}"
    echo "Submitted job for script: ${path_complete}"
done