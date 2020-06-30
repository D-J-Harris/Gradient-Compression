#!/bin/sh
python -u main.py \
--cuda \
--wandb \
--memory 'dgc' \
--compression 'dgc' \
--compression_ratio 0.001 \
--quantum_num 64 \
--project_name 'project_name' \
--experiment_name 'experiment_name' \
