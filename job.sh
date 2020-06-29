#!/bin/sh
python -u main.py \
--hidden_size 300 \
--num_workers 1 \
--batch_size_train 32 \
--tie_weights \
--cuda \
--wandb \
--memory 'dgc' \
--compression 'dgc' \
--compression_ratio 0.001 \
--quantum_num 64 \
--project_name 'project_name' \
