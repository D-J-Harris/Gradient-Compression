#!/bin/sh
python -u main.py \
--hidden_size 300 \
--num_workers 4 \
--seq_length 35 \
--num_layers 2 \
--batch_size_train 32 \
--dropout_prob 0.5 \
--initial_lr 20 \
--tie_weights \
--cuda \
--wandb \
--memory 'dgc' \
--compression 'dgc' \
--compression_ratio 0.001 \
--project_name 'project_name' \