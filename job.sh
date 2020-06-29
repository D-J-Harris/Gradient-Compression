#!/bin/sh
python -u main.py \
--hidden_size 300 \
--num_workers 1 \
--seq_length 35 \
--num_layers 2 \
--batch_size_train 32 \
--dropout_prob 0.5 \
--initial_lr 20 \
--tie_weights \
--cuda \
--wandb \
--memory 'none' \
--compression 'none' \
--project_name 'project_name' \
