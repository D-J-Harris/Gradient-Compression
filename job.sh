#!/bin/sh
python -u main.py \
--hidden_size 600 \
--num_workers 1 \
--seq_length 35 \
--num_layers 4 \
--batch_size_train 32 \
--dropout_prob 0.5 \
--initial_lr 20 \
--tie_weights \
--memory 'none' \
--compression 'none' \
--project_name 'project_name' \
