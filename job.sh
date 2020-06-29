#!/bin/sh
python -u main.py \
--hidden_size 100 \
--num_workers 1 \
--seq_length 35 \
--num_layers 2 \
--batch_size_train 32 \
--num_epochs 50 \
--dropout_prob 0.5 \
--wandb \
--initial_lr 10 \
--tie_weights \
--memory 'none' \
--compression 'none' \
--project_name 'qsgd' \
