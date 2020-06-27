#!/bin/sh
python -u main.py \
--hidden_size 100 \
--num_workers 2 \
--seq_length 35 \
--num_layers 2 \
--batch_size_train 32 \
--num_epochs 1 \
--dropout_prob 0.5 \
--tie_weights \
--seed 1111 \
--memory 'none' \
--compression 'none' \
--project_name 'temp' \

## missing --cuda, --wandb, --tie_weights and --save, among other flags

# 2>&1 means '2' 1>'1' 2>'1'
# i.e. '2' is the argument value for --tied,
# stdout (1) goes to file called '1'
# stderr (2) goes to file called '1'
