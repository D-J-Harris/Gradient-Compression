#!/bin/sh
python -u main.py \
--hidden_size 400 \
--num_workers 2 \
--num_steps 35 \
--num_layers 2 \
--batch_size_train 50 \
--num_epochs 1 \
--seed 1111 \
--memory 'none' \
--compression 'none' \
--project_name 'default' \


## missing --cuda, --wandb, --tie_weights and --save, among other flags

# 2>&1 means '2' 1>'1' 2>'1'
# i.e. '2' is the argument value for --tied,
# stdout (1) goes to file called '1'
# stderr (2) goes to file called '1'
