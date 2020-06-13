#!/bin/sh
python -u main.py \
--hidden_size 50 \
--num_workers 2 \
--num_steps 35 \
--num_layers 2 \
--batch_size_train 12 \
--num_epochs 1 \
--tie_weights \
--memory 'none' \
--compression 'none' \
--project_name 'testers' \


## missing --cuda, --wandb, --tie_weights and --save, among other flags

# 2>&1 means '2' 1>'1' 2>'1'
# i.e. '2' is the argument value for --tied,
# stdout (1) goes to file called '1'
# stderr (2) goes to file called '1'
