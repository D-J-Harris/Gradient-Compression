import os
import time
import wandb
import numpy as np

import torch.nn
import torch.nn as nn

import data_load
from model import LSTM
from argparser import get_args
from optim import DistributedSGD
from utils import repackage_hidden, batchify, get_batch, save_model

from memory.memory_chooser import memory_chooser
from compression.compression_chooser import compression_chooser

args = get_args()

def run_epoch(model, data, is_train=False):
    """Runs the model on the given data, for a single epoch"""

    # note dropout is disabled for eval() mode
    if is_train:
        model.train()
    else:
        model.eval()

    # hidden state, cell state indexed by worker number
    hiddens = {}
    for worker in range(args.num_workers):
        hiddens[str(worker)+'h'] = model.init_hidden()
        hiddens[str(worker)+'c'] = model.init_hidden()
    costs = 0.0

    epoch_size = data.size(0) // args.seq_length  # no. sequences that fit
    # loop over data in batches of sequence length defined by seq_length (bptt) parameter

    start = time.process_time()
    for batch_idx in range(epoch_size * args.num_workers):
        worker_num = (batch_idx+1) % args.num_workers
        seq_start = (batch_idx // args.num_workers) * args.seq_length

        # both in dims seq_length * batches
        inputs, targets = get_batch(data, seq_start, worker_num, args)

        hidden = repackage_hidden((hiddens[str(worker_num)+'h'], hiddens[str(worker_num)+'c']))
        outputs, hidden = model(inputs, hidden)
        hiddens[str(worker_num)+'h'] = torch.clone(hidden[0]).detach()
        hiddens[str(worker_num)+'c'] = torch.clone(hidden[1]).detach()

        loss = criterion(outputs.view(-1, vocab_size), targets)
        loss = loss / args.num_workers
        costs += loss.item()

        if is_train:
            # clear leaf nodes in graph, backward pass, compress and save grads
            # no clipping as this is not a linear operator (after grad summing only)
            model.zero_grad()
            loss.backward()
            optimizer.compress_step(worker_num)  # pass the worker number, for memory

            # step 'master model' once we have passed through n-workers' worth
            if worker_num == 0:
              optimizer.assign_grads()
              torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
              optimizer.step()

    # track the time taken to loop through this epoch
    end = time.process_time()
    return np.exp(costs / epoch_size), end-start


if __name__ == "__main__":

    ###############################################################################
    # define device and settings
    ###############################################################################

    # define the compression and residual saving techniques
    compressor = compression_chooser(args)
    memory = memory_chooser(args)
    print("Using compression: ", str(compressor))
    print("Using memory: ", str(memory))

    # Set the random seed manually for reproducibility, and device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda" if args.cuda else "cpu")

    # ensure the num_workers is a factor of batch_size_train
    if not args.batch_size_train % args.num_workers == 0:
        raise ValueError('make sure the batch size is a multiple of the number of workers')

    ###############################################################################
     # load data and define model
    ###############################################################################

    corpus = data_load.Corpus(args.data)

    print("Number of tokens:")
    print("Train: ", len(corpus.train))
    print("Valid: ", len(corpus.valid))
    print("Test:  ", len(corpus.test))

    train_data = batchify(corpus.train, device, args)
    val_data = batchify(corpus.valid, device, args)
    test_data = batchify(corpus.test, device, args, is_test=True)

    vocab_size = len(corpus.dictionary)
    print("Vocab size:\n{}".format(vocab_size))
    model = LSTM(embedding_dim=args.hidden_size, seq_length=args.seq_length,
                 batch_size=args.batch_size_train // args.num_workers,
                 num_workers=args.num_workers, vocab_size=vocab_size, num_layers=args.num_layers,
                 dropout_prob=args.dropout_prob, tie_weights=args.tie_weights)
    model.to(device)

    # initialize weights and biases for metric tracking
    if args.wandb:
        experiment_name = args.experiment_name + str(format(args.initial_lr, '.2f'))
        wandb.init(project=args.project_name, name=experiment_name, config=args, reinit=True)
        wandb.watch(model)

    print("Number of trainable model parameters:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    lr = args.initial_lr
    lr_decay_base = 1
    m_flat_lr = 14.0  # number of epochs before lr decay

    criterion = nn.CrossEntropyLoss()  # mean reduction i.e. sum over (seq_length * batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = DistributedSGD(optimizer, model.named_parameters(),
                               args.num_workers, compressor, memory)

    ###############################################################################
    # run training and save model
    ###############################################################################

    # training
    run_time = 0.0
    best_val = np.inf
    best_epoch = 0
    patience = args.patience
    for epoch in range(1, args.num_epochs + 1):
        lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
        lr = lr * lr_decay
        for g in optimizer.param_groups:
            g['lr'] = lr

        # train, log, val, log
        train_p, time_elapsed = run_epoch(model, train_data, is_train=True)
        run_time += time_elapsed
        print('\nTrain perplexity at epoch {}:      {:8.2f}'.format(epoch, train_p))
        val_p, _ = run_epoch(model, val_data)
        print('Validation perplexity at epoch {}: {:8.2f}\n'.format(epoch, val_p))

        if args.wandb:
            wandb.log({f'train perplexity': train_p})
            wandb.log({f'validation perplexity': val_p})

        # save this model, to back reference for early stopping
        save_model(args, model, epoch)

        # performing early stopping patience checks
        if val_p < best_val:
            best_val = val_p
            best_epoch = epoch
            patience = args.patience
        else:
            patience -= 1
            if patience == 0:
                break

    # print some metrics
    print('epoch size:', train_data.size(0) / args.seq_length)
    print('average time per epoch per worker:', run_time / (args.num_workers * args.num_epochs))

    # testing, set new batch size (to 1)
    # first load the best model
    model_path = os.path.join(args.save_model, 'model_'+str(best_epoch))
    model.load_state_dict(torch.load(model_path))

    model.batch_size = args.batch_size_test
    args.num_workers = 1
    test_p, _ = run_epoch(model, test_data)
    print('\nTest perplexity: {:8.2f}\n'.format(test_p))
    if args.wandb:
        wandb.log({f'test perplexity': test_p})
