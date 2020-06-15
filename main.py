import wandb
import argparse
import numpy as np

import torch
import torch.nn
import torch.nn as nn

import data_load
from model import LSTM
from optim import DistributedSGD
from utils import repackage_hidden, batchify, get_batch

from memory.memory_chooser import memory_chooser
from compression.compression_chooser import compression_chooser

parser = argparse.ArgumentParser(description='LSTM-based language model')
parser.add_argument('--data', type=str, default='./penn',
                    help='location of the data corpus')
parser.add_argument('--hidden_size', type=int, default=250,
                    help='size of word embeddings/hidden size LSTM')
parser.add_argument('--num_workers', type=int, default=1,
                    help='number of workers simulated')
parser.add_argument('--num_steps', type=int, default=35,
                    help='backpropagation through time parameter')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of LSTM layers (>1 for dropout)')
parser.add_argument('--batch_size_train', type=int, default=24,
                    help='batch size during training')
parser.add_argument('--batch_size_test', type=int, default=1,
                    help='batch size during testing')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs')
parser.add_argument('--dropout_prob', type=float, default=0.0,
                    help='dropout probability, regularisation')
parser.add_argument('--tie_weights', action='store_true',
                    help='tie weights of in_ and out_embeddings')
parser.add_argument('--initial_lr', type=float, default=20.0,
                    help='initial learning rate')
parser.add_argument('--save', type=str,  default='models_logs/lm_model.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', action='store_true',
                    help='default use CUDA')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval for measuring epoch progress')
parser.add_argument('--project_name', type=str, default="test_run",
                    help='project name for wandb instance')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed for consistent testing')
parser.add_argument('--compression', type=str, default='none',
                    help='method used for gradient compression')
parser.add_argument('--memory', type=str, default='none',
                    help='method used for memory on gradient residuals')
parser.add_argument('--wandb', action='store_true',
                    help='default use wandb for metric logging')
args = parser.parse_args()


def run_epoch(model, data, is_train=False):
    """Runs the model on the given data, for a single epoch"""

    # note dropout is disabled for eval() mode
    if is_train:
        model.train()
    else:
        model.eval()

    # epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    hidden = model.init_hidden()
    costs = 0.0
    iters = 0

    # loop over data in batches of sequence length defined by bptt parameter
    for batch, i in enumerate(range(0, data.size(0) - 1, args.num_steps)):
        inputs, targets = get_batch(args, data, i)

        hidden = repackage_hidden(hidden)
        outputs, hidden = model(inputs, hidden)

        # add/divide by num_steps is for weighting purposes
        loss = criterion(outputs.view(-1, vocab_size), targets)
        costs += loss.item() * model.num_steps
        iters += model.num_steps

        if is_train:
            # clear leaf nodes in graph, backward pass,
            # clip, compress and save grads
            model.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.compress_step()

            # step 'master model' once we have passed through n-workers' worth
            if (batch+1) % model.num_workers == 0:
                optimizer.step()

        # log in output, not to wandb though
        if batch % args.log_interval == 0 and batch > 0:
            print('epoch progress {:.3f}%  -->  ppl {:8.2f}'.format(
                i * 100.0 / data.size(0),
                np.exp(costs / iters)))

    return np.exp(costs / iters)


if __name__ == "__main__":

    ###############################################################################
    # define device and settings
    ###############################################################################

    compressor = compression_chooser(args.compression)
    memory = memory_chooser(args.memory)

    # Set the random seed manually for reproducibility.
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

    train_data = batchify(corpus.train, device, args.batch_size_train)
    val_data = batchify(corpus.valid, device, args.batch_size_train)
    test_data = batchify(corpus.test, device, args.batch_size_test)

    vocab_size = len(corpus.dictionary)
    print("Vocab size:\n{}".format(vocab_size))
    model = LSTM(embedding_dim=args.hidden_size, num_steps=args.num_steps,
                 batch_size=args.batch_size_train, num_workers=args.num_workers,
                 vocab_size=vocab_size, num_layers=args.num_layers,
                 dropout_prob=args.dropout_prob, tie_weights=args.tie_weights)
    model.to(device)

    # initialize weights and biases for metric tracking
    if args.wandb:
        wandb.init(project=args.project_name, reinit=True)
        wandb.watch(model)

    print("Number of model parameters:")
    print(sum(p.numel() for p in model.parameters()))
    print("Number of trainable model parameters:")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    lr = args.initial_lr
    lr_decay_base = 1 / 1.2
    m_flat_lr = 6.0  # number of epochs before lr decay

    criterion = nn.CrossEntropyLoss()  # reduction is mean i.e. divide by num_steps * batch_size
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = DistributedSGD(optimizer, model.named_parameters(),
                               args.num_workers, compressor, memory)

    ###############################################################################
    # run training and save model
    ###############################################################################

    # training
    for epoch in range(1, args.num_epochs + 1):
        lr_decay = lr_decay_base ** max(epoch - m_flat_lr, 0)
        lr = lr * lr_decay
        for g in optimizer.param_groups:
            g['lr'] = lr

        # train, log, val, log
        train_p = run_epoch(model, train_data, is_train=True)
        print('\nTrain perplexity at epoch {}: {:8.2f}\n'.format(epoch, train_p))
        val_p = run_epoch(model, val_data)
        print('\nValidation perplexity at epoch {}: {:8.2f}\n'.format(epoch, val_p))

        if args.wandb:
            wandb.log({f'train perplexity': train_p})
            wandb.log({f'validation perplexity': val_p})

    # testing, set new batch size (to 1)
    model.batch_size = args.batch_size_test
    test_p = run_epoch(model, test_data)
    print('\nTest perplexity: {:8.2f}\n'.format(test_p))

    if args.wandb:
        wandb.log({f'test perplexity': test_p})
        wandb.log({f'dropout': args.dropout_prob})
        wandb.log({f'num_workers': args.num_workers})
        wandb.log({f'initial_lr': args.initial_lr})
        wandb.log({f'batch size': args.batch_size_train})
        wandb.log({f'tied_weights': args.tie_weights})
        wandb.log({f'batch size': args.batch_size_train})
        wandb.log({f'compression': args.compression})
        wandb.log({f'hidden_size': args.hidden_size})

    # save the model locally
    # with open(args.save, 'wb') as f:
    #     torch.save(model.state_dict(), f)
