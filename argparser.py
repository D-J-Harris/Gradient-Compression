import argparse

# separate file for defining arguments from terminal
def get_args():

    parser = argparse.ArgumentParser(description='LSTM model, Gradient Compression')

    parser.add_argument('--data', type=str, default='./penn',
                        help='location of the data corpus')

    parser.add_argument('--hidden_size', type=int, default=300,
                        help='size of word embeddings/hidden size LSTM')

    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers simulated')

    parser.add_argument('--seq_length', type=int, default=35,
                        help='backpropagation through time parameter')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of LSTM layers (>1 for dropout)')

    parser.add_argument('--batch_size_train', type=int, default=32,
                        help='batch size during training')

    parser.add_argument('--batch_size_test', type=int, default=1,
                        help='batch size during testing')

    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs')

    parser.add_argument('--dropout_prob', type=float, default=0.5,
                        help='dropout probability, regularisation')

    parser.add_argument('--tie_weights', action='store_true',
                        help='tie weights of in_ and out_embeddings')

    parser.add_argument('--initial_lr', type=float, default=20.0,
                        help='initial learning rate')

    parser.add_argument('--cuda', action='store_true',
                        help='default use CUDA')

    parser.add_argument('--log_interval', type=int, default=100,
                        help='report interval for measuring epoch progress')

    parser.add_argument('--project_name', type=str, default="project_name",
                        help='project name for wandb instance')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for consistent testing')

    parser.add_argument('--compression', type=str, default='none',
                        help='method used for gradient compression')

    parser.add_argument('--memory', type=str, default='none',
                        help='method used for memory on gradient residuals')

    parser.add_argument('--wandb', action='store_true',
                        help='default use wandb for metric logging')

    parser.add_argument('--compression_ratio', type=float, default=0.01,
                        help='default compression ratio for sparsification techniques')

    parser.add_argument('--quantum_num', type=int, default=64,
                        help='number of quantisation levels used in QSGD')


    args = parser.parse_args()

    print('number of workers:', args.num_workers)
    print('dropout used:', args.dropout_prob)
    print('initial learning rate:', args.initial_lr)
    print('training batch size:', args.batch_size_train)
    print('were the embedding weights tied:', args.tie_weights)
    print('hidden layer size:', args.hidden_size)

    return args
