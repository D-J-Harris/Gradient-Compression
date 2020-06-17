import torch

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, device, args):
    """Cleanly divide data source into batches."""
    bsz = args.batch_size_train
    nbatch = data.size(0)  // bsz

    # trim off any extra elements that wouldn't cleanly fit (remainders)
    x = data.narrow(0, 0, bsz * nbatch)
    x = x.view(bsz, -1).t().contiguous()
    x = x.long()

    return x.to(device)


def get_batch(source, batch_idx, worker, args):
    """Returns the batch starting from position batch_idx."""
    i = batch_idx
    seq_len = min(args.seq_length, len(source) - 1 - i)
    data = source[(i * args.seq_length):(i * args.seq_length) + seq_len]
    target = source[(i * args.seq_length) + 1:(i * args.seq_length) + 1 + seq_len]

    data_split = data[:,(args.batch_size_train//args.num_workers) * worker: (args.batch_size_train//args.num_workers) * (worker+1)]
    target_split = target[:,(args.batch_size_train//args.num_workers) * worker: (args.batch_size_train//args.num_workers) * (worker+1)].reshape(-1)
    return data_split, target_split

def truncate(tensor, n_decimals):
    """Returns a tensor truncated to n_decimal places.
    Useful for ensuring equal values up to float precision"""
    return (tensor * 10 ** n_decimals).round() / (10 ** n_decimals)
