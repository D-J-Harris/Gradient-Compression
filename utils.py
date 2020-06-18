import torch

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, device, bsz):
    """Cleanly divide data source into batches."""
    nbatch = data.size(0) // bsz
    # trim off any extra elements that wouldn't cleanly fit (remainders)
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    data = data.long()
    return data.to(device)


def get_batch(args, source, i, worker):
    """Returns the batch starting from position i."""
    seq_len = min(args.num_steps, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    bsz = args.batch_size_train

    data_split = data[:,(bsz // args.num_workers) * worker: (bsz // args.num_workers) * (worker + 1)]
    target_split = target[:,(bsz // args.num_workers) * worker: (bsz // args.num_workers) * (worker + 1)].reshape(-1)
    return data_split, target_split


def truncate(tensor, n_decimals):
    """Returns a tensor truncated to n_decimal places.
    Useful for ensuring equal values up to float precision"""
    return (tensor * 10 ** n_decimals).round() / (10 ** n_decimals)
