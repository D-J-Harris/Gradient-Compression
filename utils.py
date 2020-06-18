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
    data.long()
    return data.to(device)


def get_batch(args, source, i, worker_num):
    """Returns the batch starting from position i."""
    seq_len = min(args.num_steps, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return torch.chunk(data, chunks=args.num_workers, dim=1)[worker_num],\
           torch.chunk(target, chunks=args.num_workers, dim=1)[worker_num].reshape(-1)
