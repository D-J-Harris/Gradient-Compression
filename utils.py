import torch

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, device, args):
    """Cleanly divide data source into batches."""
    nseq = (data.size(0) - 1) // args.seq_length

    # trim off any extra elements that wouldn't cleanly fit (remainders)
    x = data.narrow(0, 0, args.seq_length * nseq)
    x = x.view(-1, args.seq_length)
    x = x.long()

    y = data.narrow(0, 1, args.seq_length * nseq)
    y = y.view(-1, args.seq_length)
    y = y.long()

    return x.to(device), y.to(device)


def get_batch(source, batch_size, batch_idx):
    """Returns the batch starting from position batch_idx."""

    start = batch_size * batch_idx
    end = batch_size * (batch_idx+1)
    data = source[start:end].t().contiguous()
    return data
