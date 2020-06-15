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
    return data.to(device)


def get_batch(args, source, batch_idx):
    """Returns the batch starting from position i."""
    step_start = batch_idx * args.num_steps
    seq_len = min(args.num_steps, len(source) - 1 - step_start)
    data = source[step_start:step_start+seq_len]
    target = source[step_start+1:step_start+1+seq_len].view(-1)
    return data, target
