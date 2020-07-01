import os
import torch

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, device, args, is_test=False):
    """Cleanly divide data source into batches."""
    if is_test:
      bsz = args.batch_size_test
    else:
      bsz = args.batch_size_train
    nbatch = data.size(0)  // bsz

    # trim off any extra elements that wouldn't cleanly fit (remainders)
    x = data.narrow(0, 0, bsz * nbatch)
    x = x.view(bsz, -1).t().contiguous()
    return x.to(device)


def get_batch(source, batch_idx, worker, args):
    """Returns the batch starting from position batch_idx."""
    i = batch_idx
    bsz = args.batch_size_train
    seq_len = min(args.seq_length, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len]

    # chunk the data along batch dim, retrieving a chunk indexed by worker
    data_split = data[:,(bsz//args.num_workers) * worker: (bsz//args.num_workers) * (worker+1)]
    target_split = target[:,(bsz//args.num_workers) * worker: (bsz//args.num_workers) * (worker+1)].reshape(-1)
    return data_split, target_split


def save_model(args, model, id):
    """Save a model to the defined path"""

    if not os.path.exists(args.save_model):
        os.mkdir(args.save_model)

    file = os.path.join(args.save_model, 'model_'+str(id))
    with open(file, 'wb') as f:
        torch.save(model.state_dict(), f)