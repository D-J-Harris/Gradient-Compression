import math
import torch
import torch.nn as nn


class LSTM(nn.Module):
  """Simple LSMT-based language model"""
  def __init__(self, embedding_dim, num_steps, batch_size, num_workers, vocab_size, num_layers, dropout_prob, tie_weights):
    super(LSTM, self).__init__()
    self.embedding_dim = embedding_dim
    self.num_steps = num_steps
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.vocab_size = vocab_size
    self.num_layers = num_layers
    self.dropout_prob = dropout_prob
    self.dropout = nn.Dropout(dropout_prob)
    self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=embedding_dim,
                            num_layers=num_layers,
                            dropout=dropout_prob)

    # custom weight init in torch.nn.Embedding
    emb_w = torch.Tensor(vocab_size, embedding_dim)
    stdv = 1. / math.sqrt(emb_w.size(1))  # like in nn.Linear
    emb_w.uniform_(-stdv, stdv)
    self.embedding = nn.Embedding(vocab_size, embedding_dim, _weight=emb_w)

    # when tied embedding matrix weights, input_dim === hidden_size
    self.tie_weights = tie_weights
    if not tie_weights:
      self.decoder = nn.Linear(in_features=embedding_dim, out_features=vocab_size)
      self.decoder.weight.data.uniform_(-stdv, stdv)
      self.decoder.bias.data.fill_(0.0)


  def init_hidden(self):
    """Initialise the hidden weights in LSTM."""
    weight = next(self.parameters()).data
    return (weight.new(self.num_layers, self.batch_size, self.embedding_dim).zero_(),
            weight.new(self.num_layers, self.batch_size, self.embedding_dim).zero_())


  def forward(self, inputs, hidden):
    """Run a forward pass of the LSTM model."""
    embeds = self.dropout(self.embedding(inputs))
    lstm_out, hidden = self.lstm(embeds, hidden)
    lstm_out = self.dropout(lstm_out)

    if self.tie_weights:
      logits = lstm_out.view(-1, self.embedding_dim).mm(self.embedding.weight.t())
    else:
      logits = self.decoder(lstm_out.view(-1, self.embedding_dim))

    num_steps = lstm_out.size(0)
    return logits.view(num_steps, self.batch_size, self.vocab_size), hidden


def repackage_hidden(h):
  """Wraps hidden states in new Tensors, to detach them from their history."""
  if isinstance(h, torch.Tensor):
    return h.detach()
  else:
    return tuple(repackage_hidden(v) for v in h)
