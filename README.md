# Gradient-Compression in Distributed Deep Learning

## MSc Thesis in Artificial Intelligence, for University of Edinburgh

Investigating the application of gradient compression techniques used to speed up distributed deep learning.
Code credit to KAUST for their [GRACE framework](https://github.com/sands-lab/grace), from which compression method implementations are inspired, and to Horovod for the [base distributed optimiser code](https://github.com/horovod/horovod/blob/master/horovod/torch/optimizer.py) this work is built upon.



## Abstract

Distributed deep learning is a technique used to reduce the large training times
associated with large, complex models by training across multiple processors.
However, as these distributed computations are scaled up across more and more
processors, the communication of model training updates between these processors
can start to become a bottleneck - potentially outweighing the savings in
running costs gained by moving to a distributed environment in the first place.
Therefore, the field of communication-efficient training has become an important
research area which looks to satisfy the accelerating demand of distributed
training, and its related costs.
This work looks at gradient compression, a technique used to reduce the
memory volume of the training updates that are communicated. A comparative
survey is carried about between three of the most prominent gradient compression
techniques, Top-k, DGC and QSGD, focusing on providing a fair and
quantitative evaluation across the spectrum of metrics that defines the field.
It is found that tuning models on each compression setting separately, as opposed
to adopting baseline settings, can lead to a fairer evaluation of techniques
- sometimes even consistently beating the baseline as model size and worker
number is scaled. In addition, sparsification techniques maintain modest performance
relative to the baseline just like quantisation, but with higher gradient
compression. However, this higher compression can come at a cost of additional
computation time or losses in convergence time which negate any positive benefits
of the compression. With a range of experimental details and settings particular
to a given situation, it is ultimately concluded that there is no objectively
best compression technique of those surveyed, and consideration of the resources
at hand is the most useful indicator for which technique is best to adopt.

built on from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
additionally from https://horovod.readthedocs.io/en/latest/pytorch.html
