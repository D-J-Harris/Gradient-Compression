## built on from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
## additionally from https://horovod.readthedocs.io/en/latest/pytorch.html

import torch
from torch.optim.optimizer import Optimizer

from memory.none import NoneMemory
from compression.none import NoneCompression

class _DistributedSGD(Optimizer):
    """
    custom SGD class that additionally implements gradient compression
    before each step update
    """

    def __init__(self, params, named_parameters, num_workers=1, compression=NoneCompression(),
                 memory=NoneMemory()):
        super(self.__class__, self).__init__(params)

        # checks below taken from horovod library
        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = [('manual_named_param.%s' % i, v)
                                for param_group in self.param_groups
                                for i, v in enumerate(param_group['params'])]
        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        dups = _DistributedSGD.find_duplicates([k for k, _ in named_parameters])
        if len(dups) > 0:
            raise ValueError('Parameter names in named_parameters must be unique. '
                             'Found duplicates: %s' % ', '.join(dups))

        all_param_ids = {id(v)
                         for param_group in self.param_groups
                         for v in param_group['params']}
        named_param_ids = {id(v) for k, v in named_parameters}
        unnamed_param_ids = all_param_ids - named_param_ids
        if len(unnamed_param_ids):
            raise ValueError('named_parameters was specified, but 2 or more model '
                             'parameters were not named. Python object ids: '
                             '%s' % ', '.join(str(id) for id in unnamed_param_ids))

        self.parameter_names = {v: k for k, v in sorted(named_parameters)}
        self.num_workers = num_workers
        self.compression = compression
        self.memory = memory


    @staticmethod
    def find_duplicates(lst):
        seen = set()
        dups = set()
        for el in lst:
            if el in seen:
                dups.add(el)
            seen.add(el)
        return dups


    @torch.no_grad()
    def assign_grads(self, closure=None):
        """Loops over grads and places accumulated gradient into them.
        """

        for group in self.param_groups:
            for p in group['params']:
                name = self.parameter_names.get(p)

                # get accumulated gradients and context from memory
                d_p = self.memory.cumulative_grads[name].detach()
                p.grad = d_p

        self.memory.cumulative_grads = {}


    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        """

        for group in self.param_groups:
            for p in group['params']:

                if p.grad is None:
                    continue
                d_p = torch.clone(p.grad).detach()
                p.add_(d_p, alpha=-group['lr'])
                p.grad = None


    @torch.no_grad()
    def compress_step(self, worker):
        """
        accumulates the compressed gradients as it goes along,
        each step of this method reflects a worker compressing its
        own gradients.
        """

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                name = self.parameter_names.get(p)
                if p.grad is None:
                    continue
                d_p = p.grad

                # apply hyperparameter adjustments
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1-dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # memory compensate, grad compress, then memory update
                d_p = self.memory.compensate(d_p, name, worker)
                d_p_comp, ctx = self.compression.compress(d_p, name)
                self.memory.update(d_p, name, worker, self.compression, d_p_comp, ctx)

                # always decompress before accumulation
                # this doesnt affect sparsification, and is standard e.g. in qsgd
                d_p_comp = self.compression.decompress(d_p_comp, ctx)

                # if first worker, initialise dict of cumulative grads
                if name not in self.memory.cumulative_grads:
                    self.memory.cumulative_grads[name] = torch.clone(d_p_comp).detach()
                else:
                    self.memory.cumulative_grads[name] += torch.clone(d_p_comp).detach()


def DistributedSGD(optimizer, named_parameters=None, num_workers=1,
                   compression=NoneCompression(), memory=NoneMemory()):
    """method allowing for addition of named parameters to optimiser
    (helps for compressor memory)."""
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedSGD.__dict__))
    return cls(optimizer.param_groups, named_parameters, num_workers, compression, memory)
