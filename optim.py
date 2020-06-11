## built on from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
## additionally from https://horovod.readthedocs.io/en/latest/pytorch.html

import torch
from torch.optim.optimizer import Optimizer

from compression.none_compressor import NoneCompressor

class Compression(object):
    """Optional gradient compression algorithm used during distributed training"""
    none = NoneCompressor()


class _DistributedSGD(Optimizer):
    """
    custom SGD class that additionally implements gradient compression
    before each step update
    """

    def __init__(self, params, named_parameters, compression=Compression.none):
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
            raise ValueError('named_parameters was specified, but one or more model '
                             'parameters were not named. Python object ids: '
                             '%s' % ', '.join(str(id) for id in unnamed_param_ids))

        self._parameter_names = {v: k for k, v in sorted(named_parameters)}
        self._compression = compression

        ## @here change to the memory class
        self.cum_grad_update = {}


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
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                d_p = self.cum_grad_update[i]
                p.add_(d_p, alpha=-group['lr'])

        return loss


    @torch.no_grad()
    def compress_step(self):
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

            for i, p in enumerate(group['params']):
                name = self._parameter_names.get(p)
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                d_p_comp, ctx = self._compression.compress(d_p, name)
                if i not in self.cum_grad_update:
                    self.cum_grad_update[i] = d_p_comp
                else:
                    self.cum_grad_update[i] += d_p_comp


def DistributedSGD(optimizer, named_parameters=None, compression=Compression.none):
    """method allowing for addition of named parameters to optimiser
    (helps for compressor memory)."""
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedSGD.__dict__))
    return cls(optimizer.param_groups, named_parameters, compression)
