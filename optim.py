## built on from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py

import torch
from compress.none_compressor import NoneCompressor
from torch.optim.optimizer import Optimizer


class CustomSGD(Optimizer):
    """
    custom SGD class that additionally implements gradient compression
    before each step update
    """

    def __init__(self, params, compressor=NoneCompressor(), lr=1.0, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        self.compressor = compressor
        self.cum_grad_update = {}
        self.lr = lr
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CustomSGD, self).__init__(params, defaults)


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

        for i, p in enumerate(self.param_groups[0]['params']):
            d_p = self.cum_grad_update[i]
            p.add_(d_p, alpha=-self.lr)

        return loss

    def compress_step(self):
        """
        accumulates the compressed gradients as it goes along,
        each step of this method reflects a worker compressing its
        own gradients
        """

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for i, p in enumerate(group['params']):
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

                d_p_comp, ctx = self.compressor.compress(d_p)
                if i not in self.cum_grad_update:
                    self.cum_grad_update[i] = d_p_comp
                else:
                    self.cum_grad_update[i] += d_p_comp