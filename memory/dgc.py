import torch
from memory.memory import Memory


class DGCMemory(Memory):
    def __init__(self, momentum, gradient_clipping):
        super().__init__(cumulative_grads={}, residuals={})
        self.gradient_clipping = gradient_clipping
        self.momentum = momentum
        self.gradients = {}
        self.residuals = {}


    def __str__(self):
        return "dgc_memory"



    def compensate(self, tensor, name, worker):
        """Update the tensor with the residuals."""
        idx = name + str(worker)

        if self.gradient_clipping:
            tensor_squ_sum = torch.sum(tensor * tensor)
            clipping_val = torch.sqrt(tensor_squ_sum)
            tensor = tensor.clamp(-clipping_val, clipping_val)
        if idx in self.residuals:
            self.residuals[idx] = self.momentum * self.residuals[idx] + tensor
        else:
            self.residuals[idx] = tensor
        if idx in self.gradients:
            self.gradients[idx] += self.residuals[idx]
            tensor = self.gradients[idx]
        else:
            self.gradients[idx] = tensor
        return tensor

    def update(self, tensor, name, worker, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        idx = name + str(worker)

        shape, mask, _ = ctx
        not_mask = ~mask.view(shape)
        temp = self.residuals[idx] * not_mask
        self.residuals[idx] = temp
        temp = self.gradients[idx] * not_mask
        self.gradients[idx] = temp
