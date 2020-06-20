import torch
from memory.memory import Memory


class DGCMemory(Memory):
    def __init__(self, momentum, gradient_clipping):
        super().__init__(cumulative_grads={}, residuals={})
        self.gradient_clipping = gradient_clipping
        self.momentum = momentum


    def __str__(self):
        return "dgc_memory"


    def compensate(self, tensor, name, worker):
        """Update the tensor with the residuals."""
        idx = name+str(worker)

        if self.gradient_clipping:
            tensor_squ_sum = torch.sum(tensor * tensor)
            clipping_val = torch.sqrt(tensor_squ_sum)
            tensor = tensor.clamp(-clipping_val, clipping_val)
        if idx in self.residuals:
            self.residuals[idx] = self.momentum * self.residuals[idx].clone() + tensor.clone()
        else:
            self.residuals[idx] = tensor.clone()
        if idx in self.cumulative_grads:
            self.cumulative_grads[idx] += self.residuals[idx].clone()
            tensor = self.cumulative_grads[idx].clone()
        else:
            self.cumulative_grads[idx] = tensor.clone()
        return tensor

    def update(self, tensor, name, worker, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        idx = name+str(worker)
        shape, mask, _ = ctx
        not_mask = ~mask.view(shape)
        temp = self.residuals[idx].clone() * not_mask
        self.residuals[idx] = temp.clone()
        temp = self.cumulative_grads[idx].clone() * not_mask
        self.cumulative_grads[idx] = temp
