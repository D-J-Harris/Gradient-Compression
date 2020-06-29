import torch
from memory.memory import Memory

class ResidualMemory(Memory):
    def __init__(self, beta=1.0, gamma=1.0):
        super().__init__(cumulative_grads={}, residuals={})
        self.beta = beta
        self.gamma = gamma

    def __str__(self):
        return "residual_memory"

    def compensate(self, tensor, name, worker):
        """Update the tensor with the residuals."""
        idx = name + str(worker)

        if idx in self.residuals:
            tensor = self.beta * self.residuals[idx] + self.gamma * tensor
        return tensor

    def update(self, tensor, name, worker, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        idx = name + str(worker)

        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        residual = tensor - tensor_decompressed
        self.residuals[idx] = torch.clone(residual).detach()
