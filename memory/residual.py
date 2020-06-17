import torch
from memory.memory import Memory

class ResidualMemory(Memory):
    def __init__(self, beta=1.0, gamma=1.0):
        super().__init__(cumulative_grads={}, residuals={})
        self.beta = beta
        self.gamma = gamma

    def compensate(self, tensor, name, worker):
        """Update the tensor with the residuals."""
        if name+str(worker) in self.residuals:
            tensor = self.beta * self.residuals[name+str(worker)] + self.gamma * tensor
        return tensor

    def update(self, tensor, name, worker, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        residual = tensor - tensor_decompressed
        self.residuals[name+str(worker)] = torch.clone(residual).detach()
