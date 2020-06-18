from abc import ABC, abstractmethod


class Memory(ABC):

    def __init__(self, cumulative_grads, residuals):
        self.cumulative_grads = cumulative_grads
        self.residuals = residuals

    @abstractmethod
    def compensate(self, tensor, name, worker):
        """Update the tensor with the residuals."""
        raise NotImplemented("compensate was not implemented.")

    def update(self, tensor, name, worker, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass