from abc import ABC, abstractmethod

class Memory(ABC):

    def __init__(self, cumulative_grads):
        self.cumulative_grads = cumulative_grads

    @abstractmethod
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        raise NotImplemented("compensate was not implemented.")

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass