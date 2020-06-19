from abc import ABC, abstractmethod

class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self, is_sparse=False, is_quant=False, param_count={}):
        self.is_sparse = is_sparse
        self.is_quant = is_quant
        self.param_count = param_count

    @abstractmethod
    def compress(self, tensor, name):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compression was not implemented.")

    @abstractmethod
    def decompress(self, tensors, ctx):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")

    def update_running_ratio(self, name, ratio):
        if name not in self.param_count:
            self.param_count[name] = ratio
        else:
            self.param_count[name] += ratio
