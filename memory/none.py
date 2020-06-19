from memory.memory import Memory

class NoneMemory(Memory):

    def __init__(self):
        super().__init__(cumulative_grads={}, residuals={})

    def __str__(self):
        return "none_memory"

    def compensate(self, tensor, name, worker):
        """Update the tensor with the residuals."""
        return tensor

    def update(self, tensor, name, worker, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass
