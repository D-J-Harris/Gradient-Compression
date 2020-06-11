from memory.memory import Memory

class NoneMemory(Memory):

    def __init__(self):
        super().__init__(cumulative_grads={})

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        return tensor

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass
