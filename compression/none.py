from compression.compression import Compressor

class NoneCompression(Compressor):
    """Default no-op compression."""
    def __init__(self):
        super().__init__()
        self.is_sparse = False

    # returns tensors and context
    # where tensors = tensor, indices (for sparse)
    def compress(self, tensor, name):
        tensors = tensor, None
        return tensors, None

    def decompress(self, tensors, ctx):
        tensor, indices = tensors
        return tensor
