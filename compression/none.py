from compression.compression import Compressor

class NoneCompression(Compressor):
    """Default no compression."""
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "none_compression"

    # returns tensors and context
    # where tensors = tensor, indices (for sparse)
    def compress(self, tensor, name):
        return tensor, None

    def decompress(self, tensors, ctx):
        return tensors
