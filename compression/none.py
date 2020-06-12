from compression.compression import Compressor

class NoneCompression(Compressor):
    """Default no-op compression."""

    # returns tensors and context
    # where tensors = tensor, indices (for sparse)
    def compress(self, tensor, name):
        tensors = tensor, None
        return tensors, None

    def decompress(self, tensors, ctx):
        tensor, _ = tensors
        return tensor
