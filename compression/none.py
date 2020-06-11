from compression.compression import Compressor

class NoneCompression(Compressor):
    """Default no-op compression."""

    def compress(self, tensor, name):
        return tensor, None

    def decompress(self, tensors, ctx):
        tensor, = tensors
        return tensor
