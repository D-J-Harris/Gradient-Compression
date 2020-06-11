from compress.compressor import Compressor

class NoneCompressor(Compressor):
    """Default no-op compression."""

    def compress(self, tensor):
        return tensor, None

    def decompress(self, tensors, ctx):
        tensor, = tensors
        return tensor
