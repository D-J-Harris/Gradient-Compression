from compression.compression import Compressor

class NoneCompression(Compressor):
    """Default no-op compression."""
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "none_compression"

    # returns tensors and context
    # where tensors = tensor, indices (for sparse)
    def compress(self, tensor, name):

        # save running compression ratio for that layer
        ratio = 1.0
        self.update_running_ratio(name, ratio)

        return tensor, None

    def decompress(self, tensors, ctx):
        return tensors
