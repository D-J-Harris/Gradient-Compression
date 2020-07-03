import torch
from compression.compression import Compressor
from compression.elias_packing import recursive_encode


class QSGDCompressor(Compressor):
    """QSGD, Alistarh et al."""

    def __init__(self, quantum_num):
        super().__init__()
        self.quantum_num = quantum_num
        self.bits_packed = 0
        self.counter = 0


    def __str__(self):
        return f"qsgd_{self.quantum_num}"


    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        norm = tensor.max()  # max as per paper empirical details
        norm = norm.flatten()
        abs_gradient = tensor.abs()

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign)

        # simulate packing and unpacking, counting bits as progression goes on
        # note the quantisation has been done - simulated environments don't need actual packing
        bits = 2 * torch.floor(torch.log2(torch.abs(tensor_compressed))) + 1
        bits[bits == float('-inf')] = 1.
        self.bits_packed += (torch.sum(bits).item() + 32)
        self.counter += len(tensor_compressed)

        tensor_compressed = tensor_compressed, norm

        # the norm is 32bits, levels (would be) integer compressed bits
        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, norm = tensor_compressed

        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed
