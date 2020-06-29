import torch
from compression.compression import Compressor


class QSGDCompressor(Compressor):

    def __init__(self, quantum_num):
        super().__init__()
        self.quantum_num = quantum_num


    def __str__(self):
        return f"qsgd_{self.quantum_num}"


    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        max = tensor.max()
        max = max.flatten()
        abs_gradient = tensor.abs()

        level_float = self.quantum_num / max * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(torch.int8 if self.quantum_num < 128 else torch.half)
        tensor_compressed = tensor_compressed, max

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, max = tensor_compressed

        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = max / self.quantum_num * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed