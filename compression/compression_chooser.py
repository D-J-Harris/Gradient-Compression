from compression.none import NoneCompression
from compression.topk import TopKCompression
from compression.randomk import RandomKCompressor
from compression.dgc import DGCCompressor
from compression.qsgd import QSGDCompressor

def compression_chooser(inp, compress_ratio=0.01, quantum_num=64):
    """method for selecting compression method
    from command line argument."""

    if inp == 'none':
        return NoneCompression()

    if inp == 'topk':
        return TopKCompression(compress_ratio)

    if inp == 'randomk':
        return RandomKCompressor(compress_ratio)

    if inp == 'dgc':
        return DGCCompressor(compress_ratio)

    if inp == 'qsgd':
        return QSGDCompressor(quantum_num)

    else:
        raise ValueError('compression argument invalid')