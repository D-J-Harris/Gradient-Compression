from compression.none import NoneCompression
from compression.topk import TopKCompression
from compression.randomk import RandomKCompressor
from compression.dgc import DGCCompressor
from compression.qsgd import QSGDCompressor

def compression_chooser(args):
    """method for selecting compression method
    from command line argument."""

    inp = args.compression
    compress_ratio = args.compression_ratio
    quantum_num = args.quantum_num

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
