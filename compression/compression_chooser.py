from compression.none import NoneCompression
from compression.topk import TopKCompression
from compression.randomk import RandomKCompressor

def compression_chooser(inp, compress_ratio=0.3):
    """method for selecting compression method
    from command line argument."""

    if inp == 'none':
        return NoneCompression()

    if inp == 'topk':
        return TopKCompression(compress_ratio)

    if inp == 'randomk':
        return RandomKCompressor(compress_ratio)

    else:
        raise ValueError('compression argument invalid')