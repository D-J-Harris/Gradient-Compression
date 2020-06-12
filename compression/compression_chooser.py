from compression.none import NoneCompression
from compression.topk import TopKCompression

def compression_chooser(inp, arg=0.3):
    """method for selecting compression method
    from command line argument."""

    if inp == 'none':
        return NoneCompression()

    if inp == 'topk':
        return TopKCompression(arg)