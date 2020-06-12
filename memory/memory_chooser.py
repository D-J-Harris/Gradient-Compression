from memory.none import NoneMemory
from memory.residual import ResidualMemory

def memory_chooser(inp, arg=None):
    """method for selecting memory method
        from command line argument."""

    if inp == 'none':
        return NoneMemory()

    if inp == 'residual':
        return ResidualMemory()

    else:
        raise ValueError('memory argument invalid')
