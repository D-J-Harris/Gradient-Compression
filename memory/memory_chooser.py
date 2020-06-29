from memory.none import NoneMemory
from memory.residual import ResidualMemory
from memory.dgc import DGCMemory

def memory_chooser(args, momentum=0.1, gradient_clipping=0.25):
    """method for selecting memory method
        from command line argument."""

    inp = args.memory

    if inp == 'none':
        return NoneMemory()

    if inp == 'residual':
        return ResidualMemory()

    if inp == 'dgc':
        return DGCMemory(momentum, gradient_clipping)

    else:
        raise ValueError('memory argument invalid')
