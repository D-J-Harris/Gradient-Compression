import sys
import math

"""Taken from https://github.com/avojak/integer-compression"""

###############################################################################
# compression code
################################################################################


def binary_compression(integer):
    return "{0:b}".format(integer)

def unary_compression(integer):
    return '1' * (integer - 1) + '0'


def gamma_compression(integer):
    unary = unary_compression(1 + math.floor(math.log2(integer)))
    uniform = binary_compression(int(integer - math.pow(2, math.floor(math.log2(integer)))))
    uniform = uniform.zfill(math.floor(math.log2(integer)))
    return unary + uniform


###############################################################################
# decompression code
################################################################################


def binary_decompression(compressed):
    try:
        return int(compressed, 2)
    except ValueError:
        sys.exit('Invalid binary encoding: \"' + compressed + '\"')

def unary_decompression(compressed):
    num_ones = 0
    for bit in compressed:
        if bit == '1':
            num_ones = num_ones + 1
        elif bit == '0':
            if len(compressed) != num_ones + 1:
                sys.exit('Invalid unary encoding: \"' + compressed + '\"')
            return num_ones + 1
        else:
            sys.exit('Invalid unary encoding: \"' + compressed + '\"')
    sys.exit('Invalid unary encoding: \"' + compressed + '\"')

def gamma_decompression(compressed):
    if compressed == '00':
        return 1
    else:
        zero_loc = compressed.index('0')
        k = unary_decompression(compressed[:zero_loc + 1]) - 1
        r = binary_decompression(compressed[zero_loc + 1:zero_loc + 1 + k])
        return int(math.pow(2, k)) + r