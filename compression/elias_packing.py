# taken from https://gist.github.com/robertofraile/483003

def recursive_encode(n):
    s = ""
    if n > 1:
        b = bin(n)[2:]
        s += recursive_encode(len(b) - 1) + b
    return s


def recursive_decode(s, n):
    if s[0] == "0":
        return [n, s[1:]]
    else:
        m = int(s[:n + 1], 2)
        return recursive_decode(s[n + 1:], m)
