import torch
from utils import truncate
torch.set_printoptions(precision=16)

a = torch.load('1hiddens0')
b = torch.load('2hiddens1')

c = a['0h']
d = a['0c']
e = b['0h']
f = b['0c']
g = b['1h']
h = b['1c']

i = torch.cat((g,e), 1)
j = torch.cat((h,f), 1)
print(d-j)
