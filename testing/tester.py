import torch
import data_load
from utils import batchify

# a = torch.load('big')
# b = torch.load('small1new')
# c = torch.load('small2new')
#
# d = b+c
#
# # print(a-d)
#
# rel_error = torch.div(a-d, d)
# print(torch.max(rel_error))
# print(torch.max(a-d))