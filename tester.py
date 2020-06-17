import torch
torch.set_printoptions(threshold=5000)

a= torch.load('1')
b = torch.load('2')

c = torch.load('1inp2')
d = torch.load('2inp4')
e = torch.load('2inp5')

f = torch.load('1outp1')
g = torch.load('2outp2')
h = torch.load('2outp3')

# i = torch.load('1loss0')
# j = torch.load('2loss0')
# k = torch.load('2loss1')

l = torch.cat((d, e), 1)
m = torch.cat((g, h), 1)

# print(i)
# print(j)
# print(k)
# print(j+k)

# print(torch.max(f-m))
# print(torch.max(torch.div(f-m, m)))

# aa = torch.load('1targ0')
# bb = torch.load('2targ0')
# cc = torch.load('2targ1')

# ff = f.view(-1, 10000)
# gg = g.view(-1, 10000)
# hh = h.view(-1, 10000)
# mm = torch.cat((gg, hh), 0)
# print(torch.all(ff.eq(mm)))

hum1 = torch.load('hiddens_single')
hum2 = torch.load('hiddens')

aaa = hum1[0][0]
bbb = hum2[0][0]
ccc = hum2[1][0]

ddd = torch.cat((ccc, bbb), 1)

print(aaa-ddd)
