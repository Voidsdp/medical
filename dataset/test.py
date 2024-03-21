# from configs import model
import torch

a = torch.randn(32,1)
# print(a.shape)
x = a.view(-1)
# print(x.shape)
# print(model.checkpoint)


a = [0,0,0,0]
x = 1
y = 2
z = 3
j = 4
acc = x,y,z,j
print(a)