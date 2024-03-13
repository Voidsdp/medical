x = [1,2,3,4,5,6,7,8]
import torch
from torch import nn
# 假设有一个二维张量
tensor = torch.tensor([[1, 3, 5],
                       [2, 4, 6]])

# 在指定维度上找到最大值的索引
max_indices = torch.argmax(tensor, dim=1)

# print("原始张量:\n", tensor)
# print("在每行上的最大值的索引:\n", max_indices)


loss1 = nn.BCELoss()
loss2 = nn.CrossEntropyLoss()

x = torch.rand(32,4)  #.to(dtype=torch.long)
y = torch.randint(0,3,(32,))   #.to(dtype=torch.long)

z = loss2(x,y)

#torch.tensor(3)        类型为long
#torch.tensor(3.)       类型为float32

noise_label = torch.randint(0,1,(32,))
# print(noise_label.dtype)  

embedding = nn.Embedding(200,100)

x = torch.tensor(1)
# print(embedding(x).shape)
#Embedding可以传多维的
y = torch.tensor([1,2])
# print(embedding(y).shape)
print(y.dim())
z = torch.tensor(1)
z.unsqueeze(-1)
# print(z.shape)
print(x.dim())