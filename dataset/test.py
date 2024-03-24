# from configs import model
import torch
from torchvision import transforms
import logging


a = torch.randn(32,1)
# print(a.shape)
x = a.view(-1)
# print(x.shape)
# print(model.checkpoint)

class demo1:
    img_size = 225

    transform = transforms.Resize(img_size)
img = torch.randn(5,3,224,224)
conv = torch.nn.Conv2d(in_channels=4,out_channels=3,kernel_size=3,stride=1,padding=1)

# img = conv(img)
# print(img.shape)

noise = torch.randn(5,1,224,224)
img = torch.cat((img,noise),1)      
print(img.shape)  
img = conv(img)
print(img.shape)


