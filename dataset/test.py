# from configs import model
import torch
from torchvision import transforms
a = torch.randn(32,1)
# print(a.shape)
x = a.view(-1)
# print(x.shape)
# print(model.checkpoint)

class demo1:
    img_size = 225

    transform = transforms.Resize(img_size)

demo1.img_size = 3000

print(demo1.img_size)
print(demo1.transform.size)
