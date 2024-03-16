from torchvision import transforms

class data:
    image_size = 96
    mean = (0.5,0.5,0.5)
    std = (0.5,0.5,0.5)
    transforms = {
         'train': transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.Resize(image_size),               
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std),
         ]),   
        'val': transforms.Compose([
                    transforms.Resize(image_size),               
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std),
         ]),
         'test': transforms.Compose([
                    transforms.Resize(image_size),               
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std),
         ]),
    }              

class model:
      nz = 100 #噪声维度
      ndf = 64
      ngf = 64
      nc = 4 #分类种类

      checkpoint = {
           'discriminator': 'checkpoint/discriminator.pth',
           'generator': 'checkpoint/generator.pth',
      }