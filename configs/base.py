from torchvision import transforms
 
class model:
      noise_dim = 100 
      num_class = 7

      checkpoint = 'checkpoint'


class optimizer:
    optimizer_type = 'AdamW'
    betas = [0.5,0.999]

class Tensorboard:
      log_dir = './tensorboard'


class data:

    label_file = 'label.json'
    mean = (0.5,0.5,0.5)
    std = (0.5,0.5,0.5)
    transforms = {
         'inception_v3': { 
              'train': transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.Resize(299),               
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std),
         ]),   
               'val': transforms.Compose([
                    transforms.Resize(299),               
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std),
         ]),
               'test': transforms.Compose([
                    transforms.Resize(299),               
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std),
         ])
        },
         'others': { 
              'train': transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.Resize(224),               
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std),
         ]),   
               'val': transforms.Compose([
                    transforms.Resize(224),               
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std),
         ]),
               'test': transforms.Compose([
                    transforms.Resize(224),               
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std),
         ])
        },
    } 