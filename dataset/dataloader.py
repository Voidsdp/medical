import os
import importlib

from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import ImageDataset, CoImageDataset
from .collate_fn import get_img_collate


def build_train_valid_test_data_iterators(data_path,model_name,batch_size=1,selection='Base'):     
    model_name = model_name if model_name == 'inception_v3' else 'base'  #select config
    img_size, mean, std = get_data_cfg(model_name)
    
    train_dataset = get_dataset(data_path,'train',selection)
    val_dataset = get_dataset(data_path,'val',selection)
    test_dataset = get_dataset(data_path,'test',selection)

    train_collate = get_img_collate(get_transform(img_size,mean,std,'train'),selection)
    val_collate = get_img_collate(get_transform(img_size,mean,std,'val'),selection)
    test_collate = get_img_collate(get_transform(img_size,mean,std,'test'),selection)

    train_loader = get_dataloader(train_dataset,batch_size,shuffle=True,collate_fn=train_collate)
    val_loader = get_dataloader(val_dataset,batch_size,shuffle=False,collate_fn=val_collate)
    test_loader = get_dataloader(test_dataset,batch_size,shuffle=False,collate_fn=test_collate)

    return train_loader, val_loader, test_loader


def get_data_cfg(model_name='base'):
    cfg = importlib.import_module('configs.' + model_name).data
    return cfg.img_size, cfg.mean, cfg.std


def get_transform(img_size,mean,std,split='train'):
    if split == 'train':
       transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.Resize(img_size),               
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std),
         ])
    else:
       transform = transforms.Compose([
                    transforms.Resize(img_size),               
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean,std),
         ])
       
    return transform


def get_dataset(data_path,split='train',selection='Base'):
    path = os.path.join(data_path,split)
    if os.path.exists(path):
        return ImageDataset(path) if selection == 'Base' else CoImageDataset(path)
    else:
        return None


def get_dataloader(dataset,batch_size,shuffle=False,collate_fn=None):
    if dataset is not None:
       dataloader = DataLoader(dataset,batch_size,shuffle,collate_fn=collate_fn) 
    else:
       dataloader = None
    
    return dataloader