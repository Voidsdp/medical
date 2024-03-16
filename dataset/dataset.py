import os
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from configs import data 


def build_train_valid_test_data_iterators(data_path,batch_size=1):   
    train_dataset = get_dataset(data_path,'train')
    val_dataset = get_dataset(data_path,'val')
    test_dataset = get_dataset(data_path,'test')

    train_loader = DataLoader(train_dataset,batch_size,shuffle=True) if train_dataset is not None else None
    val_loader = DataLoader(val_dataset,batch_size,shuffle=False) if val_dataset is not None else None
    test_loader = DataLoader(test_dataset,batch_size,shuffle=False) if test_dataset is not None else None

    return train_loader, val_loader, test_loader


def get_dataset(data_path,split='train'):
    path = os.path.join(data_path,split)
    return Mydataset(path,split) if os.path.exists(path) else None


def make_dataset(data_path):
    images = []
    category_dict =  {  
        category: idx for idx,category in enumerate(sorted(os.listdir(data_path)))  #[类别:下标]字典
        if category != 'label.json'    
    } 
    with open(os.path.join(data_path,'label.json'),'w') as f:  #生成label.json文件
         json.dump(category_dict,f)

    for category in category_dict:
        category_path = os.path.join(data_path+'/',category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path+'/',img_name)
            item = (img_path,category_dict[category])
            images.append(item)
    
    return images


class Mydataset(Dataset):
    def __init__(self,data_path,split='train'):
        super().__init__()
        self.data_path = data_path
        self.imgs_path = make_dataset(data_path)  #[(图片路径，标签)]
        self.trans = data.transforms[split]

    def __getitem__(self, index):
        img = self.trans(Image.open(self.imgs_path[index][0]))
        label = torch.tensor(self.imgs_path[index][1])
        return img, label

    def __len__(self):
        return len(self.imgs_path)

