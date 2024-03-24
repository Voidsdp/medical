import os
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from configs import data
from utils import color_print, check_dir

transforms = data.transforms
label_file = data.label_file


def build_train_valid_test_data_iterators(data_path,model_name,batch_size=1):   
    check_dir(data_path,alert=True)
    
    model_name = model_name if model_name == 'inception_v3' else 'others'

    train_dataset = get_dataset(data_path,model_name,'train')
    val_dataset = get_dataset(data_path,model_name,'val')
    test_dataset = get_dataset(data_path,model_name,'test')

    train_loader = get_dataloader(train_dataset,batch_size,shuffle=True)
    val_loader = get_dataloader(val_dataset,batch_size,shuffle=False)
    test_loader = get_dataloader(test_dataset,batch_size,shuffle=False)

    return train_loader, val_loader, test_loader


def get_dataset(data_path,model_name,split='train'):
    path = os.path.join(data_path,split)
    return Mydataset(path,model_name,split) if os.path.exists(path) else None


def get_dataloader(dataset,batch_size,shuffle):
    return DataLoader(dataset,batch_size,shuffle) if dataset is not None else None


def make_dataset(data_path):
    images = []
    label_path = os.path.join(data_path,label_file)

    if os.path.exists(label_path):
       with open(label_path,'r') as f: 
            category_dict = json.load(f)
    else:
        category_dict =  {  
            category: idx for idx,category in enumerate(sorted(os.listdir(data_path)))  #[类别:下标]字典
            if category != 'label.json'    
        } 
        with open(label_path,'w') as f:  #generate label.json
            json.dump(category_dict,f)
            color_print('New label_file has been created!','red')

    for category in category_dict:
        category_path = os.path.join(data_path+'/',category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path+'/',img_name)
            item = (img_path,category_dict[category])
            images.append(item)
    
    return images, len(category_dict)


class Mydataset(Dataset):
    def __init__(self,data_path,model_name,split='train'):
        super().__init__()
        self.data_path = data_path
        self.model_name = model_name
        self.split = split
        self.imgs_path, self.num_class = make_dataset(data_path)  #[(img_path, label)]
        self.trans = transforms[model_name][split]

    def __getitem__(self, index):
        img_path, real_label = self.imgs_path[index]
        img = self.trans(Image.open(img_path)) 
    
        fake_label = torch.randint(0,self.num_class,())
        label = (real_label,fake_label)

        if self.split == 'train':
           valid = (1,0)
           return img, label, valid
        else:         
           return img, real_label
            
    def __len__(self):
        return len(self.imgs_path)
