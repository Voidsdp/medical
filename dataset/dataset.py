import os
import json

from PIL import Image
from torch.utils.data import Dataset

from configs import data
from utils import color_print


label_file = data.label_file


def make_dataset(data_path):
    images = []
    label_path = os.path.join(data_path,label_file)

    if os.path.exists(label_path):
       with open(label_path,'r') as f: 
            category_dict = json.load(f)
    else:
        category_dict =  {  
            category: idx for idx,category in enumerate(sorted(os.listdir(data_path)))  #{category:idx}
            if category != 'label.json'    
        } 
        with open(label_path,'w') as f:  #generate label.json
            json.dump(category_dict,f)
            color_print('New label_file has been created!','red')

    for category in category_dict:
        category_path = os.path.join(data_path,category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path,img_name)
            item = (img_path,category_dict[category])
            images.append(item)
    
    return images, len(category_dict)


class Imagedataset(Dataset):
    def __init__(self,data_path):
        super().__init__()
        self.data_path = data_path
        self.imgs_path, self.num_class = make_dataset(data_path)  #[(img_path, label)]


    def __getitem__(self,index):
        img_path, label = self.imgs_path[index]
        img = Image.open(img_path)

        return img, label


    def __len__(self):
        return len(self.imgs_path)
