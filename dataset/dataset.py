from torch.utils.data import Dataset, random_split
import cv2
import os
from torchvision import transforms
import torch
from PIL import Image


label_dict = {
    'EOSINOPHIL':torch.tensor(0),
    'LYMPHOCYTE':torch.tensor(1),
    'MONOCYTE':torch.tensor(2),
    'NEUTROPHIL':torch.tensor(3)
}
image_size = 96

data_transform=transforms.Compose([
       transforms.Resize(image_size),               #Resize后整体比例不变
       transforms.CenterCrop(image_size),
       transforms.ToTensor(),
       transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
def get_dataset(data_path):
    datasets = []
    label_list = os.listdir(data_path)
    for label in label_list:
        class_name_path = os.path.join(data_path, label)
        datasets.append(Mydataset(class_name_path))
        #如果ratio大于一，获得数据集的长度是ratio的两倍，数据是从两个数据集中取的
    dataset = datasets[0]
    for _dataset in datasets[1:]:
        dataset += _dataset
    return dataset

class Mydataset(Dataset):
    def __init__(self,data_path):
        super().__init__()
        
        self.data_path = data_path
        self.imgs_list = os.listdir(data_path)

        self.imgs_path = [os.path.join(data_path,img) for img in self.imgs_list]

        self.trans = data_transform
        self.label = label_dict[data_path.split('\\')[-1]]

    def __getitem__(self, index):
        img = self.trans(Image.open(self.imgs_path[index]))
        return img, self.label

    def __len__(self):
        return len(self.imgs_path)
    


if __name__ == '__main__':
    train = get_dataset("C:\\Users\\lenovo\\Desktop\\DCGAN\\Data\\dataset2-master\\images\\TRAIN")
    img, label = train[5050]
    print(img)
    print(label)     