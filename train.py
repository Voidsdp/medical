import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision import models,transforms,datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import argparse
from models.model import *
from dataset.dataset import get_dataset
from tqdm import tqdm
import random

image_size=96

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random(seed)
        torch.manual_seed(seed)


def main(args):
    set_random_seed(args.seed)

    dataset = get_dataset(args.data_path)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    G=Generator(args.num_classes).to(device)
    D=Discriminator(args.num_classes).to(device)

    adversarial_loss = torch.nn.BCELoss().to(device)
    auxiliary_loss = torch.nn.CrossEntropyLoss().to(device)
    d_optimizer=torch.optim.Adam(D.parameters(),lr=args.lr,betas=[0.5,0.999])   #修改成0.9和0.99
    g_optimizer=torch.optim.Adam(G.parameters(),lr=args.lr,betas=[0.5,0.999])

    # for i, (img,lab) in enumerate(dataloader):      #第一个参数是给的标号，后面才是数据
    #     print(i)
    #     break

    for epoch in tqdm(range(args.epochs)):
        for i,(img, label) in enumerate(dataloader):
            img = img.to(device)
            label = label.to(device)        #类型为long,[32,]
            
            noise=torch.randn(args.batch_size,nz,1,1,device=device) 
            noise_label = torch.randint(0,1,(args.batch_size,),device=device)                 #batch_size,latent_dim
            # print(noise_label.dtype)
            fake_img = G(noise,noise_label)

            valid = torch.ones(args.batch_size,1,device=device)
            fake = torch.zeros(args.batch_size,1,device=device)

            d_optimizer.zero_grad()

            real_pred, real_aux = D(img)            #第一个是真假，第二个是种类

            adversarial_real = adversarial_loss(real_pred, valid)
            auxiliary_real = auxiliary_loss(real_aux, label)
            d_real_loss = (adversarial_real + auxiliary_real) / 2
            # d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux,label)) / 2          

            fake_pred, fake_aux = D(fake_img.detach())

            adversarial_fake = adversarial_loss(fake_pred, fake)
            auxiliary_fake = auxiliary_loss(fake_aux,noise_label)
            d_fake_loss = (adversarial_fake + auxiliary_fake) / 2
            # d_fake_loss = ( + auxiliary_loss(fake_aux,noise_label)) / 2

            d_loss = (d_fake_loss + d_real_loss) / 2  # -(LS + LC)
            d_loss.backward()
            
            d_optimizer.step()

            g_optimizer.zero_grad()
            validity, pred_label = D(fake_img)
            g_loss = (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, noise_label)) / 2

            g_loss.backward()
            g_optimizer.step()

            if i % 50 == 0:
                print("Epoch[{}/{}],Step[{}/{}],d_loss:{:4f},g_loss:{:4f}"
                        .format(epoch,args.epochs,i,len(dataloader),d_loss.item(),g_loss.item()))

        # 保存模型
        noise=torch.randn(nz,1,1,device=device)   #BCHW
        fake_image = G(noise,torch.tensor(1,device=device))
        vutils.save_image(fake_image, './img/result{}.png'.format(epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',default="C:\\Users\\lenovo\\Desktop\\DCGAN\\Data\\dataset2-master\\images\\TRAIN")          #'./data/dogs-cats/train/'
    parser.add_argument('--epochs',default=1000)
    parser.add_argument('--batch_size',default=32)
    parser.add_argument('--lr',default=1e-5)
    parser.add_argument('--checkpoint',default='models/saved_model/')
    parser.add_argument('--saved_img',default='result')
    parser.add_argument('--num_classes',default=4)
    parser.add_argument('--seed',default=0)

    args = parser.parse_args()

    main(args)