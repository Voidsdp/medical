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

def get_dataset(path):
    ...
    



def main(args):
    set_random_seed(args.seed)

    train_dataset = get_dataset(args.data_path)
    val_dataset = get_dataset(args.val_data_path)
    tarin_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)


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
        total_real_correct = 0
        total_fake_correct = 0
        total_real_samples = 0
        total_fake_samples = 0
        total_real_label_correct = 0
        total_fake_label_correct = 0
        for i,(img, label) in enumerate(tarin_dataloader):
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

            # if i % 50 == 0:
            #     print("Epoch[{}/{}],Step[{}/{}],d_loss:{:4f},g_loss:{:4f}"
            #             .format(epoch,args.epochs,i,len(train_dataloader),d_loss.item(),g_loss.item()))
        with torch.no_grad():
            for img, label in val_dataloader:
                img = img.to(device)
                label = label.to(device)
                noise=torch.randn(args.batch_size,nz,1,1,device=device) 
                noise_label = torch.randint(0,1,(args.batch_size,),device=device) 

                real_pred, real_aux = D(img)
                fake_pred, fake_aux = D(G(noise,noise_label))

                # Calculate and update total correct predictions and total samples
                total_real_correct += torch.sum((real_pred > 0.5).float()).item()
                total_fake_correct += torch.sum((fake_pred <= 0.5).float()).item()
                total_real_samples += real_pred.size(0)
                total_fake_samples += fake_pred.size(0)

                # Calculate accuracy for predicting classes
                total_real_label_correct += torch.sum(torch.argmax(real_aux, dim=1) == label).item()
                total_fake_label_correct += torch.sum(torch.argmax(fake_aux, dim=1) == noise_label).item()

            # Calculate overall discriminator accuracy
            overall_real_accuracy = total_real_correct / total_real_samples * 100
            overall_fake_accuracy = total_fake_correct / total_fake_samples * 100
            overall_real_label_accuracy = total_real_label_correct / total_real_samples * 100
            overall_fake_label_accuracy = total_fake_label_correct / total_fake_samples * 100
        
        print("Real Image Accuracy: {:.2f}%, Fake Image Accuracy: {:.2f}%".format(overall_real_accuracy, overall_fake_accuracy))
        print("Classes Real Image Label Accuracy: {:.2f}%, Fake Image Label Accuracy: {:.2f}%".format(overall_real_label_accuracy, overall_fake_label_accuracy))


        # 保存模型
        noise=torch.randn(nz,1,1,device=device)   #BCHW
        fake_image = G(noise,torch.tensor(1,device=device))
        vutils.save_image(fake_image, './img/result{}.png'.format(epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',default="C:/Users/lenovo/Desktop/DCGAN/Data/dataset2-master/images/TRAIN")          #'./data/dogs-cats/train/'
    parser.add_argument('--epochs',default=1000)
    parser.add_argument('--batch_size',default=32)
    parser.add_argument('--lr',default=1e-5)
    parser.add_argument('--checkpoint',default='models/saved_model/')
    parser.add_argument('--saved_img',default='result')
    parser.add_argument('--num_classes',default=4)
    parser.add_argument('--seed',default=0)
    parser.add_argument('--val_data_path',default="C:/Users/lenovo/Desktop/DCGAN/Data/dataset2-master/images/TEST")

    args = parser.parse_args()

    main(args)