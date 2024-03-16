import os
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import torchvision.utils as vutils

from configs import model as model_cfg
from dataset import build_train_valid_test_data_iterators
from models import build_discriminator_generator_net
from evaluator import discrimiator_acc_compute
from visualize import generate_img

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random(seed)
        torch.manual_seed(seed)


def get_models(checkpoint):
    checkpoint = model_cfg.checkpoint if args.checkpoint else None
    D, G = build_discriminator_generator_net(checkpoint)
    D.to(device), G.to(device)
    return D, G


def get_optimizers(models,lr):
    D, G = models
    d_optimizer=AdamW(D.parameters(),lr,betas=[0.5,0.999])     #修改成0.9和0.99
    g_optimizer=AdamW(G.parameters(),lr,betas=[0.5,0.999])

    return d_optimizer, g_optimizer


def get_criterion():
    adversarial_criterion = torch.nn.BCELoss().to(device)
    auxiliary_criterion = torch.nn.CrossEntropyLoss().to(device)
    return adversarial_criterion, auxiliary_criterion


def loss_compute(img,model,valid,label):
    adversarial_criterion, auxiliary_criterion = get_criterion()

    valid_pred, label_pred = model(img) #[B,1] [B,nc]
    adversarial_loss = adversarial_criterion(valid_pred, valid)
    auxiliary_loss = auxiliary_criterion(label_pred, label)
    loss = (adversarial_loss + auxiliary_loss) / 2

    return loss


#训练
def train(train_loader,models,optimizers):
    #解包
    D, G = models
    d_optimizer, g_optimizer = optimizers
    acc = (0.,0.,0.,0.)

    for real_img, real_label in train_loader:
        real_label = real_label.to(device)    #[B]    
        fake_label = torch.randint(0,model_cfg.nc,(real_label.shape[0],),device=device)   #[B] to consistent with label's shape instead of batch_size 
                                                                                          #    considered 'drop_last' of dataloader 
        real_img = real_img.to(device)   #[B,3,96,96]     
        fake_img = G(fake_label)         #generate fake image before to save computation after

        real_valid = torch.ones(args.batch_size,1,device=device)  #[B,1]
        fake_valid = torch.zeros(args.batch_size,1,device=device) #[B,1]

        '''
        ------D start------
        '''
        d_optimizer.zero_grad()
                
        d_real_loss = loss_compute(real_img,D,real_valid,real_label)  
        d_fake_loss = loss_compute(fake_img.detach(),D,fake_valid,fake_label)  
        d_loss = (d_fake_loss + d_real_loss) / 2  # -(LS + LC)

        d_loss.backward()
        d_optimizer.step()

        '''
        ------D end and G start------
        '''
        g_optimizer.zero_grad()

        g_loss = loss_compute(fake_img,D,fake_valid,fake_label)
     
        g_loss.backward()
        g_optimizer.step()
        
        '''
        ------G end------
        '''   
        print('D_LOSS: {:.2f}%,G_LOSS: {:.2f}%'.format(d_loss, g_loss))

#valid
def eval(val_loader,models):
    D, G = models
    acc = (0.,0.,0.,0.)

    with torch.no_grad():
        for real_img, real_label in val_loader:
            real_label = real_label.to(device)
            fake_label = torch.randint(0,model_cfg.nc,(real_label.shape[0],),device=device) 

            real_img = real_img.to(device)
            fake_img = G(fake_label)

            pred = *D(real_img), *D(fake_img)           #[B,1] [B,nc]
            label = (real_label,fake_label)
            acc = discrimiator_acc_compute(pred,label,acc)

        #mean
        real_valid_acc, fake_valid_acc, real_label_acc, fake_label_acc = [
               i / len(val_loader) * 100  for i in acc 
        ]
        valid_acc = (real_valid_acc + fake_valid_acc)/2
        label_acc = (real_label_acc + fake_label_acc)/2

        print("real_valid_acc: {:.2f}%, fake_valid_acc: {:.2f}%".format(real_valid_acc,fake_valid_acc))
        print("real_label_acc: {:.2f}%, fake_label_acc: {:.2f}%".format(real_label_acc,fake_label_acc))
        print("valid_acc: {:.2f}%, label_acc: {:.2f}%".format(valid_acc,label_acc))


def main(args):
    #seed
    set_random_seed(args.seed)
    
    #cross validation datasets
    train_loader, val_loader, test_loader = build_train_valid_test_data_iterators(args.data_path,args.batch_size)
   
    #models and optimizers
    models = get_models(args.checkpoint)
    optimizers = get_optimizers(models,args.lr)

    for epoch in tqdm(range(args.epochs)):
        #train
        train(train_loader,models,optimizers)
        #eval
        if args.is_eval:
           eval(val_loader,models)
        #visualize
        if args.is_visualize:
           G = models[1]
           generate_img(G,torch.tensor(1,device=device),'images/{}'.format(epoch))
        
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',default='data/pathology/')        
    parser.add_argument('--saved_img',default='result')
    parser.add_argument('--checkpoint',action='store_ture')
    parser.add_argument('--is_eval',action='store_false')
    parser.add_argument('--is_visualize',action='store_true')

    parser.add_argument('--seed',default=0)
    parser.add_argument('--epochs',default=1000)
    parser.add_argument('--batch_size',default=32)
    parser.add_argument('--lr',default=1e-5)

    args = parser.parse_args()
    main(args)