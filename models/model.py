import torch
from torch import nn

from configs import model
from backbone import get_discriminator_backbone,get_generator_backbone

nz = model.nz
nc = model.nc
default_checkpoint = model.checkpoint


def build_discriminator_generator_net(checkpoint=None):
    """
    Tips:
    return D, G
    """
    D, G = Discriminator(), Generator()
    if checkpoint is not None:
       if checkpoint == 'default':
          checkpoint = default_checkpoint
          
       D.load_state_dict(torch.load(checkpoint['discriminator']))
       G.load_state_dict(torch.load(checkpoint['generator']))

    return D, G 

# 3 96 96
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=get_discriminator_backbone()
        self.valid_linear = nn.Linear(512 * 4 * 4 ,1)  #[B,1]
        self.label_linear = nn.Linear(512 * 4 * 4,nc)  #[B,nc]

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    #input = [B,3,96,96]
    def forward(self,x):  
        x = self.backbone(x)               #[B,512,4,4]
        x = torch.flatten(x, 1)            #[B,512*4*4]

        validity = self.valid_linear(x)         #[B,1]
        validity = self.sigmoid(validity)  #[B,1]      #输出真假的可能性

        label = self.label_linear(x)            #[B,nc]
        label = self.softmax(label)        #[B,nc]     #输出各种类的可能性

        return validity,label


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.features = get_generator_backbone() 
        self.embedding = nn.Embedding(nc, nz)

    #input:[B] dtype: long;
    def forward(self,label):    
        noise = torch.randn(label.shape[0],nz,1,1)  #[B,nz,1,1]             
        label = self.embedding(label)               #[B,nz]
        label = label.view(-1,nz,1,1)               #[B,nz,1,1]
  
        noise = torch.cat((noise, label),dim=1)     #[B,2 * nz,1,1]  
        fake_img = self.features(noise)             #[B,3,96,96]

        return fake_img



