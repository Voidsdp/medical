import os
import torch
from torch import nn

from configs import model
from utils import check_dir
from .backbone import get_discriminator_backbone,get_generator_backbone


noise_dim = model.noise_dim
num_class = model.num_class
default_checkpoint = model.checkpoint

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_discriminator_generator_net(model_name='cnn',checkpoint=None,backbone_pretrained=False):
    """
    Tips:
    return D, G
    """
    D, G = Discriminator(model_name,backbone_pretrained), Generator(model_name)
    if checkpoint is not None:
       if checkpoint == 'default':
          checkpoint = os.path.join(default_checkpoint,model_name)
          
       check_dir(checkpoint,alert=True)
       D.load_state_dict(torch.load(checkpoint + '/D.pth'))
       G.load_state_dict(torch.load(checkpoint + '/G.pth'))

    return D, G 


class Discriminator(nn.Module):
    def __init__(self,model_name,backbone_pretrained=False):
        super(Discriminator,self).__init__()
        self.model_name = model_name
        self.backbone, out_features = get_discriminator_backbone(model_name,backbone_pretrained)
        self.valid_linear = nn.Linear(out_features,1)  #[B,1]
        self.label_linear = nn.Linear(out_features,num_class)  #[B,nc]

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    #input = [B,3,96,96]
    def forward(self,x):  
        x = self.backbone(x)                    #[B,512,4,4]
        x = torch.flatten(x, 1)                 #[B,512*4*4]

        validity = self.valid_linear(x)         #[B,1]
        validity = self.sigmoid(validity)       #[B,1]      #输出真假的可能性

        label = self.label_linear(x)            #[B,nc]
        label = self.softmax(label)             #[B,nc]     #输出各种类的可能性

        return validity,label


class Generator(nn.Module):
    def __init__(self,model_name):
        super(Generator,self).__init__()
        self.model_name = model_name
        self.features = get_generator_backbone() 
        self.embedding = nn.Embedding(num_class,noise_dim)

    #input:[B] dtype: long;
    def forward(self,label):    
        noise = torch.randn(label.shape[0],noise_dim,1,1)  #[B,nz,1,1]    
        label = self.embedding(label)               #[B,nz]
        label = label.view(-1,noise_dim,1,1)               #[B,nz,1,1]

        noise = noise.to(label.device)
        
        noise = torch.cat((noise, label),dim=1)     #[B,2 * nz,1,1]
        # noise = noise.to(device)  
        fake_img = self.features(noise)             #[B,3,96,96]

        return fake_img



