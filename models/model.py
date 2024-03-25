import torch
from torch import nn

from configs import model
from .backbone import get_discriminator_backbone, get_generator_backbone, get_style_backbone

noise_dim = model.noise_dim
num_class = model.num_class
default_checkpoint = model.checkpoint

def build_discriminator_generator_net(model_name,checkpoint=None,backbone_pretrained=False):
    """
    Tips:
    return D, G
    """
    D, G = Discriminator(model_name,backbone_pretrained), Generator(model_name)
    if checkpoint is not None:
       D.load_state_dict(torch.load(checkpoint + '/D.pth'))
       G.load_state_dict(torch.load(checkpoint + '/G.pth'))

    return D, G 


def build_style_net(model_name,checkpoint=None):
    style_net = StyleNet(model_name)
    if checkpoint is not None:
       style_net.load_state_dict(torch.load(checkpoint + '/S.pth'))
    return style_net


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
        self.backbone = get_generator_backbone(model_name) 
        self.embedding = nn.Embedding(num_class,noise_dim)

    #input:[B] dtype: long;
    def forward(self,label):    
        noise = torch.randn(label.shape[0],noise_dim,1,1,device=label.device)   #[B,nz,1,1]    
        label = self.embedding(label)                                     #[B,nz]
        label = label.view(-1,noise_dim,1,1)                              #[B,nz,1,1]
        
        noise = torch.cat((noise, label),dim=1)  #[B,2 * nz,1,1]
        fake_img = self.backbone(noise)          #[B,3,96,96]

        return fake_img


class StyleNet(nn.Module):
    def __init__(self,model_name):
        super(StyleNet,self).__init__()
        self.backbone = get_style_backbone(model_name)
        
    def forward(self,img): 
        noise = torch.randn(img.shape[0],1,img.shape[2],img.shape[3],device=img.device)  #as same shape as img
        img = torch.cat((img,noise),1)        
        img = self.backbone(img)

        return img

