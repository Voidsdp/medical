import torch
from torch import nn

from configs import model

nz = model.nz
ndf = model.ndf
ngf = model.ngf
nc = model.nc
checkpoint = model.checkpoint


def build_discriminator_generator_net(checkpoint=None):
    """
    Tips:
    return D, G
    """
    D, G = Discriminator(), Generator() 
    if checkpoint is not None:
       D.load_state_dict(torch.load(checkpoint['discriminator']))
       G.load_state_dict(torch.load(checkpoint['generator']))
       
    return D, G 


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=ndf,kernel_size=5, stride=3,padding=1,bias=False),    #[B,64,32,32]
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(in_channels=ndf,out_channels=ndf*2,kernel_size=4, stride=2,padding=1,bias=False),#[B,128,16,16]
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=ndf*2,out_channels=ndf*4,kernel_size=4, stride=2,padding=1,bias=False),#[B,256,8,8]
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=ndf*4,out_channels=ndf*8,kernel_size=4, stride=2,padding=1,bias=False),#[B,512,4,4]
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),
            )
        
        self.linear1 = nn.Linear(512 * 4 * 4 ,1)  #[B,1]
        self.linear2 = nn.Linear(512 * 4 * 4,nc)  #[B,nc]

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    #input = [B,3,96,96]
    def forward(self,x):  
        x = self.features(x)               #[B,512,4,4]
        x = torch.flatten(x, 1)            #[B,512*4*4]

        validity = self.linear1(x)         #[B,1]
        validity = self.sigmoid(validity)  #[B,1]      #输出真假的可能性

        label = self.linear2(x)            #[B,nc]
        label = self.softmax(label)        #[B,nc]     #输出各种类的可能性

        return validity,label


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.features=nn.Sequential(
            #channels_in,输出面out，卷积核数，步长，padding，bias
            nn.ConvTranspose2d(in_channels=2 * nz,out_channels=ngf*8,kernel_size=4,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=ngf*8,out_channels=ngf*4,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=ngf*4,out_channels=ngf*2,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=ngf*2,out_channels=ngf,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=ngf,out_channels=3,kernel_size=5,stride=3,padding=1,bias=False),
            nn.Tanh()
            )
        self.embedding = nn.Embedding(nc, nz)

    #input:[B] dtype: long;
    def forward(self,label):    
        noise = torch.randn(label.shape[0],nz,1,1)  #[B,nz,1,1]             
        label = self.embedding(label)               #[B,nz]
        label = label.view(-1,nz,1,1)               #[B,nz,1,1]
  
        noise = torch.cat((noise, label),dim=1)     #[B,2 * nz,1,1]  
        fake_img = self.features(noise)             #[B,3,96,96]

        return fake_img



