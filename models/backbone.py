import torch
from torch import nn

from configs import model

nz = model.nz
ndf = model.ndf
ngf = model.ngf

def get_discriminator_backbone(model='cnn'):
    if model == 'cnn':
       backbone = nn.Sequential( 
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
    return backbone

def get_generator_backbone(model='decnn'):
    if model == 'decnn':
       backbone = nn.Sequential(
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
       
       return backbone