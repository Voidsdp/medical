import torch
from torch import nn

nz=100
ngf=64
ndf=64
#input = [B,3,96,96]
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.features=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=ndf,kernel_size=5, stride=3,padding=1,bias=False),  #[B,64,32,32]
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(in_channels=ndf,out_channels=ndf*2,kernel_size=4, stride=2,padding=1,bias=False),#[B,128,16,16]
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=ndf*2,out_channels=ndf*4,kernel_size=4, stride=2,padding=1,bias=False),#[B,256,8,8]
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(in_channels=ndf*4,out_channels=ndf*8,kernel_size=4, stride=2,padding=1,bias=False),#[B,512,4,4]
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=False),
            )
        
        self.linear1 = nn.Linear(512 * 4 * 4 ,1)
        self.linear2 = nn.Linear(512 * 4 * 4,num_classes)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        label = self.linear1(x)         
        label = self.sigmoid(label)             #输出真假的可能性

        validity = self.linear2(x)
        validity = self.softmax(validity)            #输出各种类的可能性
        # validity = torch.argmax(validity, dim=1)

        return label,validity
    
class Generator(nn.Module):
    def __init__(self,num_classes, latent_dim=100):
        super(Generator,self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.features=nn.Sequential(
            #channels_in,输出面out，卷积核数，步长，padding，bias
            nn.ConvTranspose2d(in_channels=2 * latent_dim,out_channels=ngf*8,kernel_size=4,stride=1,padding=0,bias=False),
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
        self.embedding = nn.Embedding(num_classes, latent_dim)
        
        
    def forward(self,noise, label):             #[B,latent_dim,1,1]  [B,]            
            noise = noise if noise.dim() == 4 else noise.unsqueeze(0)             
            label = label if label.dim() == 1 else label.unsqueeze(0)
            # print(label.shape)
            label = self.embedding(label)       #[B,latent_dim]
            # print(label.shape)
            label = label.unsqueeze(-1).unsqueeze(-1)     #[B,latent_dim,1,1]

            z = torch.cat((noise, label),dim=1)         #[B,2 * latent_dim,1,1]  
            z = self.features(z)
            return z

if __name__ == '__main__':
    noise = torch.randn(100,1,1)        #[B,C,H,W]，噪音在C通道
    # label = torch.randn(()).to(dtype=torch.long)            #randn生成0维的需要传一个括号
    noise_ = torch.randn(1,200,1,1)
    
    label = torch.tensor(1)
    G = Generator(4)
    G.features(noise_)
    y = G(noise, label)
    # print(y.shape)
    

