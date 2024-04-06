import torch
from torch import nn
import os
import torch.nn.functional as fn

from configs import model
from .backbone import get_backbone, get_generator_backbone, get_style_backbone

noise_dim = model.noise_dim
num_class = model.num_class
default_checkpoint = model.checkpoint
k = model.k

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_base_net(model_name,checkpoint=None,backbone_pretrained=False):
    base_net = BaseModel(model_name,backbone_pretrained)
    if checkpoint is not None:
       base_net.load_state_dict(torch.load(os.path.join(checkpoint, type(base_net).__name__ + '.pth')))     ###
    
    return base_net


def build_discriminator_generator_net(model_name,checkpoint=None,backbone_pretrained=False):
    """
    Tips:
    return D, G
    """
    D, G = Discriminator(model_name,backbone_pretrained), Generator(model_name)
    if checkpoint is not None:
       D.load_state_dict(torch.load(os.path.join(checkpoint, type(D).__name__ + '.pth')))
       G.load_state_dict(torch.load(os.path.join(checkpoint, type(G).__name__ + '.pth')))

    return D, G 


def build_style_net(model_name,checkpoint=None):
    style_net = StyleNet(model_name)
    if checkpoint is not None:
       style_net.load_state_dict(torch.load(os.path.join(checkpoint, type(style_net).__name__ + '.pth')))
    
    return style_net


def build_coattention_net(model_name,checkpoint=None,backbone_pretrained=False):
    co_model = CoModel(model_name,backbone_pretrained)
    if checkpoint is not None:
       co_model.load_state_dict(torch.load(os.path.join(checkpoint, type(co_model).__name__ + '.pth')))
    
    return co_model


class Discriminator(nn.Module):
    def __init__(self,model_name,backbone_pretrained=False):
        super(Discriminator,self).__init__()
        self.model_name = model_name
        self.backbone, out_features = get_backbone(model_name,backbone_pretrained)
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


class BaseModel(nn.Module):
    def __init__(self,model_name,backbone_pretrained=False):
        super(BaseModel,self).__init__()
        self.model_name = model_name
        self.backbone, out_features = get_backbone(model_name,backbone_pretrained)
        self.label_linear = nn.Linear(out_features,num_class)
    def forward(self,x):
        x = self.backbone(x)
        label = self.label_linear(x)
        
        return label
    

class CoModel(nn.Module):
    def __init__(self,model_name,backbone_pretrained=False,k=k):
        super(CoModel,self).__init__()
        self.model_name = model_name
        self.backbone, out_feature = get_backbone(model_name,backbone_pretrained)
        self.k = k
        self.backbone.classifier = nn.Sequential()

        self.tanh = nn.Tanh()
                                                                    #需要根据不同模型的架构进行修改
        self.W_b = nn.Parameter(torch.randn(7 * 7, 7 * 7))          #[D, D]
        self.W_v = nn.Parameter(torch.randn(k, 7 * 7))              #[k, D]
        self.W_q = nn.Parameter(torch.randn(k, 7 * 7))              #[k, D]
        self.w_hv = nn.Parameter(torch.randn(k, 1))                 
        self.w_hq = nn.Parameter(torch.randn(k, 1))

        self.fc = nn.Linear(98, num_class)
    def forward(self, image_1, image_2):                    
        image_1 = self.backbone(image_1)
        image_2 = self.backbone(image_2)
        image_1, image_2 = image_1.reshape(-1,7 * 7,512), image_2.reshape(-1,7 * 7, 512)

        v, q = self.parallel_co_attention(image_1,image_2)

        v_h = torch.concat((v,q),dim=1)
        h_w = self.tanh(v_h)
        label = self.fc(h_w)

        return label


    def parallel_co_attention(self, V, Q):  # V : B x D x C, Q : B x D x C 
        C = torch.matmul(Q.permute(0, 2, 1), torch.matmul(self.W_b, V)) # B x C x C

        H_v = self.tanh(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q), C))                  #B x K x C                  
        H_q = self.tanh(torch.matmul(self.W_q, Q) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1))) # B x K x C

        a_v = fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)      # B x 1 x C
        a_q = fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)      # B x 1 x C

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)),dim=1)            # B x D
        q = torch.squeeze(torch.matmul(a_q, Q.permute(0, 2, 1)),dim=1)            # B x D

        return v, q