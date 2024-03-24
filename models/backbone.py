from torch import nn
from torchvision import models
import timm

from configs import model

def get_discriminator_backbone(model_name='cnn',pretrained=False):
    if model_name == 'vgg16':
        backbone = models.vgg16(pretrained=pretrained)
        out_features = backbone.classifier[6].in_features
        del backbone.classifier[6]
        
    elif model_name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
        out_features = backbone.fc.in_features
        backbone.fc = nn.Sequential()

    elif model_name == 'inception_v3':
        backbone = models.inception_v3(pretrained=pretrained)

    elif model_name == 'densenet121':
        backbone =  models.densenet121(pretrained=pretrained)
        out_features = backbone.classifier.in_features
        del backbone.classifier

    elif model_name == 'Swin-T':
        backbone = timm.create_model('swin_tiny_patch4_window7_224',
                                  pretrained=pretrained)
        out_features = backbone.head.fc.in_features
        backbone.head.fc = nn.Sequential()
        backbone.head.flatten = nn.Sequential()       
        out_features = 768

    elif model_name == 'Swin-S':
        backbone = timm.create_model('swin_small_patch4_window7_224',
                                  pretrained=pretrained)
        out_features = backbone.head.fc.in_features
        backbone.head.fc = nn.Sequential()
        backbone.head.flatten = nn.Sequential() 
        out_features = 768

    elif model_name == 'Swin-B':
        backbone = timm.create_model('swin_base_patch4_window7_224',
                                  pretrained=pretrained)
        out_features = backbone.head.fc.in_features
        backbone.head.fc = nn.Sequential()
        backbone.head.flatten = nn.Sequential() 
        out_features = 1024
        
    elif model_name == 'Swin-L':
        backbone = timm.create_model('swin_large_patch4_window7_224',
                                  pretrained=pretrained)
        
        out_features = backbone.head.fc.in_features
        backbone.head.fc = nn.Sequential()
        backbone.head.flatten = nn.Sequential() 
        out_features = 1536
        
    else:
       backbone = get_backbone(model_name)
       out_features = 512*4*4

    return backbone, out_features


def get_generator_backbone(model_name='decnn'):
    backbone = get_backbone(model_name)
    return backbone
    

#customed net
noise_dim = model.noise_dim

def get_backbone(model_name='cnn'):
    ndf = 64
    ngf = 64
    if model_name == 'decnn':
            backbone = nn.Sequential(
                nn.ConvTranspose2d(in_channels=2 * noise_dim, out_channels=ngf*8, kernel_size=7, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(ngf*8),          #[B,7,7]
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels=ngf*8, out_channels=ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf*4),          #[B,14,14]
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels=ngf*4, out_channels=ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf*2),          #[B,28,28]
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels=ngf*2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf),            #[B,56,56]
                nn.ReLU(True),
                
                nn.ConvTranspose2d(in_channels=ngf, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf),            #[B,112,112]
                nn.ReLU(True),
                
                nn.ConvTranspose2d(in_channels=ngf, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
                nn.Tanh()                       #[B,224,224]
            )
    return backbone