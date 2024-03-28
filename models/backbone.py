from torch import nn
from torchvision import models
import timm

from configs import model as model_cfg

def get_discriminator_backbone(model_name='vgg16',pretrained=False):
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
        out_features = backbone.norm.normlized_shape[0]
        backbone.head.fc = nn.Sequential()
        backbone.head.flatten = nn.Sequential()       

    elif model_name == 'Swin-S':
        backbone = timm.create_model('swin_small_patch4_window7_224',
                                  pretrained=pretrained)
        out_features = backbone.norm.normlized_shape[0]
        backbone.head.fc = nn.Sequential()
        backbone.head.flatten = nn.Sequential() 

    elif model_name == 'Swin-B':
        backbone = timm.create_model('swin_base_patch4_window7_224',
                                  pretrained=pretrained)
        out_features = backbone.norm.normlized_shape[0]
        backbone.head.fc = nn.Sequential()
        backbone.head.flatten = nn.Sequential() 
        
    elif model_name == 'Swin-L':
        backbone = timm.create_model('swin_large_patch4_window7_224',
                                  pretrained=pretrained)
        
        out_features = backbone.norm.normlized_shape[0]
        backbone.head.fc = nn.Sequential()
        backbone.head.flatten = nn.Sequential() 
        
    return backbone, out_features

    
def get_generator_backbone(model_name):
    ngf = 64
    model_name = model_name if model_name == 'inception_v3' else 'base'  # change equal to in later
    noise_dim = model_cfg.noise_dim

    if model_name == 'base':
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
    else:
            backbone = nn.Sequential(
                nn.ConvTranspose2d(in_channels=2 * noise_dim, out_channels=ngf*8, kernel_size=7, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(ngf*8),          #[B,C,7,7]
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels=ngf*8, out_channels=ngf*4, kernel_size=7, stride=3, padding=1, bias=False),
                nn.BatchNorm2d(ngf*4),          #[B,C,23,23]
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels=ngf*4, out_channels=ngf*2, kernel_size=7, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf*2),          #[B,C,49,49]
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels=ngf*2, out_channels=ngf, kernel_size=7, stride=3, padding=1, bias=False),
                nn.BatchNorm2d(ngf),            #[B,C,149,149]
                nn.ReLU(True),
                
                nn.ConvTranspose2d(in_channels=ngf, out_channels=3, kernel_size=5, stride=2, padding=1, bias=False),
                nn.Tanh()                       #[B,C,299,299]
            )

    return backbone
    

def get_style_backbone(model_name):
    model_name = model_name if model_name == 'inception_v3' else 'base' #to do or the same

    backbone = nn.Sequential(
            nn.Conv2d(in_channels=4,out_channels=3,kernel_size=3,stride=1,padding=1),
            nn.Tanh()
        )
    
    return backbone