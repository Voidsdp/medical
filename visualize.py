import torch
import torchvision.utils as vutils

from configs import model as model_cfg 
from models import Generator

def generate_img(G,label,result_dir):
    fake_image = G(label)
    vutils.save_image(fake_image,result_dir)


if __name__ == '__main__':
   device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
   G = Generator().to(device)
   fake_image = G(torch.tensor(1,device=device))
   vutils.save_image(fake_image, 'result.png')
