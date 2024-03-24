import torch
import torchvision.utils as vutils

from models import Generator
from utils import writer

def generate_img(G,label,result_dir,epoch):
    fake_image = G(label)
    vutils.save_image(fake_image,result_dir)
    # writer.add_image('fake image',fake_image,epoch)


if __name__ == '__main__':
   device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
   G = Generator().to(device)
   fake_image = G(torch.tensor(1,device=device))
   vutils.save_image(fake_image, 'result.png')
