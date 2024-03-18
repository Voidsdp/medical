import torch
from utils import acc_print,discrimiator_acc_compute


def eval(val_loader,models):
    D, G = models
    device = next(D.parameters()).device
    acc = [0.,0.,0.,0.]
  
    with torch.no_grad():
        for img, label in val_loader:
            real_label, fake_label = [item.to(device) for item in label]  
            real_img, fake_img = img.to(device), G(fake_label)

            pred = *D(real_img), *D(fake_img)           #[B,1] [B,nc]
            label = real_label, fake_label
            discrimiator_acc_compute(pred,label,acc,len(val_loader))

        #print mean_acc 
        acc_print(acc,hightlight=True)

        return acc[1]
