import torch
from utils import acc_print, acc_compute, writer


def eval(val_loader,models,epoch):
    D, G = models
    device = next(D.parameters()).device
    acc = 0.

    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device) 

            pred = D(img)[1]          #[B,1] [B,nc]
            acc += acc_compute(pred,label)
        #print mean_acc
        acc_print(acc / len(val_loader.dataset) * 100,hightlight=True)

        writer.add_scalar('VAL acc',acc / len(val_loader.dataset) * 100,epoch)
        return acc / len(val_loader.dataset) * 100
