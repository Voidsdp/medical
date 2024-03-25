import torch
from metrics import acc_compute


def eval_acgan(val_loader,models):
    D, _ = models
    device = next(D.parameters()).device
    acc = 0

    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device) 

            pred = D(img)[1]              #[B,1] [B,nc]
            acc += acc_compute(pred,label)
        
        acc = acc / len(val_loader.dataset) * 100

        return acc


def eval_sgan(val_loader,models):
    D, _, _ = models
    device = next(D.parameters()).device
    acc = 0

    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device) 

            pred = D(img)[1]               #[B,1] [B,nc]
            acc += acc_compute(pred,label)
 
        return acc / len(val_loader.dataset) * 100