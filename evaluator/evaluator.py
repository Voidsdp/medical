import torch
from .metrics import acc_compute


def eval_base(val_loader,model):
    base_model = model
    device = next(base_model.parameters()).device
    acc = 0

    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)    
            
            label_pred = base_model(img)
            acc += acc_compute(label_pred, label)

        acc = acc / len(val_loader.dataset) * 100

        return acc


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


def eval_co(val_loader,models):
    co_model = models
    device = next(co_model.parameters()).device
    acc = 0

    with torch.no_grad():
        for pathlolgy_img,Imaging_img, label in val_loader:
            pathlolgy_img, Imaging_img, label = pathlolgy_img.to(device),Imaging_img.to(device) ,label.to(device) 
            pred = co_model(pathlolgy_img,Imaging_img)             
                        
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