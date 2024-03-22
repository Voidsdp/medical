import torch
from .criterion import get_criterion


def loss_compute(img,model,valid,label,return_pred=False):
    adversarial_criterion, auxiliary_criterion = get_criterion()  #as same device as data
    adversarial_criterion, auxiliary_criterion = adversarial_criterion.to(img.device), auxiliary_criterion.to(img.device)
   
    valid_pred, label_pred = model(img) #[B,1] [B,nc]
    valid_pred = valid_pred.view(-1)
    adversarial_loss = adversarial_criterion(valid_pred, valid)
    auxiliary_loss = auxiliary_criterion(label_pred, label)
    loss = (adversarial_loss + auxiliary_loss) / 2
    
    if return_pred:
       return loss, (valid_pred,label_pred)
    else:
       return loss
