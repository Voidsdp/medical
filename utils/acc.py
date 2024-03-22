import torch

def acc_compute(pred,label=None):
    if label is None:
       acc = torch.sum((pred > 0.5).float()).item()     #pred [B,1] or [B]
    else:
       acc = torch.sum((torch.argmax(pred, dim=1) == label).float()).item() #pred [B,nc]
    return acc

