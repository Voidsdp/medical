import torch

BCE = torch.nn.BCELoss()
CE = torch.nn.CrossEntropyLoss()

def loss_compute(valid_pred,label_pred,valid,label):
    device = label.device
    valid_criterion, label_criterion = BCE.to(device), CE.to(device)  #as same device as data

    valid_pred, valid = valid_pred.view(-1), valid.view(-1)

    valid_loss = valid_criterion(valid_pred, valid)
    label_loss = label_criterion(label_pred, label)
    loss = (valid_loss + label_loss) / 2

    return loss
