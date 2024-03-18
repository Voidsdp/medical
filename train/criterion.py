import torch

def get_criterion():
    adversarial_criterion = torch.nn.BCELoss()
    auxiliary_criterion = torch.nn.CrossEntropyLoss()
    return adversarial_criterion, auxiliary_criterion