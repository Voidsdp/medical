from torch.optim import AdamW
from configs import optimizer

optimzer_type = optimizer.optimizer_type
betas = optimizer.betas

def get_optimizers(models,lr):
    optimizers = []
    if optimzer_type == 'AdamW':
        for model in models:
            optimizer=AdamW(model.parameters(),lr,betas=betas)     #0.9 and 0.99 to do 
            optimizers.append(optimizer)

    return optimizers
