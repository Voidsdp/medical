from torch.optim import AdamW
from configs import optimizer

optimzer_type = optimizer.optimizer_type
betas = optimizer.betas

def get_optimizers(models,lr):
    D, G = models
    if optimzer_type == 'AdamW':
        d_optimizer=AdamW(D.parameters(),lr,betas=betas)     #0.9 and 0.99 to do 
        g_optimizer=AdamW(G.parameters(),lr,betas=betas)

    return d_optimizer, g_optimizer
