from torch.optim import AdamW

def get_optimizers(models,lr):
    D, G = models
    d_optimizer=AdamW(D.parameters(),lr,betas=[0.5,0.999])     #0.9 and 0.99 to do 
    g_optimizer=AdamW(G.parameters(),lr,betas=[0.5,0.999])

    return d_optimizer, g_optimizer
