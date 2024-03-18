import torch
from utils import acc_print
from .utils import discrimiator_acc_compute


#valid
def eval(val_loader,models):
    D, G = models
    device = next(D.parameters()).device
    nc = D.label_linear.out_features
    acc = [0.,0.,0.,0.]
  
    with torch.no_grad():
        for real_img, real_label in val_loader:
            batch_size = real_label.shape[0]  #current_batch_size is not as same as args.batch_size

            real_label = real_label.to(device)
            fake_label = torch.randint(0,nc,(batch_size),device=device) 

            real_img = real_img.to(device)
            fake_img = G(fake_label)

            pred = *D(real_img), *D(fake_img)           #[B,1] [B,nc]
            label = real_label, fake_label
            discrimiator_acc_compute(pred,label,acc,len(val_loader))

        #print mean_acc 
        acc_print(acc,hightlight=True)

        return acc[1]