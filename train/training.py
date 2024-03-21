from train import loss_compute
from utils import discrimiator_acc_compute,acc_print, writer


def train(train_loader,models,optimizers):
    D, G = models
    device = next(D.parameters()).device
    d_optimizer, g_optimizer = optimizers

    acc = [0.,0.,0.,0.]

    for img, label, valid in train_loader:
        real_label, fake_label = [item.to(device) for item in label]                                                                                      #    considered 'drop_last' of dataloader 
        real_img, fake_img = img.to(device), G(fake_label)           #[B,3,96,96]     
        real_valid, fake_valid = [item.to(device) for item in valid] 
        real_valid = real_valid.float()
        fake_valid = fake_valid.float() 

        '''
        ------D start------
        '''
        d_optimizer.zero_grad()
                
        d_real_loss, real_pred = loss_compute(real_img,D,real_valid,real_label,return_pred=True)  
        d_fake_loss, fake_pred = loss_compute(fake_img.detach(),D,fake_valid,fake_label,return_pred=True)  
        d_loss = (d_fake_loss + d_real_loss) / 2  # -(LS + LC)
        d_loss.backward()
        d_optimizer.step()

        '''
        ------D end and G start------
        '''
        g_optimizer.zero_grad()

        g_loss = loss_compute(fake_img,D,fake_valid,fake_label)
     
        g_loss.backward()
        g_optimizer.step()
        
        '''
        ------G end------
        '''   
        # print('D_LOSS: {:.2f}%,G_LOSS: {:.2f}%'.format(d_loss, g_loss))

        '''
        ------ acc compute ------
        '''
        pred = *real_pred, *fake_pred
        label = real_label, fake_label

        discrimiator_acc_compute(pred,label,acc,len(train_loader))
        # print(acc)
    acc_print(acc,hightlight=True)