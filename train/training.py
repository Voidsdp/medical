from train import loss_compute
from utils import acc_compute,acc_print, writer


def train_acgan(train_loader,models,optimizers,epoch):
    D, G = models
    device = next(D.parameters()).device
    d_optimizer, g_optimizer = optimizers

    acc = 0.

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
        d_fake_loss = loss_compute(fake_img.detach(),D,fake_valid,fake_label)  
        d_loss = (d_fake_loss + d_real_loss) / 2  # -(LS + LC)
        d_loss.backward()
        d_optimizer.step()

        '''
        ------D end and G start------
        '''
        g_optimizer.zero_grad()

        g_loss = loss_compute(fake_img,D,real_valid,fake_label)
     
        g_loss.backward()
        g_optimizer.step()
        
        '''
        ------G end------
        '''   
        # print('D_LOSS: {:.2f}%,G_LOSS: {:.2f}%'.format(d_loss, g_loss))
        writer.add_scalars('Train Loss',{'g_loss':g_loss, 'd_loss':d_loss},epoch)
        '''
        ------ acc compute ------
        '''
        pred = real_pred[1]
        
        acc += acc_compute(pred,real_label)
    
    writer.add_scalar('TRAIN acc',acc / len(train_loader.dataset) * 100,epoch)
    acc_print(acc / len(train_loader.dataset) * 100,hightlight=True)
    

def train_sgan(train_loader,models,optimizers,epoch):
    D, G, S = models
    device = next(D.parameters()).device
    d_optimizer, g_optimizer, s_optimizer = optimizers

    acc = 0.

    for img, label, valid in train_loader:
        real_label, fake_label = [item.to(device) for item in label]                                                                                      #    considered 'drop_last' of dataloader 
        real_img, s_fake_img = img.to(device), S(G(fake_label))           #[B,3,96,96]     
        s_real_img =  S(real_img)
        
        real_valid, fake_valid = [item.to(device) for item in valid] 
        real_valid = real_valid.float()
        fake_valid = fake_valid.float() 

        '''
        ------D start------
        '''
        d_optimizer.zero_grad()

        d_real_loss, real_pred = loss_compute(real_img,D,real_valid,real_label,return_pred=True)  
        d_s_fake_loss = loss_compute(s_fake_img.detach(),D,fake_valid,fake_label)  
        d_s_real_loss = loss_compute(s_real_img.detach(),D,real_valid,real_label)
        d_loss = (d_s_fake_loss + d_real_loss + d_s_real_loss) / 3  
        
        d_loss.backward()
        d_optimizer.step()

        '''
        ------D end and G start------
        '''
        g_optimizer.zero_grad()
        s_optimizer.zero_grad()

        g_s_fake_loss = loss_compute(s_fake_img,D,real_valid,fake_label)
        s_loss = loss_compute(s_real_img,D,fake_valid,real_label)
        g_s_loss = (g_s_fake_loss + s_loss) / 2
        
        g_s_loss.backward()
        g_optimizer.step()
        s_optimizer.step()
        
        '''
        ------G end------
        '''   
        # print('D_LOSS: {:.2f}%,G_LOSS: {:.2f}%'.format(d_loss, g_loss))
        writer.add_scalars('Train Loss',{'g_loss':g_s_loss, 'd_loss':d_loss},epoch)
        '''
        ------ acc compute ------
        '''
        pred = real_pred[1]
        
        acc += acc_compute(pred,real_label)
    
    writer.add_scalar('TRAIN acc',acc / len(train_loader.dataset) * 100,epoch)
    acc_print(acc / len(train_loader.dataset) * 100,hightlight=True)