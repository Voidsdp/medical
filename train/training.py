import torch

from configs import model
from evaluator import acc_compute
from .loss import loss_compute, loss_compute_base

num_class = model.num_class


def train_base(train_loader,models,optimizers):
    base_model = models
    device = next(base_model.parameters()).device
    optimizer = optimizers

    loss, acc = 0, 0

    for img, label in train_loader:
        optimizer.zero_grad()
        img, label = img.to(device), label.to(device)
        label_pred = base_model(img)

        loss_base = loss_compute_base(label_pred, label)
        loss_base.backward()

        optimizer.step()
        
        acc += acc_compute(label_pred, label)
        loss += loss_base

    loss = loss.item() / len(train_loader.dataset)  
    acc = acc / len(train_loader.dataset) * 100

    return loss, acc


def train_co(train_loader,models,optimizers):
    co_model = models
    device = next(co_model.parameters()).device
    optimizer = optimizers

    loss, acc = 0, 0

    for pathlolgy_img,Imaging_img, label in train_loader:
        optimizer.zero_grad()
        pathlolgy_img, Imaging_img, label = pathlolgy_img.to(device),Imaging_img.to(device) ,label.to(device)
        label_pred = co_model(pathlolgy_img,Imaging_img)

        loss_base = loss_compute_base(label_pred, label)
        loss_base.backward()

        optimizer.step()
        
        acc += acc_compute(label_pred, label)
        loss += loss_base 

    loss = loss.item() / len(train_loader.dataset)  
    acc = acc / len(train_loader.dataset) * 100

    return loss, acc


def train_acgan(train_loader,models,optimizers):
    D, G = models
    device = next(D.parameters()).device
    d_optimizer, g_optimizer = optimizers

    loss, acc = 0, 0

    for img, label in train_loader:
        batch_size = img.shape[0] # considered 'drop_last' of dataloader 

        real_label = label.to(device)
        fake_label = torch.randint(0,num_class,(batch_size,),device=device)   
                                                                                         
        real_img = img.to(device) # [B,3,96,96]    
        fake_img = G(fake_label)

        real_valid = torch.ones((batch_size,1),dtype=torch.float32,device=device)  #[B,1]
        fake_valid = torch.zeros((batch_size,1),dtype=torch.float32,device=device) #[B,1]

        '''
        D start
        '''
        d_optimizer.zero_grad()

        real_valid_pred, real_label_pred = D(real_img)          # [B*1] [B*nc]

        d_real_loss = loss_compute(real_valid_pred,real_label_pred,real_valid,real_label)  
        d_fake_loss = loss_compute(*D(fake_img.detach()) ,fake_valid,fake_label)  
        d_loss = (d_fake_loss + d_real_loss) / 2 
        
        d_loss.backward()
        d_optimizer.step()

        loss += d_loss

        '''
        D end and G start
        '''
        g_optimizer.zero_grad()
        
        g_loss = loss_compute(*D(fake_img),real_valid,fake_label)
     
        g_loss.backward()
        g_optimizer.step()
        
        '''
        G end
        '''   
        acc += acc_compute(real_label_pred,real_label)
    
    loss = loss.item() / len(train_loader.dataset)
    acc = acc / len(train_loader.dataset) * 100

    return loss, acc
     

def train_sgan(train_loader,models,optimizers,epoch):
    D, G, S = models
    device = next(D.parameters()).device
    d_optimizer, g_optimizer, s_optimizer = optimizers

    loss, acc = 0, 0

    for img, label in train_loader:
        batch_size = img.shape[0] # considered 'drop_last' of dataloader 

        real_label = label.to(device)
        fake_label = torch.randint(0,num_class,(batch_size,),device=device)   
                                                                                         
        real_img = img.to(device) # [B,3,96,96]    
        s_real_img = S(real_img)
        s_fake_img = S(G(fake_label))

        real_valid = torch.ones((batch_size,1),dtype=torch.float32,device=device)  #[B,1]
        fake_valid = torch.zeros((batch_size,1),dtype=torch.float32,device=device) #[B,1]
    
        '''
        D start
        '''
        d_optimizer.zero_grad()

        real_valid_pred, real_label_pred = D(real_img)

        d_real_loss = loss_compute(real_valid_pred,real_label_pred,real_valid,real_label)  
        d_s_real_loss = loss_compute(*D(s_real_img.detach()),real_valid,real_label)
        d_s_fake_loss = loss_compute(*D(s_fake_img.detach()),fake_valid,fake_label)  
 
        d_loss = (d_s_fake_loss + d_real_loss + d_s_real_loss) / 3  
        
        d_loss.backward()
        d_optimizer.step()
    
        loss += d_loss
        '''
        D end and G start
        '''
        g_optimizer.zero_grad()
        s_optimizer.zero_grad()

        g_s_fake_loss = loss_compute(*D(s_fake_img),real_valid,fake_label)
        s_loss = loss_compute(*D(s_real_img),fake_valid,real_label)
        g_s_loss = (g_s_fake_loss + s_loss) / 2
        
        g_s_loss.backward()
        g_optimizer.step()
        s_optimizer.step()
        
        '''
        G end
        '''   
        acc += acc_compute(real_label_pred,real_label)

    loss = loss.item() / len(train_loader.dataset)
    acc = acc / len(train_loader.dataset) * 100

    return loss, acc
