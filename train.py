import argparse
from tqdm import tqdm
import torch

from configs import model as model_cfg
from dataset import build_train_valid_test_data_iterators
from models import build_discriminator_generator_net
from train import loss_compute, get_optimizers
from evaluator import discrimiator_acc_compute,eval
from utils import args_print, acc_print, set_random_seed
from visualize import generate_img

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


#train
def train(train_loader,models,optimizers):
    D, G = models
    d_optimizer, g_optimizer = optimizers

    nc = D.label_linear.out_features
    acc = (0.,0.,0.,0.)

    for real_img, real_label in train_loader:
        real_label = real_label.to(device)    #[B]    
        fake_label = torch.randint(0,nc,(real_label.shape[0],),device=device)   #[B] to consistent with label's shape instead of batch_size 
                                                                                          #    considered 'drop_last' of dataloader 
        real_img = real_img.to(device)   #[B,3,96,96]     
        fake_img = G(fake_label)         #generate fake image before to save computation after

        real_valid = torch.ones(args.batch_size,1,device=device)  #[B,1]
        fake_valid = torch.zeros(args.batch_size,1,device=device) #[B,1]

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
        print('D_LOSS: {:.2f}%,G_LOSS: {:.2f}%'.format(d_loss, g_loss))

        '''
        ------ acc compute ------
        '''
        pred = *real_pred, *fake_pred
        label = real_label, fake_label
        acc = discrimiator_acc_compute(pred,label,acc,len(train_loader))

    acc_print(acc,hightlight=True)

def main(args):

    #seed
    set_random_seed(args.seed)
    
    #cross validation datasets
    train_loader, val_loader, test_loader = build_train_valid_test_data_iterators(args.data_path,args.batch_size)
   
    #models and optimizers
    models = build_discriminator_generator_net(args.checkpoint)
    D, G = [model.to(device) for model in models]
    optimizers = get_optimizers(models,args.lr)

    best_acc, acc = 0, 0
    for epoch in tqdm(range(args.epochs)):
        #train
        train(train_loader,models,optimizers)
        
        #eval
        if args.is_eval:
           acc = eval(val_loader,models)
        
        #save model
        if args.save_model:
           if acc > best_acc or not args.is_eval:
              best_acc = acc
              torch.save(D.state_dict(),model_cfg.checkpoint['discriminator'])
              torch.save(G.state_dict(),model_cfg.checkpoint['generator'])
            
        #visualize
        if args.is_visualize:
           generate_img(G,torch.tensor(1,device=device),'images/{}'.format(epoch))
        
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',default='data/pathology/')        
    parser.add_argument('--checkpoint',default=None,help='None, default, custom path.')
    parser.add_argument('--is_eval',default=True)
    parser.add_argument('--save_model',default=True)
    parser.add_argument('--is_visualize',default=False)

    parser.add_argument('--seed',default=0)
    parser.add_argument('--epochs',default=1000)
    parser.add_argument('--batch_size',default=32)
    parser.add_argument('--lr',default=1e-5)

    args = parser.parse_args()
    args_print(args,'green')
    main(args)