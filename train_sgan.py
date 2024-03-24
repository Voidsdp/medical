import os
import argparse
from tqdm import tqdm
import torch

from configs import model as model_cfg
from dataset import build_train_valid_test_data_iterators
from models import build_discriminator_generator_net, build_style_net
from optimizer import get_optimizers
from train import  train_sgan
from evaluator import eval_sgan
from utils import args_print, set_random_seed, check_dir, writer
from visualize import generate_img


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    #seed
    set_random_seed(args.seed)
    
    #cross validation datasets
    train_loader, val_loader, test_loader = build_train_valid_test_data_iterators(args.data_path,args.model_name,args.batch_size)
   
    #models and optimizers
    D, G = build_discriminator_generator_net(args.model_name,args.load_checkpoint,args.backbone_pretrained)
    S = build_style_net(args.model_name,args.load_checkpoint)
    models = (D,G,S)
    D, G, S = [model.to(device) for model in models]
    optimizers = get_optimizers(models,args.lr)
 
    best_acc, acc = 0, 0
    for epoch in tqdm(range(args.epochs)):
        #train
        train_sgan(train_loader,models,optimizers,epoch)
        
        #eval
        if args.is_eval:
           acc = eval_sgan(val_loader,models,epoch)
        
        if acc > best_acc:
           best_acc = acc
           
           writer.add_scalar('best acc',best_acc,epoch)
           #save model
           if args.save_checkpoint is not None:
               if args.save_checkpoint == 'default':
                  checkpoint = os.path.join(model_cfg.checkpoint,D.model_name)
               
               check_dir(checkpoint,True)
               torch.save(D.state_dict(),checkpoint + '/D.pth')
               torch.save(G.state_dict(),checkpoint + '/G.pth')
          
        print('best_acc',best_acc,flush=True)
        
        #visualize
        if args.is_visualize:
           visualize_path = 'images'
           check_dir(visualize_path,True)
           generate_img(G,torch.tensor(1,device=device),visualize_path+'/{}.jpg'.format(epoch),epoch)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',default='data/cancer/')       
    parser.add_argument('--model_name',default='Swin-T',choices=['vgg16','resnet50','inception_v3','densenet121',
                                                              'Swin-T','Swin-S','Swin-B','Swin-L'])    
    parser.add_argument('--load_checkpoint',default=None,help='None, default, custom path.')
    parser.add_argument('--save_checkpoint',default=None,help='None, default, custom path.')
    parser.add_argument('--backbone_pretrained',default=True)
    parser.add_argument('--is_eval',default=True)
    parser.add_argument('--is_visualize',default=False)

    parser.add_argument('--seed',default=0)
    parser.add_argument('--epochs',default=1000)
    parser.add_argument('--batch_size',default=4)
    parser.add_argument('--lr',default=1e-5)

    args = parser.parse_args()
    args_print(args,color='green')
    main(args)