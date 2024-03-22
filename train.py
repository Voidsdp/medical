import os
import argparse
from tqdm import tqdm
import torch

from configs import model as model_cfg
from dataset import build_train_valid_test_data_iterators
from models import build_discriminator_generator_net
from optimizer import get_optimizers
from train import  train
from evaluator import eval
from utils import args_print, set_random_seed, check_dir, writer
from visualize import generate_img


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    #seed
    set_random_seed(args.seed)
    
    #cross validation datasets
    train_loader, val_loader, test_loader = build_train_valid_test_data_iterators(args.data_path,args.batch_size)
   
    #models and optimizers
    models = build_discriminator_generator_net(args.model_name,args.load_checkpoint,args.backbone_pretrained)
    D, G = [model.to(device) for model in models]
    optimizers = get_optimizers(models,args.lr)

    best_acc, acc = 0, 0
    for epoch in tqdm(range(args.epochs)):
        #train
        train(train_loader,models,optimizers)
        
        #eval
        if args.is_eval:
           acc = eval(val_loader,models)
        
        if acc > best_acc:
           best_acc = acc
           print('best_acc',best_acc,flush=True)

           #save model
           if args.save_checkpoint is not None:
               if args.save_checkpoint == 'default':
                  checkpoint = os.path.join(model_cfg.checkpoint,D.model_name)
               
               check_dir(checkpoint,True)
               torch.save(D.state_dict(),checkpoint + '/D.pth')
               torch.save(G.state_dict(),checkpoint + '/G.pth')
            
        #visualize
        if args.is_visualize:
           visualize_path = 'images'
           check_dir(visualize_path,True)
           generate_img(G,torch.tensor(1,device=device),visualize_path+'/{}.jpg'.format(epoch))
        
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',default='data/cells/')       
    parser.add_argument('--model_name',default='resnet50',choices=['cnn','vgg16','resnet50','inception_v3','densenet121',
                                                              'Swin-T','Swin-S','Swin-B','Swin-L'])    
    parser.add_argument('--load_checkpoint',default=None,help='None, default, custom path.')
    parser.add_argument('--save_checkpoint',default=None,help='None, default, custom path.')
    parser.add_argument('--backbone_pretrained',default=True)
    parser.add_argument('--is_eval',default=True)
    parser.add_argument('--is_visualize',default=False)

    parser.add_argument('--seed',default=0)
    parser.add_argument('--epochs',default=1000)
    parser.add_argument('--batch_size',default=32)
    parser.add_argument('--lr',default=1e-5)

    args = parser.parse_args()
    args_print(args,color='green')
    main(args)