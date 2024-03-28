import torch
from tqdm import tqdm

from arguments import parse_args
from dataset import build_train_valid_test_data_iterators
from models import build_discriminator_generator_net, build_style_net
from optimizer import get_optimizers
from train import  train_sgan
from evaluator import eval_sgan
from utils import log_scalar, set_random_seed
from visualize import generate_img

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    #seed
    set_random_seed(args.seed)
    
    #cross validation datasets
    train_loader, val_loader, _ = build_train_valid_test_data_iterators(args.data_path,args.model_name,args.batch_size)
   
    #models and optimizers
    D, G = build_discriminator_generator_net(args.model_name,args.load_checkpoint,args.backbone_pretrained)
    S = build_style_net(args.model_name,args.load_checkpoint)
    models = (D,G,S)
    D, G, S = [model.to(device) for model in models]

    optimizers = get_optimizers(models,args.lr)
 
    best_acc, train_acc, val_acc = 0, 0, 0

    for epoch in tqdm(range(args.epochs)):
        #train
        loss, train_acc = train_sgan(train_loader,models,optimizers,args.epochs)
        log_scalar('train loss',loss,epoch,tensorboard=True)
        log_scalar('train acc',train_acc,epoch,tensorboard=True)

        #eval
        if args.is_eval:
           val_acc = eval_sgan(val_loader,models)
           log_scalar('val acc',val_acc,epoch,tensorboard=True)
        
        #save model
        if args.save_checkpoint is not None:
           if (args.is_eval and val_acc > best_acc) or (not args.is_eval):
              best_acc = val_acc
              torch.save(D.state_dict(),args.save_checkpoint + '/D.pth')
              torch.save(G.state_dict(),args.save_checkpoint + '/G.pth')
              torch.save(G.state_dict(),args.save_checkpoint + '/S.pth')
         
        log_scalar('best val acc',best_acc,color='red',tensorboard=True)     

        #visualize
        if args.visualize is not None:
           generate_img(G,torch.tensor(1,device=device),args.visualize+'/{}.jpg'.format(epoch))

if __name__ == '__main__':
    args = parse_args()
    main(args)