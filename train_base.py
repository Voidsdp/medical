import torch
from tqdm import tqdm
import os

from arguments import parse_args
from dataset import build_train_valid_test_data_iterators
from models import build_base_net
from optimizer import get_optimizers
from train import  train_base
from evaluator import eval_base
from utils import log_scalar, set_random_seed


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    #seed
    set_random_seed(args.seed)
    
    #cross validation datasets
    train_loader, val_loader, _ = build_train_valid_test_data_iterators(args.data_path,args.model_name,args.batch_size)
   
    #models and optimizers
    model = build_base_net(args.model_name,args.load_checkpoint,args.backbone_pretrained)
    model = model.to(device)
    optimizers = get_optimizers(model,args.lr)
    
    best_acc, train_acc, val_acc = 0, 0, 0
    
    for epoch in tqdm(range(args.epochs)):
        #train
        loss, train_acc = train_base(train_loader,model,optimizers)
        log_scalar('train loss',loss,epoch,tensorboard=True)
        log_scalar('train acc',train_acc,epoch,tensorboard=True)
   
        #eval
        if args.is_eval:
           val_acc = eval_base(val_loader,model)
           log_scalar('val acc',val_acc,epoch,tensorboard=True)
           
        #save model
        if args.is_eval and val_acc > best_acc or not args.is_eval:
            best_acc = val_acc
            if args.save_checkpoint is not None:                
              torch.save(model.state_dict(),os.path.join(args.save_checkpoint, type(model).__name__ + '.pth'))

        log_scalar('best val acc',best_acc,color='red',tensorboard=True)   


if __name__ == '__main__':
    args = parse_args()
    main(args)