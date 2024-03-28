import os
import argparse

from configs import model as model_cfg
from utils import args_print, check_dir

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',default='data/cancer/')       
    parser.add_argument('--model_name',default='Swin-T',choices=['vgg16','resnet50','inception_v3','densenet121',
                                                              'Swin-T','Swin-S','Swin-B','Swin-L'])    
    parser.add_argument('--load_checkpoint',default=None,help='None, default, custom path.')
    parser.add_argument('--save_checkpoint',default=None,help='None, default, custom path.')
    parser.add_argument('--visualize',default=None,help='None, default, custom path')
    parser.add_argument('--backbone_pretrained',default=True)
    parser.add_argument('--is_eval',default=True)

    parser.add_argument('--seed',default=0)
    parser.add_argument('--epochs',default=1000)
    parser.add_argument('--batch_size',default=4)
    parser.add_argument('--lr',default=1e-5)

    args = parser.parse_args()
    
    #check path
    check_dir(args.data_path,alert=True)

    if args.load_checkpoint == 'default':
       args.load_checkpoint = os.path.join(model_cfg.checkpoint,args.model_name)    
    elif args.load_checkpoint is not None :
       check_dir(args.load_checkpoint,True)
    
    if args.save_checkpoint == 'default':
       args.save_checkpoint = os.path.join(model_cfg.checkpoint,args.model_name)    
    elif args.save_checkpoint is not None :
       check_dir(args.save_checkpoint,True)

    if args.visualize == 'default':
       args.visualize = 'images'
    elif args.load_checkpoint is not None :
       check_dir(args.load_checkpoint,True)
    
    args_print(args,color='green')

    return args


