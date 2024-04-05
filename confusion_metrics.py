import torch
import os
import json


from arguments import parse_args
from models import build_discriminator_generator_net
from dataset import build_train_valid_test_data_iterators
from configs import model
from evaluator import ConfusionMatrix

num_class = model.num_class

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_labels(folder_path):
    folder_path = os.path.join(folder_path,'train')
    files = os.listdir(folder_path)

    for file_name in files:
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
    labels = [label for label, _ in json_data.items()]
    return labels


def main(args):

    train_loader, val_loader, _ = build_train_valid_test_data_iterators(args.data_path,args.model_name,args.batch_size)
    D, G = build_discriminator_generator_net(args.model_name,args.load_checkpoint,args.backbone_pretrained)
    D = D.to(device)

    labels = get_labels(args.data_path)

    confusion = ConfusionMatrix(num_class,labels)

    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(device), label.to(device) 

            pred = D(img)[1]              #[B,1] [B,nc]
            
            confusion.update(pred.cpu().detach().numpy(), label.cpu().detach().numpy())
        confusion.summary()

if __name__ == '__main__':
    args = parse_args()
    main(args)