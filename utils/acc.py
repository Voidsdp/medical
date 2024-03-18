import torch

def acc_compute(pred,label=None):
    if label is None:
       acc = torch.sum((pred > 0.5).float()).item()     #pred [B,1] or [B]
    else:
       acc = torch.sum((torch.argmax(pred, dim=1) == label).float()).item() #pred [B,nc]
    return acc

'''
format
pred: (real_valid_pred, real_label_pred, fake_valid_pred, fake_label_pred)
label: (real_label, fake_label)
acc: (real_valid_pred, real_label_pred, fake_valid_pred, fake_label_pred)

warning: pred and acc is same sort
'''
def discrimiator_acc_compute(pred,label,acc,dataset_length):
    real_valid_pred, real_label_pred, fake_valid_pred, fake_label_pred = pred
    real_label, fake_label = label
    
    real_valid_acc, real_label_acc, fake_valid_acc, fake_label_acc = acc

    #valid acc count
    real_valid_acc += acc_compute(real_valid_pred) / dataset_length * 100 
    fake_valid_acc += (real_label.shape[0] - acc_compute(fake_valid_pred)) / dataset_length * 100 

    #label acc count
    real_label_acc += acc_compute(real_label_pred,real_label) / dataset_length * 100 
    fake_label_acc += acc_compute(fake_label_pred,fake_label) / dataset_length * 100 
 
