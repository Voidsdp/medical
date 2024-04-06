import torch
import numpy as np
from prettytable import PrettyTable
from thop import profile

from models import build_discriminator_generator_net

def acc_compute(pred,label=None):
    if label is None:
       acc = torch.sum((pred > 0.5).float()).item()                         #pred [B,1] or [B]
    else:
       acc = torch.sum((torch.argmax(pred, dim=1) == label).float()).item() #pred [B,nc]
    return acc


class ConfusionMatrix:
   def __init__(self,num_classes,labels):
      self.num_classes = num_classes
      self.labels = labels

      self.matrix = np.zeros((num_classes, num_classes))
   def update(self, preds, labels):
      preds = np.asarray(preds, dtype=np.int32)
      labels = np.asarray(labels, dtype=np.int32)

      if len(preds) != len(labels):
         print("Error: preds and labels must have the same length.")
         return

      for p, t in zip(preds, labels):
         self.matrix[p, t] += 1
   def summary(self):
      sum_TP = 0
      for i in range(self.num_classes):
         sum_TP += self.matrix[i, i]
      acc = sum_TP / np.sum(self.matrix)
      table = PrettyTable()
      table.field_names = ['', 'Precision', 'Recall', 'F1 Score']

      for i in range(self.num_classes):
         TP = self.matrix[i, i]
         FP = np.sum(self.matrix[i, :]) - TP 
         FN = np.sum(self.matrix[:, i]) - TP

         Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0
         Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0
         F1 = round(2 * (Precision * Recall) / (Precision + Recall), 3) if Precision + Recall != 0 else 0

         table.add_row([self.labels[i], Precision, Recall, F1])
      
      print(table)
      print('acc:',acc)


def calculate_model_complexity():
   
   D, G = build_discriminator_generator_net('Swin-L',None,False)


   flops, params = profile(D,inputs=(torch.randn(1,3,224,224),))

   print(f"FLOPs: {flops / 1e9} G")  # 打印计算量（以十亿次浮点运算为单位）  
   print(f"Params: {params / 1e6} M")  # 打印参数量（以百万为单位）



