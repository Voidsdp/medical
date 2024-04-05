import torch
import numpy as np
from prettytable import PrettyTable


def acc_compute(pred,label=None):
    if label is None:
       acc = torch.sum((pred > 0.5).float()).item()     #pred [B,1] or [B]
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