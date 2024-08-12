'''
Metrics used to claculate the mean pixel accuracy. They allow all wound classes to be weight equally, instead of common wound classes dominating the metric.
'''

import torch
import torch.nn.functional as F
import numpy as np


# used to calculate the pixel accuracy per class and batch (-------------------change to per image?)
def image_acc_metric(pred, target, n_classes):
    
    accs = []

    pred_max = torch.argmax(pred, axis=1)
    target_max = torch.argmax(target, axis=1)
    
    pred = torch.movedim(F.one_hot(pred_max, num_classes = n_classes), 3, 1)
    target = torch.movedim(F.one_hot(target_max, num_classes = n_classes), 3, 1)

    for cl in range(0, n_classes):
        pred_inds = pred[:,cl,:,:]
        target_inds = target[:,cl,:,:]
        correct = (pred_inds[target_inds.bool()]).long().sum().data.cpu()
        total = target_inds.long().sum().data.cpu()
        
        if total == 0:
            accs.append(float('nan'))  # If there is no ground truth e.g. no wound of that type -> not included in the evaluation
        else:
            accs.append(float(correct) / float(total))
            
    return np.array(accs)

# used to calculate the pixel accuracy per class
def class_acc_metric(pred, target, n_classes):
    
    corr = []
    tot = []

    pred_max = torch.argmax(pred, axis=1)
    target_max = torch.argmax(target, axis=1)
    
    pred = torch.movedim(F.one_hot(pred_max, num_classes = n_classes),3,1)
    target = torch.movedim(F.one_hot(target_max, num_classes = n_classes),3,1)

    # Ignore IoU for background class ("0")
    for cl in range(0, n_classes):
        pred_inds = pred[:,cl,:,:]
        target_inds = target[:,cl,:,:]
        correct = (pred_inds[target_inds.bool()]).long().sum().data.cpu()
        total = target_inds.long().sum().data.cpu()
        
        if total == 0:
            tot.append(float('nan'))
            corr.append(float("nan"))
        else:
            tot.append(float(total))
            corr.append(float(correct))
                       
    return np.array(corr), np.array(tot)
