'''
Code used to train all models described on the paper "Segmentation and classification of seven common wounds in forensic medicine". 
The aim is to segment and classify wounds on 2D images taken during the forensic medical exams of injured persons.
'''

# import necessary libraries

from loss import custom_weighted_BCE, sqrt_custom_weighted_BCE, weighted_BCE, sqrt_weighted_BCE, weighted_FTL, weighted_FTL_BCE
from dataloader_and_preprocessing import Dataset, Dataset_imlarge
from metrics import image_acc_metric, class_acc_metric
from augmentation import training_augmentation, training_augmentation_imlarge, valid_augmentation_imlarge #, valid_augmentation
from epochs import TrainEpoch, ValidEpoch, TrainEpoch_imlarge, ValidEpoch_imlarge
from config import config
# from select_model import MODEL
import subprocess

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import sys
import os
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader                                                                                                                                                                                                                                                                                                                                             
from torch.utils.data import Dataset as BaseDataset
import torch.nn.functional as F
import kornia as K                                                                                                                                                                                                
import shutil
import time

# Train on GPU if available / otherwise train on CPU

print("GPU available:", torch.cuda.is_available())

DEVICE = config.DEVICE
now = time.strftime("_%Y_%m_%d_%H")

# set variables
path = config.path
model_version = config.model_version
ENCODER = config.ENCODER
ENCODER_WEIGHTS = config.ENCODER_WEIGHTS
ACTIVATION = config.ACTIVATION
BATCHSIZE = config.BATCHSIZE

source_folder = r"E:\ForMaLTeC\Tools\Wound_segmentation\Code\Wound_segmentation_main"
destination_folder = os.path.join(path, "runs", "paper", model_version + now, "code_and_files")

# Load all annotation data (VGG Image Annotator format)
with open(os.path.join(path, "Annotations/annotation.json")) as json_file:
    data = json.load(json_file)

# Get names of images
names = list(data["_via_img_metadata"].keys())
    
# Wound_classes without "Stich/Schnitt-Kombination" 
wound_classes = config.wound_classes
print(wound_classes)
    
# Load image names from the test dataset
with open(os.path.join(path, "Text_files/test.txt"), "r") as filehandle:
    test = json.load(filehandle)
    
#%%

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    
conda_file = os.path.join(destination_folder, "environment.yaml")
# save a copy to the results folder for reproducebility
command = f"conda env export -n WoundSeg -f {conda_file}"
result = subprocess.run(command, stderr=subprocess.PIPE)
    
with open(os.path.join(destination_folder, "annotation.json"), "w") as json_file:
    json.dump(data, json_file)

with open(os.path.join(destination_folder, "test.txt"), "w") as json_file:
    json.dump(test, json_file)

file_list = os.listdir(source_folder)

python_files = [file for file in file_list if file.endswith(".py")]

for file in python_files:
    
    source_file = os.path.join(source_folder, file)
    
    
    shutil.copy(source_file, \
                destination_folder)
        
#%%  
# remove test from train and val images
names = [x for x in names if x not in test]

# split data for training and validation and seven folds
kf = KFold(n_splits = 7, shuffle = True, random_state = 42)
for ti, vi in kf.split(names):
    print(len(ti), len(vi))

print("Number of images in train and validation: ", len(names))
print("Number of images in test dataset: ", len(test))
print("Wound classes: " , wound_classes)

Loss_functions = [custom_weighted_BCE, sqrt_custom_weighted_BCE, weighted_BCE, 
                  sqrt_weighted_BCE, weighted_FTL, weighted_FTL_BCE]

for loss_function in Loss_functions: # not super elegant-------------------------------
    if loss_function.__name__ == config.loss:
        loss = loss_function()
        break

# test dataset without transformations
test_dataset = Dataset(
    test,
    dir_path = path,
    augmentation=False,
    preprocessing=True,
)

# run the training for seven folds
for fold, (train_index, val_index) in enumerate(kf.split(names)):
    if fold > 0:
        break
    
    MODEL = smp.FPN(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=2, 
        activation=ACTIVATION
    )
    
    # the model is newly initialized every fold to restart training from scratch 
    model = MODEL
    
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=config.optimizer_lr),
    ])
    lr_scheduler = lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.lr_scheduler_gamma, verbose=True)
    
    train = [names[i] for i in train_index]
    valid = [names[i] for i in val_index]
    
    # large images need to be loaded, so that we need another training pipeline
    if ENCODER == "hd_model": 
        train_epoch = TrainEpoch_imlarge(
            model,
            loss=loss,
            metrics=config.metrics,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )
        
        valid_epoch = ValidEpoch_imlarge(
            model,
            loss=loss,
            metrics=config.metrics,
            device=DEVICE,
            verbose=True,
        )
        train_dataset = Dataset_imlarge(
            train[:],
            dir_path = path,
            augmentation=training_augmentation_imlarge(),
            preprocessing=True,
        )
        
        valid_dataset = Dataset_imlarge(
            valid[:],
            dir_path = path,
            augmentation=valid_augmentation_imlarge(),
            preprocessing=True,
        )
    

    else:
        train_epoch = TrainEpoch(
            model,
            loss=loss,
            # metrics=config.metrics,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )
        
        valid_epoch = ValidEpoch(
            model,
            loss=loss,
            # metrics=config.metrics,
            device=DEVICE,
            verbose=True,
        )
        
        train_dataset = Dataset(
            train[:], # + valid[:],
            dir_path = path,
            augmentation=training_augmentation(),
            preprocessing=True,
        )
        
        valid_dataset = Dataset(
            valid[:],
            dir_path = path,
            augmentation=False,
            preprocessing=True,
        )
    
    
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True) #, num_workers=4) 
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False) #, num_workers=4)

    max_score = 0
    if not os.path.exists(os.path.join(path, "runs", "paper", model_version + now)):
        os.makedirs(os.path.join(path, "runs", "paper", model_version + now))
    writer = [[] for _ in range(13)]
    
    for i in range(0, 150):
    
        print('\nEpoch: {}'.format(i))
        print('\nFold: {}'.format(fold))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
    
        # the mean class accuracy of the seven most common wound classes is used to determine the best model, which is saved
        new_max_score = np.nanmean([valid_logs["acc"]])
        
        if max_score < new_max_score:
            max_score = new_max_score
            torch.save(model, os.path.join(path, "runs", "paper", model_version + now, model_version) \
                        + "_best_model_fold_" + str(fold) + ".pth")
            print('Model saved!')
    
        if i==50:
            torch.save(model, os.path.join(path, "runs", "paper", model_version + now, model_version + "_75_epoch_model_fold_" + str(fold) + ".pth"))
            
        # writer[0].append(valid_logs["dice_loss"])
        # writer[1].append(train_logs["dice_loss"])
        # writer[2].append(valid_logs['iou_score'])
        # writer[3].append(train_logs["iou_score"])
        
        writer[4].append(valid_logs["loss"])
        writer[5].append(train_logs["loss"])
        writer[6].append(valid_logs['acc'])
        writer[7].append(train_logs["acc"])
        writer[8].append(valid_logs["mean_acc_per_image"])
        writer[9].append(train_logs["mean_acc_per_image"])
        
        lr_scheduler.step()
        
        # every 25 epochs the confusion matrix for the test dataset is calculated to give some idea of the models performance on the test set
        if i%25 == 0:
            
            y_true = []
            y_pred = []
            y_pred_best = []
            
            class_accs_test = np.full((1, len(wound_classes)), np.nan)
            class_correct_test = np.full((1, len(wound_classes)), np.nan)
            class_total_test = np.full((1, len(wound_classes)), np.nan)
            
            for n in range(len(test_dataset)):
            
                image, y_gt = test_dataset[n]
         
                w, h = y_gt.size()[1], y_gt.size()[2]
                
                x_tensor = image.unsqueeze(0)
                pr_mask = model.forward(x_tensor)
    
                # y_gt = torch.cat((torch.full((1,w,h), 0.8), y_gt.cpu().detach()))
                
                class_accs_test = np.concatenate((class_accs_test, 
                                image_acc_metric(pr_mask, torch.unsqueeze(y_gt, 0), len(wound_classes))[None,:]))
                
     
                cor, tot = class_acc_metric(pr_mask, torch.unsqueeze(y_gt, 0), len(wound_classes))
                class_correct_test = np.concatenate((class_correct_test, cor[None,:]))
                class_total_test = np.concatenate((class_total_test, tot[None,:]))
                
                # get values necessary to calculate the confusion matrix
                y_gt = torch.argmax(y_gt, dim = 0)
                pr_mask = torch.argmax(pr_mask, dim = 1)
    
                y_gt = y_gt.flatten().tolist()
                y_true.extend(y_gt)
                pr_mask = pr_mask.flatten().tolist()
                y_pred.extend(pr_mask)
                
                
            print(np.nansum(class_correct_test, axis=0)/np.nansum(class_total_test, axis=0))    # values of cm diagonal 
            # This includes all 10 wound classes
            print(np.nanmean(class_accs_test, axis = 0))
            
            cm = confusion_matrix(y_true, y_pred, normalize="true") # For all classes
        
            df_cm = pd.DataFrame(cm, index = [i for i in wound_classes],
                              columns = [i for i in wound_classes])
            
            plt.figure(figsize = (10,7))
            sns.heatmap(df_cm, annot=True)
            plt.title('Confusion matrix , Average Test IoU: ' + str(np.round(np.trace(cm[:,:])/len(wound_classes)*100,2)) + "%")
            plt.tight_layout()
            plt.show()
            
            writer[10].append(list(np.round(np.nanmean(class_accs_test, axis = 0), 3)))
            writer[11].append(list(np.round(np.diagonal(cm), 3)))
            writer[12].append(cm.tolist())
            

            
    keys = ["dice_valid", "dice_train", "iou_valid", "iou_train", "loss_valid", "loss_train", 
    "acc_valid", "acc_train", "mean_acc_valid_per_image", "mean_acc_train_per_image", 
     "mean_acc_test_per_image", "cm_diagonal", "cm"] # cm_diagonal = acc_test
    logs = dict(zip(keys, writer))

    with open(os.path.join(path, "runs", "paper", model_version + now, "fold_" + str(fold) + "_logs.json"), 'w') as fp:
        json.dump(logs, fp)
    
        
    torch.save(model, os.path.join(path, "runs", "paper", model_version + now, model_version + "_last_epoch_model_fold_" + str(fold) + ".pth"))

