#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference and plotting of results
"""
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import segmentation_models_pytorch as smp
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
import torch.nn.functional as F
import kornia as K
import segmenter

if torch.cuda.is_available():
    DEVICE = 0
    
path = "/home/zino/Data"

#%%

plt.figure(figsize=(35,23))
plt.rcParams["font.size"]="25"

model_versions = ["Seresnext_FPN_BCE_100_ip", "Segformer_pretrained_100"]
box = ["Se-ResNext-50 FPN",  "Segformer_b2 FPN"]

for b, model_version in enumerate(model_versions):

    allogs = []
    for fold in range(5):
        with open(os.path.join(path, "runs", "Project", model_version, "fold_" + str(fold) + "_logs.json")) as fp:
            logs = json.load(fp)
            allogs.append(logs)

    averagev = []
    averaget =[]
    for logs in allogs:
        averagev.append(logs["iou_valid"])
        reducedv = [np.array(logs["acc_valid"])[:,i] for i in [1,2,3,5,6,8,10]]
        reducedv = np.mean(reducedv, axis=0)
        averaget.append(reducedv)

    plt.plot(np.mean(np.array(averagev)*100, 0), label = "IoU validation set; " + box[b], linewidth=4)
    plt.plot(np.mean(np.array(averaget)*100, 0), label = "Mean pixel accuracy validation set; " + box[b], linewidth=4)
plt.xlabel('epochs', size = 45, labelpad=30)
plt.ylabel('mean pixel accuracy and IoU (%)', size = 45, labelpad=30)

plt.title("Intersection over union and mean pixel accuracy validation set", size = 60, y=1.02)
plt.legend(prop = {"size": 40})
plt.tight_layout()
plt.show()

#%%

# Claculate the mean maximal validation accuracy
for model_version in model_versions: 
    logs = []
    for fold in range(7):
        with open(os.path.join(path, "runs", "Project", model_version, "fold_" + str(fold) + "_logs.json"), 'r') as fp:
            log = json.load(fp)
            logs.append(log)
     
    valid_accs = [log["acc_valid"] for log in logs]
    valid_accn = np.transpose(np.array(valid_accs), (2,0, 1))
    valid_accn = np.vstack((valid_accn[1:4], valid_accn[5:7], [valid_accn[8]], [valid_accn[10]]))
    max_accs = np.nanmean(valid_accn, 0) #mean over wound classes
    max_accs = np.max(max_accs, 1)
    
    print(model_version, " mean max validation accuracy: ", np.mean(max_accs))
    
    
#%%

# Plot mean test pixel accuracy
plt.figure(figsize=(35,23))

model_versions = ["Segformer_pretrained_100"]
 
wound_class_reduced = ['hematoma', 'cut', 'stab', 'dermatorrhagia', 'abrasion', 'contused-lacerated', 'thermal']
sns.set(font_scale=3.3)
for model_version in model_versions:
    cm = np.load(os.path.join(path, "runs", "Project", model_version, "average_test_cm.npy"))
    df_cm = pd.DataFrame(cm, index = [i for i in ["background"] + wound_class_reduced],
                  columns = [i for i in ["background"] + wound_class_reduced])

    sns.heatmap(df_cm, annot=True, vmin=0, vmax=1)
    plt.title("Mean pixel accuracy test set: " + str(np.round(np.trace(cm[1:,1:])/len(wound_class_reduced)*100,2)) + "%"\
             , size=50, pad=30, y = 1.01)
    plt.xlabel("Predicted classification")
    plt.ylabel("Actual classification")
    plt.tight_layout()
    plt.show()
    
#%%
class Dataset(BaseDataset):

    def __init__(
            self,
            names_dir,
            dir_path,
            classes=None,
            augmentation=None,
            preprocessing=None,
        ):

        self.image_ids = [os.path.join(dir_path, "Images_512_1024", image_id) + ".png" for image_id in names_dir]
        self.mask_ids = [os.path.join(dir_path, "Masks_512_1024", mask_id) + ".png" for mask_id in names_dir]
        self.certainty_ids = [os.path.join(dir_path, "Masks_512_1024_certainty", mask_id) + ".png" for mask_id in names_dir]
        self.alternative_ids = [os.path.join(dir_path, "Masks_512_1024_alternatives", mask_id) + ".png" for mask_id in names_dir]
        self.wound_classes = classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.path = dir_path
        self.mean = torch.Tensor([0.55540512, 0.46654144, 0.42994756])
        self.std = torch.Tensor([0.21014232, 0.21117639, 0.22304179])

    def __getitem__(self, ind):
        
        image: np.ndarray = cv2.imread(self.image_ids[ind], cv2.IMREAD_COLOR)
        image: torch.Tensor = K.image_to_tensor(image)
        image = image.type(torch.float16).to(DEVICE)

        mask: np.ndarray = cv2.imread(self.mask_ids[ind], cv2.IMREAD_GRAYSCALE)/20
        mask: torch.tensor =  K.image_to_tensor(mask)
        mask = mask.type(torch.float16).to(DEVICE)
        

        if self.augmentation: 
            image, mask = self.augmentation(image, mask)

        target_bce = F.one_hot(mask[0].long(), num_classes = 11 + 2 )[:,:,1:]
        target_bce = target_bce[:,:,:11]

        if self.preprocessing:

            image = image / 255.0
            
            image[0] -= self.mean[0]
            image[1] -= self.mean[1]
            image[2] -= self.mean[2]
            
            image[0] /= self.std[0]
            image[1] /= self.std[1]
            image[2] /= self.std[2]
            
            target_bce = target_bce.permute((2, 0, 1))
            
        
        if not self.preprocessing and not self.augmentation:
            return image, target_bce[1:] # Used for testing Hautrötung entfernt
        
        return image.type(torch.float32), target_bce[1:] # Hautrötung entfernt

    def __len__(self):
        return len(self.image_ids)
    
with open('/home/zino/Data/text_files/test.txt', 'r') as filehandle:
    test = json.load(filehandle)  
    
with open("/home/zino/Data/Annotation/Annotation.json") as json_file:
    data = json.load(json_file)
    

# test dataset without transformations for image visualization
test_dataset = Dataset(
    test,
    dir_path = path,
    augmentation=False,
    preprocessing=True,
    classes=11,
)


with open('/home/zino/Data/text_files/test_optimal_images.txt', 'r') as filehandle:
    test_optimal = json.load(filehandle)
    

optimal_dataset = Dataset(
    test_optimal,
    dir_path = path,
    augmentation=False,
    preprocessing=True,
    classes=11,
)

def image_acc_metric(pred, target, n_classes=11):
    
    accs = []

    pred_max = torch.argmax(pred, axis=1)
    target_max = torch.argmax(target, axis=1)
    
    pred = torch.movedim(F.one_hot(pred_max, num_classes = n_classes),3,1)
    target = torch.movedim(F.one_hot(target_max, num_classes = n_classes),3,1)

    for cl in range(0, n_classes):
        pred_inds = pred[:,cl,:,:]
        target_inds = target[:,cl,:,:]
        correct = (pred_inds[target_inds.bool()]).long().sum().data.cpu() # Cast to long to prevent overflows
        total = target_inds.long().sum().data.cpu()
        
        if total == 0:
            accs.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            accs.append(float(correct) / float(total))
            
    return np.array(accs)


def class_acc_metric(pred, target, n_classes):
    
    corr = []
    tot = []

    pred_max = torch.argmax(pred, axis=1)
    target_max = torch.argmax(target, axis=1)
    
    pred = torch.movedim(F.one_hot(pred_max, num_classes = n_classes),3,1)
    target = torch.movedim(F.one_hot(target_max, num_classes = n_classes),3,1)

    # Ignore IoU for background class ("0")
    for cl in range(0, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred[:,cl,:,:]
        target_inds = target[:,cl,:,:]
        correct = (pred_inds[target_inds.bool()]).long().sum().data.cpu() # Cast to long to prevent overflows
        total = target_inds.long().sum().data.cpu()
        
        if total == 0:
            tot.append(float('nan'))  # If there is no ground truth, do not include in evaluation
            corr.append(float("nan"))
        else:
            tot.append(float(total))
            corr.append(float(correct))
                       
    return np.array(corr), np.array(tot)

test_dataset_vis = Dataset(
    test,
    dir_path = path,
    classes=11,
)

#%%
plt.figure(figsize=(25,15))
plt.rcParams.update({"font.size": 5})

#%%

# Plot both the predictions of the model, as well as saliency maps
import matplotlib.colors as colors
cmap = colors.ListedColormap(["black", "red", "green","orange", "white", "yellow", "magenta", "lime", "cyan", "indigo", "blue"])
boundaries = [0,1,2,3,4,5,6,7,8,9,10,11]


model_versions = ["Segformer_pretrained_100"]

norm = colors.BoundaryNorm(boundaries, cmap.N, clip = True)
# take model with best validation accuracy per image in the common wound classes for testing
for model_version in model_versions:

    DEVICE = "cuda"
    y_true = []
    y_pred = []
    y_pred_best = []
    print(len(test_dataset))

    for n in range(len(test_dataset)):
        saliencies = []
        predictions = []
        
        for f in range(7):
            best_model = torch.load(os.path.join(path, "runs", "Project", model_version, \
                    model_version + "_best_model_fold_" + str(f) + ".pth"), map_location=torch.device(0))
                
            image_vis = test_dataset_vis[n][0].cpu().numpy().astype('uint8')
            image, gt_mask = test_dataset[n]
    
            y_gt = gt_mask.cpu()
    
            w, h = y_gt.size()[1], y_gt.size()[2]
    
    
            x_tensor = image.to(DEVICE).unsqueeze(0)
            x_tensor.requires_grad_(True)
            
            pr_mask = best_model.forward(x_tensor)
    
            mmask = torch.sum(torch.amax(pr_mask[0,1:], axis = 0))
            mmask.backward()
            saliency, _ = torch.max(x_tensor.grad.data.abs(),dim=1)
            
            saliencies.append(saliency[0].cpu().numpy())

 
            predictions.append(pr_mask.cpu().detach().numpy())

        # code to plot the saliency map as a heatmap
        plt.imshow(np.mean(saliencies, 0), cmap=plt.cm.hot, vmin=0, vmax=100)
        plt.axis('off')
        
        y_gt = torch.cat((torch.full((1, w,h), 0.8), y_gt))

        y_gt = torch.argmax(y_gt, dim = 0)
        pr_mask = np.argmax(np.mean(predictions, 0), 1)[0] # average of all predictions
        pr_mask[pr_mask == 4] = 0
        pr_mask[pr_mask == 7] = 0
        pr_mask[pr_mask == 9] = 0


        plt.imshow(np.transpose(image_vis, (1,2,0)), alpha=0.2)
        plt.show()
        
        plt.imshow(np.transpose(image_vis, (1,2,0)))
        plt.axis('off')
        plt.show()

        plt.imshow(y_gt, cmap=cmap, norm=norm, interpolation="nearest")
        plt.axis('off')
        plt.show()

        plt.imshow(pr_mask, cmap=cmap, norm=norm, interpolation="nearest")
        plt.axis('off')
        plt.show()

#%%

# calculate the mean IoU and mean pixel accuracy for different best and last_epoch models

# model_versions = ["Seresnext_FPN_BCE_100_ip", "Efficientnet_FPN_BCE_100_ip", "Resnet_FPN_BCE_100_ip", "Resnest_FPN_BCE_100_ip", \
#                   "Seresnext_UNET_BCE_100_ip", "Efficientnet_Unet_BCE_100_ip", "Resnet_UNET_BCE_100_ip", "Resnest_UNET_BCE_100_ip", \
#                   "Seresnext_FPN_FTLBCE_100_ip", "Seresnext_FPN_FTL_100_ip", "Seresnext_FPN_sqrtBCE_100_ip", \
#                       "Seresnext_FPN_myloss_100_ip","Seresnext_FPN_BCE_100_ip_twow", "Seresnext_FPN_myloss2_100_ip",\
#                           "Seresnext_UNET_sqrtBCE_100_ip", "Seresnext_UNET_myloss_100_ip", "Seresnext_UNET_myloss2_100_ip", \
#                               "Seresnext_UNET_FTL4_100_ip", "Seresnext_UNET_FTLBCE_100_ip", "Seresnext_UNET_BCE_100_ip_twow"]

model_versions = ["Segformer_pretrained_100"]
plt.rcParams["font.size"]="25"

for model_version in model_versions: 
    results = {}
    mean_ious = []
    cms = []
    for fold in range(7):
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = "imagenet"
        ACTIVATION = "softmax2d"
        model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=10, # Background Klasse kommt hinzu
        activation=ACTIVATION,
        )
        model = torch.load(os.path.join(path, "runs", "Project", str(model_version), str(model_version) +\
                                        "_last_epoch_model_fold_" + str(fold) + ".pth"), map_location = torch.device(DEVICE))
        # print(model)

        y_true = []
        y_pred = []
        y_pred_best = []
    
        class_accs_test = np.full((1,11), np.nan)
        class_correct_test = np.full((1,11), np.nan)
        class_total_test = np.full((1,11), np.nan)
    
        for n in range(len(test_dataset)):

            image, y_gt = test_dataset[n]
         
            w, h = y_gt.size()[1], y_gt.size()[2]
            
            x_tensor = image.unsqueeze(0)
            pr_mask = model(x_tensor)
        
            y_gt = torch.cat((torch.full((1,w,h), 0.8), y_gt.cpu().detach()))
            
            class_accs_test = np.concatenate((class_accs_test, 
                            image_acc_metric(pr_mask, torch.unsqueeze(y_gt, 0), 11)[None,:]))
            
         
            cor, tot = class_acc_metric(pr_mask, torch.unsqueeze(y_gt, 0), 11)
            class_correct_test = np.concatenate((class_correct_test, cor[None,:]))
            class_total_test = np.concatenate((class_total_test, tot[None,:]))
            
            y_gt = torch.argmax(y_gt, dim = 0)
            pr_mask = torch.argmax(pr_mask, dim = 1)
        
            y_gt = y_gt.flatten().tolist()
            y_true.extend(y_gt)
            pr_mask = pr_mask.flatten().tolist()
            y_pred.extend(pr_mask)
        
        y_pred = np.array(y_pred)
        y_pred[y_pred == 4] = 0
        y_pred[y_pred == 7] = 0
        y_pred[y_pred == 9] = 0
        
        y_true = np.array(y_true)
        y_true[y_true == 4] = 0
        y_true[y_true == 7] = 0
        y_true[y_true == 9] = 0
        
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        
        cms.append(cm)
        df_cm = pd.DataFrame(cm, index = [i for i in ["background"] + wound_class_reduced],
                          columns = [i for i in ["background"] + wound_class_reduced])
        
        plt.figure(figsize=(20,14))
        sns.heatmap(df_cm, annot=True)
        plt.title('Confusion matrix , Average Test IoU: ' + str(np.round(np.trace(cm[1:,1:])/len(wound_class_reduced)*100,2)) + "%")
        plt.tight_layout()
        plt.show()
        
        cm_ious = confusion_matrix(y_true, y_pred) #not normalized
        ious= []
        for i in range(8):
            ious.append(cm_ious[i,i]/(np.sum(cm_ious[i])+np.sum(cm_ious[:,i])-cm_ious[i,i]))
  
        mean_ious.append(np.mean(ious[1:]))
        
 
    results["last fold iou test"] = np.mean(mean_ious)

    df_cm = pd.DataFrame(sum(cms)/len(cms), index = [i for i in ["background"] + wound_class_reduced],
                      columns = [i for i in ["background"] + wound_class_reduced])
    cm=sum(cms)/len(cms)
    plt.figure(figsize=(20,14))
    sns.heatmap(df_cm, annot=True)
    plt.title('Confusion matrix , Average Test IoU: '+ str(model_version) + str(np.round(np.trace(cm[1:,1:])/len(wound_class_reduced)*100,2)) + "%")
    plt.tight_layout()
    plt.show()
    
    results["last fold acc test"] = np.round(np.trace(cm[1:,1:])/len(wound_class_reduced)*100,2)

    
    mean_ious = []
    cms = []
    for fold in range(7):
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = "imagenet"
        ACTIVATION = "softmax2d"
        model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=10, # Background Klasse kommt hinzu
        activation=ACTIVATION,
        )
        model = torch.load(os.path.join(path, "runs", "Project", str(model_version), str(model_version) +\
                                        "_best_model_fold_" + str(fold) + ".pth"), map_location = torch.device(DEVICE))

        y_true = []
        y_pred = []
        y_pred_best = []
    
        class_accs_test = np.full((1,11), np.nan)
        class_correct_test = np.full((1,11), np.nan)
        class_total_test = np.full((1,11), np.nan)
    
        for n in range(len(test_dataset)):
        
            image, y_gt = test_dataset[n]
         
            w, h = y_gt.size()[1], y_gt.size()[2]
            
            x_tensor = image.unsqueeze(0)
            pr_mask = model(x_tensor)
        
            y_gt = torch.cat((torch.full((1,w,h), 0.8), y_gt.cpu().detach()))
            
            class_accs_test = np.concatenate((class_accs_test, 
                            image_acc_metric(pr_mask, torch.unsqueeze(y_gt, 0), 11)[None,:]))
         
            cor, tot = class_acc_metric(pr_mask, torch.unsqueeze(y_gt, 0), 11)
            class_correct_test = np.concatenate((class_correct_test, cor[None,:]))
            class_total_test = np.concatenate((class_total_test, tot[None,:]))
            
            y_gt = torch.argmax(y_gt, dim = 0)
            pr_mask = torch.argmax(pr_mask, dim = 1)
        
            y_gt = y_gt.flatten().tolist()
            y_true.extend(y_gt)
            pr_mask = pr_mask.flatten().tolist()
            y_pred.extend(pr_mask)
            
        y_pred = np.array(y_pred)
        y_pred[y_pred == 4] = 0
        y_pred[y_pred == 7] = 0
        y_pred[y_pred == 9] = 0
        
        y_true = np.array(y_true)
        y_true[y_true == 4] = 0
        y_true[y_true == 7] = 0
        y_true[y_true == 9] = 0
        
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        
        cms.append(cm)
        df_cm = pd.DataFrame(cm, index = [i for i in ["background"] + wound_class_reduced],
                          columns = [i for i in ["background"] + wound_class_reduced])
        
        plt.figure(figsize=(20,14))
        sns.heatmap(df_cm, annot=True)
        plt.title('Confusion matrix , Average Test IoU: ' + str(np.round(np.trace(cm[1:,1:])/len(wound_class_reduced)*100,2)) + "%")
        plt.tight_layout()
        plt.show()
        
        cm_ious = confusion_matrix(y_true, y_pred) #not normalized
        ious= []
        for i in range(8):
            ious.append(cm_ious[i,i]/(np.sum(cm_ious[i])+np.sum(cm_ious[:,i])-cm_ious[i,i]))
  
        mean_ious.append(np.mean(ious[1:]))
        
    results["best fold iou test"] = np.mean(mean_ious)

    df_cm = pd.DataFrame(sum(cms)/len(cms), index = [i for i in ["background"] + wound_class_reduced],
                      columns = [i for i in ["background"] + wound_class_reduced])
    cm=sum(cms)/len(cms)
     
    plt.figure(figsize=(20,14))
    sns.heatmap(df_cm, annot=True)
    plt.title('Confusion matrix , Average Test IoU: '+ str(model_version) + str(np.round(np.trace(cm[1:,1:])/len(wound_class_reduced)*100,2)) + "%")
    plt.tight_layout()
    plt.show()
    
    results["best fold acc test"] = np.round(np.trace(cm[1:,1:])/len(wound_class_reduced)*100,2)
    
    np.save(os.path.join(path, "runs", "Project", model_version, "average_test_cm"), cm)
    
    mean_ious = []
    cms = []
    for fold in range(7):
        ENCODER = 'se_resnext50_32x4d'
        ENCODER_WEIGHTS = "imagenet"
        ACTIVATION = "softmax2d"
        model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=10, # Background Klasse kommt hinzu
        activation=ACTIVATION,
        )
        model = torch.load(os.path.join(path, "runs", "Project", str(model_version), str(model_version) +\
                                        "_best_model_fold_" + str(fold) + ".pth"), map_location = torch.device(DEVICE))
        
        y_true = []
        y_pred = []
        y_pred_best = []
        
        class_accs_test = np.full((1,11), np.nan)
        class_correct_test = np.full((1,11), np.nan)
        class_total_test = np.full((1,11), np.nan)
    
        for n in range(len(optimal_dataset)):
        
            image, y_gt = optimal_dataset[n]
         
            w, h = y_gt.size()[1], y_gt.size()[2]
            
            x_tensor = image.unsqueeze(0)
            pr_mask = model(x_tensor)
        
            y_gt = torch.cat((torch.full((1,w,h), 0.8), y_gt.cpu().detach()))
            
            class_accs_test = np.concatenate((class_accs_test, 
                            image_acc_metric(pr_mask, torch.unsqueeze(y_gt, 0), 11)[None,:]))
            
         
            cor, tot = class_acc_metric(pr_mask, torch.unsqueeze(y_gt, 0), 11)
            class_correct_test = np.concatenate((class_correct_test, cor[None,:]))
            class_total_test = np.concatenate((class_total_test, tot[None,:]))
            
            y_gt = torch.argmax(y_gt, dim = 0)
            pr_mask = torch.argmax(pr_mask, dim = 1)
        
            y_gt = y_gt.flatten().tolist()
            y_true.extend(y_gt)
            pr_mask = pr_mask.flatten().tolist()
            y_pred.extend(pr_mask)
            
        y_pred = np.array(y_pred)
        y_pred[y_pred == 4] = 0
        y_pred[y_pred == 7] = 0
        y_pred[y_pred == 9] = 0
        
        y_true = np.array(y_true)
            
        y_true[y_true == 4] = 0
        y_true[y_true == 7] = 0
        y_true[y_true == 9] = 0
        
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        
        cms.append(cm)
        df_cm = pd.DataFrame(cm, index = [i for i in ["background"] + wound_class_reduced],
                          columns = [i for i in ["background"] + wound_class_reduced])
        
        plt.figure(figsize=(20,14))
        sns.heatmap(df_cm, annot=True)
        plt.title('Confusion matrix , Average Test IoU: ' + str(np.round(np.trace(cm[1:,1:])/len(wound_class_reduced)*100,2)) + "%")
        plt.tight_layout()
        plt.show()
        
        cm_ious = confusion_matrix(y_true, y_pred) #not normalized
        ious= []
        for i in range(8):
            ious.append(cm_ious[i,i]/(np.sum(cm_ious[i])+np.sum(cm_ious[:,i])-cm_ious[i,i]))

        mean_ious.append(np.mean(ious[1:]))
        
    results["best fold iou optimal"] = np.mean(mean_ious)

    df_cm = pd.DataFrame(sum(cms)/len(cms), index = [i for i in ["background"] + wound_class_reduced],
                      columns = [i for i in ["background"] + wound_class_reduced])
    cm=sum(cms)/len(cms)
    plt.figure(figsize=(20,14))
    sns.heatmap(df_cm, annot=True)
    plt.title('Confusion matrix , Average Test IoU: '+ str(model_version) + str(np.round(np.trace(cm[1:,1:])/len(wound_class_reduced)*100,2)) + "%")
    plt.tight_layout()
    plt.show()
    
    results["best fold acc optimal"] = np.round(np.trace(cm[1:,1:])/len(wound_class_reduced)*100,2)
    print(results)
    np.save(os.path.join(path, "runs", "Project", model_version, "results"), results)
    
#%%
results = np.load(os.path.join(path, "runs", "Project", model_version, "results.npy"), allow_pickle=True)
print(results)
