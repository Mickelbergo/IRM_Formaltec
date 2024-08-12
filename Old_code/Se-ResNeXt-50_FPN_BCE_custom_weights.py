"""
Example of all necessary code in 1 file for training Se-ResNeXt-50_FPN for the classification and segmentation of forensic wounds. It uses a weighted
BCE loss function with additional weights according to the subjective certainty of classification. For more information look at the paper
"Segmentation and classification of seven common wounds in forensic medicine". This code was used to train the model "SE-RESNEXT-50-FPN BCE-CERTAINTY",
wich achieved the highest mean pixel accuracy on the test dataset.
"""

# import necessary libraries

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

# Train on GPU if available / otherwise train on CPU

print("Current GPU: ", torch.cuda.current_device())
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    DEVICE = 0
else:
    DEVICE = "cpu"

# set variables
path = "/home/zino/Data"
model_version = "Seresnext_FPN_github"
ENCODER = "se_resnext50_32x4d"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "softmax2d"
with_stich_schnitt = False # If True images showing "Stich/Schnittkombinationen" are not removed
optimal_dataset = True # If True ~50 images with well-defined wound boundaries are removed from the train/val datasets as additional test set

# Load all annotation data (VGG Image Annotator format)
with open(os.path.join(path, "Annotation/Annotation.json")) as json_file:
    data = json.load(json_file)

# Get names of images
names = list(data["_via_img_metadata"].keys())
    
# Load wound_classes without "Stich/Schnitt-Kombination" 
with open(os.path.join(path, "text_files/wound_classes_reduced.txt"), "r") as filehandle:
    wound_classes = json.load(filehandle)  
wound_class_reduced = wound_classes.copy()[1:] # without Hautrötung
    
# Load image names from the test dataset
with open(os.path.join(path, "text_files/test.txt"), "r") as filehandle:
    test = json.load(filehandle)
    
# remove test from train and val images
names = [x for x in names if x not in test]

if optimal_dataset:
    # Load image names from the optimal small test dataset (with well-defined wound boundaries)
    with open(os.path.join(path, "text_files/test_optimal_images.txt"), "r") as filehandle:
        test_optimal = json.load(filehandle)

    # remove optimal test images from train and val images
    names = [x for x in names if x not in test_optimal]

# Filter out all Stich-/Schnittkombinationen
ssk = []
if not with_stich_schnitt:
    for i, name in enumerate(names):
        for w in data["_via_img_metadata"][name]["regions"]:
            if  "Stich/Schnittkombination" == w["region_attributes"]["Wound"]:
                ssk.append(name)
print("Number of images with Stich-/Schnittkombinationen removed: ", len(ssk))
names = [x for x in names if x not in ssk]

# split data for training and validation and seven folds
kf = KFold(n_splits = 7, shuffle = True, random_state = 42)
for ti, vi in kf.split(names):
    print(len(ti), len(vi))

# wound areas used in the loss function (pixels_per_class/total_pixels)
areas = np.array([2.02112547e+09, 1.43418300e+06, 2.15444820e+07, 1.52634900e+06, 8.17287000e+05, 3.01713000e+05, \
                  3.66251100e+06, 1.09538880e+07, 2.83833000e+05, 6.64005000e+05, 2.32836000e+05, 4.49165100e+06])/2067038208
wound_areas = dict(zip(wound_classes[1:], np.round(0.01/areas[2:], 1)))
print("Number of images in train and validation: ", len(names))
print("Number of images in test dataset: ", len(test))
print("Number of images in optimal test dataset: ", len(test_optimal))
print("Wound classes: " , wound_classes)
print("Wound classes : ", "Weights")
[print(key, ":", value) for key, value in wound_areas.items()]


# Load and prepare the data

class Dataset(BaseDataset):

    def __init__(
            self,
            names_dir,
            dir_path,
            augmentation=None,
            preprocessing=None,
        ):

        self.image_ids = [os.path.join(dir_path, "Images_512_1024", image_id) + ".png" for image_id in names_dir]
        self.mask_ids = [os.path.join(dir_path, "Masks_512_1024", mask_id) + ".png" for mask_id in names_dir]
        self.certainty_ids = [os.path.join(dir_path, "Masks_512_1024_certainty", mask_id) + ".png" for mask_id in names_dir]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.path = dir_path
        self.mean = torch.Tensor([0.55540512, 0.46654144, 0.42994756]) # mean calculated for our wound dataset
        self.std = torch.Tensor([0.21014232, 0.21117639, 0.22304179]) # std calculated for our wound dataset

    def __getitem__(self, ind):
        
        # read data
        image: np.ndarray = cv2.imread(self.image_ids[ind], cv2.IMREAD_COLOR)
        image: torch.Tensor = K.image_to_tensor(image)
        image = image.type(torch.float16).to(DEVICE)

        mask: np.ndarray = cv2.imread(self.mask_ids[ind], cv2.IMREAD_GRAYSCALE)/20 # masks saved with values corresponding to the wound_class * 20
        mask: torch.tensor = K.image_to_tensor(mask)
        mask = mask.type(torch.float16).to(DEVICE)
        
        certainty_masks: np.ndarray = cv2.imread(self.certainty_ids[ind], cv2.IMREAD_COLOR) # (x, y, 3)
        certainty_masks: torch.tensor =  K.image_to_tensor(certainty_masks)
        certainty_masks = certainty_masks.type(torch.long).to(DEVICE)
        
        weights = torch.ones([11, mask.shape[1], mask.shape[2]]).to(DEVICE)

        # very certain wounds are weighted 1.5 times as much
        vcmask = certainty_masks[2:3]
        vcmask = torch.where(vcmask > 0, 1, vcmask)
        vcmask=vcmask.expand(11,-1,-1)
        weights += vcmask * 0.5
        
        # not certain wounds only 0.5 times as much
        ncmask = certainty_masks[0:1]
        ncmask = torch.where(ncmask > 0, 1, ncmask)
        ncmask=ncmask.expand(11,-1,-1)
        weights -= ncmask * 0.5

        if self.augmentation:
            image, mask, weights = self.augmentation(image, mask, weights)

        target_bce = F.one_hot(mask[0].long(), num_classes = len(wound_classes) + 2 )[:,:,1:] # Hautrötung removed
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
            return image, target_bce[1:] # Used for testing, Hautrötung removed
        
        return image.type(torch.float32), target_bce[1:], weights[1:] # Hautrötung removed

    def __len__(self):
        return len(self.image_ids)

    
class training_augmentation(nn.Module):
    
  def __init__(self):
    super(training_augmentation, self).__init__()

    self.k1 = K.augmentation.RandomHorizontalFlip(p=0.5)
    self.k2 = K.augmentation.RandomVerticalFlip(p=0.2)
    self.k3 = K.augmentation.RandomAffine(degrees = 10, translate=0.1, scale=(0.7, 1.8), p = 1)
    self.k4 = K.augmentation.RandomCrop(size=(512,512), p = 1)
  
  def forward(self, img: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:

    img_out = self.k4(self.k3(self.k2(self.k1(img))))

    mask_out = self.k4(self.k3(self.k2(self.k1(mask, self.k1._params),  self.k2._params),  self.k3._params),  self.k4._params)
    
    weights_out = self.k4(self.k3(self.k2(self.k1(weights, self.k1._params),  self.k2._params),  self.k3._params),  self.k4._params)

    return img_out[0], mask_out[0], weights_out[0]


# Loss function: weighted BCE including pixelwise weights according to the subjective certainty of classification during labeling
class weightedBCE(nn.Module):
    def __init__(self):
        super(weightedBCE, self).__init__()
        self._name = "myloss"
        weights = torch.tensor(0.01/areas[2:]) # without Hautrötung and background
        self.weights = weights.to(DEVICE)


    @property
    def __name__(self):
        return self._name

    def forward(self, inputs, targets, pixel_weights):

        reduce_axis = list(range(2, len(inputs.shape)))
        
        eps = 1e-3  # to avoid log(<=0)
        BCE = - self.weights * torch.mean(pixel_weights * (targets * torch.log(inputs + eps)), axis=reduce_axis) \
            - torch.mean(pixel_weights * ((1 - targets) * torch.log(1 - inputs + eps)), axis=reduce_axis)
        BCE = BCE.mean()

        return BCE


loss = weightedBCE()

# Because of large class imbalances these are not very reliable, but give a rough estimate of the quality of the predictions
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.losses.DiceLoss(),
]


# used to calculate the pixel accuracy per class and batch
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


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device=None, verbose=True, 
                 image_acc = image_acc_metric, class_acc = class_acc_metric, num_classes = 11):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.acc_per_image = image_acc
        self.acc_per_class = class_acc
        self.num_classes = num_classes # 10 wound classes + 1 background class
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def batch_update(self, x, y, w):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = smp.utils.meter.AverageValueMeter()
        metrics_meters = {metric.__name__: smp.utils.meter.AverageValueMeter() for metric in self.metrics}
        class_accs = np.full((1,11), np.nan) # nan because nan values are ignored in the calculation of the mean accuracy 
        class_correct = np.full((1,11), np.nan)
        class_total = np.full((1,11), np.nan)
        
        with tqdm(dataloader, ncols=180, desc=self.stage_name, file=sys.stdout, 
                  position = 0, leave= True, disable=not (self.verbose)) as iterator:
            for count, (x, gbce, weights) in enumerate(iterator):
                    
                x, gbce, weights = x.to(self.device), gbce.to(self.device), weights.to(self.device)
                
                loss, y_pred = self.batch_update(x, gbce, weights)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean.round(3)}
                logs.update(loss_logs)

                # update metrics logs

                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred[:,1:,:,:], gbce).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)

                metrics_logs = {k: round(v.mean, 3) for k, v in metrics_meters.items()}
                logs.update(metrics_logs)
            
                b, w, h = gbce.size()[0], gbce.size()[2], gbce.size()[3]

                gbce = torch.cat((torch.full((b,1,w,h), 0.8).to(self.device), gbce), dim=1) 
                # 0.8 makes sure that the background is only argmax if there is no wound with gbce=1
                # value 0.8 is a random value >0 and <1
                
                # For each epoch the pixel accuracies of the wounds per batch are collected
                # -> small wound areas gain more influence on the final pixel accuracy
                class_accs = np.concatenate((class_accs, 
                        self.acc_per_image(y_pred, gbce, self.num_classes)[None,:]), axis = 0)
                
                # For each epoch the number of all correct pixel classifications per class are collected
                # Additionally, for each epoch the number of all pixels per class are collected
                cor, tot = self.acc_per_class(y_pred, gbce, self.num_classes)
                class_correct = np.concatenate((class_correct, cor[None,:])) 
                class_total = np.concatenate((class_total, tot[None,:])) 

                logs.update({"acc" : list(np.round(np.nansum(class_correct[1:], axis=0)/np.nansum(class_total[1:], axis=0), 3))})
                logs.update({"mean_acc_per_image" : list(np.round(np.nanmean(class_accs[1:], axis = 0), 3))})
                
                if self.verbose:
                    iterator.set_postfix_str(logs)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device=None, verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, w):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)

        loss = self.loss(prediction[:,1:,:,:], y, w)

        loss.backward()

        self.optimizer.step()
            
        return loss, prediction

class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, w):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction[:,1:,:,:], y, w)

        return loss, prediction
    

# test dataset without transformations
test_dataset = Dataset(
    test,
    dir_path = path,
    augmentation=False,
    preprocessing=True,
)

# run the training for seven folds
for fold, (train_index, val_index) in enumerate(kf.split(names)):
    
    # the model is newly initialized every fold to restart training from scratch 
    model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(wound_classes),
    activation=ACTIVATION,
    )
    
    optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
    ])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, verbose=True)
    
    train_epoch = TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )
    
    valid_epoch = ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )
    
    train = [names[i] for i in train_index]
    valid = [names[i] for i in val_index]

    train_dataset = Dataset(
        train[:],
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
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True) #, num_workers=4) 
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False) #, num_workers=4)
    

    max_score = 0
    if not os.path.exists(os.path.join(path, "runs", "paper", model_version)):
        os.makedirs(os.path.join(path, "runs", "paper", model_version))
    writer = [[] for _ in range(12)]
    
    for i in range(0, 100):
    
        print('\nEpoch: {}'.format(i))
        print('\nFold: {}'.format(fold))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
    
        # the mean class accuracy of the seven most common wound classes is used to determine the best model, which is saved
        new_max_score = np.nanmean([valid_logs["acc"][i] for i in [1,2,3,5,6,8,10]])
        
        print(new_max_score)
        if max_score < new_max_score:
            max_score = new_max_score
            torch.save(model, os.path.join(path, "runs", "paper", model_version, model_version) \
                       + "_best_model_fold_" + str(fold) + ".pth")
            print('Model saved!')
    
    
        writer[0].append(valid_logs["dice_loss"])
        writer[1].append(train_logs["dice_loss"])
        writer[2].append(valid_logs['iou_score'])
        writer[3].append(train_logs["iou_score"])
        writer[4].append(valid_logs["myloss"])
        writer[5].append(train_logs["myloss"])
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
            
            class_accs_test = np.full((1,11), np.nan)
            class_correct_test = np.full((1,11), np.nan)
            class_total_test = np.full((1,11), np.nan)
            
            for n in range(len(test_dataset)):
            
                image, y_gt, w = test_dataset[n]
         
                w, h = y_gt.size()[1], y_gt.size()[2]
                
                x_tensor = image.unsqueeze(0)
                pr_mask = model.predict(x_tensor)
    
                y_gt = torch.cat((torch.full((1,w,h), 0.8), y_gt.cpu().detach()))
                
                class_accs_test = np.concatenate((class_accs_test, 
                                image_acc_metric(pr_mask, torch.unsqueeze(y_gt, 0), 11)[None,:]))
                
     
                cor, tot = class_acc_metric(pr_mask, torch.unsqueeze(y_gt, 0), 11)
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
            # for all 10 wound classes.
            print(np.nanmean(class_accs_test, axis = 0))
            
            cm = confusion_matrix(y_true, y_pred, normalize="true")
        
            df_cm = pd.DataFrame(cm, index = [i for i in ["background"] + wound_class_reduced],
                              columns = [i for i in ["background"] + wound_class_reduced])
            
            plt.figure(figsize = (10,7))
            sns.heatmap(df_cm, annot=True)
            plt.title('Confusion matrix , Average Test IoU: ' + str(np.round(np.trace(cm[1:,1:])/len(wound_class_reduced)*100,2)) + "%")
            plt.tight_layout()
            plt.show()
            
            writer[10].append(list(np.round(np.nanmean(class_accs_test, axis = 0), 3)))
            writer[11].append(list(np.round(np.diagonal(cm), 3)))
            

            
    keys = ["dice_valid", "dice_train", "iou_valid", "iou_train", "loss_valid", "loss_train", 
    "acc_valid", "acc_train", "mean_acc_valid_per_image", "mean_acc_train_per_image", 
     "mean_acc_test_per_image", "cm_diagonal"] # cm_diagonal = acc_test
    logs = dict(zip(keys, writer))

    with open(os.path.join(path, "runs", "paper", model_version, "fold_" + str(fold) + "_logs.json"), 'w') as fp:
        json.dump(logs, fp)
        
    torch.save(model, os.path.join(path, "runs", "paper", model_version, model_version + "_last_epoch_model_fold_" + str(fold) + ".pth"))
