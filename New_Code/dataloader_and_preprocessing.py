'''
Loading and preparing the data for training. The preprocessing is included in the Dataset class. 
'''

from config import config

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch.nn.functional as F
import cv2
import os
import kornia as K
import numpy as np
import matplotlib.pyplot as plt

DEVICE = config.DEVICE
wound_classes = config.wound_classes

# Dataloader for high resolution images
class Dataset_imlarge(BaseDataset):

    def __init__(
            self,
            names_dir,
            dir_path,
            augmentation=None,
            preprocessing=None,
        ):

        self.image_ids = [os.path.join(dir_path, "Images_512_1024", image_id) + ".png" for image_id in names_dir]
        self.large_image_ids = [os.path.join(dir_path, "Images_2048_4096", image_id) + ".png" for image_id in names_dir]
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
        
        image_large: np.ndarrray = cv2.imread(self.large_image_ids[ind], cv2.IMREAD_COLOR)
        image_large: torch.Tensor = K.image_to_tensor(image_large)
        image_large = image_large.type(torch.float16).to(DEVICE)

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
            image, image_large, mask, weights = self.augmentation(image, image_large, mask, weights)

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
            
            image_large = image_large / 255.0

            image_large[0] -= self.mean[0]
            image_large[1] -= self.mean[1]
            image_large[2] -= self.mean[2]
            
            image_large[0] /= self.std[0]
            image_large[1] /= self.std[1]
            image_large[2] /= self.std[2]

            target_bce = target_bce.permute((2, 0, 1))

            
        if not self.preprocessing and not self.augmentation:
            return image, target_bce[1:] # Used for testing, Hautrötung removed
        
        return image.type(torch.float32), image_large.type(torch.float32), target_bce[1:], weights[1:] # Hautrötung removed

    def __len__(self):
        return len(self.image_ids)
    
# Regular dataloader
class Dataset(BaseDataset):

    def __init__(
            self,
            names_dir,
            dir_path,
            augmentation=None,
            preprocessing=None,
        ):

        # self.image_ids = [os.path.join(dir_path, "Images_512_1024", image_id) + ".png" for image_id in names_dir]
        # self.mask_ids = [os.path.join(dir_path, "Masks_512_1024", mask_id) + ".png" for mask_id in names_dir]
        # self.certainty_ids = [os.path.join(dir_path, "Masks_certainty_512_1024", mask_id) + ".png" for mask_id in names_dir]
        self.image_ids = [os.path.join(dir_path, "Images_640_1280", image_id) + ".png" for image_id in names_dir]
        self.mask_ids = [os.path.join(dir_path, "Masks_640_1280", mask_id) + ".png" for mask_id in names_dir]
        # self.certainty_ids = [os.path.join(dir_path, "Masks_certainty_640_1280", mask_id) + ".png" for mask_id in names_dir]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.path = dir_path
        self.mean = torch.Tensor([0.55540512, 0.46654144, 0.42994756]) # mean calculated for our wound dataset
        self.std = torch.Tensor([0.21014232, 0.21117639, 0.22304179]) # std calculated for our wound dataset

    def __getitem__(self, ind):
        
        # read data
        image: np.ndarray = cv2.imread(self.image_ids[ind], cv2.IMREAD_COLOR)
        image: torch.Tensor = K.image_to_tensor(image)
        image = image.type(torch.float32).to(DEVICE)

        mask: np.ndarray = cv2.imread(self.mask_ids[ind], cv2.IMREAD_GRAYSCALE)/15 # masks saved with values corresponding to the wound_class * 15
        mask: torch.tensor = K.image_to_tensor(mask)
        mask = mask.type(torch.float32).to(DEVICE)
        
        # certainty_masks: np.ndarray = cv2.imread(self.certainty_ids[ind], cv2.IMREAD_GRAYSCALE)
        # certainty_masks: torch.tensor =  K.image_to_tensor(certainty_masks)
        # certainty_masks = certainty_masks.type(torch.float32).to(DEVICE)/100 # saved with weights * 100
        
        # plt.imshow(torch.permute(image.cpu().to(torch.uint8), (1,2,0)))
        # plt.show()
        
        if self.augmentation:
            image, mask = self.augmentation(image, mask) #, certainty_masks)
            
        # plt.imshow(torch.permute(image.cpu().to(torch.uint8), (1,2,0)))
        # plt.show()
        # very certain wounds are weighted 1.5 times as much, not certain wounds 0.5
        # if weights from masks certainty to be changed: cmask = torch.where(certainty_masks == 1.5, 2, certainty_masks), cmask = torch.where(certainty_masks == 0.5, 0.3, certainty_masks)
        # cmask = torch.where(certainty_masks == 0, 1, certainty_masks)
        # weights = cmask.expand(9,-1,-1)

        # target_one_hot = F.one_hot(mask[0].long(), num_classes = len(wound_classes) + 5)
        mask[mask!=0] = 1
        background_bce = 1 - mask
        target_bce = torch.cat([background_bce, mask], dim=0)

        
        # # Wundkombinationen
        # uh = [wound_classes.index("Ungeformter_Bluterguss"), wound_classes.index("Hautabschuerfung")] # mask class 12
        # gh = [wound_classes.index("Geformter_Bluterguss"), wound_classes.index("Hautabschuerfung")] # mask class 13
        # th = [wound_classes.index("Thermische_Gewalt"), wound_classes.index("Hautabschuerfung")] # mask class 14

        # for index in uh:
        #     target_bce[:,:,index] += target_one_hot[:,:,12]
        # for index in gh:
        #     target_bce[:,:,index] += target_one_hot[:,:,13]
        # for index in th:
        #     target_bce[:,:,index] += target_one_hot[:,:,14]
            
        if self.preprocessing:

            image = image / 255.0
            
            image[0] -= self.mean[0]
            image[1] -= self.mean[1]
            image[2] -= self.mean[2]
            
            image[0] /= self.std[0]
            image[1] /= self.std[1]
            image[2] /= self.std[2]
            
            # target_bce = target_bce.permute((2, 0, 1))
        
        return image.type(torch.float32), target_bce.type(torch.float32) #, weights.type(torch.float32)

    def __len__(self):
        return len(self.image_ids)

