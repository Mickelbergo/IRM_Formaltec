'''
Loading and preparing the data for training. The preprocessing is included in the Dataset class. 
'''


import sys
import os

# Get the root directory (one level up from the current file's directory)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)


from configs import config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch.nn.functional as F
import cv2

import kornia as K
import numpy as np
import matplotlib.pyplot as plt




DEVICE = config.DEVICE
wound_classes = config.wound_classes


# Regular dataloader
class Dataset(BaseDataset):

    def __init__(
            self,
            names_dir,
            dir_path,
            augmentation=None,
            preprocessing=None,
        ):

        self.image_ids = [os.path.join(dir_path, "Images_640_1280", image_id) + ".png" for image_id in names_dir]
        self.mask_ids = [os.path.join(dir_path, "Masks_640_1280", mask_id) + ".png" for mask_id in names_dir]


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
        
        
        if self.augmentation:
            image, mask = self.augmentation(image, mask) #, certainty_masks)
            

        mask[mask!=0] = 1
        background_bce = 1 - mask
        target_bce = torch.cat([background_bce, mask], dim=0)

            
        if self.preprocessing:

            image = image / 255.0
            
            image[0] -= self.mean[0]
            image[1] -= self.mean[1]
            image[2] -= self.mean[2]
            
            image[0] /= self.std[0]
            image[1] /= self.std[1]
            image[2] /= self.std[2]
            

        return image.type(torch.float32), target_bce.type(torch.float32) #, weights.type(torch.float32)

    def __len__(self):
        return len(self.image_ids)

