import os
import json
import torch
import cv2
import kornia as K
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from scipy import stats

# Load preprocessing config
with open('preprocessing_config.json') as f:
    preprocessing_config = json.load(f)

DEVICE = torch.device(preprocessing_config["device"])

class Dataset(BaseDataset):
    def __init__(self, dir_path, split="train", augmentation=None, preprocessing=True):
        self.image_ids = sorted(os.listdir(os.path.join(dir_path, split, "images")))
        self.mask_ids = sorted(os.listdir(os.path.join(dir_path, split, "masks")))
        self.dir_path = dir_path
        self.split = split
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, ind):
        # Load image and mask
        image = cv2.imread(os.path.join(self.dir_path, self.split, "images", self.image_ids[ind]), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.dir_path, self.split, "masks", self.mask_ids[ind]), cv2.IMREAD_GRAYSCALE)
        
        # Convert mask values to classes
        mask_class = mask // 15  # Assuming mask values are wound_class * 15
        
        # Determine the dominant class in the mask (for classification)
        dominant_class = int(stats.mode(mask_class.flatten())[0])
        
        # Convert to tensor
        image = K.image_to_tensor(image).float().to(DEVICE)
        mask_class = K.image_to_tensor(mask_class).long().to(DEVICE)
        
        # Apply preprocessing if necessary
        if self.preprocessing:
            image = self._preprocess(image)
        
        return image, mask_class, dominant_class
    
    def _preprocess(self, image):
        mean = torch.tensor(preprocessing_config["mean"]).view(3, 1, 1).to(DEVICE)
        std = torch.tensor(preprocessing_config["std"]).view(3, 1, 1).to(DEVICE)
        image = (image - mean) / std
        return image

    def __len__(self):
        return len(self.image_ids)
