import os
import torch
import cv2
import kornia as K
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from scipy import stats
from augmentations import resize_and_pad  # Import the resize_and_pad function

# Load preprocessing config
with open('New_Code/configs/preprocessing_config.json') as f:
    preprocessing_config = json.load(f)

DEVICE = torch.device(preprocessing_config["device"])

class Dataset(BaseDataset):
    def __init__(self, dir_path, image_ids, mask_ids, augmentation=None, preprocessing=True, target_size=(640, 640)):
        self.image_ids = image_ids
        self.mask_ids = mask_ids
        self.dir_path = dir_path
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.target_size = target_size

    def __getitem__(self, ind):
        # Load image and mask
        image_path = os.path.join(self.dir_path, "images", self.image_ids[ind])
        mask_path = os.path.join(self.dir_path, "masks", self.mask_ids[ind])
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply resizing and padding
        image, mask = resize_and_pad(image, mask, self.target_size)
        
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
