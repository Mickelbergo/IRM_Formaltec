import os
import torch
import cv2
import kornia as K
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from scipy import stats
from augmentations import resize_and_pad  # Import the resize_and_pad function
import json

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
        image_path = os.path.join(self.dir_path, "images_640_1280", self.image_ids[ind])
        mask_path = os.path.join(self.dir_path, "masks_640_1280", self.mask_ids[ind])
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        #  # Check if image or mask is None
        # if image is None:
        #     raise ValueError(f"Image at path {image_path} could not be loaded.")
        # if mask is None:
        #     raise ValueError(f"Mask at path {mask_path} could not be loaded.")
        
        if image is None or mask is None:
            return None  # or you could return an empty tensor, or raise an exception
    
        # Apply resizing and padding
        image, mask = resize_and_pad(image, mask, self.target_size)
        
        # Convert mask to binary segmentation mask
        binary_mask = (mask > 0).astype(np.uint8)  # All non-zero (wound) pixels become 1
        
        # Convert mask values to class labels
        mask_class = (mask // 15)  # Assuming mask values are wound_class * 15
        
        # Filter out background (class 0) and get non-background classes
        non_background_pixels = mask_class[mask_class != 0]
        
        if len(non_background_pixels) > 0:
            # Determine the most frequent non-background class in the mask
            dominant_class = int(stats.mode(non_background_pixels.flatten())[0])
        else:
            # Handle edge case where the mask is entirely background
            dominant_class = 0  # or consider ignoring this image in training
        
        # Convert to tensor
        image = K.image_to_tensor(image).float().to(DEVICE)
        binary_mask = K.image_to_tensor(binary_mask).long().to(DEVICE)  # Binary mask for segmentation
        mask_class = K.image_to_tensor(mask_class).long().to(DEVICE)  # Class-specific mask
        
        # Apply preprocessing if necessary
        if self.preprocessing:
            image = self._preprocess(image)
        

        return image, binary_mask, mask_class, dominant_class
    
    def _preprocess(self, image):
        mean = torch.tensor(preprocessing_config["mean"]).view(3, 1, 1).to(DEVICE)
        std = torch.tensor(preprocessing_config["std"]).view(3, 1, 1).to(DEVICE)
        image = (image - mean) / std
        return image

    def __len__(self):
        return len(self.image_ids)
