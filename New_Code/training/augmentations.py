import cv2
from torchvision.transforms import v2
import torch
import torchvision.transforms.functional as F
from torch.nn.functional import interpolate
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import json

with open('New_Code/configs/training_config.json') as f:
    train_config = json.load(f)


class Augmentation:
    def __init__(self, target_size):
        self.target_size = target_size

        # Normalization only for images
        self.normalize = v2.Compose([
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def augment(self, image, mask):

        # Ensure the image and mask are Torch tensors
        if not isinstance(image, torch.Tensor):
            raise TypeError("Image should be a tensor")
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Mask should be a tensor")


        image = image / 255.0 #convert to float image

        if train_config["object_detection"] == "False":

            # Get the parameters for RandomResizedCrop (apply same params to both image and mask)
            # params = v2.RandomResizedCrop.get_params(image, scale=(1.0, 1.0), ratio=(1.0, 1.0))
            
            # # Apply RandomResizedCrop to both image and mask using the same parameters
            # image = F.resized_crop(image, *params, size= self.target_size, interpolation=v2.InterpolationMode.BILINEAR)
            # mask = F.resized_crop(mask, *params, size= self.target_size, interpolation=v2.InterpolationMode.NEAREST)

            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0).float()
            image = interpolate(image, size= self.target_size, mode = "bilinear", align_corners=False)
            mask = interpolate(mask, size= self.target_size, mode = "nearest")
            image = image.squeeze(0)
            mask = mask.squeeze(0).long()
        # Apply the same flipping transformations to both image and mask
        if torch.rand(1) < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if torch.rand(1) < 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)

        # Normalize the image (but not the mask)
        image = self.normalize(image)

        # mask = mask.permute(1,2,0)
        # plt.imshow(mask, cmap = 'gray')
        # plt.show()

        # image = image.permute(1,2,0)
        # plt.imshow(image, cmap = 'gray')
        # plt.show()

        return image, mask

class ValidationAugmentation:
    def __init__(self, target_size):
        # Define fixed resizing for validation (no randomness)
        self.transforms = v2.Compose([
            v2.ToDtype(torch.float32),
            v2.Resize(size= target_size),  # Resize to the same size as in training
        ])

        # Normalization only for images (same as training)
        self.normalize = v2.Compose([
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def augment(self, image, mask):
        # Ensure the image and mask are Torch tensors
        if not isinstance(image, torch.Tensor):
            raise TypeError("Image should be a tensor")
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Mask should be a tensor")
        
        image = image/255.0 #convert to float image

        
        # Resize both image and mask (without randomness)
        image = self.transforms(image)
        mask = self.transforms(mask)

        # Normalize the image (but not the mask)
        image = self.normalize(image)

        return image, mask.long()






