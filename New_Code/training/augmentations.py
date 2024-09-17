import cv2
from torchvision.transforms import v2
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

import cv2
import torch



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

        # Get the parameters for RandomResizedCrop (apply same params to both image and mask)
        params = v2.RandomResizedCrop.get_params(image, scale=(1.0, 1.0), ratio=(1.0, 1.0))
        
        # Apply RandomResizedCrop to both image and mask using the same parameters
        image = F.resized_crop(image, *params, size= self.target_size, interpolation=v2.InterpolationMode.BILINEAR)
        mask = F.resized_crop(mask, *params, size= self.target_size, interpolation=v2.InterpolationMode.NEAREST)

        # Apply the same flipping transformations to both image and mask
        if torch.rand(1) < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if torch.rand(1) < 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)

        # Normalize the image (but not the mask)
        image = self.normalize(image)

        return image, mask

class ValidationAugmentation:
    def __init__(self, target_size):
        # Define fixed resizing for validation (no randomness)
        self.transforms = v2.Compose([
            v2.Resize(size= target_size),  # Resize to the same size as in training
        ])

        # Normalization only for images (same as training)
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

        # Resize both image and mask (without randomness)
        image = self.transforms(image)
        mask = self.transforms(mask)

        # Normalize the image (but not the mask)
        image = self.normalize(image)

        return image, mask

################################## DEPRECATED ##################################################


def resize_and_pad(image, mask, target_size=(640, 640)):
    """
    Resize the image and mask to maintain aspect ratio and then pad to the target size.
    
    Args:
        image (np.ndarray): The original image.
        mask (np.ndarray): The corresponding mask.
        target_size (tuple): The target size (height, width) to resize and pad the image and mask to.
        
    Returns:
        padded_image (np.ndarray): The resized and padded image.
        padded_mask (np.ndarray): The resized and padded mask.
    """
    target_height, target_width = target_size
    
    # Get current dimensions
    height, width = image.shape[:2]
    
    # Resize the image and mask while keeping the aspect ratio
    if width > height:
        scale = target_width / width
        new_width = target_width
        new_height = int(height * scale)
    else:
        scale = target_height / height
        new_height = target_height
        new_width = int(width * scale)
    
    resized_image = cv2.resize(image, (new_width, new_height))
    resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # Pad the resized image and mask to the target size
    delta_w = target_width - new_width
    delta_h = target_height - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    padded_mask = cv2.copyMakeBorder(resized_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    
    return padded_image, padded_mask






