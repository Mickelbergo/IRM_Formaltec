import torch
import numpy as np
import torch
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Augmentation:
    def __init__(self, target_size, preprocessing_fn=None):
        self.target_size = target_size
        self.preprocessing_fn = preprocessing_fn


        # Define the augmentation pipeline
        self.transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.OneOf([A.ElasticTransform(p = 0.5),
                    A.OpticalDistortion(p=0.5),
                    A.GridDistortion(p=0.5)]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def augment(self, image, mask):
        # Convert tensors to numpy arrays
        image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)

        # Apply augmentations
        augmented = self.transform(image=image_np, mask=mask_np)
        image_aug = augmented['image']
        mask_aug = augmented['mask']

        # If a preprocessing function is provided, apply it
        if self.preprocessing_fn is not None:
            image_aug = image_aug.permute(1, 2, 0).numpy()
            image_aug = self.preprocessing_fn(image_aug)
            image_aug = torch.from_numpy(image_aug).permute(2, 0, 1).float()
        else:
            #transformer does not use a preprocessing_fn
            image_aug = image_aug.float()
            image_aug = A.Normalize()(image=image_aug.permute(1, 2, 0).numpy())['image']
            image_aug = torch.from_numpy(image_aug).permute(2, 0, 1).float()

        mask_aug = mask_aug.long()

        return image_aug, mask_aug


class ValidationAugmentation:
    def __init__(self, target_size, preprocessing_fn=None):
        self.target_size = target_size
        self.preprocessing_fn = preprocessing_fn

        # Define the validation transformation pipeline
        self.transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})

    def augment(self, image, mask):
        # Convert tensors to numpy arrays
        image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)

        # Apply augmentations
        augmented = self.transform(image=image_np, mask=mask_np)
        image_aug = augmented['image']
        mask_aug = augmented['mask']

        # If a preprocessing function is provided, apply it
        if self.preprocessing_fn is not None:
            image_aug = image_aug.permute(1, 2, 0).numpy()
            image_aug = self.preprocessing_fn(image_aug)
            image_aug = torch.from_numpy(image_aug).permute(2, 0, 1).float()
        else:
            #transformer does not use a preprocessing_fn
            image_aug = image_aug.float()
            image_aug = A.Normalize()(image=image_aug.permute(1, 2, 0).numpy())['image']
            image_aug = torch.from_numpy(image_aug).permute(2, 0, 1).float()

        mask_aug = mask_aug.long()

        return image_aug, mask_aug

