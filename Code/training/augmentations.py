import torch
import numpy as np
import torch
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Augmentation:
    def __init__(self, target_size, preprocessing_fn=None, config_path='Code/configs/preprocessing_config.json'):
        self.target_size = target_size
        self.preprocessing_fn = preprocessing_fn
        with open(config_path) as f:
            config = json.load(f)
        aug_cfg = config.get('augmentation_settings', {})
        transforms = []
        # Resize
        if aug_cfg.get('resize', {}).get('enabled', True):
            transforms.append(A.Resize(height=target_size[0], width=target_size[1]))
        # Horizontal Flip
        if aug_cfg.get('horizontal_flip', {}).get('enabled', False):
            transforms.append(A.HorizontalFlip(p=aug_cfg['horizontal_flip'].get('p', 0.5)))
        # Vertical Flip
        if aug_cfg.get('vertical_flip', {}).get('enabled', False):
            transforms.append(A.VerticalFlip(p=aug_cfg['vertical_flip'].get('p', 0.5)))
        # Random Rotate 90
        if aug_cfg.get('random_rotate_90', {}).get('enabled', False):
            transforms.append(A.RandomRotate90(p=aug_cfg['random_rotate_90'].get('p', 0.5)))
        # Color Jitter
        if aug_cfg.get('color_jitter', {}).get('enabled', False):
            cj = aug_cfg['color_jitter']
            transforms.append(A.ColorJitter(
                brightness=cj.get('brightness', 0.2),
                contrast=cj.get('contrast', 0.2),
                saturation=cj.get('saturation', 0.2),
                hue=cj.get('hue', 0.2),
                p=cj.get('p', 0.5)
            ))
        # Random Brightness Contrast
        if aug_cfg.get('random_brightness_contrast', {}).get('enabled', False):
            transforms.append(A.RandomBrightnessContrast(p=aug_cfg['random_brightness_contrast'].get('p', 0.5)))
        # Hue Saturation Value
        if aug_cfg.get('hue_saturation_value', {}).get('enabled', False):
            transforms.append(A.HueSaturationValue(p=aug_cfg['hue_saturation_value'].get('p', 0.0)))
        # Distortions
        distortion_transforms = []
        if aug_cfg.get('elastic_transform', {}).get('enabled', False):
            distortion_transforms.append(A.ElasticTransform(p=aug_cfg['elastic_transform'].get('p', 0.5)))
        if aug_cfg.get('optical_distortion', {}).get('enabled', False):
            distortion_transforms.append(A.OpticalDistortion(p=aug_cfg['optical_distortion'].get('p', 0.5)))
        if aug_cfg.get('grid_distortion', {}).get('enabled', False):
            distortion_transforms.append(A.GridDistortion(p=aug_cfg['grid_distortion'].get('p', 0.5)))
        if distortion_transforms:
            transforms.append(A.OneOf(distortion_transforms))
        # Coarse Dropout
        if aug_cfg.get('coarse_dropout', {}).get('enabled', False):
            cd = aug_cfg['coarse_dropout']
            transforms.append(A.CoarseDropout(
                num_holes_range=cd.get('num_holes_range', [1,8]),
                hole_height_range=cd.get('hole_height_range', [1,32]),
                hole_width_range=cd.get('hole_width_range', [1,32]),
                p=cd.get('p', 0.3)
            ))
        # Gaussian Blur
        if aug_cfg.get('gaussian_blur', {}).get('enabled', False):
            gb = aug_cfg['gaussian_blur']
            transforms.append(A.GaussianBlur(
                blur_limit=tuple(gb.get('blur_limit', [3, 7])),
                p=gb.get('p', 0.2)
            ))
        # ToTensor
        transforms.append(ToTensorV2())
        self.transform = A.Compose(transforms, additional_targets={'mask': 'mask'})

    def augment(self, image, mask):
        image_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)
        augmented = self.transform(image=image_np, mask=mask_np)
        image_aug = augmented['image']
        mask_aug = augmented['mask']
        if self.preprocessing_fn is not None:
            image_aug = image_aug.permute(1, 2, 0).numpy()
            image_aug = self.preprocessing_fn(image_aug)
            image_aug = torch.from_numpy(image_aug).permute(2, 0, 1).float()
        else:
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

