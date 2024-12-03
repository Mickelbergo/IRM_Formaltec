import os
import torch
import cv2
import kornia as K
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from scipy import stats
from augmentations import Augmentation, ValidationAugmentation 
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

class Dataset(BaseDataset):
    def __init__(self, dir_path, image_ids, mask_ids, detection_model = None, augmentation=None, preprocessing_fn = None, target_size=(640, 640), preprocessing_config = None, train_config = None, device = None):
        self.image_ids = image_ids
        self.mask_ids = mask_ids
        self.dir_path = dir_path
        self.augmentation = augmentation
        self.detection_model = detection_model
        self.target_size = target_size
        self.preprocessing_fn = preprocessing_fn
        self.preprocessing_config = preprocessing_config
        self.train_config = train_config
        self.device = device


    def detect_and_crop(self, image, mask, margin=200):
        """Detect regions using YOLO, add a margin around them, and crop image and mask."""
        if self.detection_model:
            results = self.detection_model.predict(source=np.array(image), device=self.device, save=False, verbose=False)
            detections = results[0].boxes

            if len(detections) > 0:
                # Get the bounding box with the highest confidence
                x1, y1, x2, y2 = map(int, detections.xyxy[0].cpu().numpy())

                # Add margin
                left_margin = np.random.randint(0, margin)
                top_margin = np.random.randint(0, margin)
                right_margin = np.random.randint(0, margin)
                bottom_margin = np.random.randint(0, margin)
                x1 = max(0, x1 - left_margin)
                y1 = max(0, y1 - top_margin)
                x2 = min(image.width, x2 + right_margin)
                y2 = min(image.height, y2 + bottom_margin)

                # Crop and resize image and mask
                cropped_image = image.crop((x1, y1, x2, y2)).resize(self.target_size)
                cropped_mask = mask.crop((x1, y1, x2, y2)).resize(self.target_size, Image.NEAREST)
            else:
                # If no detection, fallback to resizing the entire image
                #print("No YOLO detections found. Using full image resize.")
                return self.resize(image, mask)
        else:
            # If no detection model is provided, resize the full image and mask
            return self.resize(image, mask)

        return cropped_image, cropped_mask


    def resize(self, image, mask):
        """Resize image and mask to the target size."""
        resized_image = image.resize(self.target_size)
        resized_mask = mask.resize(self.target_size, Image.NEAREST)
        return resized_image, resized_mask

    def extract_background(self, image, mask, patch_size=(224, 224)):
        """Extract a patch of the background (class 0) from the image."""
        # Convert mask to a NumPy array for easier processing
        mask_array = np.array(mask)

        # Find all background pixels (label 0)
        background_coords = np.argwhere(mask_array == 0)

        if len(background_coords) == 0:
            # If no background is found, fall back to center crop
            print("No background found in mask. Performing center crop.")
            return self.resize(image, mask)

        # Randomly sample a background coordinate
        random_index = np.random.choice(len(background_coords)) 
        center_y, center_x = background_coords[random_index]

        # Define patch boundaries
        half_h, half_w = patch_size[0] // 2, patch_size[1] // 2
        x1 = max(0, center_x - half_w)
        y1 = max(0, center_y - half_h)
        x2 = min(image.width, center_x + half_w)
        y2 = min(image.height, center_y + half_h)

        # Crop the image and mask around the sampled background patch
        cropped_image = image.crop((x1, y1, x2, y2)).resize(self.target_size)
        cropped_mask = mask.crop((x1, y1, x2, y2)).resize(self.target_size, Image.NEAREST)

        return cropped_image, cropped_mask

    def __getitem__(self, ind):
        # Load image and mask
        image_path = os.path.sep.join([self.dir_path, "new_images_640_1280", self.image_ids[ind]])
        mask_path = os.path.sep.join([self.dir_path, "new_masks_640_1280", self.mask_ids[ind]])

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError("mask is none")
        

        # Convert mask to binary segmentation mask
        binary_mask = (mask > 0).astype(np.uint8)
        # Convert mask values to class labels

        multiclass_mask = (mask // 15)  # Assuming mask values are wound_class * 15

        multiclass_mask[np.isin(multiclass_mask, [11, 12, 13, 14])] = 6 #this gets rid of classes 11-14. #remove this (and recalculate weights) to revert

        # Filter out background (class 0) and get non-background classes
        non_background_pixels = multiclass_mask[multiclass_mask != 0]
        dominant_class = 0



        image = Image.fromarray(image)
        binary_mask = Image.fromarray(binary_mask)
        multiclass_mask = Image.fromarray(multiclass_mask)


        #change this if you want
        mode = np.random.choice(["yolo", "resize"], p = [0.2,0.8])

        if self.augmentation == "train":
            
            if self.preprocessing_config["segmentation"] == "binary":
                if mode == "yolo":
                    image, binary_mask= self.detect_and_crop(image, binary_mask)
                elif mode == "resize":
                    image, binary_mask= self.resize(image, binary_mask)
                elif mode == "background":
                    image, binary_mask= self.extract_background(image, binary_mask)
                # else:
                #     raise ValueError(f"Unknown mode: {mode}")

            elif self.preprocessing_config["segmentation"] == 'multiclass':
                if mode == "yolo":
                    image, multiclass_mask = self.detect_and_crop(image, multiclass_mask)
                elif mode == "resize":
                    image, multiclass_mask = self.resize(image, multiclass_mask)
                elif mode == "background":
                    image, multiclass_mask = self.extract_background(image, multiclass_mask)
                # else:
                #     raise ValueError(f"Unknown mode: {mode}")
            
            else:
                raise ValueError(f'segmentation must either be "binary" or "multiclass"')


        
        # Convert to tensor 
        #note that Kornia permutes the images directly, no need to manually permute
        image = np.array(image)
        binary_mask = np.array(binary_mask)
        multiclass_mask = np.array(multiclass_mask)

        image = K.image_to_tensor(image).float().to(self.device)
        binary_mask = K.image_to_tensor(binary_mask).long().to(self.device)  # Binary mask for segmentation
        multiclass_mask = K.image_to_tensor(multiclass_mask).long().to(self.device)  # Multiclass segmentation

        #self.visualize_sample(image, multiclass_mask, binary_mask)

        # Apply augmentations
        if self.augmentation == 'train':
            if self.preprocessing_config["segmentation"] == "binary":
                image, binary_mask = Augmentation(self.target_size, self.preprocessing_fn).augment(image, binary_mask)
            else: image, multiclass_mask = Augmentation(self.target_size, self.preprocessing_fn).augment(image, multiclass_mask)
        
        if self.augmentation == 'validation':
            if self.preprocessing_config["segmentation"] == "binary":
                image, binary_mask = ValidationAugmentation(self.target_size, self.preprocessing_fn).augment(image, binary_mask)
            else: image, multiclass_mask = Augmentation(self.target_size, self.preprocessing_fn).augment(image, multiclass_mask)

        return image, binary_mask, multiclass_mask, 0
    

    def __len__(self):
        return len(self.image_ids)


    def visualize_sample(self, image, multiclass_mask, binary_mask):
        # Convert tensors back to numpy
        image_np = K.tensor_to_image(image.cpu().long())
        binary_mask_np = binary_mask.cpu().numpy()[0]  # Assuming the mask has shape [1, H, W]
        multiclass_mask_np = multiclass_mask.cpu().numpy()[0]  # Assuming the mask has shape [1, H, W]

        # Get the unique class labels in the multiclass mask
        unique_classes = np.unique(multiclass_mask_np)

        # Plot the image, binary mask, and multiclass mask
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_np)
        axes[0].set_title("Image")

        axes[1].imshow(binary_mask_np, cmap="gray")
        axes[1].set_title("Binary Mask")

        axes[2].imshow(multiclass_mask_np, cmap="nipy_spectral")
        axes[2].set_title(f"Multiclass Mask - Classes: {unique_classes}")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()


