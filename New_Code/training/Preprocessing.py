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

    def detect_and_crop(self, image, mask):
        image_tensor = K.image_to_tensor(np.array(image)).float().unsqueeze(0).to(self.device)
        self.detection_model.to(self.device)
        # Perform detection
        with torch.no_grad():
            detections = self.detection_model(image_tensor)[0]

        # Check if any detections are made
        if len(detections['boxes']) > 0:
            # Get the bounding box with the highest confidence score
            scores = detections['scores'].cpu().numpy()
            max_index = scores.argmax()
            box = detections['boxes'][max_index].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)

            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.width, x2), min(image.height, y2)

            # Crop both image and mask using the detected bounding box
            cropped_image = image.crop((x1, y1, x2, y2)).resize(self.target_size)
            cropped_mask = mask.crop((x1, y1, x2, y2)).resize((self.target_size), Image.NEAREST)
        else:
            # No detections: perform center crop
            print(f"No detection found for image. Performing center crop.")
            width, height = image.size
            new_width, new_height = 224, 224
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height

            # Crop both image and mask using the center crop
            cropped_image = image.crop((left, top, right, bottom))
            cropped_mask = mask.crop((left, top, right, bottom))

        return cropped_image, cropped_mask


    def __getitem__(self, ind):
        # Load image and mask
        image_path = os.path.sep.join([self.dir_path, "new_images_640_1280", self.image_ids[ind]])
        mask_path = os.path.sep.join([self.dir_path, "new_masks_640_1280", self.mask_ids[ind]])


        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError("mask is none")
        
        #OBJECT DETECTION
        if self.train_config["object_detection"]:
            # Convert NumPy arrays to PIL Images
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
            cropped_image, cropped_mask = self.detect_and_crop(image, mask)

            if cropped_image is None or cropped_mask is None:
                return None

            image, mask = np.array(cropped_image), np.array(cropped_mask)


        # Convert mask to binary segmentation mask
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Convert mask values to class labels
        multiclass_mask = (mask // 15)  # Assuming mask values are wound_class * 15


        # Filter out background (class 0) and get non-background classes
        non_background_pixels = multiclass_mask[multiclass_mask != 0]
        

        # if len(non_background_pixels) > 0:
        #     # Determine the most frequent non-background class in the mask
        #     dominant_class = int(torch.mode(non_background_pixels.flatten())[0])
        # else:
        #     # Handle edge case where the mask is entirely background
        #     dominant_class = 0

        dominant_class = 0

        # Convert to tensor 
        #note that Kornia permutes the images directly, no need to manually permute
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



        return image, binary_mask, multiclass_mask, dominant_class
    

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

