import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
import json
import torch
import random
import sys
from tqdm import tqdm

with open('Code/configs/training_config.json') as f:
    train_config = json.load(f)
preprocess_path = train_config["data"]["preprocess_path"]

sys.path.append(preprocess_path)
from Preprocessing import Dataset


def precompute_image_weights(dataset, class_weights, save_path):
    """
    Precompute image-level weights based on pixel class distributions and class weights.

    Args:
        dataset (Dataset): A dataset that returns (image, binary_mask, multiclass_mask, _).
        class_weights (torch.Tensor): Tensor of class weights [num_classes].
        save_path (str): Path to save the computed image weights.

    Returns:
        None
    """
    image_weights = []

    print("Precomputing image weights...")
    for i in tqdm(range(len(dataset))):
        _, _, mask_classes, _ = dataset[i]  # Assuming the third output is the segmentation mask
        mask = mask_classes.cpu().numpy()

        # Calculate class distribution in the mask
        unique, counts = np.unique(mask, return_counts=True)
        class_distribution = dict(zip(unique, counts))

        # Compute image weight as the weighted average of class weights
        total_pixels = mask.size
        image_weight = sum((class_distribution.get(cls, 0) / total_pixels) * class_weights[cls].item() 
                           for cls in range(len(class_weights)))

        image_weights.append(image_weight)

    # Convert to a tensor and save
    image_weights_tensor = torch.tensor(image_weights)
    torch.save(image_weights_tensor, save_path)

    print(f"Image weights saved to {save_path}")
    print("Image Weights:", image_weights_tensor)


def calculate_class_distribution(dataset):
    class_counts = Counter()
    total_pixels = 0
    all_classes = set()

    # Iterate through dataset and count class occurrences
    for i in range(len(dataset)):
        _, _, mask_classes, _ = dataset[i]  # Get mask_classes for each image
        unique, counts = np.unique(mask_classes.cpu().numpy(), return_counts=True)

        all_classes.update(unique)

        class_counts.update(dict(zip(unique, counts)))
        total_pixels += mask_classes.numel()

    # Calculate frequency of each class
    class_distribution = {cls: count / total_pixels for cls, count in class_counts.items()}

    print(f'all classes found: {sorted(all_classes)}')
    return class_counts, class_distribution

def calculate_class_weights(class_counts, total_pixels, total_classes):
    """
    Calculate the original class weights inversely proportional to class distribution.
    Background (class 0) weight remains 1.

    Parameters:
        class_counts (dict): Count of pixels for each class.
        total_pixels (int): Total number of pixels in the dataset.
        total_classes (int): Total number of classes.

    Returns:
        dict: Original class weights for all classes.
    """
    # class_weights = {}

    # # Exclude background class (class 0) for the calculation
    # for cls in range(total_classes):
    #     if cls in class_counts and cls != 0:
    #         distribution = class_counts[cls] / total_pixels
    #         weight = 1 / (distribution + 1e-6)  # Inverse of class distribution
    #         class_weights[cls] = weight
    #     else:
    #         class_weights[cls] = 1 if cls == 0 else 0  # Fixed weight for background and 0 for missing classes

    # return class_weights

    """
    Precompute class weights inversely proportional to class distribution, excluding the background from total count.

    Parameters:
        class_counts (dict): Count of pixels for each class.
        total_classes (int): Total number of classes.

    Returns:
        dict: Original class weights (inverse of class distribution), with background fixed at 1.
    """
    # Total pixels for non-background classes
    total_non_background_pixels = sum(count for cls, count in class_counts.items() if cls != 0)
    
    class_weights = {}
    for cls in range(total_classes):
        if cls == 0:
            class_weights[cls] = 1.0  # Fix background weight at 1
        elif cls in class_counts:
            distribution = class_counts[cls] / total_non_background_pixels
            class_weights[cls] = 1 / (distribution + 1e-6)  # Inverse class distribution
        else:
            class_weights[cls] = 0  # Assign weight 0 to missing classes
    
    return class_weights


def plot_class_distribution(class_distribution):
    classes = list(class_distribution.keys())
    frequencies = list(class_distribution.values())

    plt.bar(classes[1:], frequencies[1:])
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Class Distribution in Segmentation Masks')
    plt.show()

def save_class_weights(weights, save_path):
    torch.save(weights, save_path)

def load_class_weights(save_path):
    return torch.load(save_path)


# Load configurations
with open('Code/configs/training_config.json') as f:
    train_config = json.load(f)

with open('Code/configs/preprocessing_config.json') as f:
    preprocessing_config = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# Set paths
path = train_config["data"]["path"]
model_version = train_config["model"]["version"]

# Load all image and mask paths
image_dir = os.path.join(path, "new_images_640_1280")
mask_dir = os.path.join(path, "new_masks_640_1280")

valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
image_ids = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)])

# Shuffle and split the dataset into training and validation sets
random.seed(42)
random.shuffle(image_ids)

split_ratio = 0.8
split_index = int(len(image_ids) * split_ratio)

train_ids = image_ids[:split_index]
valid_ids = image_ids[split_index:]

# Create dataset instances (using the whole dataset)
full_dataset = Dataset(
    dir_path=path,
    image_ids=image_ids,  # Use all images (training + validation)
    mask_ids=image_ids,   # Use all masks
    augmentation=None,
    target_size=tuple(preprocessing_config["target_size"])
)

# Class weights file path
class_weights_path = os.path.join(path, "class_weights.pth")
image_weights_path = os.path.join(path, "image_weights.pth")


# Check if class weights are already saved
if os.path.exists(class_weights_path):
    print(f"Loading class weights from {class_weights_path}")
    class_weights_tensor = load_class_weights(class_weights_path).to(DEVICE)
    print(class_weights_tensor)

    # Precompute and save image weights
    precompute_image_weights(full_dataset, class_weights_tensor, image_weights_path)

    class_counts, class_distribution = calculate_class_distribution(full_dataset)
    plot_class_distribution(class_distribution)
    
else:
    print("Calculating class weights for the whole dataset...")
    class_counts, class_distribution = calculate_class_distribution(full_dataset)

    # Plot class distribution (optional)
    #plot_class_distribution(class_distribution)

    # Calculate class weights
    total_pixels = sum(class_counts.values())
    total_classes = train_config["segmentation_classes"]
    class_weights = calculate_class_weights(class_counts, total_pixels, total_classes)
    # Convert class weights to tensor and save them
    class_weights_tensor = torch.tensor([class_weights[cls] for cls in sorted(class_weights.keys())]).to(DEVICE)
    save_class_weights(class_weights_tensor, class_weights_path)

    print(f"Class weights saved to {class_weights_path}")
    precompute_image_weights(full_dataset, class_weights_tensor, image_weights_path)
    print(f'Image weights saved to {image_weights_path}')

    print("class weights:", class_weights_tensor)