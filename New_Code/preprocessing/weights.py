import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
import json
import torch
import random
import sys
preprocess_path = "E:/ForMalTeC/Wound_segmentation_III/GIT/IRM_Formaltec/New_Code/training"
sys.path.append(preprocess_path)
from Preprocessing import Dataset


def calculate_class_distribution(dataset):
    class_counts = Counter()
    total_pixels = 0

    # Iterate through dataset and count class occurrences
    for i in range(len(dataset)):
        _, _, mask_classes, _ = dataset[i]  # Get mask_classes for each image
        unique, counts = np.unique(mask_classes.cpu().numpy(), return_counts=True)
        class_counts.update(dict(zip(unique, counts)))
        total_pixels += mask_classes.numel()

    # Calculate frequency of each class
    class_distribution = {cls: count / total_pixels for cls, count in class_counts.items()}
    return class_counts, class_distribution


def calculate_class_weights(class_counts, total_pixels):
    # Inverse frequency weighting
    class_weights = {cls: total_pixels / (count + 1e-6) for cls, count in class_counts.items()}  # Avoid division by zero
    
    max_cap = 300  # Define the maximum weight cap
    class_weights = {cls: min(weight, max_cap) for cls, weight in class_weights.items()}
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
with open('New_Code/configs/training_config.json') as f:
    train_config = json.load(f)

with open('New_Code/configs/preprocessing_config.json') as f:
    preprocessing_config = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# Set paths
path = train_config["path"]
model_version = train_config["model_version"]

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

# Check if class weights are already saved
if os.path.exists(class_weights_path):
    print(f"Loading class weights from {class_weights_path}")
    class_weights_tensor = load_class_weights(class_weights_path).to(DEVICE)
    print(class_weights_tensor)
else:
    print("Calculating class weights for the whole dataset...")
    class_counts, class_distribution = calculate_class_distribution(full_dataset)

    # Plot class distribution (optional)
    plot_class_distribution(class_distribution)

    # Calculate class weights
    total_pixels = sum(class_counts.values())
    class_weights = calculate_class_weights(class_counts, total_pixels)

    # Convert class weights to tensor and save them
    class_weights_tensor = torch.tensor([class_weights[cls] for cls in sorted(class_weights.keys())]).to(DEVICE)
    save_class_weights(class_weights_tensor, class_weights_path)

    print(f"Class weights saved to {class_weights_path}")