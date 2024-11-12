import os
import json
import numpy as np
import cv2
import random
import shutil

# Load configuration
with open('New_Code/configs/training_config.json') as f:
    train_config = json.load(f)
preprocess_path = train_config["preprocess_path"]
path = train_config["path"]

with open('New_Code/configs/preprocessing_config.json') as f:
    preprocessing_config = json.load(f)

yolo_path = preprocessing_config["yolo_path"]
labels_dir = os.path.join(yolo_path, "labels")
os.makedirs(labels_dir, exist_ok=True)

# Load all image and mask paths
image_dir = os.path.join(path, "new_images_640_1280")
mask_dir = os.path.join(path, "new_masks_640_1280")
image_ids = sorted([f for f in os.listdir(image_dir)])
mask_ids = sorted([f for f in os.listdir(mask_dir)])

def generate_bounding_boxes(mask_path, image_path, labels_dir, image_name):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or image is None:
        print(f"Error: Unable to read image or mask at {image_path} or {mask_path}")
        return

    binary_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    img_height, img_width = image.shape[:2]

 
    yolo_label_path = os.path.join(labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
    with open(yolo_label_path, "w") as label_file:
        for x, y, w, h in bounding_boxes:
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height
            class_id = 0  
            label_file.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")


for image_name, mask_name in zip(image_ids, mask_ids):
    mask_path = os.path.join(mask_dir, mask_name)
    image_path = os.path.join(image_dir, image_name)
    generate_bounding_boxes(mask_path, image_path, labels_dir, image_name)

# Organize images and labels into train and val folders

train_images_dir = os.path.join(yolo_path, 'images/train')
val_images_dir = os.path.join(yolo_path, 'images/val')

train_labels_dir = os.path.join(labels_dir, 'train')
val_labels_dir = os.path.join(labels_dir, 'val')

# Create directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# List all images and shuffle for random splitting
all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
random.shuffle(all_images)

# Split data (80% train, 20% val)
split_index = int(0.8 * len(all_images))
train_images = all_images[:split_index]
val_images = all_images[split_index:]

# Move files
for image in train_images:
    image_path = os.path.join(image_dir, image)
    label_path = os.path.join(labels_dir, f"{os.path.splitext(image)[0]}.txt")
    
    # Move image and corresponding label to train directories
    shutil.copy(image_path, train_images_dir)
    shutil.move(label_path, train_labels_dir)

for image in val_images:
    image_path = os.path.join(image_dir, image)
    label_path = os.path.join(labels_dir, f"{os.path.splitext(image)[0]}.txt")
    
    # Move image and corresponding label to val directories
    shutil.copy(image_path, val_images_dir)
    shutil.move(label_path, val_labels_dir)
