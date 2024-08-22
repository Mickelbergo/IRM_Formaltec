import os
import torch
import random
import json
import torch.nn as nn

from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from preprocessing2 import Dataset
from epochs2 import TrainEpoch, ValidEpoch

# Load configurations
with open('New_Code/configs/training_config.json') as f:
    train_config = json.load(f)

with open('New_Code/configs/preprocessing_config.json') as f:
    preprocessing_config = json.load(f)

# Device configuration
DEVICE = torch.device(train_config["device"])

# Set paths
path = train_config["path"]
model_version = train_config["model_version"]

# Load all image and mask paths
image_dir = os.path.join(path, "images")
mask_dir = os.path.join(path, "masks")

image_ids = sorted(os.listdir(image_dir))

# Shuffle and split the dataset into training and validation sets
random.seed(42)  # For reproducibility
random.shuffle(image_ids)

split_ratio = 0.8  # 80% training, 20% validation
split_index = int(len(image_ids) * split_ratio)

train_ids = image_ids[:split_index]
valid_ids = image_ids[split_index:]

# Create dataset instances
train_dataset = Dataset(
    dir_path=path,
    image_ids=train_ids,
    mask_ids=train_ids,  # Assuming mask names match image names
    augmentation=preprocessing_config["augmentation"],
    preprocessing=True,
    target_size=(640, 640)  # You can adjust this size as needed
)

valid_dataset = Dataset(
    dir_path=path,
    image_ids=valid_ids,
    mask_ids=valid_ids,  # Assuming mask names match image names
    augmentation=None,
    preprocessing=True,
    target_size=(640, 640)  # Ensure it's the same size as for training
)

train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=train_config["batch_size"], shuffle=False)

# Define model
model = smp.Unet(
    encoder_name=train_config["encoder"],
    encoder_weights=train_config["encoder_weights"],
    classes=2,  # Segmentation classes (e.g., wound vs. background)
    activation=train_config["activation"]
)
model = model.to(DEVICE)

# Define optimizer, loss, and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=train_config["optimizer_lr"])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_config["lr_scheduler_gamma"])
segmentation_loss_fn = nn.CrossEntropyLoss()  # Segmentation loss
classification_loss_fn = nn.CrossEntropyLoss()  # Classification loss

# Define training and validation epochs
train_epoch = TrainEpoch(model, segmentation_loss_fn, classification_loss_fn, optimizer, device=DEVICE)
valid_epoch = ValidEpoch(model, segmentation_loss_fn, classification_loss_fn, device=DEVICE)

# Training loop
max_score = 0
for epoch in range(train_config["num_epochs"]):
    print(f"Epoch {epoch + 1}/{train_config['num_epochs']}")
    
    # Train model
    train_logs = train_epoch.run(train_loader)
    
    # Validate model
    valid_logs = valid_epoch.run(valid_loader)
    
    # Save best model
    if valid_logs['iou_score'] > max_score:
        max_score = valid_logs['iou_score']
        torch.save(model, os.path.join(path, f"best_model_{model_version}.pth"))
        print("Best model saved!")
    
    # Update learning rate
    scheduler.step()

    # Print metrics
    print(f"Train Loss: {train_logs['loss']:.4f}, Valid Loss: {valid_logs['loss']:.4f}")
    print(f"Train Acc: {train_logs['accuracy']:.4f}, Valid Acc: {valid_logs['accuracy']:.4f}")
    print(f"Train IoU: {train_logs['iou_score']:.4f}, Valid IoU: {valid_logs['iou_score']:.4f}")

# Save the final model
torch.save(model, os.path.join(path, f"final_model_{model_version}.pth"))
print("Final model saved!")
