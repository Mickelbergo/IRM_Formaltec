import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from preprocessing import Dataset
from epochs import TrainEpoch, ValidEpoch

# Load configurations
with open('training_config.json') as f:
    config = json.load(f)

# Device configuration
DEVICE = torch.device(config["device"])

# Set paths
path = config["path"]
model_version = config["model_version"]

# Load datasets
train_dataset = Dataset(
    dir_path=path,
    split="train",
    augmentation=config["augmentation"],
    preprocessing=True
)

valid_dataset = Dataset(
    dir_path=path,
    split="val",
    augmentation=None,
    preprocessing=True
)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False)

# Define model
model = smp.Unet(
    encoder_name=config["encoder"],
    encoder_weights=config["encoder_weights"],
    classes=2,  # Segmentation classes (e.g., wound vs. background)
    activation=config["activation"]
)
model = model.to(DEVICE)

# Define optimizer, loss, and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer_lr"])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_scheduler_gamma"])
segmentation_loss_fn = nn.CrossEntropyLoss()  # Segmentation loss
classification_loss_fn = nn.CrossEntropyLoss()  # Classification loss

# Define training and validation epochs
train_epoch = TrainEpoch(model, segmentation_loss_fn, classification_loss_fn, optimizer, device=DEVICE)
valid_epoch = ValidEpoch(model, segmentation_loss_fn, classification_loss_fn, device=DEVICE)

# Training loop
max_score = 0
for epoch in range(config["num_epochs"]):
    print(f"Epoch {epoch + 1}/{config['num_epochs']}")
    
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
