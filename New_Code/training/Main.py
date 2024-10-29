import os
import torch
import random
import json
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Preprocessing import Dataset
from Epochs import TrainEpoch, ValidEpoch
from model import UNetWithClassification, UNetWithSwinTransformer, Faster_RCNN
from preprocessing_memory import Memory_dataset

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
def main():

    # Load configurations
    with open('New_Code/configs/training_config.json') as f:
        train_config = json.load(f)

    with open('New_Code/configs/preprocessing_config.json') as f:
        preprocessing_config = json.load(f)

    # Device configuration
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
    random.seed(42)  # For reproducibility
    random.shuffle(image_ids)

    split_ratio = 0.8  # 80% training, 20% validation
    split_index = int(len(image_ids) * split_ratio)

    train_ids = image_ids[:split_index]
    valid_ids = image_ids[split_index:]
    
    #OBJECT DETECTION -> FASTER R-CNN
    detection_model = Faster_RCNN().get_faster_rcnn().to(DEVICE)
    detection_model.eval()

    # Create dataset instances
    train_dataset = Dataset(
        dir_path=path,
        image_ids=train_ids,
        mask_ids=train_ids,  # Assuming mask names match image names
        augmentation= 'train',
        detection_model = detection_model,
        target_size= tuple(preprocessing_config["target_size"])
    )

    valid_dataset = Dataset(
        dir_path=path,
        image_ids=valid_ids,
        mask_ids=valid_ids,  # Assuming mask names match image names
        augmentation='validation',
        detection_model = detection_model,
        target_size= tuple(preprocessing_config["target_size"])
    )

    train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers= train_config["num_workers"])
    valid_loader = DataLoader(valid_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers= train_config["num_workers"])

    # Define model
    if train_config["encoder"] == "transformer": #using SWIN transformer from huggingface with pretrained weights
        model = UNetWithSwinTransformer(classes = train_config["segmentation_classes"], activation = train_config["activation"])
    else:
        model = UNetWithClassification(
            encoder_name=train_config["encoder"],
            encoder_weights=train_config["encoder_weights"],
            classes= train_config["segmentation_classes"],  # Segmentation classes (e.g., wound vs. background),  # Replace with the actual number of wound classes
            activation=train_config["activation"]
        )
    model = model.to(DEVICE)

    # Define optimizer, loss, and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["optimizer_lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_config["lr_scheduler_gamma"])
    encoder = train_config["encoder"]
    # Define the type of segmentation, the corresponding loss function and the weights
    segmentation = preprocessing_config["segmentation"] #either 'binary' or 'multiclass'
    class_weights_multiclass = torch.load(os.path.join(path, "class_weights.pth")).float().to(DEVICE)
    class_weights = torch.tensor(train_config["class_weights"]).to(DEVICE)
    if segmentation == "binary": 
        CE_Loss = nn.CrossEntropyLoss(weight = class_weights) #use the predefined weights for background vs wound
    else:
        CE_Loss = nn.CrossEntropyLoss(weight = class_weights_multiclass) #no weights yet

    if train_config["dice"]:
        DICE_Loss = smp.losses.DiceLoss(mode = "multiclass")
    else: 
        DICE_Loss = None

    #Segmentation loss function is either only weighted BCE or weighted BCE + DICE


    if train_config["display_image"]:
        display_image = True
    else: display_image = False

    # Define training and validation epochs
    train_epoch = TrainEpoch(model, CE_Loss, DICE_Loss, segmentation, optimizer, device=DEVICE, grad_clip_value = train_config["grad_clip_value"], display_image = display_image)
    valid_epoch = ValidEpoch(model, CE_Loss, DICE_Loss, segmentation, device=DEVICE, display_image = display_image)

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
            torch.save(model, os.path.join(path, f"best_model_{model_version}_{epoch}_{encoder}.pth"))
            print("Best model saved!")
        
        # Update learning rate
        scheduler.step()

        # Print metrics
        print(f"Train Loss: {train_logs['loss']:.4f}, Valid Loss: {valid_logs['loss']:.4f}")
        print(f"Train Acc: {train_logs['accuracy']:.4f}, Valid Acc: {valid_logs['accuracy']:.4f}")
        print(f"Train IoU: {train_logs['iou_score']:.4f}, Valid IoU: {valid_logs['iou_score']:.4f}")

    # Save the final model
    torch.save(model, os.path.join(path, f"final_model_{model_version}_{150}_{encoder}.pth"))
    print("Final model saved!")

if __name__ == "__main__":
    main()