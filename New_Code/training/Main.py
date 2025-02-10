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
from model import UNetWithClassification, UNetWithSwinTransformer
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import ssl
import numpy as np
from ultralytics import YOLO
from kornia.losses import FocalLoss

ssl._create_default_https_context = ssl._create_unverified_context

def rescale_weights(original_weights, weight_range=(50, 200)):
    """
    Rescale precomputed class weights to a specified range, keeping class 0 weight fixed at 1.
    """
    # Convert dictionary to tensor if needed
    if isinstance(original_weights, dict):
        original_weights = torch.tensor([original_weights[cls] for cls in sorted(original_weights.keys())])
    
    min_weight, max_weight = weight_range

    # Ensure class 0 (background) remains fixed at 1
    rescaled_weights = original_weights.clone()
    non_background_weights = original_weights[1:]  # Exclude class 0

    if len(non_background_weights) > 0:
        min_original_weight = non_background_weights.min()
        max_original_weight = non_background_weights.max()

        if max_original_weight > min_original_weight:
            rescaled_weights[1:] = (non_background_weights - min_original_weight) / (max_original_weight - min_original_weight)
            rescaled_weights[1:] = rescaled_weights[1:] * (max_weight - min_weight) + min_weight
        else:
            rescaled_weights[1:] = min_weight  # Default to minimum weight if all are the same

    rescaled_weights[0] = 1.0  # Fix class 0 weight at 1.0

    return rescaled_weights

def worker_init_fn(worker_id): #initialize random seed for each worker
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)

def main():

    # Load configurations
    with open('New_Code/configs/training_config.json') as f:
        train_config = json.load(f)

    with open('New_Code/configs/preprocessing_config.json') as f:
        preprocessing_config = json.load(f)

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Initializing Device: {DEVICE}')

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
    
    #OBJECT DETECTION -> YOLO
    detection_model = YOLO(os.path.join(preprocessing_config["yolo_path"], "attempt_1/weights/best.pt"))

    if train_config["encoder"] != "transformer":
        preprocessing_fn = get_preprocessing_fn(train_config["encoder"], pretrained= train_config["encoder_weights"])
    else:
        preprocessing_fn = None    
    # Create dataset instances
    train_dataset = Dataset(
        dir_path=path,
        image_ids=train_ids,
        mask_ids=train_ids,  # Assuming mask names match image names
        augmentation= 'train',
        preprocessing_fn= preprocessing_fn,
        detection_model=detection_model,
        target_size= tuple(preprocessing_config["target_size"]),
        preprocessing_config = preprocessing_config,
        train_config = train_config,
        device = DEVICE,
        exclude_images_with_classes=preprocessing_config["exclude_images_with_classes"],
        classes_to_exclude=preprocessing_config["classes_to_exclude"])


    valid_dataset = Dataset(
        dir_path=path,
        image_ids=valid_ids,
        mask_ids=valid_ids,
        detection_model=detection_model,  # Assuming mask names match image names
        augmentation='validation',
        preprocessing_fn= preprocessing_fn,
        target_size= tuple(preprocessing_config["target_size"]),
        preprocessing_config = preprocessing_config,
        train_config = train_config,
        device = DEVICE,
        exclude_images_with_classes=preprocessing_config["exclude_images_with_classes"],
        classes_to_exclude=preprocessing_config["classes_to_exclude"])
    

    train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True, num_workers= train_config["num_workers"], worker_init_fn= worker_init_fn, persistent_workers= True)
    valid_loader = DataLoader(valid_dataset, batch_size=train_config["batch_size"], shuffle=False, num_workers= train_config["num_workers"], worker_init_fn= worker_init_fn, persistent_workers=True)

    # Define model
    if train_config["encoder"] == "transformer": #using SWIN transformer from huggingface with pretrained weights
        model = UNetWithSwinTransformer(classes = train_config["segmentation_classes"], activation = train_config["activation"])
    else:
        model = UNetWithClassification(
            encoder_name=train_config["encoder"],
            encoder_weights=train_config["encoder_weights"],
            classes= train_config["segmentation_classes"],  # Segmentation classes (e.g., wound vs. background),  # Replace with the actual number of wound classes
            activation=None #crossentropy loss expects raw logits
        )
    model = model.to(DEVICE)

    # Define optimizer, loss, and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["optimizer_lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_config["lr_scheduler_gamma"])
    encoder = train_config["encoder"]

    # Define the type of segmentation, the corresponding loss function and the weights

    segmentation = preprocessing_config["segmentation"] #either 'binary' or 'multiclass'
    #class_weights_multiclass = torch.load(os.path.join(path, "class_weights.pth"), weights_only= True).float().to(DEVICE)
    #class_weights = torch.tensor(train_config["class_weights"]).to(DEVICE)

    weight_range = (50,200)
    non_scaled_weights = torch.load(os.path.join(path, "class_weights.pth"), weights_only=True).float().to(DEVICE)
    image_weights_tensor = torch.load(os.path.join(path, "image_weights.pth"), weights_only=True).float().to(DEVICE)
    class_weights_multiclass = rescale_weights(non_scaled_weights, weight_range=weight_range)
    class_weights_binary = torch.tensor(train_config["class_weights"]).float().to(DEVICE)

    if segmentation == "binary": 
        CE_Loss = nn.CrossEntropyLoss(weight = class_weights_binary) #use the predefined weights for background vs wound
    else:
        CE_Loss = nn.CrossEntropyLoss(weight = class_weights_multiclass) 

    if train_config["dice"]:
        DICE_Loss = smp.losses.DiceLoss(mode = "multiclass")
    else: 
        DICE_Loss = None
    if train_config["focal"]:
        Focal_loss = FocalLoss(alpha=0.5, gamma=4.0, reduction='mean', weight=class_weights_multiclass)
    else:
        Focal_loss = None

    #Segmentation loss function is either only weighted BCE or weighted BCE + DICE

    if train_config["display_image"]:
        display_image = True
    else: display_image = False

    # Define training and validation epochs
    
    train_epoch = TrainEpoch(model=model, CE_Loss=CE_Loss, DICE_Loss=DICE_Loss, Focal_loss=Focal_loss, segmentation=segmentation, optimizer=optimizer, device=DEVICE, grad_clip_value = train_config["grad_clip_value"], 
                            display_image = display_image, nr_classes = train_config["segmentation_classes"],scheduler=scheduler)
    valid_epoch = ValidEpoch(model=model, CE_Loss = CE_Loss, DICE_Loss = DICE_Loss, Focal_loss = Focal_loss, segmentation = segmentation, device=DEVICE, display_image = display_image, nr_classes = train_config["segmentation_classes"], scheduler=scheduler)

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

