import os
import torch
import random
import json
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from itertools import product
from visualize_gradcam import visualize_gradcam
from collections import Counter
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context

# Utility function for config access
# Use new structured config if available, else fallback to legacy

def get_config(cfg, *keys, legacy_key=None):
    d = cfg
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return cfg.get(legacy_key or keys[-1])
    return d

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

def worker_init_fn(worker_id):
    # Initialize random seed for each worker
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)

def get_most_frequent_foreground_class(mask_path):
    mask = np.array(Image.open(mask_path).convert("RGB"))
    class_mask = mask // 15
    if class_mask.ndim == 3:
        class_mask = class_mask[..., 0]
    flat = class_mask.flatten()
    foreground = flat[flat > 0]
    if len(foreground) == 0:
        return 0
    most_common = Counter(foreground).most_common(1)[0][0]
    return int(most_common)

def train_once(train_config, preprocessing_config, train_ids, valid_ids, path, preprocessing_fn, detection_model, DEVICE,
               lambdaa, sampler_option, weight_range, loss_combination, lr, optimizer_choice, grid_search=False):
    """
    Train the model once with the provided hyperparameters.
    We always save each new best model (based on IoU), including IoU and F1 in the filename along with all other parameters.
    After training, return the best IoU, best F1, and the state dict of the best model.
    """

    # Create dataset instances
    train_dataset = Dataset(
        dir_path=path,
        image_ids=train_ids,
        mask_ids=train_ids,
        augmentation='train',
        preprocessing_fn=preprocessing_fn,
        detection_model=detection_model,
        target_size=tuple(preprocessing_config["target_size"]),
        preprocessing_config=preprocessing_config,
        train_config=train_config,
        device=DEVICE,        
        exclude_images_with_classes=preprocessing_config["exclude_images_with_classes"],
        classes_to_exclude=preprocessing_config["classes_to_exclude"]
    )

    valid_dataset = Dataset(
        dir_path=path,
        image_ids=valid_ids,
        mask_ids=valid_ids,
        detection_model=detection_model,
        augmentation='validation',
        preprocessing_fn=preprocessing_fn,
        target_size=tuple(preprocessing_config["target_size"]),
        preprocessing_config=preprocessing_config,
        train_config=train_config,
        device=DEVICE,
        exclude_images_with_classes=preprocessing_config["exclude_images_with_classes"],
        classes_to_exclude=preprocessing_config["classes_to_exclude"]
    )

    # Define model
    if get_config(train_config, "model", "encoder", legacy_key="encoder") == "transformer":
        model = UNetWithSwinTransformer(classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"), activation=get_config(train_config, "model", "activation", legacy_key="activation"))
    else:
        model = UNetWithClassification(
            encoder_name=get_config(train_config, "model", "encoder", legacy_key="encoder"),
            encoder_weights=get_config(train_config, "model", "encoder_weights", legacy_key="encoder_weights"),
            classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"),
            activation=None #crossentropy loss expects raw logits
        )
    model = model.to(DEVICE)

    # Define optimizer
    if optimizer_choice == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_choice == "sgd":  # 'sgd'
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError("Supported optimizers: 'adamw', 'sgd' ")
    

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=get_config(train_config, "training", "lr_scheduler_gamma", legacy_key="lr_scheduler_gamma"))
    encoder = get_config(train_config, "model", "encoder", legacy_key="encoder")
    segmentation = preprocessing_config["segmentation"]

    # Loading weights
    non_scaled_weights = torch.load(os.path.join(path, "class_weights.pth"), weights_only=True).float().to(DEVICE)
    image_weights_tensor = torch.load(os.path.join(path, "image_weights.pth"), weights_only=True).float().to(DEVICE)

    class_weights_multiclass = rescale_weights(non_scaled_weights, weight_range=weight_range)
    class_weights_binary = torch.tensor(get_config(train_config, "model", "class_weights", legacy_key="class_weights")).float().to(DEVICE)

    # Determine loss combination
    if loss_combination == 'focal+ce':
        focal_loss_flag = True
        dice_loss_flag = False
    elif loss_combination == "dice+ce":  # 'dice+ce'
        focal_loss_flag = False
        dice_loss_flag = True
    else:
        raise ValueError("Supported loss combinations: 'focal+ce', 'dice+ce' ")

    if segmentation == "binary":
        CE_Loss = nn.CrossEntropyLoss(weight=class_weights_binary)
    else:
        CE_Loss = nn.CrossEntropyLoss(weight=class_weights_multiclass) 

    DICE_Loss = smp.losses.DiceLoss(mode="multiclass") if dice_loss_flag else None
    Focal_loss = FocalLoss(alpha=0.5, gamma=4.0, reduction='mean', weight=class_weights_multiclass) if focal_loss_flag else None

    # Handling sampler
    image_ids_total = train_ids + valid_ids
    if sampler_option:
        assert len(image_weights_tensor) == len(image_ids_total), "Mismatch between image weights and dataset size!"
        image_id_to_weight = dict(zip(image_ids_total, image_weights_tensor))

        if(preprocessing_config["exclude_images_with_classes"]): #if we excluded some image ids, we have to take this into account for the sampler
            train_ids = train_dataset.image_ids

        train_weights = torch.tensor([image_id_to_weight[img_id] for img_id in train_ids]).to(DEVICE)
        sampler = WeightedRandomSampler(weights=train_weights.tolist(), num_samples=len(train_weights), replacement=True)
    else:
        sampler = None

    train_loader = DataLoader(train_dataset, batch_size=get_config(train_config, "training", "batch_size", legacy_key="batch_size"), shuffle=False, 
                              num_workers=get_config(train_config, "training", "num_workers", legacy_key="num_workers"), worker_init_fn=worker_init_fn, 
                              persistent_workers=True, sampler=sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=get_config(train_config, "training", "batch_size", legacy_key="batch_size"), shuffle=False, 
                              num_workers=get_config(train_config, "training", "num_workers", legacy_key="num_workers"), worker_init_fn=worker_init_fn, 
                              persistent_workers=True)

    train_epoch = TrainEpoch(model, CE_Loss, DICE_Loss, Focal_loss, lambdaa, segmentation, optimizer, device=DEVICE,
                             grad_clip_value=get_config(train_config, "training", "grad_clip_value", legacy_key="grad_clip_value"),
                             display_image=get_config(train_config, "training", "display_image", legacy_key="display_image"), 
                             nr_classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"), 
                             scheduler=scheduler, mixed_prec=get_config(train_config, "training", "mixed_precision", legacy_key="mixed_precision"))
    valid_epoch = ValidEpoch(model, CE_Loss, DICE_Loss, Focal_loss, lambdaa, segmentation, device=DEVICE,
                             display_image=get_config(train_config, "training", "display_image", legacy_key="display_image"), 
                             nr_classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"), 
                             scheduler=scheduler, mixed_prec=get_config(train_config, "training", "mixed_precision", legacy_key="mixed_precision"))

    max_score = 0.0
    best_f1 = 0.0
    best_model_state = None

    # Convert weight_range to a string for naming
    wr_str = f"{weight_range[0]}_{weight_range[1]}"

    for epoch in range(get_config(train_config, "training", "num_epochs", legacy_key="num_epochs")):

        print(f"Epoch {epoch + 1}/{get_config(train_config, "training", "num_epochs", legacy_key="num_epochs")}  (lambda={lambdaa}, sampler={sampler_option}, weights={weight_range}, loss_combo={loss_combination}, lr={lr}, opt={optimizer_choice})")

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        current_iou = valid_logs['iou_score']
        current_f1 = valid_logs.get('f1_score', 0.0)  # Ensure f1_score is available

        if current_iou > max_score:
            max_score = current_iou
            best_f1 = current_f1
            best_model_state = model.state_dict()

            # Save each new best model with all parameters in name
            model_filename = (f"best_model_{get_config(train_config, "model", "version", legacy_key="model_version")}_epoch{epoch}_"
                              f"encoder_{encoder}_seg_{segmentation}_lambda{lambdaa}_opt{optimizer_choice}_lr{lr}_"
                              f"{loss_combination}_wr{wr_str}_sampler{sampler_option}_"
                              f"iou{current_iou:.4f}_f1{current_f1:.4f}.pth")
            torch.save(best_model_state, os.path.join(path, model_filename))
            print(f"Best model saved as {model_filename}!")

        # Print metrics
        print(f"Train Loss: {train_logs['loss']:.4f}, Valid Loss: {valid_logs['loss']:.4f}")
        print(f"Train Acc: {train_logs['accuracy']:.4f}, Valid Acc: {valid_logs['accuracy']:.4f}")
        print(f"Train IoU: {train_logs['iou_score']:.4f}, Valid IoU: {current_iou:.4f}")
        print(f"Valid F1: {current_f1:.4f}")

        # Grad-CAM visualization (every 5 epochs, if enabled)
        if get_config(train_config, 'training', 'gradCAM', legacy_key='gradCAM') and (epoch % 5 == 0):
            os.makedirs('gradcam_outputs', exist_ok=True)
            # Save current model weights
            temp_model_path = 'gradcam_outputs/temp_model.pth'
            torch.save(model.state_dict(), temp_model_path)
            # Create a new model instance and load weights
            if get_config(train_config, "model", "encoder", legacy_key="encoder") == "transformer":
                gradcam_model = UNetWithSwinTransformer(classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"), activation=get_config(train_config, "model", "activation", legacy_key="activation"))
                target_layer = "up4.0"
            else:
                gradcam_model = UNetWithClassification(
                    encoder_name=get_config(train_config, "model", "encoder", legacy_key="encoder"),
                    encoder_weights=get_config(train_config, "model", "encoder_weights", legacy_key="encoder_weights"),
                    classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"),
                    activation= None
                )
                target_layer = "segmentation_model.decoder.seg_blocks.3.block.0.block.2"
            gradcam_model.load_state_dict(torch.load(temp_model_path, map_location=DEVICE))
            gradcam_model = gradcam_model.to(DEVICE)
            gradcam_model.eval()
            # Get a batch from the validation loader
            val_batch = next(iter(valid_loader))
            images = val_batch['image'] if isinstance(val_batch, dict) else val_batch[0]
            # Try to get image_ids for mask lookup
            image_ids_batch = val_batch['image_id'] if isinstance(val_batch, dict) and 'image_id' in val_batch else valid_ids[:images.shape[0]]
            mask_dir = os.path.join(path, "new_masks_640_1280")
            for idx in range(min(2, images.shape[0])):
                input_tensor = images[idx:idx+1].to(DEVICE)
                img_name = image_ids_batch[idx] if idx < len(image_ids_batch) else None
                if img_name is not None:
                    mask_path = os.path.join(mask_dir, img_name)
                    if os.path.exists(mask_path):
                        target_class = get_most_frequent_foreground_class(mask_path)
                    else:
                        target_class = 0
                else:
                    target_class = 0
                save_path = f'gradcam_outputs/gridsearch_epoch{epoch+1}_sample{idx+1}.png'
                visualize_gradcam(gradcam_model, input_tensor, target_class, target_layer, save_path=save_path, show=False)
                print(f"Saved Grad-CAM visualization: {save_path} (target_class={target_class})")

    return max_score, best_f1, best_model_state

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
    path = get_config(train_config, "data", "path", legacy_key="path")
    model_version = get_config(train_config, "model", "version", legacy_key="model_version")

    # Load all image and mask paths
    image_dir = os.path.join(path, "new_images_640_1280")
    valid_extensions = tuple(get_config(train_config, "training", "valid_extensions", legacy_key="valid_extensions"))
    image_ids = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)])

    # Shuffle and split the dataset into training and validation sets
    random.seed(get_config(train_config, "training", "random_seed", legacy_key="random_seed"))  # For reproducibility
    random.shuffle(image_ids)

    split_ratio = get_config(train_config, "training", "split_ratio", legacy_key="split_ratio")
    split_index = int(len(image_ids) * split_ratio)

    train_ids = image_ids[:split_index]
    valid_ids = image_ids[split_index:]

    # OBJECT DETECTION -> YOLO
    detection_model = YOLO(os.path.join(preprocessing_config["yolo_path"], "attempt_1/weights/best.pt"))

    if get_config(train_config, "model", "encoder", legacy_key="encoder") != "transformer":
        preprocessing_fn = get_preprocessing_fn(get_config(train_config, "model", "encoder", legacy_key="encoder"), pretrained=get_config(train_config, "model", "encoder_weights", legacy_key="encoder_weights"))
    else:
        preprocessing_fn = None

    # Hyperparameter sets for grid search
    lambdaa_values = [1.0, 3, 5,10]
    sampler_options = [False]
    weight_ranges = [(50, 200), (70, 120), (40,250)]
    loss_combinations = ['focal+ce', 'dice+ce']
    learning_rates = [1e-4, 5e-4, 1e-3, 3e-4]
    optimizers = ['adamw', 'sgd']

    if get_config(train_config, "training", "grid_search", legacy_key="grid_search"):
        global_best_iou = 0.0
        global_best_f1 = 0.0
        global_best_model_state = None

        for (lambdaa, sampler_option, weight_range, loss_combination, lr, optimizer_choice) in product(
            lambdaa_values, sampler_options, weight_ranges, loss_combinations, learning_rates, optimizers
        ):
            best_iou, best_f1, final_model_state = train_once(
                train_config, preprocessing_config, train_ids, valid_ids, path, preprocessing_fn, detection_model, DEVICE,
                lambdaa, sampler_option, weight_range, loss_combination, lr, optimizer_choice, grid_search=True
            )
            # Track the global best model based on IoU
            if best_iou > global_best_iou and final_model_state is not None:
                global_best_iou = best_iou
                global_best_f1 = best_f1
                global_best_model_state = final_model_state

        # After completing the grid search, save only the best model
        if global_best_model_state is not None:
            final_filename = f"final_model_{global_best_iou:.4f}_{global_best_f1:.4f}.pth"
            torch.save(global_best_model_state, os.path.join(path, final_filename))
            print(f"Global best model from grid search saved as {final_filename}!")
        else:
            print("No improvements found during grid search.")

    else:
        # Run a single training session with the original hyperparameters
        lambdaa = get_config(train_config, "training", "lambda_loss", legacy_key="lambda")
        sampler_option = get_config(train_config, "training", "sampler", legacy_key="sampler")
        weight_range = get_config(train_config, "training", "weight_range_multiclass", legacy_key="weight_range_multiclass")
        loss_combination = 'focal+ce' if get_config(train_config, "training", "use_focal_loss", legacy_key="focal") else 'dice+ce'
        lr = get_config(train_config, "training", "learning_rate", legacy_key="optimizer_lr")
        optimizer_choice = get_config(train_config, "training", "optimizer", legacy_key="optimizer")
        best_iou, best_f1, best_model_state = train_once(
            train_config, preprocessing_config, train_ids, valid_ids, path, preprocessing_fn, detection_model, DEVICE,
            lambdaa, sampler_option, weight_range, loss_combination, lr, optimizer_choice, grid_search=False
        )

        # After single run training completes, we could also save a final model if needed:
        if best_model_state is not None:
            final_filename = f"final_model_{best_iou:.4f}_{best_f1:.4f}.pth"
            torch.save(best_model_state, os.path.join(path, final_filename))
            print(f"Final best model saved as {final_filename}!")

if __name__ == "__main__":
    main()
