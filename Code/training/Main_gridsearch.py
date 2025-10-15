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
from model import UNetWithClassification, UNetWithSwinTransformer, UNetWithViT
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import ssl
import numpy as np
from ultralytics import YOLO
from kornia.losses import FocalLoss
from itertools import product
from visualize_gradcam import visualize_gradcam
from collections import Counter
from PIL import Image
import warnings
import time
import signal
import threading


ssl._create_default_https_context = ssl._create_unverified_context


def get_config(cfg, *keys, legacy_key=None):
    """
    Utility function for config access
    Uses new structured config if available, else fallback to legacy
    """
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
    if isinstance(original_weights, dict):
        original_weights = torch.tensor([original_weights[cls] for cls in sorted(original_weights.keys())])
    
    min_weight, max_weight = weight_range
    rescaled_weights = original_weights.clone()
    non_background_weights = original_weights[1:]

    if len(non_background_weights) > 0:
        min_original_weight = non_background_weights.min()
        max_original_weight = non_background_weights.max()

        if max_original_weight > min_original_weight:
            rescaled_weights[1:] = (non_background_weights - min_original_weight) / (max_original_weight - min_original_weight)
            rescaled_weights[1:] = rescaled_weights[1:] * (max_weight - min_weight) + min_weight
        else:
            rescaled_weights[1:] = min_weight

    rescaled_weights[0] = 1.0
    return rescaled_weights

def worker_init_fn(worker_id):
    """Initialize random seed for each worker"""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)

def get_most_frequent_foreground_class(mask_path):
    """Get the most frequent foreground class from a mask"""
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


def get_vit_model_name(train_config):
    """Get the configured ViT model name from config or fallback to default"""
    vit_config = get_config(train_config, "model", "vit_config")
    if vit_config and "model_name" in vit_config:
        return vit_config["model_name"]
    return "facebook/dinov2-large"  # fallback

def get_vit_target_layer(train_config):
    """Get the configured ViT target layer from config or fallback to default"""
    vit_config = get_config(train_config, "model", "vit_config")
    if vit_config and "target_layer" in vit_config:
        return vit_config["target_layer"]
    return "decoder1.0"  # fallback

def create_model(train_config):
    """Create and return the appropriate model based on configuration"""
    encoder = get_config(train_config, "model", "encoder", legacy_key="encoder")
    
    if encoder == "transformer":
        model = UNetWithSwinTransformer(
            classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"),
            activation=get_config(train_config, "model", "activation", legacy_key="activation")
        )
        use_progressive = False
    elif encoder == "vit":
        model = UNetWithViT(
            classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"),
            activation=get_config(train_config, "model", "activation", legacy_key="activation"),
            model_name=get_vit_model_name(train_config),
            dropout_rate=get_config(train_config, "model", "vit_config", "dropout_rate", legacy_key="dropout_rate")
        )
        use_progressive = True
        
        # ADD DEBUG SECTION
        print("=== Model Parameter Status After Creation ===")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"Frozen params: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        
        if trainable_params == total_params:
            print("⚠️  WARNING: All parameters are trainable! Encoder is NOT frozen!")
        else:
            print("✅ Encoder is properly frozen")
    else:
        model = UNetWithClassification(
            encoder_name=encoder,
            encoder_weights=get_config(train_config, "model", "encoder_weights", legacy_key="encoder_weights"),
            classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"),
            activation=None
        )
        use_progressive = False
    
    return model, use_progressive

def create_datasets(train_config, preprocessing_config, train_ids, valid_ids, path, preprocessing_fn, detection_model, device):
    """Create and return train and validation datasets"""
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
        device=device,        
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
        device=device,
        exclude_images_with_classes=preprocessing_config["exclude_images_with_classes"],
        classes_to_exclude=preprocessing_config["classes_to_exclude"]
    )
    
    return train_dataset, valid_dataset

def create_optimizer_and_scheduler(model, optimizer_choice, lr, train_config, grid_search):
    """Create optimizer and scheduler"""
    if optimizer_choice == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=get_config(train_config, "training", "weight_decay", legacy_key="weight_decay"))
    elif optimizer_choice == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError("Supported optimizers: 'adamw', 'sgd'")
    
    # ADD COSINE ANNEALING OPTION
    scheduler_type = get_config(train_config, "training", "scheduler_type", legacy_key="scheduler_type")
    
    if scheduler_type == "cosine_restarts":
        cosine_config = get_config(train_config, "training", "cosine_restart_config") or {}
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cosine_config.get("T_0", 15),
            T_mult=cosine_config.get("T_mult", 1),
            eta_min=cosine_config.get("eta_min", 1e-7)
        )
    else:
        # Original exponential scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=get_config(train_config, "training", "non_grid_search", "lr_scheduler_gamma", legacy_key="lr_scheduler_gamma")
            if not grid_search else get_config(train_config, "training", "lr_scheduler_gamma", legacy_key="lr_scheduler_gamma")
        )
    
    return optimizer, scheduler

def create_loss_functions(train_config, preprocessing_config, loss_combination, weight_range, path, device):
    """Create and return loss functions"""
    # Load weights
    non_scaled_weights = torch.load(os.path.join(path, "class_weights.pth"), weights_only=True).float().to(device)
    class_weights_multiclass = rescale_weights(non_scaled_weights, weight_range=weight_range)
    class_weights_binary = torch.tensor(get_config(train_config, "model", "class_weights", legacy_key="class_weights")).float().to(device)

    # Determine loss flags
    focal_loss_flag = loss_combination == 'focal+ce'
    dice_loss_flag = loss_combination == "dice+ce"

    if preprocessing_config["segmentation"] == "binary":
        CE_Loss = nn.CrossEntropyLoss(weight=class_weights_binary)
    else:
        CE_Loss = nn.CrossEntropyLoss(weight=class_weights_multiclass)

    DICE_Loss = smp.losses.DiceLoss(mode="multiclass") if dice_loss_flag else None

    focal_cfg = get_config(train_config, "training", "focal_loss") or {}
    Focal_loss = FocalLoss(
        alpha=focal_cfg.get("alpha", 0.5),
        gamma=focal_cfg.get("gamma", 4.0),
        reduction=focal_cfg.get("reduction", "mean"),
        weight=class_weights_multiclass
    ) if focal_loss_flag else None

    return CE_Loss, DICE_Loss, Focal_loss

def create_data_loaders(train_dataset, valid_dataset, train_config, preprocessing_config, train_ids, originals_fnames, image_weights_tensor, sampler_option):
    """Create and return data loaders with optional weighted sampling"""
    def entry_to_orig_filename(entry):
        name = entry[1] if isinstance(entry, (list, tuple)) else entry
        base, ext = os.path.splitext(name)
        if base.endswith("_gen"):
            base = base[:-4]
        return base + ext

    # Handle sampler
    sampler = None
    if sampler_option:
        if preprocessing_config["exclude_images_with_classes"]:
            train_ids = train_dataset.image_ids

        image_id_to_weight = {fname: w.item() for fname, w in zip(originals_fnames, image_weights_tensor)}
        default_w = float(torch.mean(image_weights_tensor).item())
        train_weights = [image_id_to_weight.get(entry_to_orig_filename(e), default_w) for e in train_ids]
        sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=get_config(train_config, "training", "batch_size", legacy_key="batch_size"), 
        shuffle=False, 
        num_workers=get_config(train_config, "training", "num_workers", legacy_key="num_workers"), 
        worker_init_fn=worker_init_fn, 
        persistent_workers=True, 
        sampler=sampler
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=get_config(train_config, "training", "batch_size", legacy_key="batch_size"), 
        shuffle=False, 
        num_workers=get_config(train_config, "training", "num_workers", legacy_key="num_workers"), 
        worker_init_fn=worker_init_fn, 
        persistent_workers=True
    )
    
    return train_loader, valid_loader

class TrainingController:
    def __init__(self, train_config):
        self.display_image = get_config(train_config, "training", "display_image", legacy_key="display_image")
        self.save_models = get_config(train_config, "training", "save_intermediate_models", legacy_key="save_models")
        self.run_gradcam = get_config(train_config, "training", "gradCAM", legacy_key="gradCAM")
        
    def toggle_display(self):
        self.display_image = not self.display_image
        print(f"Image display {'enabled' if self.display_image else 'disabled'}")
    
    def toggle_saving(self):
        self.save_models = not self.save_models
        print(f"Model saving {'enabled' if self.save_models else 'disabled'}")
    
    def toggle_gradcam(self):
        self.run_gradcam = not self.run_gradcam
        print(f"GradCAM {'enabled' if self.run_gradcam else 'disabled'}")


def setup_keyboard_controls():
    """Setup keyboard controls for training"""
    def signal_handler(signum, frame):
        print("\n=== Training Controls ===")
        print("Press 'd' + Enter to toggle image display")
        print("Press 's' + Enter to toggle model saving") 
        print("Press 'g' + Enter to toggle GradCAM")
        print("Press 'c' + Enter to continue training")
        print("Press 'q' + Enter to quit")
        
        while True:
            try:
                choice = input("Choice: ").lower().strip()
                if choice == 'd':
                    training_controller.toggle_display()
                    break
                elif choice == 's':
                    training_controller.toggle_saving()
                    break
                elif choice == 'g':
                    training_controller.toggle_gradcam()
                    break
                elif choice == 'c':
                    print("Continuing training...")
                    break
                elif choice == 'q':
                    print("Exiting training...")
                    exit(0)
                else:
                    print("Invalid choice. Try again.")
            except KeyboardInterrupt:
                print("\nExiting training...")
                exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

def create_epoch_runners(model, CE_Loss, DICE_Loss, Focal_loss, lambdaa, segmentation, optimizer, scheduler, train_config, device):
    """Create train and validation epoch runners"""
    train_epoch = TrainEpoch(
        model, CE_Loss, DICE_Loss, Focal_loss, lambdaa, segmentation, optimizer, device=device,
        grad_clip_value=get_config(train_config, "training", "grad_clip_value", legacy_key="grad_clip_value"),
        display_image=training_controller.display_image, 
        nr_classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"), 
        scheduler=scheduler, 
        mixed_prec=get_config(train_config, "training", "mixed_precision", legacy_key="mixed_precision")
    )
    
    valid_epoch = ValidEpoch(
        model, CE_Loss, DICE_Loss, Focal_loss, lambdaa, segmentation, device=device,
        display_image=training_controller.display_image, 
        nr_classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"), 
        scheduler=scheduler, 
        mixed_prec=get_config(train_config, "training", "mixed_precision", legacy_key="mixed_precision")
    )
    
    return train_epoch, valid_epoch

def save_best_model(model_state, train_config, encoder, segmentation, lambdaa, optimizer_choice, lr, loss_combination, weight_range, sampler_option, epoch, iou, f1, path, stage=""):
    """Save the best model with descriptive filename"""

    if not training_controller.save_models:
        print("Model saving is disabled. Skipping save.")
        return
    
    wr_str = f"{weight_range[0]}_{weight_range[1]}"
    stage_str = f"_{stage}" if stage else ""
    
    model_filename = (f"best_model_{get_config(train_config, 'model', 'version', legacy_key='model_version')}{stage_str}_epoch{epoch}_"
                      f"encoder_{encoder}_seg_{segmentation}_lambda{lambdaa}_opt{optimizer_choice}_lr{lr}_"
                      f"{loss_combination}_wr{wr_str}_sampler{sampler_option}_"
                      f"iou{iou:.4f}_f1{f1:.4f}.pth")
    
    os.makedirs(f'{path}/models', exist_ok=True)
    torch.save(model_state, os.path.join(f'{path}/models', model_filename))
    print(f"Best model saved as {model_filename}!")
    return model_filename

def run_gradcam_visualization(model, train_config, valid_loader, valid_ids, path, device, epoch, stage=""):
    """Run GradCAM visualization if enabled"""
    if not training_controller.run_gradcam:
        print("GradCAM is disabled. Skipping GradCAM visualization.")
        return
    
    if not get_config(train_config, 'training', 'gradCAM', legacy_key='gradCAM') or epoch % 5 != 0:
        return

    def _entry_to_mask_filename(entry):
        """Convert entry to mask filename"""
        if isinstance(entry, (list, tuple)):
            if len(entry) == 3:
                return entry[2]
            elif len(entry) == 2:
                name = entry[1]
            else:
                name = str(entry)
        else:
            name = entry
        base, _ext = os.path.splitext(name)
        if base.endswith("_gen"):
            base = base[:-4]
        return base + ".png"
    
    # Setup directories and get encoder type
    os.makedirs('gradcam_outputs', exist_ok=True)
    encoder = get_config(train_config, "model", "encoder", legacy_key="encoder")
    
    # Get target layer based on encoder type
    if encoder == "vit":
        target_layer = get_vit_target_layer(train_config)
    elif encoder == "transformer":
        target_layer = "up4.0"
    else:
        target_layer = "segmentation_model.decoder.seg_blocks.3.block.0.block.2"
    
    # Get stage string and validation batch
    stage_str = f"_{stage}" if stage else ""
    val_batch = next(iter(valid_loader))
    images = val_batch['image'] if isinstance(val_batch, dict) else val_batch[0]
    image_ids_batch = val_batch['image_id'] if isinstance(val_batch, dict) and 'image_id' in val_batch else valid_ids[:images.shape[0]]
    mask_dir = os.path.join(path, "new_masks_640_1280")
    
    # Create separate model instance for GradCAM to avoid hook conflicts
    try:
        if encoder == "transformer":
            gradcam_model = UNetWithSwinTransformer(
                classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"),
                activation=get_config(train_config, "model", "activation", legacy_key="activation")
            )
        elif encoder == "vit":
            gradcam_model = UNetWithViT(
                classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"),
                activation=get_config(train_config, "model", "activation", legacy_key="activation"),
                model_name=get_vit_model_name(train_config)
            )
        else:
            # Standard segmentation models (efficientnet, resnet, etc.)
            gradcam_model = UNetWithClassification(
                encoder_name=encoder,
                encoder_weights=get_config(train_config, "model", "encoder_weights", legacy_key="encoder_weights"),
                classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"),
                activation=None
            )
        
        # Load current weights and set to evaluation mode
        gradcam_model.load_state_dict(model.state_dict())
        gradcam_model = gradcam_model.to(device)
        gradcam_model.eval()
        
        # Process samples for GradCAM
        for idx in range(min(2, images.shape[0])):
            input_tensor = images[idx:idx+1].to(device)
            entry = image_ids_batch[idx] if idx < len(image_ids_batch) else None
            
            # Determine target class
            if entry is not None:
                mask_name = _entry_to_mask_filename(entry)
                mask_path = os.path.join(mask_dir, mask_name)
                target_class = get_most_frequent_foreground_class(mask_path) if os.path.exists(mask_path) else 0
            else:
                target_class = 0

            save_path = f'gradcam_outputs/gridsearch{stage_str}_epoch{epoch+1}_sample{idx+1}.png'
            
            # Generate GradCAM visualization
            try:
                visualize_gradcam(gradcam_model, input_tensor, target_class, target_layer, save_path=save_path, show=False)
                print(f"Saved Grad-CAM visualization: {save_path} (target_class={target_class})")
            except Exception as e:
                print(f"GradCAM failed for {save_path}: {e}")
                
    except Exception as e:
        print(f"Failed to create GradCAM model for encoder '{encoder}': {e}")
        
    finally:
        # Clean up the separate model
        if 'gradcam_model' in locals():
            del gradcam_model
        torch.cuda.empty_cache()
def train_single_epoch(model, train_epoch, valid_epoch, train_loader, valid_loader, epoch, total_epochs, hyperparams, stage=""):
    """Train a single epoch and return logs"""
    stage_str = f" ({stage})" if stage else ""
    print(f"Epoch {epoch + 1}/{total_epochs}{stage_str}  (lambda={hyperparams['lambdaa']}, sampler={hyperparams['sampler_option']}, "
          f"weights={hyperparams['weight_range']}, loss_combo={hyperparams['loss_combination']}, "
          f"lr={hyperparams['lr']}, opt={hyperparams['optimizer_choice']})")
    
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    current_iou = valid_logs['iou_score']
    current_f1 = valid_logs.get('f1_score', 0.0)

    # Print metrics
    print(f"Train Loss: {train_logs['loss']:.4f}, Valid Loss: {valid_logs['loss']:.4f}")
    print(f"Train Acc: {train_logs['accuracy']:.4f}, Valid Acc: {valid_logs['accuracy']:.4f}")
    print(f"Train IoU: {train_logs['iou_score']:.4f}, Valid IoU: {current_iou:.4f}")
    print(f"Valid F1: {current_f1:.4f}")

    return current_iou, current_f1
def train_progressive(model, train_epoch, valid_epoch, train_loader, valid_loader, total_epochs, hyperparams, train_config, valid_ids, path, device):
    """Progressive training for ViT models"""
    print("Starting progressive training for ViT model...")
    
    max_score = 0.0
    best_f1 = 0.0
    best_model_state = None

    vit_config = get_config(train_config, "model", "vit_config")

    # Define stage epochs
    stage1_epochs = vit_config.get("stage1_epochs", 40)  # Default 40 epochs for Stage 1
    stage2_epochs = vit_config.get("stage2_epochs", 30)  # Default 30 epochs for Stage 2  
    stage3_epochs = total_epochs - stage1_epochs - stage2_epochs  # Remaining epochs for Stage 3
    
    # Stage 1: Frozen encoder
    print(f"Stage 1: Training with frozen encoder for {stage1_epochs} epochs...")
    for epoch in range(stage1_epochs):
        current_iou, current_f1 = train_single_epoch(
            model, train_epoch, valid_epoch, train_loader, valid_loader, 
            epoch, stage1_epochs, hyperparams, "Stage 1"
        )
        
        if current_iou > max_score:
            max_score = current_iou
            best_f1 = current_f1
            best_model_state = model.state_dict()

            save_best_model(best_model_state, train_config, hyperparams['encoder'], hyperparams['segmentation'],
                          hyperparams['lambdaa'], hyperparams['optimizer_choice'], hyperparams['lr'],
                          hyperparams['loss_combination'], hyperparams['weight_range'], hyperparams['sampler_option'],
                          epoch, current_iou, current_f1, path, "stage1")

        run_gradcam_visualization(model, train_config, valid_loader, valid_ids, path, device, epoch, "stage1")
    
    # Stage 2: Unfreeze encoder
    print("Stage 2: Unfreezing encoder for full model fine-tuning...")
    if hasattr(model, 'unfreeze_encoder'):
        # CHANGE THIS LINE - add configuration option
        vit_config = get_config(train_config, "model", "vit_config") or {}
        unfreeze_layers = vit_config.get("unfreeze_layers_stage2", "all")  # "all" or number
        
        if unfreeze_layers == "all":
            model.unfreeze_encoder(num_layers=None)  # Unfreeze ALL layers
        else:
            model.unfreeze_encoder(num_layers=int(unfreeze_layers))  # Unfreeze specific number
        
        print("=== Model Parameter Status After Unfreezing ===")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"After unfreezing - Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    

    # Create Stage 2 optimizer with reduced learning rate
    stage2_lr_factor = vit_config.get("stage2_lr_factor", 0.5)  # 50% of original LR
    stage2_lr = hyperparams['lr'] * stage2_lr_factor
    
    if hyperparams['optimizer_choice'] == 'adamw':
        optimizer_stage2 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=stage2_lr, 
            weight_decay=get_config(train_config, "training", "weight_decay", legacy_key="weight_decay")
        )
    else:
        optimizer_stage2 = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=stage2_lr, momentum=0.9, weight_decay=1e-4
        )
    
    scheduler_stage2 = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_stage2,
        gamma=get_config(train_config, "training", "lr_scheduler_gamma", legacy_key="lr_scheduler_gamma")
    )
    
    # Recreate loss functions and epoch runners for Stage 2
    CE_Loss, DICE_Loss, Focal_loss = create_loss_functions(
        train_config, {'segmentation': hyperparams['segmentation']},
        hyperparams['loss_combination'], hyperparams['weight_range'], path, device
    )
    
    train_epoch_s2, valid_epoch_s2 = create_epoch_runners(
        model, CE_Loss, DICE_Loss, Focal_loss, hyperparams['lambdaa'], 
        hyperparams['segmentation'], optimizer_stage2, scheduler_stage2, train_config, device
    )
    
    # Stage 2 hyperparams
    hyperparams_s2 = hyperparams.copy()
    hyperparams_s2['lr'] = stage2_lr
    
    for epoch in range(stage2_epochs):
        current_iou, current_f1= train_single_epoch(
            model, train_epoch_s2, valid_epoch_s2, train_loader, valid_loader,
            epoch, stage2_epochs, hyperparams_s2, "Stage2"
        )
        
        if current_iou > max_score:
            max_score = current_iou
            best_f1 = current_f1
            best_model_state = model.state_dict()
            save_best_model(best_model_state, train_config, hyperparams['encoder'], hyperparams['segmentation'],
                          hyperparams['lambdaa'], hyperparams['optimizer_choice'], stage2_lr,
                          hyperparams['loss_combination'], hyperparams['weight_range'], hyperparams['sampler_option'],
                          stage1_epochs + epoch, current_iou, current_f1, path, "stage2")
        
        run_gradcam_visualization(model, train_config, valid_loader, valid_ids, path, device, stage1_epochs + epoch, "stage2")
    
    # ===== STAGE 3: More encoder unfreezing =====
    print(f"\n🎯 STAGE 3: Extended encoder unfreezing for {stage3_epochs} epochs...")
    
    # Unfreeze more layers in Stage 3
    if hasattr(model, 'unfreeze_encoder'):
        unfreeze_layers_stage3 = vit_config.get("unfreeze_layers_stage3", 8)  # More layers in Stage 3
        model.unfreeze_encoder(num_layers=unfreeze_layers_stage3)
        print(f"   Unfroze {unfreeze_layers_stage3} encoder layers total")
        
        # Debug parameter status
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Stage 3 - Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Create Stage 3 optimizer with even lower learning rate
    stage3_lr_factor = vit_config.get("stage3_lr_factor", 0.2)  # 20% of original LR
    stage3_lr = hyperparams['lr'] * stage3_lr_factor
    
    if hyperparams['optimizer_choice'] == 'adamw':
        optimizer_stage3 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=stage3_lr, 
            weight_decay=get_config(train_config, "training", "weight_decay", legacy_key="weight_decay") * 1.5  # Higher weight decay
        )
    else:
        optimizer_stage3 = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=stage3_lr, momentum=0.9, weight_decay=1e-4
        )
    
    scheduler_stage3 = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_stage3,
        gamma=get_config(train_config, "training", "lr_scheduler_gamma", legacy_key="lr_scheduler_gamma")
    )
    
    # Recreate epoch runners for Stage 3
    train_epoch_s3, valid_epoch_s3 = create_epoch_runners(
        model, CE_Loss, DICE_Loss, Focal_loss, hyperparams['lambdaa'], 
        hyperparams['segmentation'], optimizer_stage3, scheduler_stage3, train_config, device
    )
    
    # Stage 3 hyperparams
    hyperparams_s3 = hyperparams.copy()
    hyperparams_s3['lr'] = stage3_lr
    
    # Reset early stopping for Stage 3
    consecutive_large_gaps = 0
    max_allowed_gap_s3 = 0.08  # Very strict gap limit for Stage 3
    
    for epoch in range(stage3_epochs):
        current_iou, current_f1= train_single_epoch(
            model, train_epoch_s3, valid_epoch_s3, train_loader, valid_loader,
            epoch, stage3_epochs, hyperparams_s3, "Stage3"
        )
        
        if current_iou > max_score:
            max_score = current_iou
            best_f1 = current_f1
            best_model_state = model.state_dict()
            save_best_model(best_model_state, train_config, hyperparams['encoder'], hyperparams['segmentation'],
                          hyperparams['lambdaa'], hyperparams['optimizer_choice'], stage3_lr,
                          hyperparams['loss_combination'], hyperparams['weight_range'], hyperparams['sampler_option'],
                          stage1_epochs + stage2_epochs + epoch, current_iou, current_f1, path, "stage3")
        
        run_gradcam_visualization(model, train_config, valid_loader, valid_ids, path, device, 
                                stage1_epochs + stage2_epochs + epoch, "stage3")
    
    print(f"\n✅ 3-stage training completed! Best IoU: {max_score:.4f}, Best F1: {best_f1:.4f}")
   
    return max_score, best_f1, best_model_state

def train_standard(model, train_epoch, valid_epoch, train_loader, valid_loader, total_epochs, hyperparams, train_config, valid_ids, path, device):
    """Standard training for non-ViT models"""
    max_score = 0.0
    best_f1 = 0.0
    best_model_state = None
    
    for epoch in range(total_epochs):
        current_iou, current_f1 = train_single_epoch(
            model, train_epoch, valid_epoch, train_loader, valid_loader,
            epoch, total_epochs, hyperparams, stage= "Standard"
        )
        
        if current_iou > max_score:
            max_score = current_iou
            best_f1 = current_f1
            best_model_state = model.state_dict()
            save_best_model(best_model_state, train_config, hyperparams['encoder'], hyperparams['segmentation'],
                          hyperparams['lambdaa'], hyperparams['optimizer_choice'], hyperparams['lr'],
                          hyperparams['loss_combination'], hyperparams['weight_range'], hyperparams['sampler_option'],
                          epoch, current_iou, current_f1, path)
        
        run_gradcam_visualization(model, train_config, valid_loader, valid_ids, path, device, epoch)


    return max_score, best_f1, best_model_state

def train_once(train_config, preprocessing_config, train_ids, valid_ids, path, preprocessing_fn, detection_model, device,
               lambdaa, sampler_option, weight_range, loss_combination, lr, optimizer_choice, grid_search=False, originals_fnames=None):
    """Main training function - now much more modular"""
    
    # Check for missing generated masks
    for entry in train_ids:
        fname = entry[1] if isinstance(entry, (list, tuple)) else entry
        if isinstance(fname, tuple):
            fname = fname[1]
        if isinstance(entry, (list, tuple)) and len(entry) == 3:
            continue
        if isinstance(fname, str) and (fname.endswith("_gen.png") or fname.endswith("_gen.jpg")):
            mask_name = fname.replace("_gen.png", ".png").replace("_gen.jpg", ".jpg")
            mask_path = os.path.join(path, "new_masks_640_1280", mask_name)
            if not os.path.exists(mask_path):
                warnings.warn(f"[Dataset] No mask found for generated image {fname}", RuntimeWarning)

    # Create model
    model, use_progressive_training = create_model(train_config)
    model = model.to(device)
    
    # Create datasets
    train_dataset, valid_dataset = create_datasets(
        train_config, preprocessing_config, train_ids, valid_ids, path, preprocessing_fn, detection_model, device
    )
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, optimizer_choice, lr, train_config, grid_search)
    
    # Create loss functions
    CE_Loss, DICE_Loss, Focal_loss = create_loss_functions(
        train_config, preprocessing_config, loss_combination, weight_range, path, device
    )
    
    # Load image weights for sampling
    image_weights_tensor = torch.load(os.path.join(path, "image_weights.pth"), weights_only=True).float().to(device)
    
    # Create data loaders
    train_loader, valid_loader = create_data_loaders(
        train_dataset, valid_dataset, train_config, preprocessing_config, train_ids, originals_fnames, image_weights_tensor, sampler_option
    )
    
    # Create epoch runners
    train_epoch, valid_epoch = create_epoch_runners(
        model, CE_Loss, DICE_Loss, Focal_loss, lambdaa, preprocessing_config["segmentation"],
        optimizer, scheduler, train_config, device
    )
    
    # Prepare hyperparameters dict
    hyperparams = {
        'lambdaa': lambdaa,
        'sampler_option': sampler_option,
        'weight_range': weight_range,
        'loss_combination': loss_combination,
        'lr': lr,
        'optimizer_choice': optimizer_choice,
        'encoder': get_config(train_config, "model", "encoder", legacy_key="encoder"),
        'segmentation': preprocessing_config["segmentation"]
    }
    
    total_epochs = get_config(train_config, "training", "num_epochs", legacy_key="num_epochs")
    
    # Train based on model type
    if use_progressive_training:
        max_score, best_f1, best_model_state = train_progressive(
            model, train_epoch, valid_epoch, train_loader, valid_loader, total_epochs,
            hyperparams, train_config, valid_ids, path, device
        )
    else:
        max_score, best_f1, best_model_state = train_standard(
            model, train_epoch, valid_epoch, train_loader, valid_loader, total_epochs,
            hyperparams, train_config, valid_ids, path, device
        )
    
    return max_score, best_f1, best_model_state

def setup_data_splits(train_config, path):
    """Setup train/validation data splits with optional randomness"""
    image_dir = os.path.join(path, "new_images_640_1280")
    valid_extensions = tuple(get_config(train_config, "training", "valid_extensions", legacy_key="valid_extensions"))
    image_ids = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)])

    # ADD RANDOMNESS: Use current time if random_split enabled
    base_seed = get_config(train_config, "training", "random_seed", legacy_key="random_seed")
    use_random_split = get_config(train_config, "training", "random_split", legacy_key="random_split")
    
    if use_random_split:
        import time
        actual_seed = int(time.time() * 1000) % 100000  # Different seed each run
        print(f"🎲 Using random split with seed: {actual_seed}")
    else:
        actual_seed = base_seed
        print(f"🔒 Using fixed split with seed: {actual_seed}")
    
    random.seed(actual_seed)
    random.shuffle(image_ids)

    split_ratio = get_config(train_config, "training", "split_ratio", legacy_key="split_ratio")
    split_index = int(len(image_ids) * split_ratio)

    train_ids = [(image_dir, f) for f in image_ids[:split_index]]
    valid_ids = [(image_dir, f) for f in image_ids[split_index:]]
    originals_fnames = [f for _, f in train_ids + valid_ids]

    # Add generated images if enabled
    if get_config(train_config, "training", "use_diffusion_images"):
        gen_image_dir = os.path.join(path, "generated_samples")
        if os.path.isdir(gen_image_dir):
            gen_images = []
            for f in sorted(os.listdir(gen_image_dir)):
                if f.lower().endswith(valid_extensions):
                    mask_name = f.replace("_gen.png", ".png").replace("_gen.jpg", ".jpg")
                    gen_images.append((gen_image_dir, f, mask_name))
            train_ids.extend(gen_images)

    return train_ids, valid_ids, originals_fnames


def run_grid_search(train_config, preprocessing_config, train_ids, valid_ids, path, preprocessing_fn, detection_model, device, originals_fnames):
    """Run grid search over hyperparameters"""
    gs_params = get_config(train_config, "training", "grid_search_parameters") or {}
    ngs = get_config(train_config, "training", "non_grid_search") or {}
    
    learning_rates = gs_params.get("learning_rate", [1e-4, 3e-4, 1e-3])
    optimizers = gs_params.get("optimizer", ["adamw", "sgd"])
    gammas = gs_params.get("lr_scheduler_gamma", [0.99, 0.999, 0.9999])
    use_focal_list = gs_params.get("use_focal_loss", [True, False])
    loss_functions = gs_params.get("loss_functions", ["dice", "focal_loss"])
    lambda_list = gs_params.get("lambda_loss", [1, 5, 10])

    global_best_iou = 0.0
    global_best_f1 = 0.0
    global_best_model_state = None

    for (lr, optimizer_choice, gamma, use_focal, lf, lambdaa) in product(
        learning_rates, optimizers, gammas, use_focal_list, loss_functions, lambda_list
    ):
        loss_combo = "focal+ce" if use_focal else "dice+ce"
        weight_range = tuple(ngs.get("weight_range_multiclass", [50, 200]))
        train_config["training"]["lr_scheduler_gamma"] = gamma

        print(f"[Grid] lr={lr}, opt={optimizer_choice}, gamma={gamma}, use_focal={use_focal}, lf={lf}, lambda={lambdaa}, weight_range={weight_range}")
        
        best_iou, best_f1, final_model_state = train_once(
            train_config, preprocessing_config, train_ids, valid_ids, path, preprocessing_fn, detection_model, device,
            lambdaa, get_config(train_config, "training", "sampler"), weight_range, loss_combo, lr, optimizer_choice, 
            grid_search=True, originals_fnames=originals_fnames
        )
        
        if best_iou > global_best_iou and final_model_state is not None:
            global_best_iou = best_iou
            global_best_f1 = best_f1
            global_best_model_state = final_model_state

    return global_best_iou, global_best_f1, global_best_model_state

def run_single_training(train_config, preprocessing_config, train_ids, valid_ids, path, preprocessing_fn, detection_model, device, originals_fnames):
    """Run single training configuration"""
    ngs = get_config(train_config, "training", "non_grid_search") or {}
    
    lambdaa = ngs.get("lambda_loss", 5)
    sampler_option = get_config(train_config, "training", "sampler", legacy_key="sampler")
    weight_range = tuple(ngs.get("weight_range_multiclass", [50, 200]))
    use_focal_loss = bool(ngs.get("use_focal_loss", True))
    lr = ngs.get("learning_rate", 3e-4)
    optimizer_choice = ngs.get("optimizer", "adamw")
    loss_combination = 'focal+ce' if use_focal_loss else 'dice+ce'

    train_config["training"]["lr_scheduler_gamma"] = ngs.get("lr_scheduler_gamma", 0.999)

    print(f"[Run] lr={lr}, opt={optimizer_choice}, gamma={train_config['training']['lr_scheduler_gamma']}, "
          f"use_focal={use_focal_loss}, lambda={lambdaa}, weight_range={weight_range}, loss_combo={loss_combination}")
    
    return train_once(
        train_config, preprocessing_config, train_ids, valid_ids, path, preprocessing_fn, detection_model, device,
        lambdaa, sampler_option, weight_range, loss_combination, lr, optimizer_choice, 
        grid_search=False, originals_fnames=originals_fnames
    )

def print_training_setup(train_config, preprocessing_config):
    """Print training configuration"""
    print("\n==== Training Setup ====")
    print(f"Batch size: {get_config(train_config, 'training', 'batch_size')}")
    print(f"Num epochs: {get_config(train_config, 'training', 'num_epochs')}")
    print(f"Segmentation: {preprocessing_config.get('segmentation')}")
    print(f"Encoder: {get_config(train_config, 'model', 'encoder')}")
    print(f"Encoder weights: {get_config(train_config, 'model', 'encoder_weights')}")
    print(f"Display image: {get_config(train_config, 'training', 'display_image')}")
    print(f"Num workers: {get_config(train_config, 'training', 'num_workers')}")
    print(f"Mixed precision: {get_config(train_config, 'training', 'mixed_precision')}")
    print(f"Use diffusion images: {get_config(train_config, 'training', 'use_diffusion_images')}")
    print(f"GradCAM: {get_config(train_config, 'training', 'gradCAM')}")
    print(f"Metrics: {get_config(train_config, 'training', 'metrics')}")
    print(f"Sampler: {get_config(train_config, 'training', 'sampler')}")
    print(f"Split ratio: {get_config(train_config, 'training', 'split_ratio')}")
    print(f"Grad clip value: {get_config(train_config, 'training', 'grad_clip_value')}")
    print(f"Grid-search enabled: {get_config(train_config, 'training', 'grid_search_enabled')}")
    print("==== ============== ====\n")

def main():
    """Main training orchestration function"""
    # Load configurations
    with open('Code/configs/training_config.json') as f:
        train_config = json.load(f)

    with open('Code/configs/preprocessing_config.json') as f:
        preprocessing_config = json.load(f)

    # Setup training controller and keyboard controls
    global training_controller
    training_controller = TrainingController(train_config)

    setup_keyboard_controls()


    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Initializing Device: {device}')

    # Print setup
    print_training_setup(train_config, preprocessing_config)
    time.sleep(2)

    # Setup paths and data
    path = get_config(train_config, "data", "path", legacy_key="path")
    train_ids, valid_ids, originals_fnames = setup_data_splits(train_config, path)

    # Setup YOLO and preprocessing
    detection_model = YOLO(os.path.join(preprocessing_config["yolo_path"], "attempt_1/weights/best.pt"))
    
    if get_config(train_config, "model", "encoder", legacy_key="encoder") not in ["transformer", "vit"]:
        preprocessing_fn = get_preprocessing_fn(
            get_config(train_config, "model", "encoder", legacy_key="encoder"), 
            pretrained=get_config(train_config, "model", "encoder_weights", legacy_key="encoder_weights")
        )
    else:
        preprocessing_fn = None


    # press ctryl+c to access training controls
    print("🎮 TRAINING CONTROLS: Press Ctrl+C anytime during training to access controls!")

    # Run training
    grid_enabled = bool(get_config(train_config, "training", "grid_search_enabled"))
    
    if grid_enabled:
        best_iou, best_f1, best_model_state = run_grid_search(
            train_config, preprocessing_config, train_ids, valid_ids, path, preprocessing_fn, detection_model, device, originals_fnames
        )
    else:
        best_iou, best_f1, best_model_state = run_single_training(
            train_config, preprocessing_config, train_ids, valid_ids, path, preprocessing_fn, detection_model, device, originals_fnames
        )

    # Save final model
    if best_model_state is not None:
        final_filename = f"final_model_{best_iou:.4f}_{best_f1:.4f}.pth"
        torch.save(best_model_state, os.path.join(path, final_filename))
        print(f"Final best model saved as {final_filename}!")

if __name__ == "__main__":
    main()