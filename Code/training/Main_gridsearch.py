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
import warnings
import time

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
               lambdaa, sampler_option, weight_range, loss_combination, lr, optimizer_choice, grid_search=False, originals_fnames=None):
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

    # Warn if any _gen mask is missing
    for entry in train_ids:
        fname = entry[1] if isinstance(entry, (list, tuple)) else entry
        if isinstance(fname, tuple):  # defensive
            fname = fname[1]
        if isinstance(entry, (list, tuple)) and len(entry) == 3:
            # we already have explicit mask name, skip check here
            continue
        if isinstance(fname, str) and (fname.endswith("_gen.png") or fname.endswith("_gen.jpg")):
            mask_name = fname.replace("_gen.png", ".png").replace("_gen.jpg", ".jpg")
            mask_path = os.path.join(path, "new_masks_640_1280", mask_name)
            if not os.path.exists(mask_path):
                warnings.warn(f"[Dataset] No mask found for generated image {fname}", RuntimeWarning)

    # Define model
    if get_config(train_config, "model", "encoder", legacy_key="encoder") == "transformer":
        model = UNetWithSwinTransformer(
            classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"),
            activation=get_config(train_config, "model", "activation", legacy_key="activation")
        )
    else:
        model = UNetWithClassification(
            encoder_name=get_config(train_config, "model", "encoder", legacy_key="encoder"),
            encoder_weights=get_config(train_config, "model", "encoder_weights", legacy_key="encoder_weights"),
            classes=get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes"),
            activation=None
        )
    model = model.to(DEVICE)

    # Define optimizer
    if optimizer_choice == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_choice == "sgd":  # 'sgd'
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError("Supported optimizers: 'adamw', 'sgd' ")
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=get_config(train_config, "training", "non_grid_search", "lr_scheduler_gamma", legacy_key="lr_scheduler_gamma")
        if not grid_search else get_config(train_config, "training", "lr_scheduler_gamma", legacy_key="lr_scheduler_gamma")
    )
    encoder = get_config(train_config, "model", "encoder", legacy_key="encoder")
    segmentation = preprocessing_config["segmentation"]

    # Loading weights
    non_scaled_weights = torch.load(os.path.join(path, "class_weights.pth"), weights_only=True).float().to(DEVICE)
    image_weights_tensor = torch.load(os.path.join(path, "image_weights.pth"), weights_only=True).float().to(DEVICE)

    class_weights_multiclass = rescale_weights(non_scaled_weights, weight_range=weight_range)
    class_weights_binary = torch.tensor(get_config(train_config, "model", "class_weights", legacy_key="class_weights")).float().to(DEVICE)

    # Determine loss combination (keep your two presets)
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

    focal_cfg = get_config(train_config, "training", "focal_loss") or {}
    focal_alpha = focal_cfg.get("alpha", 0.5)
    focal_gamma = focal_cfg.get("gamma", 4.0)
    focal_reduction = focal_cfg.get("reduction", "mean")
    Focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=focal_reduction, weight=class_weights_multiclass) if focal_loss_flag else None

    # Handling sampler
    def entry_to_orig_filename(entry):
        name = entry[1] if isinstance(entry, (list, tuple)) else entry
        base, ext = os.path.splitext(name)
        if base.endswith("_gen"):
            base = base[:-4]
        return base + ext
    
    #Handling grad-cam
    def _entry_to_mask_filename(entry):
        # Accepts: str | (folder, image) | (folder, image, mask)
        import os
        if isinstance(entry, (list, tuple)):
            if len(entry) == 3:
                # We stored the original mask name explicitly for generated samples
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

    assert originals_fnames is not None, "originals_fnames must be provided"
    assert len(image_weights_tensor) == len(originals_fnames), "Mismatch between image weights and original images"
    image_id_to_weight = {fname: w.item() for fname, w in zip(originals_fnames, image_weights_tensor)}

    image_ids_total = train_ids + valid_ids

    if sampler_option:

        if(preprocessing_config["exclude_images_with_classes"]): #if we excluded some image ids, we have to take this into account for the sampler
            train_ids = train_dataset.image_ids

        default_w = float(torch.mean(image_weights_tensor).item())
        train_weights = [image_id_to_weight.get(entry_to_orig_filename(e), default_w) for e in train_ids]
        sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)
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
            os.makedirs(f'{path}/models', exist_ok=True)
            torch.save(best_model_state, os.path.join(f'{path}/models', model_filename))
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
                entry = image_ids_batch[idx] if idx < len(image_ids_batch) else None
                if entry is not None:
                    mask_name = _entry_to_mask_filename(entry)
                    mask_path = os.path.join(mask_dir, mask_name)
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
    with open('Code/configs/training_config.json') as f:
        train_config = json.load(f)

    with open('Code/configs/preprocessing_config.json') as f:
        preprocessing_config = json.load(f)

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Initializing Device: {DEVICE}')

    # Pretty print of all relevant parameters (using new structure)
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

    time.sleep(2)
    print("\n-- Training --")

    print(f"Split ratio: {get_config(train_config, 'training', 'split_ratio')}")
    print(f"Grad clip value: {get_config(train_config, 'training', 'grad_clip_value')}")
    print("\n-- Grid-search enabled --")
    print(get_config(train_config, 'training', 'grid_search_enabled'))
    print("\n-- Non-grid-search cfg --")
    print(get_config(train_config, 'training', 'non_grid_search'))
    print("\n-- Grid-search parameters --")
    print(get_config(train_config, 'training', 'grid_search_parameters'))
    print("==== ============== ====\n")

    time.sleep(2)
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

    train_ids = [(image_dir, f) for f in image_ids[:split_index]]
    valid_ids = [(image_dir, f) for f in image_ids[split_index:]]

    originals_fnames = [f for _, f in train_ids + valid_ids]

    if get_config(train_config, "training", "use_diffusion_images"):
        gen_image_dir = os.path.join(path, "generated_samples")
        if os.path.isdir(gen_image_dir):
            gen_images = []
            for f in sorted(os.listdir(gen_image_dir)):
                if f.lower().endswith(valid_extensions):
                    # strip "_gen" so the mask path matches original mask file
                    mask_name = f.replace("_gen.png", ".png").replace("_gen.jpg", ".jpg")
                    gen_images.append((gen_image_dir, f, mask_name))
            # add only to TRAIN
            train_ids.extend(gen_images)

    # OBJECT DETECTION -> YOLO
    detection_model = YOLO(os.path.join(preprocessing_config["yolo_path"], "attempt_1/weights/best.pt"))

    if get_config(train_config, "model", "encoder", legacy_key="encoder") != "transformer":
        preprocessing_fn = get_preprocessing_fn(get_config(train_config, "model", "encoder", legacy_key="encoder"), pretrained=get_config(train_config, "model", "encoder_weights", legacy_key="encoder_weights"))
    else:
        preprocessing_fn = None

    # Hyperparameter sets (from config)
    grid_enabled = bool(get_config(train_config, "training", "grid_search_enabled"))
    gs_params = get_config(train_config, "training", "grid_search_parameters") or {}
    ngs = get_config(train_config, "training", "non_grid_search") or {}

    if grid_enabled:
        # derive search spaces; provide sane fallbacks
        learning_rates = gs_params.get("learning_rate", [1e-4, 3e-4, 1e-3])
        optimizers = gs_params.get("optimizer", ["adamw", "sgd"])
        gammas = gs_params.get("lr_scheduler_gamma", [0.99, 0.999, 0.9999])
        use_focal_list = gs_params.get("use_focal_loss", [True, False])
        loss_functions = gs_params.get("loss_functions", ["dice", "focal_loss"])
        lambda_list = gs_params.get("lambda_loss", [1, 5, 10])

        # We keep your two-mode "loss_combination" mapping for Trainer:
        # - if 'dice' in loss_functions AND use_focal_loss False => "dice+ce"
        # - if use_focal_loss True => "focal+ce"
        global_best_iou = 0.0
        global_best_f1 = 0.0
        global_best_model_state = None

        for (lr, optimizer_choice, gamma, use_focal, lf, lambdaa) in product(
            learning_rates, optimizers, gammas, use_focal_list, loss_functions, lambda_list
        ):
            # choose loss combination
            loss_combo = "focal+ce" if use_focal else "dice+ce"

            # choose weight range from non_grid_search or default
            weight_range = tuple(ngs.get("weight_range_multiclass", [50, 200]))

            # scheduler gamma override (we pass in train_once via train_config read, but we set gamma here for print)
            # The actual scheduler reads gamma from train_config; to reflect search gamma, we temporarily inject:
            train_config["training"]["lr_scheduler_gamma"] = gamma

            print(f"[Grid] lr={lr}, opt={optimizer_choice}, gamma={gamma}, use_focal={use_focal}, lf={lf}, lambda={lambdaa}, weight_range={weight_range}")
            best_iou, best_f1, final_model_state = train_once(
                train_config, preprocessing_config, train_ids, valid_ids, path, preprocessing_fn, detection_model, DEVICE,
                lambdaa, get_config(train_config, "training", "sampler"), weight_range, loss_combo, lr, optimizer_choice, grid_search=True, originals_fnames=originals_fnames
            )
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
        # Non-grid single run
        lambdaa = ngs.get("lambda_loss", 5)
        sampler_option = get_config(train_config, "training", "sampler", legacy_key="sampler")
        weight_range = tuple(ngs.get("weight_range_multiclass", [50, 200]))
        use_focal_loss = bool(ngs.get("use_focal_loss", True))
        lr = ngs.get("learning_rate", 3e-4)
        optimizer_choice = ngs.get("optimizer", "adamw")
        # keep your two-mode combination
        loss_combination = 'focal+ce' if use_focal_loss else 'dice+ce'

        # Ensure scheduler gamma is present for the non-grid path
        train_config["training"]["lr_scheduler_gamma"] = ngs.get("lr_scheduler_gamma", 0.999)

        print(f"[Run] lr={lr}, opt={optimizer_choice}, gamma={train_config['training']['lr_scheduler_gamma']}, use_focal={use_focal_loss}, lambda={lambdaa}, weight_range={weight_range}, loss_combo={loss_combination}")
        best_iou, best_f1, best_model_state = train_once(
            train_config, preprocessing_config, train_ids, valid_ids, path, preprocessing_fn, detection_model, DEVICE,
            lambdaa, sampler_option, weight_range, loss_combination, lr, optimizer_choice, grid_search=False, originals_fnames=originals_fnames
        )

        # After single run training completes, we could also save a final model if needed:
        if best_model_state is not None:
            final_filename = f"final_model_{best_iou:.4f}_{best_f1:.4f}.pth"
            torch.save(best_model_state, os.path.join(path, final_filename))
            print(f"Final best model saved as {final_filename}!")

if __name__ == "__main__":
    main()
