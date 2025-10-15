# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **forensic wound segmentation** research project that uses deep learning to automatically detect, classify, and segment wound types from medical images. The pipeline supports binary and multiclass segmentation with 11 wound classes (background + 10 wound types including dermatorrhagia, hematoma, stab, cut, thermal injuries, etc.).

**Key Architecture**: The project uses a U-Net based segmentation pipeline with support for multiple encoders:
- Vision Transformers (ViT/DINOv2) with progressive 3-stage training
- Swin Transformers
- CNN encoders (EfficientNet, ResNet, etc.)

**Data**: ~2000 forensic wound images (not publicly available due to privacy). Models evaluated using IoU and F1 score.

---

## Development Commands

### Preprocessing
Run these in order before training:
```bash
# 1. Resize images and save preprocessed versions
python Code/preprocessing/preprocessing.py

# 2. Calculate class weights for balanced training
python Code/preprocessing/weights.py

# 3. Train YOLO model for wound detection/cropping
python Code/preprocessing/yolo.py
```

### Training

**Main training script** (supports both grid search and single runs):
```bash
python Code/training/Main_gridsearch.py
```

**Transformer-specific training** (for Swin Transformer models):
```bash
python Code/training/Main_transformer.py
```

**Inference/Prediction**:
```bash
python Code/training/prediction.py
```

**Visualization** (Grad-CAM):
```bash
python Code/training/visualize_gradcam.py
```

---

## Configuration System

All hyperparameters are controlled via JSON configs in `Code/configs/`:

### `training_config.json`
- **Model selection**: `model.encoder` - Choose "vit", "transformer", or CNN encoder names
- **ViT progressive training**: `model.vit_config` contains 3-stage training parameters:
  - `stage1_epochs`, `stage2_epochs` - Epoch allocation per stage
  - `unfreeze_layers_stage2/3` - How many encoder layers to unfreeze
  - `stage2_lr_factor`, `stage3_lr_factor` - Learning rate multipliers
  - `dropout_rate` - Dropout for regularization
- **Training params**: batch size, epochs, learning rate, optimizer, weight decay
- **Loss configuration**:
  - `non_grid_search.loss_functions` - "dice" or "focal"
  - `non_grid_search.lambda_loss` - Weight for loss combination
  - `non_grid_search.weight_range_multiclass` - Class weight rescaling range
- **Scheduler**: `scheduler_type` ("cosine_restarts" or exponential)
- **Grid search**: Enable via `grid_search_enabled: true`
- **Data augmentation**: Controlled in preprocessing_config
- **Metrics**: `f1_average` - "macro", "micro", or "weighted"
- **GradCAM**: `gradCAM: true` to enable interpretability visualizations

### `preprocessing_config.json`
- **Augmentations**: All augmentation types with probabilities (horizontal_flip, color_jitter, elastic_transform, etc.)
- **Normalization**: ImageNet mean/std
- **Target size**: Image resolution [384, 384]
- **Segmentation mode**: "binary" or "multiclass"
- **Class exclusion**: `exclude_images_with_classes` and `classes_to_exclude` to filter rare classes

---

## Code Architecture

### Training Pipeline (`Code/training/Main_gridsearch.py`)

**High-level flow**:
1. Load configs → Setup device → Split data
2. Load YOLO detection model for preprocessing
3. Create model based on encoder type
4. Choose training mode: **progressive** (ViT) or **standard** (CNN/Swin)
5. Run grid search or single training
6. Save best model with descriptive filename

**Progressive Training** (ViT only):
- **Stage 1**: Frozen encoder, train decoder only (~30 epochs)
- **Stage 2**: Unfreeze last N encoder layers, reduced LR (~30 epochs)
- **Stage 3**: Unfreeze more layers, very low LR (remaining epochs)
- Each stage uses separate optimizer/scheduler
- Monitors IoU/F1, saves best model per stage

**Key utility functions**:
- `get_config(cfg, *keys, legacy_key)` - Hierarchical config access with fallback
- `rescale_weights(weights, range)` - Rescale class weights to specified range
- `train_once()` - Main training orchestrator
- `create_model()` - Factory for model instantiation
- `save_best_model()` - Saves with full hyperparameter naming

**Interactive training controls**: Press Ctrl+C during training to:
- Toggle image display ('d')
- Toggle model saving ('s')
- Toggle GradCAM ('g')
- Continue ('c') or quit ('q')

### Models (`Code/training/model.py`)

**Three model classes**:

1. **`UNetWithClassification`**: U-Net++ with CNN encoders (ResNet, EfficientNet, etc.)
   - Uses segmentation_models_pytorch library
   - Standard encoder-decoder architecture

2. **`UNetWithViT`**: U-Net with DINOv2 Vision Transformer encoder
   - Multi-scale feature extraction from transformer layers
   - Custom decoder with skip connections
   - Methods:
     - `freeze_encoder()` - Freeze all encoder params
     - `unfreeze_encoder(num_layers)` - Unfreeze last N layers (or all if None)
     - `extract_multiscale_features()` - Get features from early/mid/late/final layers
   - Supports progressive training (frozen → partial unfreeze → full unfreeze)

3. **`UNetWithSwinTransformer`**: U-Net with Swin Transformer V2 encoder
   - Uses torchvision Swin V2 Base
   - Feature extraction from 4 stages with skip connections

### Data Pipeline (`Code/training/Preprocessing.py`)

**Dataset classes**:
- Standard `Dataset` for CNN encoders
- `TransformerDataset` for transformer models (uses HuggingFace processors)

**Key features**:
- YOLO-based wound detection and cropping
- Albumentations-based augmentation pipeline
- Class exclusion filtering
- Support for diffusion-generated synthetic images
- WeightedRandomSampler for class imbalance

### Training Loop (`Code/training/Epochs.py`)

**`TrainEpoch` and `ValidEpoch`**:
- Mixed precision training support
- Gradient clipping
- Combined loss functions (Dice+CE or Focal+CE)
- Metrics: Accuracy, IoU, F1 (macro/micro/weighted)
- Learning rate scheduling (ExponentialLR or CosineAnnealingWarmRestarts)

---

## Model Naming Convention

Best models are saved with full hyperparameter info:
```
best_model_{version}_{stage}_epoch{N}_encoder_{encoder}_seg_{mode}_lambda{λ}_opt{opt}_lr{lr}_{loss}_wr{w1}_{w2}_sampler{bool}_iou{iou}_f1{f1}.pth
```

Example:
```
best_model_v1.5_epoch21_encoder_timm-efficientnet-l2_seg_binary_lambda5_optadamw_lr0.0003_dice+ce_wr50_200_samplerFalse_iou0.8005_f10.8746.pth
```

---

## Known Best Configurations

From `TODO.txt` and README:

**Current best (ViT DINOv2-giant)**:
- Encoder: `facebook/dinov2-giant`
- Learning rate: 0.0001
- Loss: Dice + CE (not focal)
- Lambda: 10
- Weight range: [30, 60]
- Grad clip: 0.1
- Scheduler: Cosine restarts (T_0=15)
- Progressive training: 3 stages (30/30/40 epochs)
- No diffusion images
- Optimizer: AdamW

**Grid search insights**:
- Higher lambda (10+) works better
- Dice loss outperforms focal
- Lower lr_scheduler_gamma (0.999-0.9999) is better
- Weight decay: 1e-5

---

## Class Information

11 classes total (0-10):
- 0: Background
- 1: Dermatorrhagia (ungeformter bluterguss)
- 2: Hematoma (geformter bluterguss)
- 3: Stab (stich) - RARE
- 4: Cut (schnitt)
- 5: Thermal (thermische gewalt)
- 6: Skin abrasion (hautabschürfung)
- 7: Puncture/gun shot - RARE
- 8: Contused-lacerated (quetsch-riss wunden)
- 9: Semisharp force (halbscharfe gewalt) - RARE
- 10: Lacerations (risswunden)

**Rare classes** (3, 7, 9) can be excluded via `classes_to_exclude` in preprocessing_config.

Classes 11-14 were merged into class 6.

---

## Important Implementation Details

### ViT Progressive Training
- **Stage 1**: Encoder frozen, only decoder trains
- **Stage 2**: Unfreeze last ~6-8 encoder layers, LR reduced by `stage2_lr_factor` (0.3)
- **Stage 3**: Unfreeze last ~12-16 layers, LR reduced by `stage3_lr_factor` (0.05)
- Each stage saves its own best model
- Total epochs distributed across stages via `stage1_epochs`, `stage2_epochs` in vit_config

### Loss Functions
Two combinations supported:
1. **Dice + CE**: `lambda * DICE_Loss(pred, mask) + CE_Loss(pred, mask)`
2. **Focal + CE**: `lambda * Focal_Loss(pred, mask) + CE_Loss(pred, mask)`

Lambda controls the balance (typical: 5-20).

### Class Weights
- Loaded from `class_weights.pth` (computed by `weights.py`)
- Rescaled to `weight_range` (e.g., [30, 60]) to control influence
- Background (class 0) always weighted at 1.0
- Applied to CE and Focal losses

### YOLO Integration
- YOLO model detects wound bounding boxes
- Used in Dataset to crop/zoom images dynamically
- Trained separately via `Code/preprocessing/yolo.py`
- Model path: `{yolo_path}/attempt_1/weights/best.pt`

### Grad-CAM
- Enabled via `gradCAM: true` in training_config
- Runs every 5 epochs
- Saves visualizations to `gradcam_outputs/`
- Target layer configurable per encoder type:
  - ViT: `decoder1.0` (configurable in vit_config)
  - Swin: `up4.0`
  - CNN: `segmentation_model.decoder.seg_blocks.3.block.0.block.2`
- Creates separate model instance to avoid hook conflicts

### Data Split
- Default: 80/20 train/val split
- Random split enabled via `random_split: true` (uses timestamp seed)
- Fixed split uses `random_seed: 42`
- Supports generated/diffusion images via `use_diffusion_images: true`

---

## Development Notes

- The codebase uses a hierarchical config system with legacy fallbacks
- Model versioning tracked via `model.version` (current: v1.6)
- Mixed precision training enabled by default for faster training
- Persistent workers in DataLoader for efficiency
- All paths are Windows-style (E:/ drive)
- Data directory is separate from code: `E:/projects/Wound_Segmentation_III/Data`

### Common Gotchas
1. **ViT encoder freezing**: Must call `freeze_encoder()` after model creation (line 113 in model.py)
2. **Generated image masks**: Masks for `*_gen.png` images map to `*.png` (without `_gen` suffix)
3. **Grid search**: Iterates over all hyperparameter combinations - can be very long
4. **Image weights**: Required for WeightedRandomSampler, loaded from `image_weights.pth`

---

## File Structure Overview

```
IRM_Formaltec/
├── Code/
│   ├── configs/               # JSON configuration files
│   ├── preprocessing/         # Data preprocessing scripts
│   ├── training/             # Training, models, epochs, prediction
│   └── Legacy/               # Old/deprecated code
├── diffusion_model/          # Diffusion model weights/configs
├── gradcam_outputs/          # Grad-CAM visualizations
├── images/                   # Documentation images
├── TODO.txt                  # Research notes and best configs
└── README.md                 # Project documentation
```

Data directory (external):
```
Data/
├── new_images_640_1280/      # Preprocessed images
├── new_masks_640_1280/       # Ground truth masks
├── generated_samples/        # Diffusion-generated images
├── YOLO/                     # YOLO training data
├── class_weights.pth         # Computed class weights
└── image_weights.pth         # Per-image weights for sampling
```
