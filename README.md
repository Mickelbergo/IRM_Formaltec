# Forensic Wound Segmentation Using Vision Transformers

## Overview

I'm developing a deep learning system for automated forensic wound analysis that can detect, classify, and segment 11 different types of wounds from medical images. This project combines state-of-the-art Vision Transformers (specifically DINOv2) with advanced data augmentation techniques including diffusion-based synthetic image generation to tackle the challenging problem of forensic wound classification.

**Current Performance**: IoU ~0.5, F1 ~0.6 on 11-class multiclass segmentation

**Dataset**: ~2000 forensic wound images (not publicly available due to privacy constraints)

---

## Motivation

Forensic wound analysis is a critical but time-consuming task that requires expert knowledge. Many wound types are rare, leading to severe class imbalance and making it difficult to train robust models. My goals with this project are:

1. **Automate wound detection and classification** to assist forensic pathologists
2. **Handle severe class imbalance** through intelligent preprocessing and synthetic data generation
3. **Achieve interpretable predictions** using Grad-CAM visualizations
4. **Build a scalable training pipeline** that supports multiple architectures and can be easily extended

The challenge lies not just in achieving high accuracy, but in handling rare wound classes (like stab wounds, gunshot wounds) that appear infrequently in the dataset while maintaining good performance on common classes.

---

## Technical Approach

### 1. Data Preprocessing Pipeline

My preprocessing workflow addresses the unique challenges of forensic medical imagery:

**Stage 1: Image Standardization** ([preprocessing.py](Code/preprocessing/preprocessing.py))
- Resize all images to 640×1280 while preserving aspect ratio
- Rotate landscape images to portrait orientation for consistency
- Use nearest-neighbor interpolation for masks to preserve class indices
- Handle 11 distinct wound classes (0-10, where 0 is background)

**Stage 2: Class Weight Calculation** ([weights.py](Code/preprocessing/weights.py))
- Compute inverse frequency weights to handle severe class imbalance
- Generate per-image weights for WeightedRandomSampler
- Background class (0) always weighted at 1.0 for stability
- Weights are rescalable to configurable ranges (e.g., [30, 60])

**Stage 3: YOLO Wound Detection** ([yolo.py](Code/preprocessing/yolo.py))
- Train a YOLO model to detect wound bounding boxes
- Used during training for dynamic wound cropping and zoom
- Helps the model focus on relevant wound regions
- Reduces computational cost by cropping to regions of interest

**Data Augmentation** (Albumentations-based):
- Geometric: horizontal/vertical flips, rotations
- Photometric: color jitter, brightness/contrast adjustments
- Distortions: elastic transforms, optical distortion, grid distortion
- Regularization: coarse dropout for robustness
- All augmentations applied consistently to both images and masks

### 2. Diffusion Model for Synthetic Data Generation

One of my key innovations is using diffusion models to generate synthetic training data. I've implemented multiple diffusion variants with increasing sophistication:

**Basic DDPM** ([diffusion_dummy.py](Code/training/diffusion_dummy.py))
- Simple denoising diffusion probabilistic model
- UNet2D architecture with DDPMScheduler
- Baseline for comparison

**Mask-Conditional DDPM** ([diffusion.py](Code/training/diffusion.py))
- One-hot encoded mask conditioning
- v-prediction with SNR (p2) loss weighting
- Exponential Moving Average (EMA) for cleaner weights
- Classifier-free guidance (20% condition dropout)
- DPMSolverMultistep for fast inference
- Paired geometric augmentations for masks
- Auto-detects class labels from mask palettes

**Enhanced Diffusion Model** ([diffusion_enhanced.py](Code/training/diffusion_enhanced.py))

This is my most advanced implementation, incorporating 7 key improvements from recent research:

1. **Cross-Attention Conditioning** (Rombach et al. 2022)
   - Dedicated mask encoder network
   - Cross-attention layers at multiple scales
   - Better semantic understanding vs. simple concatenation

2. **Deeper Architecture**
   - 6 blocks with cross-attention: [128, 256, 512, 512, 768, 768]
   - Multi-head attention (8 heads)
   - Operates at 512×512 resolution

3. **Perceptual Loss (LPIPS)** (Zhang et al. 2018)
   - VGG-based perceptual loss weighted at 0.1
   - Ensures photorealistic wound textures
   - Complements pixel-level MSE loss

4. **Adversarial Training** (Isola et al. 2017)
   - PatchGAN discriminator for local realism
   - Starts at epoch 50 after diffusion stabilizes
   - Weight: 0.01 to avoid mode collapse
   - Produces sharper edges and better textures

5. **Medical Stain Augmentation** (Tellez et al. 2018)
   - Brightness, contrast, saturation, hue variations
   - Simulates different lighting and tissue conditions
   - Critical for medical image diversity

6. **Multi-Scale Training** (Karras et al. 2018)
   - Three resolutions: 384×384, 512×512, 640×640
   - 30% probability of alternate scales
   - Better generalization (optional, requires batch_size=1)

7. **Improved Conditioning Strategy**
   - Reduced dropout to 10% (medical images need stronger guidance)
   - Guidance scale: 3.0 for better mask adherence

**Training Requirements**: 48GB VRAM (A6000/A100), mixed precision, gradient checkpointing

**Synthetic Image Generation** ([generate_synthetic.py](Code/training/generate_synthetic.py))
- Load trained diffusion checkpoint
- Generate wound images conditioned on existing masks
- Used for data augmentation during segmentation training

### 3. Segmentation Model Architectures

I support three main encoder types, each with different strengths:

**Vision Transformer (ViT) - DINOv2** ([model.py](Code/training/model.py))
```python
class UNetWithViT:
    # DINOv2-giant: 1.1B parameters, 1536-dimensional embeddings
    # Multi-scale feature extraction from transformer layers
    # Custom decoder with skip connections
```

My current best model uses `facebook/dinov2-giant` with:
- Multi-scale features from layers 10, 20, 30, 40
- Projection layers to match decoder dimensions
- Stochastic depth (DropPath) for regularization
- Support for freezing/unfreezing encoder layers

**Available DINOv2 variants**:
- `dinov2-small`: 22M params, 384 dim
- `dinov2-base`: 86M params, 768 dim
- `dinov2-large`: 300M params, 1024 dim
- `dinov2-giant`: 1.1B params, 1536 dim (BEST)

**Swin Transformer V2** ([model.py](Code/training/model.py))
```python
class UNetWithSwinTransformer:
    # Swin V2 Base from torchvision
    # 4-stage hierarchical feature extraction
    # Natural skip connections from stages
```

**CNN Encoders** (U-Net++) ([model.py](Code/training/model.py))
```python
class UNetWithClassification:
    # Uses segmentation_models_pytorch library
    # Supports: ResNet, EfficientNet, MiT, etc.
    # Standard encoder-decoder architecture
```

### 4. Progressive 3-Stage Training (ViT Only)

One of my key innovations for training large Vision Transformers is progressive unfreezing:

**Stage 1: Frozen Encoder** (~30 epochs)
- Entire DINOv2 encoder frozen (requires_grad=False)
- Only decoder trains
- Purpose: Learn basic reconstruction without disrupting pretrained features
- Learning rate: Base LR (e.g., 0.0001)

**Stage 2: Partial Unfreeze** (~30 epochs)
- Unfreeze last 6-8 encoder layers
- Upper transformer blocks adapt to wound features
- Learning rate: Base LR × 0.3 (reduced)
- Allows task-specific fine-tuning of upper layers

**Stage 3: Full/Extended Unfreeze** (remaining epochs)
- Unfreeze last 12-16 encoder layers
- Deep fine-tuning of transformer
- Learning rate: Base LR × 0.05 (heavily reduced)
- Final optimization for maximum performance

**Implementation**:
- Separate optimizer and scheduler per stage
- Each stage saves its own best model
- Monitors IoU and F1 score
- Prevents catastrophic forgetting while enabling adaptation

### 5. Loss Functions and Optimization

I use combined loss functions to leverage complementary strengths:

**Loss Combinations**:
```python
# Option 1: Dice + Cross Entropy (current best)
total_loss = lambda * DiceLoss(pred, mask) + CrossEntropyLoss(pred, mask)

# Option 2: Focal + Cross Entropy (for extreme imbalance)
total_loss = lambda * FocalLoss(pred, mask) + CrossEntropyLoss(pred, mask)
```

**Key Parameters**:
- `lambda`: Balance between segmentation and pixel-wise losses (10-20 works best)
- Focal loss: `alpha=0.5, gamma=4.0` for hard example mining
- Cross entropy: Label smoothing (0.1) for regularization
- Class weights: Rescaled to [30, 60] range, applied to CE and Focal

**Optimization Strategy**:
- Optimizer: AdamW (weight decay 1e-5)
- Scheduler: CosineAnnealingWarmRestarts (T_0=15, smooth cycles)
- Gradient clipping: 0.1 (prevents exploding gradients in large models)
- Mixed precision: FP16 for faster training
- Batch size: 8 (effective on 48GB VRAM)

**Regularization Techniques**:
- Dropout: 0.4-0.5 in decoder
- Stochastic depth: 0.2 in ViT encoder
- Label smoothing: 0.1
- Early stopping: Patience 15 epochs
- Exponential Moving Average (EMA): 0.9999 decay

### 6. Training Infrastructure

**Grid Search Capability** ([Main_gridsearch.py](Code/training/Main_gridsearch.py))
- Automatically test hyperparameter combinations
- Parameters: learning rate, optimizer, lambda, loss functions
- Saves best model with descriptive naming

**Model Naming Convention**:
```
best_model_v1.6_epoch21_encoder_dinov2-giant_seg_multiclass_lambda10_optadamw_lr0.0001_dice+ce_wr30_60_samplerFalse_iou0.5003_f10.6124.pth
```
This includes: version, epoch, encoder, segmentation mode, hyperparameters, and metrics.

**Training Monitoring**:
- Real-time Streamlit dashboard ([dashboard.py](Code/training/dashboard.py))
- Metrics: Loss, IoU, F1 (macro/micro/weighted), per-class IoU
- Interactive controls: Toggle image display, model saving, GradCAM during training
- Training logs saved to `training_logs/` with JSON metrics

**Interpretability**:
- Grad-CAM visualizations ([visualize_gradcam.py](Code/training/visualize_gradcam.py))
- Runs every 5 epochs
- Target layers configurable per encoder type
- Saves heatmaps overlaid on original images

**Test-Time Augmentation (TTA)** ([tta_inference.py](Code/inference/tta_inference.py))
- Augmentations: original, horizontal flip, vertical flip, rotations
- Average predictions across all augmentations
- Expected improvement: +2-4% IoU

---

## Wound Classes

I'm working with 11 wound classes total (0-10):

| Class | Name | German Name | Frequency | Status |
|-------|------|-------------|-----------|---------|
| 0 | Background | Hintergrund | - | Active |
| 1 | Dermatorrhagia | Ungeformter Bluterguss | Common | Active |
| 2 | Hematoma | Geformter Bluterguss | Common | Active |
| 3 | Stab | Stich | Rare | Active |
| 4 | Cut | Schnitt | Common | Active |
| 5 | Thermal | Thermische Gewalt | Moderate | Excluded* |
| 6 | Skin Abrasion | Hautabschürfung | Common | Active |
| 7 | Puncture/Gun Shot | Schusswunden | Rare | Excluded* |
| 8 | Contused-Lacerated | Quetsch-Riss Wunden | Moderate | Active |
| 9 | Semisharp Force | Halbscharfe Gewalt | Rare | Excluded* |
| 10 | Lacerations | Risswunden | Moderate | Excluded* |

*By default, I exclude classes [5, 7, 9, 10] from training due to extreme rarity. This can be configured via `classes_to_exclude` in [preprocessing_config.json](Code/configs/preprocessing_config.json).

**Note**: Classes 11-14 were merged into class 6 during data cleaning.

---

## Project Structure

```
IRM_Formaltec/
├── Code/
│   ├── training/                    # Main training pipeline
│   │   ├── Main_gridsearch.py       # Primary training script (grid search + single runs)
│   │   ├── Main_transformer.py      # Swin Transformer training
│   │   ├── model.py                 # Model architectures (ViT, Swin, CNN)
│   │   ├── Epochs.py                # Training/validation epoch logic
│   │   ├── Preprocessing.py         # Dataset classes and data loading
│   │   ├── augmentations.py         # Albumentations pipelines
│   │   ├── ema.py                   # Exponential Moving Average
│   │   ├── training_logger.py       # Comprehensive experiment tracking
│   │   ├── dashboard.py             # Streamlit real-time monitoring
│   │   ├── visualize_gradcam.py     # Grad-CAM interpretability
│   │   ├── prediction.py            # Standard inference
│   │   ├── diffusion_dummy.py       # Basic DDPM diffusion
│   │   ├── diffusion.py             # Mask-conditional DDPM
│   │   ├── diffusion_enhanced.py    # Advanced diffusion (7 improvements)
│   │   ├── diffusion_fast.py        # Fast inference diffusion
│   │   ├── generate_synthetic.py    # Generate synthetic wound images
│   │   └── compare_diffusion.py     # Compare diffusion variants
│   │
│   ├── preprocessing/               # Data preprocessing
│   │   ├── preprocessing.py         # Image resizing and standardization
│   │   ├── weights.py               # Class weight calculation
│   │   └── yolo.py                  # YOLO bounding box generation
│   │
│   ├── inference/                   # Inference utilities
│   │   └── tta_inference.py         # Test-Time Augmentation
│   │
│   └── configs/                     # JSON configuration files
│       ├── training_config.json     # Training hyperparameters
│       └── preprocessing_config.json # Augmentation and data settings
│
├── diffusion_model/                 # Trained diffusion model weights
│   ├── unet/                        # Diffusion U-Net weights
│   ├── scheduler/                   # DDPM scheduler config
│   └── samples/                     # Generated sample images
│
├── diffusion_model_enhanced/        # Enhanced diffusion weights (if trained)
├── diffusion_model_fast/            # Fast diffusion weights (if trained)
│
├── training_logs/                   # Training run logs and metrics
│   └── vit_YYYYMMDD_HHMMSS/         # Per-run directories
│       ├── hyperparameters.json
│       ├── metrics.json
│       ├── model_info.json
│       └── predictions/
│
├── gradcam_outputs/                 # Grad-CAM visualizations
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── MANUAL.md                        # Operational manual
├── CLAUDE.md                        # AI assistant reference
└── TODO.txt                         # Research notes and best configs
```

**External Data Directory** (not in repo): `E:/projects/Wound_Segmentation_III/Data/`
```
Data/
├── new_images_640_1280/             # Preprocessed images
├── new_masks_640_1280/              # Ground truth masks
├── generated_samples/               # Synthetic images from diffusion
├── YOLO/                            # YOLO training data
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   └── labels/val/
├── class_weights.pth                # Computed class weights
└── image_weights.pth                # Per-image sampling weights
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (48GB recommended for diffusion training)
- CUDA 11.8+
- 16GB+ RAM (64GB+ recommended)
- Windows (project uses Windows-style paths)

### Installation

1. **Clone the repository**:
```bash
git clone <repo-url>
cd IRM_Formaltec
```

2. **Create virtual environment**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch>=2.0.0` - PyTorch deep learning framework
- `transformers>=4.30.0` - HuggingFace transformers (DINOv2)
- `segmentation-models-pytorch>=0.3.3` - CNN encoders
- `diffusers>=0.35.1` - Diffusion models
- `albumentations>=1.3.0` - Data augmentation
- `ultralytics>=8.0.0` - YOLO detection
- `streamlit>=1.24.0` - Training dashboard
- `accelerate` - Multi-GPU training for diffusion

4. **Prepare data** (if you have access):
- Place raw images in `E:/projects/Wound_Segmentation_III/Data/images/`
- Place masks in `E:/projects/Wound_Segmentation_III/Data/masks/`

---

## Usage

### Data Preparation

Run these in order **before training**:

```bash
# Step 1: Resize and standardize images
python Code/preprocessing/preprocessing.py
# Output: new_images_640_1280/, new_masks_640_1280/

# Step 2: Calculate class weights for balanced training
python Code/preprocessing/weights.py
# Output: class_weights.pth, image_weights.pth

# Step 3: Generate YOLO bounding boxes (optional)
python Code/preprocessing/yolo.py
# Output: YOLO/labels/ with bounding box annotations
```

### Training Segmentation Model

**Basic Training** (uses settings from config):
```bash
python Code/training/Main_gridsearch.py
```

**Monitor Training** (separate terminal):
```bash
streamlit run Code/training/dashboard.py
# Access at http://localhost:8501
```

**Interactive Controls During Training**:
- Press `Ctrl+C` to pause training
- `d` - Toggle image display
- `s` - Toggle model saving
- `g` - Toggle Grad-CAM visualization
- `c` - Continue training
- `q` - Quit training

**Configuration** ([training_config.json](Code/configs/training_config.json)):
```json
{
  "model": {
    "encoder": "vit",
    "vit_config": {
      "model_name": "facebook/dinov2-giant",
      "dropout_rate": 0.5,
      "stochastic_depth_rate": 0.2
    }
  },
  "training": {
    "batch_size": 8,
    "num_epochs": 100,
    "grid_search_enabled": false,
    "progressive_training": true,
    "non_grid_search": {
      "learning_rate": 0.0001,
      "optimizer": "adamw",
      "loss_functions": ["dice"],
      "lambda_loss": 10,
      "weight_range_multiclass": [30, 60]
    }
  }
}
```

### Training Diffusion Model

**Basic Diffusion** (mask-conditional):
```bash
python Code/training/diffusion.py
```

**Enhanced Diffusion** (with all 7 improvements):
```bash
cd Code/training
accelerate launch diffusion_enhanced.py
# Requires 48GB VRAM
```

**Generate Synthetic Images**:
```bash
python Code/training/generate_synthetic.py
# Loads trained diffusion model
# Generates wound images from existing masks
# Saves to generated_samples/
```

### Inference

**Standard Inference**:
```bash
python Code/training/prediction.py
```

**Test-Time Augmentation** (TTA):
```bash
python Code/inference/tta_inference.py \
  --model_path models/best_model.pth \
  --image_path test_image.jpg \
  --augmentations original hflip vflip rot90
# Expected improvement: +2-4% IoU
```

**Grad-CAM Visualization**:
```bash
python Code/training/visualize_gradcam.py
# Or enable in training_config.json: "gradCAM": true
# Runs automatically every 5 epochs during training
```

---

## Current Best Configuration

Based on extensive grid search and experimentation, my best performing setup is:

**Model Architecture**:
- Encoder: `facebook/dinov2-giant` (1.1B parameters)
- Decoder: Custom U-Net with multi-scale features
- Progressive training: 3 stages (30/30/40 epochs)

**Training Hyperparameters**:
- Learning rate: 0.0001
- Optimizer: AdamW (weight decay 1e-5)
- Loss: Dice + CE (lambda=10)
- Weight range: [30, 60]
- Gradient clipping: 0.1
- Scheduler: CosineAnnealingWarmRestarts (T_0=15)

**Regularization**:
- Dropout: 0.5
- Stochastic depth: 0.2
- Label smoothing: 0.1
- Early stopping: 15 epochs patience
- EMA: 0.9999 decay

**Data**:
- No diffusion-generated images (real data only performs better currently)
- YOLO detection probabilities: [0.2, 0.8]
- Excluded classes: [5, 7, 9, 10]

**Performance**: IoU ~0.5, F1 ~0.6 (multiclass, 11 classes)

**Key Insights from Grid Search**:
- Higher lambda (10-20) works better than lower values
- Dice loss outperforms Focal loss in most cases
- Lower lr_scheduler_gamma (0.999-0.9999) gives better results
- Progressive training is crucial for ViT models
- Gradient clipping at 0.1 prevents instability

---

## Current Status and Next Steps

### What's Working

✅ Preprocessing pipeline (resize, weights, YOLO)
✅ Progressive 3-stage ViT training
✅ Grid search with automatic hyperparameter tuning
✅ Real-time training dashboard with Streamlit
✅ Grad-CAM interpretability visualizations
✅ Test-Time Augmentation (TTA)
✅ Multiple diffusion model variants
✅ Synthetic image generation from masks
✅ Comprehensive training logging
✅ Multi-encoder support (ViT, Swin, CNN)

### Known Issues

⚠️ **Class Imbalance**: Rare classes (3, 7, 9) still challenging despite all techniques
⚠️ **Diffusion-Generated Images**: Currently don't improve segmentation performance (under investigation)
⚠️ **Grad-CAM Hooks**: Occasional hook registration conflicts when switching encoders
⚠️ **Memory Usage**: DINOv2-giant requires 48GB VRAM for comfortable training

### Next Steps

📋 **Immediate Priorities** (from TODO.txt):
1. Investigate why diffusion-generated images don't improve performance
2. Test higher lambda values (15-25) in grid search
3. Experiment with different learning rate schedules per progressive training stage
4. Try lower weight decay in stages 2-3 of progressive training
5. Verify Grad-CAM works correctly with all encoder types

📋 **Future Improvements**:
1. Implement hierarchical classification (coarse-to-fine wound types)
2. Add attention mechanisms to decoder
3. Try ensemble methods (multiple models)
4. Implement uncertainty quantification
5. Add more medical-specific augmentations
6. Explore self-supervised pretraining on unlabeled forensic images

---

## Performance Benchmarks

**Hardware**: NVIDIA RTX 6000 ADA (48GB VRAM), 512GB RAM

**Training Time**:
- DINOv2-giant: ~10-12 hours for 100 epochs (batch size 8)
- Swin Transformer: ~6-8 hours for 100 epochs
- EfficientNet-B7: ~4-6 hours for 100 epochs

**Inference Speed**:
- DINOv2-giant: ~150ms per image (384×384)
- With TTA (4 augmentations): ~600ms per image
- Batch inference (batch size 8): ~1200ms for 8 images

**Model Sizes**:
- DINOv2-giant: ~4.2GB
- Swin V2 Base: ~350MB
- EfficientNet-B7: ~250MB

---

## Acknowledgments and References

This project builds on cutting-edge research in computer vision and medical imaging:

**Vision Transformers**:
- DINOv2: Oquab et al. (2023) - "DINOv2: Learning Robust Visual Features without Supervision"
- Swin Transformer: Liu et al. (2021) - "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"

**Diffusion Models**:
- DDPM: Ho et al. (2020) - "Denoising Diffusion Probabilistic Models"
- Improved DDPM: Nichol & Dhariwal (2021) - "Improved Denoising Diffusion Probabilistic Models"
- Latent Diffusion: Rombach et al. (2022) - "High-Resolution Image Synthesis with Latent Diffusion Models"
- Classifier-Free Guidance: Ho & Salimans (2022)

**Loss Functions and Training**:
- Focal Loss: Lin et al. (2017) - "Focal Loss for Dense Object Detection"
- Dice Loss: Milletari et al. (2016) - "V-Net: Fully Convolutional Neural Networks"
- Label Smoothing: Szegedy et al. (2016) - "Rethinking the Inception Architecture"

**Medical Image Analysis**:
- Perceptual Loss (LPIPS): Zhang et al. (2018) - "The Unreasonable Effectiveness of Deep Features"
- Stain Augmentation: Tellez et al. (2018) - "Quantifying the effects of data augmentation"
- Test-Time Augmentation: Various (2018) - "Test-Time Augmentation for Deep Learning"

**Segmentation Architectures**:
- U-Net: Ronneberger et al. (2015) - "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- U-Net++: Zhou et al. (2018) - "UNet++: A Nested U-Net Architecture"

**Libraries and Tools**:
- PyTorch: Paszke et al. (2019)
- Hugging Face Transformers: Wolf et al. (2020)
- Albumentations: Buslaev et al. (2020)
- Segmentation Models PyTorch: Yakubovskiy (2020)
- Diffusers: von Platen et al. (2022)

---

## License

This is a research project for forensic medical analysis. The dataset is not publicly available due to privacy and ethical constraints related to forensic medical imagery.

---

## Contact

For questions about this research project, please open an issue in the repository.

---

**Last Updated**: November 2025
**Project Version**: v1.6
**Status**: Active Development