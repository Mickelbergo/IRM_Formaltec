# Development Guide

Complete technical documentation for the forensic wound segmentation system.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Setup](#setup)
4. [Training](#training)
5. [Inference & API](#inference--api)
6. [Docker](#docker)
7. [Configuration](#configuration)
8. [Improvements](#improvements)
9. [Testing](#testing)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

### What It Does

Segments and classifies 11 types of forensic wounds using deep learning:
- DINOv2 Vision Transformer (1.1B parameters)
- U-Net decoder with skip connections
- Progressive 3-stage training
- REST API for deployment

**Current Performance**:
- Best IoU: 0.5
- Best F1: 0.6
- 11 classes (multiclass segmentation)

### Technologies

- PyTorch 2.0+
- DINOv2, UNet++, Swin Transformer
- FastAPI
- Streamlit (dashboard)
- Docker + Docker Compose
- PostgreSQL, Redis
- Prometheus, Grafana

### Project Structure

```
IRM_Formaltec/
├── Code/
│   ├── training/
│   │   ├── Main_gridsearch.py     # Main training script
│   │   ├── model.py                # Model architectures
│   │   ├── Epochs.py               # Training/validation loops
│   │   ├── Preprocessing.py        # Data loading
│   │   ├── training_logger.py      # Metrics logging
│   │   ├── dashboard.py            # Streamlit dashboard
│   │   └── visualize_gradcam.py    # Grad-CAM visualization
│   ├── inference/
│   │   ├── api_server.py           # FastAPI REST API
│   │   ├── tta_inference.py        # Test-time augmentation
│   │   └── local_llm.py            # Local report generation
│   ├── preprocessing/
│   │   ├── preprocessing.py        # Image preprocessing
│   │   ├── weights.py              # Class weight calculation
│   │   └── yolo.py                 # YOLO wound detection
│   └── configs/
│       ├── training_config.json
│       └── preprocessing_config.json
├── tests/                          # Unit tests
├── docker-compose.yml              # Full stack
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Architecture

### Model Architecture

```
Input Image (384x384x3)
         ↓
    YOLO Detection (crop wound)
         ↓
┌────────────────────────┐
│  DINOv2-giant Encoder  │ (1.1B params)
│  - 40 transformer blocks
│  - Multi-scale features │
└────────┬───────────────┘
         │
    Feature extraction:
    - Early (layer 10)
    - Mid (layer 20)
    - Late (layer 30)
    - Final (layer 40)
         ↓
┌────────────────────────┐
│    U-Net Decoder       │
│  - Skip connections    │
│  - Upsampling blocks   │
└────────┬───────────────┘
         ↓
  Segmentation Head (1x1 conv)
         ↓
  Output Mask (384x384x11)
```

### Progressive Training (ViT only)

- **Stage 1** (30 epochs): Encoder frozen, train decoder
- **Stage 2** (30 epochs): Unfreeze last 8 layers, LR × 0.3
- **Stage 3** (40 epochs): Unfreeze last 16 layers, LR × 0.05

### API Architecture

```
Client
  ↓
FastAPI Gateway
  ├─→ Inference Service (DINOv2 + TTA)
  ├─→ LLM Service (local reports)
  ├─→ PostgreSQL (history)
  └─→ Redis (cache)
  ↓
Monitoring (Prometheus + Grafana)
```

---

## Setup

### Prerequisites

**Hardware:**
- NVIDIA GPU with 8GB+ VRAM
- 16GB+ RAM
- 50GB+ disk space

**Software:**
- Python 3.10+
- CUDA 11.8+
- Docker (optional)

### Installation

```bash
# Clone
git clone <repo-url>
cd IRM_Formaltec

# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install
pip install -r requirements.txt

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Training

### Quick Start

```bash
python Code/training/Main_gridsearch.py
```

### Data Preprocessing

Run once before training:

```bash
# 1. Resize images
python Code/preprocessing/preprocessing.py

# 2. Calculate class weights
python Code/preprocessing/weights.py

# 3. Train YOLO detector
python Code/preprocessing/yolo.py
```

### Configuration

Edit `Code/configs/training_config.json`:

```json
{
  "model": {
    "version": "v1.6",
    "encoder": "vit",
    "vit_config": {
      "model_name": "facebook/dinov2-giant",
      "stage1_epochs": 30,
      "stage2_epochs": 30,
      "unfreeze_layers_stage2": 8,
      "unfreeze_layers_stage3": 16,
      "stage2_lr_factor": 0.3,
      "stage3_lr_factor": 0.05,
      "dropout_rate": 0.5,
      "stochastic_depth_rate": 0.1,
      "early_stopping_patience": 15
    },
    "segmentation_classes": 11
  },
  "training": {
    "batch_size": 8,
    "num_epochs": 100,
    "learning_rate": 0.0002,
    "weight_decay": 1e-5,
    "label_smoothing": 0.1,
    "mixed_precision": true,
    "grad_clip_value": 0.1,
    "progressive_training": true,
    "scheduler_type": "exponential",
    "non_grid_search": {
      "optimizer": "adamw",
      "lr_scheduler_gamma": 0.999,
      "lambda_loss": 10,
      "loss_functions": ["dice"],
      "weight_range_multiclass": [30, 60]
    }
  }
}
```

### Key Parameters

**Model:**
- `encoder`: "vit", "transformer", "timm-efficientnet-l2", etc.
- `segmentation_classes`: 11 (background + 10 wound types)

**Training:**
- `batch_size`: 8 (reduce if OOM)
- `num_epochs`: 100
- `learning_rate`: 0.0002
- `label_smoothing`: 0.1 (prevents overfitting)

**ViT Config:**
- `stochastic_depth_rate`: 0.1 (encoder regularization)
- `early_stopping_patience`: 15 (auto-stop)
- `dropout_rate`: 0.5

**Loss:**
- `loss_functions`: ["dice"] or ["focal"]
- `lambda_loss`: 10 (balance dice+CE)
- `weight_range_multiclass`: [30, 60] (class weight scaling)

### Run Training

```bash
# Basic
python Code/training/Main_gridsearch.py

# Specific GPU
CUDA_VISIBLE_DEVICES=0 python Code/training/Main_gridsearch.py

# Grid search
# Edit training_config.json: "grid_search_enabled": true
python Code/training/Main_gridsearch.py
```

### Monitor Training

```bash
streamlit run Code/training/dashboard.py
```

Shows:
- Loss curves (train/val)
- IoU and F1 scores
- Learning rate schedule
- Per-class IoU heatmap
- Sample predictions

### Training Output

**Console:**
```
Loading facebook/dinov2-giant...
Stochastic Depth enabled: drop_path_rate=0.1
Total params: 1,139,854,347
Trainable params: 8,234,512

Stage 1: Training with frozen encoder for 30 epochs...
Epoch 1/30 (Stage 1)
Train Loss: 0.4521, Valid Loss: 0.5123
Train IoU: 0.6234, Valid IoU: 0.4521
Valid F1: 0.5123

[...]

Early stopping triggered in Stage 2 after 18 epochs
Best validation IoU: 0.5023

3-stage training completed! Best IoU: 0.5023, Best F1: 0.6012
```

**Saved Model:**
```
E:/projects/Wound_Segmentation_III/Data/models/
└── best_model_v1.6_stage3_epoch78_encoder_vit_seg_multiclass_lambda10_optadamw_lr0.0002_dice+ce_wr30_60_samplerFalse_iou0.5023_f10.6012.pth
```

### Training Tips

**Overfitting:**
- Increase `stochastic_depth_rate` (0.1 → 0.15)
- Increase `dropout_rate` (0.5 → 0.6)
- Increase `label_smoothing` (0.1 → 0.15)
- Decrease `early_stopping_patience` (15 → 10)

**Faster Training:**
- Increase `batch_size` (8 → 16)
- Increase `num_workers` (2 → 4)
- Enable `mixed_precision`: true
- Decrease `early_stopping_patience`

**Better Accuracy:**
- Use TTA at inference
- Train longer (`num_epochs`: 150)
- Lower learning rate (0.0001)
- Increase class weights (`weight_range_multiclass`: [50, 100])

---

## Inference & API

### API Server

```bash
# Start
python Code/inference/api_server.py

# Custom port
uvicorn Code.inference.api_server:app --port 8001

# With reload (dev)
uvicorn Code.inference.api_server:app --reload
```

**Output:**
```
============================================================
Forensic Wound Segmentation API Server
============================================================
Model Path: E:/projects/.../best_model.pth
Device: cuda
TTA Enabled: True
============================================================

Loading facebook/dinov2-giant...
Model loaded successfully!
  - Encoder: vit
  - Parameters: 1,139,854,347
  - Device: cuda
  - TTA: True

INFO:     Uvicorn running on http://0.0.0.0:8000
```

### API Endpoints

Interactive docs: http://localhost:8000/docs

**1. Health Check**
```bash
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

**2. Model Info**
```bash
GET /api/v1/models

Response:
{
  "model_name": "vit",
  "version": "v1.6",
  "encoder": "vit",
  "num_classes": 11,
  "parameters": 1139854347,
  "device": "cuda"
}
```

**3. Segment Image**
```bash
POST /api/v1/segment
Content-Type: multipart/form-data

Parameters:
- file: image file
- use_tta: bool (default: true)
- return_confidence: bool (default: true)
- generate_report: bool (default: false)

Response:
{
  "success": true,
  "prediction": {
    "shape": [384, 384],
    "unique_classes": [0, 1, 4, 6],
    "class_names": {
      "1": "dermatorrhagia",
      "4": "cut",
      "6": "skin_abrasion"
    }
  },
  "confidence_scores": {
    "cut": 0.873,
    "skin_abrasion": 0.125
  },
  "processing_time_ms": 1834.2
}
```

**4. Get Visualization**
```bash
POST /api/v1/segment/image

Returns: PNG image
```

### Python Client

```python
import requests

# Basic
response = requests.post(
    "http://localhost:8000/api/v1/segment",
    files={"file": open("wound.jpg", "rb")},
    params={"use_tta": True}
)

result = response.json()
print(f"Classes: {result['prediction']['class_names']}")
print(f"Confidence: {result['confidence_scores']}")

# With report
params = {"use_tta": True, "generate_report": True}
response = requests.post(url, files=files, params=params)
print(result['report'])

# Get visualization
response = requests.post(
    "http://localhost:8000/api/v1/segment/image",
    files={"file": open("wound.jpg", "rb")}
)
with open("output.png", "wb") as f:
    f.write(response.content)
```

### Test-Time Augmentation

```bash
python Code/inference/tta_inference.py \
  --model_path models/best_model.pth \
  --image_path test.jpg \
  --output_dir tta_outputs \
  --augmentations original hflip vflip rot90
```

**Programmatic:**
```python
from tta_inference import TTAWrapper, load_model

model = load_model("models/best_model.pth", "Code/configs/training_config.json")
tta_model = TTAWrapper(model, device="cuda", augmentations=["original", "hflip", "vflip", "rot90"])

import torch
image = torch.randn(1, 3, 384, 384).cuda()
prediction = tta_model(image)
```

---

## Docker

### Quick Start

```bash
docker-compose up -d
docker-compose ps
docker-compose logs -f api
docker-compose down
```

### Services

| Service | Port | Purpose |
|---------|------|---------|
| api | 8000 | FastAPI REST API |
| dashboard | 8501 | Streamlit dashboard |
| postgres | 5432 | Database |
| redis | 6379 | Cache |
| prometheus | 9090 | Metrics |
| grafana | 3000 | Monitoring |

### Environment Variables

Create `.env`:

```bash
# Model
MODEL_PATH=./models/best_model.pth
CONFIG_PATH=Code/configs/training_config.json

# LLM (runs locally by default)
LOCAL_LLM_ONLY=true
LLM_TYPE=template

# Performance
ENABLE_TTA=true

# Database
DB_PASSWORD=your_password
DATABASE_URL=postgresql://admin:password@postgres:5432/wound_analysis

# Redis
REDIS_URL=redis://redis:6379/0
```

### Docker Commands

```bash
# Build
docker-compose build
docker-compose build api
docker-compose build --no-cache

# Start/stop
docker-compose up -d
docker-compose restart api
docker-compose down
docker-compose down -v  # Remove volumes

# Logs
docker-compose logs -f api
docker-compose logs --tail=100 api

# Execute commands
docker-compose exec api bash
docker-compose exec postgres psql -U admin -d wound_analysis
```

---

## Configuration

### Training Config

`Code/configs/training_config.json`

**Model:**
```json
{
  "model": {
    "version": "v1.6",
    "encoder": "vit",
    "vit_config": {
      "model_name": "facebook/dinov2-giant",
      "dropout_rate": 0.5,
      "stochastic_depth_rate": 0.1,
      "early_stopping_patience": 15
    }
  }
}
```

**Training:**
```json
{
  "training": {
    "batch_size": 8,
    "num_epochs": 100,
    "learning_rate": 0.0002,
    "weight_decay": 1e-5,
    "label_smoothing": 0.1,
    "mixed_precision": true,
    "grad_clip_value": 0.1
  }
}
```

**Loss:**
```json
{
  "training": {
    "non_grid_search": {
      "loss_functions": ["dice"],
      "lambda_loss": 10,
      "weight_range_multiclass": [30, 60]
    }
  }
}
```

### Preprocessing Config

`Code/configs/preprocessing_config.json`

```json
{
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "augmentation_settings": {
    "horizontal_flip": {"enabled": true, "p": 0.5},
    "vertical_flip": {"enabled": true, "p": 0.5},
    "random_rotate_90": {"enabled": true, "p": 0.5},
    "color_jitter": {"enabled": true, "p": 0.5},
    "elastic_transform": {"enabled": true, "p": 0.2},
    "gaussian_blur": {"enabled": true, "p": 0.1}
  },
  "target_size": [384, 384],
  "segmentation": "multiclass",
  "classes_to_exclude": [3, 7, 9]
}
```

---

## Improvements

### Implemented Techniques

**1. Stochastic Depth (DropPath)**

Randomly drops transformer blocks during training.

Configuration:
```json
"stochastic_depth_rate": 0.1
```

Impact: +2-5% IoU

**2. Label Smoothing**

Softens hard labels [0, 1] to [ε, 1-ε].

Configuration:
```json
"label_smoothing": 0.1
```

Impact: +1-3% IoU

**3. Early Stopping**

Stops training when validation plateaus.

Configuration:
```json
"early_stopping_patience": 15
```

Impact: -20-30% training time

**4. Test-Time Augmentation**

Averages predictions from multiple augmented versions.

Usage:
```python
params = {"use_tta": True}
```

Impact: +2-4% IoU (inference only)

---

## Testing

### Unit Tests

```bash
pytest tests/ -v
pytest tests/ --cov=Code --cov-report=html
pytest tests/test_model.py -v
```

### Integration Tests

```bash
docker-compose up -d postgres redis
pytest tests/integration/ -v
docker-compose down
```

### Load Testing

```bash
pip install locust
locust -f tests/load_test.py --host=http://localhost:8000
# Open http://localhost:8089
```

---

## Troubleshooting

### CUDA Out of Memory

```json
"batch_size": 4  // Reduce
"gradient_checkpointing": true  // Enable
```

### Model File Not Found

```bash
ls E:/projects/Wound_Segmentation_III/Data/models/best_model.pth
export MODEL_PATH="/full/path/to/model.pth"
```

### Port Already in Use

```bash
# Find process
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Use different port
uvicorn Code.inference.api_server:app --port 8001
```

### Training Loss NaN

```json
"learning_rate": 0.00005  // Lower LR
"grad_clip_value": 0.5  // Increase clipping
```

### Docker GPU Not Working

```bash
# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
```

---

## Additional Notes

### Model Naming Convention

Best models saved as:
```
best_model_{version}_{stage}_epoch{N}_encoder_{encoder}_seg_{mode}_lambda{λ}_opt{opt}_lr{lr}_{loss}_wr{w1}_{w2}_samplerFalse_iou{iou}_f1{f1}.pth
```

### Wound Classes

| ID | Name | Frequency |
|----|------|-----------|
| 0 | Background | - |
| 1 | Dermatorrhagia | Common |
| 2 | Hematoma | Common |
| 3 | Stab | Rare |
| 4 | Cut | Common |
| 5 | Thermal | Moderate |
| 6 | Skin Abrasion | Common |
| 7 | Puncture/Gun Shot | Rare |
| 8 | Contused-Lacerated | Moderate |
| 9 | Semisharp Force | Rare |
| 10 | Lacerations | Moderate |

### System Requirements

**Minimum**: RTX 3060 (8GB), 16GB RAM, 50GB disk
**Recommended**: RTX 3090 (24GB), 32GB RAM, 100GB disk

**Timing**:
- Training: 4-8 hours for 100 epochs (RTX 3090)
- Inference: ~1 second per image (standard), ~2 seconds (TTA)
- Model size: 4.2GB (DINOv2-giant)

### Data Processing

All processing runs locally on your GPU/CPU. No external API calls by default.

---

**Last Updated**: 2025-10-29
**Status**: Active Development
