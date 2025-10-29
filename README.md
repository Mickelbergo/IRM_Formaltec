# Forensic Wound Segmentation

Deep learning system for automated forensic wound analysis using Vision Transformers (DINOv2).

## Overview

Segments and classifies 11 types of forensic wounds from medical images using U-Net with DINOv2-giant encoder (1.1B parameters).

**Current Best Model**: IoU 0.5, F1 0.6 (multiclass)

## Features

- DINOv2-giant, UNet++, Swin Transformer support
- 11 wound classes (background + 10 wound types)
- Binary and multiclass segmentation
- YOLO-based wound detection and cropping
- Progressive 3-stage training for ViT models
- REST API with FastAPI
- Real-time training dashboard
- Grad-CAM visualization
- Test-time augmentation
- Docker deployment

## Quick Start

### Training

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python Code/training/Main_gridsearch.py

# Monitor training (separate terminal)
streamlit run Code/training/dashboard.py
```

### API Server

```bash
# Start server (runs locally by default)
python Code/inference/api_server.py

# Test inference
curl -X POST "http://localhost:8000/api/v1/segment" \
  -F "file=@test_image.jpg"

# Interactive docs
# http://localhost:8000/docs
```

### Docker

```bash
# Start all services
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - Grafana: http://localhost:3000
```

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+
- 16GB+ RAM

### Setup

```bash
git clone <repo-url>
cd IRM_Formaltec

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## Training

### Basic Training

```bash
python Code/training/Main_gridsearch.py
```

All regularization improvements are enabled by default:
- Stochastic depth (0.1)
- Label smoothing (0.1)
- Early stopping (patience 15)
- Progressive training (for ViT)

### Configuration

Edit `Code/configs/training_config.json`:

```json
{
  "model": {
    "encoder": "vit",
    "vit_config": {
      "model_name": "facebook/dinov2-giant",
      "stochastic_depth_rate": 0.1,
      "early_stopping_patience": 15,
      "dropout_rate": 0.5
    }
  },
  "training": {
    "batch_size": 8,
    "num_epochs": 100,
    "learning_rate": 0.0002,
    "label_smoothing": 0.1
  }
}
```

### Monitoring

```bash
streamlit run Code/training/dashboard.py
```

Shows real-time:
- Loss curves
- IoU and F1 scores
- Learning rate schedule
- Per-class performance

## Inference

### Python API

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/segment",
    files={"file": open("wound.jpg", "rb")},
    params={"use_tta": True}
)

result = response.json()
print(f"Classes: {result['prediction']['class_names']}")
print(f"Confidence: {result['confidence_scores']}")
```

### Test-Time Augmentation

```bash
python Code/inference/tta_inference.py \
  --model_path models/best_model.pth \
  --image_path test.jpg \
  --augmentations original hflip vflip rot90
```

### API Endpoints

- `GET /health` - Health check
- `GET /api/v1/models` - Model info
- `POST /api/v1/segment` - Segment image
- `POST /api/v1/segment/image` - Get visualization

## Wound Classes

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

## Project Structure

```
IRM_Formaltec/
├── Code/
│   ├── training/          # Training pipeline
│   ├── inference/         # API and inference
│   ├── preprocessing/     # Data preprocessing
│   └── configs/           # JSON configs
├── tests/                 # Unit tests
├── docker-compose.yml     # Docker deployment
├── requirements.txt       # Dependencies
├── README.md              # This file
└── DEVELOPMENT.md         # Detailed guide
```

## Documentation

- [DEVELOPMENT.md](DEVELOPMENT.md) - Complete technical guide
- [CLAUDE.md](CLAUDE.md) - Project overview for AI assistants

## Commands Reference

```bash
# Training
python Code/training/Main_gridsearch.py

# Dashboard
streamlit run Code/training/dashboard.py

# API
python Code/inference/api_server.py

# TTA
python Code/inference/tta_inference.py --model_path model.pth --image_path test.jpg

# Docker
docker-compose up -d
docker-compose logs -f api
docker-compose down

# Tests
pytest tests/ -v
```

## Notes

- Data is processed locally on your GPU/CPU
- Model is ~4.2GB (DINOv2-giant)
- Training takes 4-8 hours on RTX 3090
- Inference ~1 second per image
- Dataset not publicly available due to privacy reasons
