# 🎨 Training Dashboard Setup Guide

## Overview

I've created a **comprehensive real-time training visualization dashboard** for your wound segmentation model. This system includes:

1. **TrainingLogger** - Logs all metrics during training
2. **Streamlit Dashboard** - Beautiful real-time visualization
3. **Automatic metric tracking** - Loss, IoU, F1, LR, per-class IoU
4. **Prediction visualization** - See your model's predictions live

---

## 📦 Files Created

### 1. `Code/training/training_logger.py`
Complete logging system that tracks:
- Training & validation metrics (loss, IoU, F1, accuracy)
- Learning rate schedule
- Per-class IoU scores
- Model info & hyperparameters
- Prediction samples

### 2. `Code/training/dashboard.py`
Streamlit dashboard with:
- **Real-time metric plots** (4 charts: Loss, IoU, F1, Accuracy)
- **Learning rate visualization** (log scale)
- **Per-class IoU heatmap** (see which classes are learning)
- **Latest per-class IoU bar chart**
- **Prediction samples** (input/ground truth/prediction)
- **Auto-refresh** option (every 10s)
- **Multiple run comparison**

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install streamlit plotly pandas
```

### Step 2: Integrate Logger into Training

Add these lines to your `Main_gridsearch.py`:

**At the top (after imports):**
```python
from training_logger import TrainingLogger

# Global logger
training_logger = None
```

**In `main()` function, after loading configs:**
```python
def main():
    # ... existing code ...

    # Initialize logger
    global training_logger
    from datetime import datetime
    run_name = f"vit_giant_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_logger = TrainingLogger(run_name=run_name)

    # Log hyperparameters
    hyperparams_log = {
        'encoder': get_config(train_config, 'model', 'encoder'),
        'learning_rate': get_config(train_config, 'training', 'non_grid_search', 'learning_rate'),
        'lambda': get_config(train_config, 'training', 'non_grid_search', 'lambda_loss'),
        'weight_range': get_config(train_config, 'training', 'non_grid_search', 'weight_range_multiclass'),
        'batch_size': get_config(train_config, 'training', 'batch_size'),
        'num_epochs': get_config(train_config, 'training', 'num_epochs'),
        'scheduler_type': get_config(train_config, 'training', 'scheduler_type'),
    }
    training_logger.log_hyperparameters(hyperparams_log)

    # ... rest of main() ...
```

**In `create_model()` function, after model creation:**
```python
def create_model(train_config):
    # ... existing code ...

    # Log model info
    if training_logger:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        encoder = get_config(train_config, "model", "encoder", legacy_key="encoder")
        training_logger.log_model_info(model, encoder, total_params, trainable_params)

    return model, use_progressive
```

**In `train_single_epoch()` function, after getting metrics:**
```python
def train_single_epoch(model, train_epoch, valid_epoch, train_loader, valid_loader, epoch, total_epochs, hyperparams, stage=""):
    # ... existing code to run training ...

    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # Log metrics
    if training_logger:
        training_logger.log_epoch_start(epoch, stage)

        # Get per-class IoU if available
        train_per_class = train_logs.get('per_class_iou', None)
        valid_per_class = valid_logs.get('per_class_iou', None)

        training_logger.log_metrics(
            'train',
            train_logs['loss'],
            train_logs['accuracy'],
            train_logs['iou_score'],
            train_logs['f1_score'],
            lr=train_logs.get('lr', None),
            per_class_iou=train_per_class
        )

        training_logger.log_metrics(
            'valid',
            valid_logs['loss'],
            valid_logs['accuracy'],
            valid_logs['iou_score'],
            valid_logs['f1_score'],
            per_class_iou=valid_per_class
        )

    # ... rest of function ...
```

**In `train_progressive()`, log stage changes:**
```python
def train_progressive(...):
    # Stage 1
    if training_logger:
        training_logger.log_stage_change("Stage 1 - Frozen Encoder")

    # ... Stage 1 training ...

    # Stage 2
    if training_logger:
        training_logger.log_stage_change("Stage 2 - Partial Unfreeze", unfreeze_layers=8)

    # ... Stage 2 training ...

    # Stage 3
    if training_logger:
        training_logger.log_stage_change("Stage 3 - Extended Unfreeze", unfreeze_layers=16)

    # ... Stage 3 training ...
```

**Optional: Log prediction samples**
```python
# In train_single_epoch, after validation:
if training_logger and epoch % 5 == 0:  # Every 5 epochs
    # Get a batch from validation
    val_batch = next(iter(valid_loader))
    images = val_batch['image'][:2]  # First 2 samples
    gt_masks = val_batch['multiclass_mask'][:2]

    # Get predictions
    model.eval()
    with torch.no_grad():
        pred_masks = model(images.to(device)).argmax(dim=1)

    # Log samples
    for idx in range(2):
        training_logger.log_prediction_sample(
            images[idx],
            gt_masks[idx].squeeze(),
            pred_masks[idx],
            sample_idx=idx,
            phase='valid'
        )
```

**At the end of training:**
```python
if training_logger:
    summary = training_logger.finalize()
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"Best Valid IoU: {summary['best_valid_iou']:.4f}")
    print(f"Best Valid F1: {summary['best_valid_f1']:.4f}")
    print(f"{'='*50}")
```

---

## 🎨 Launch Dashboard

### Method 1: Watch Training in Real-Time

**Terminal 1 - Start Training:**
```bash
python Code/training/Main_gridsearch.py
```

**Terminal 2 - Launch Dashboard:**
```bash
streamlit run Code/training/dashboard.py
```

The dashboard will auto-update as training progresses!

### Method 2: View After Training

```bash
streamlit run Code/training/dashboard.py
```

Then select your training run from the dropdown.

---

## 📊 Dashboard Features

### 1. **Performance Metrics** (Top Cards)
- Current Train/Valid IoU
- Best IoU achieved
- Current F1 score
- Total epochs completed

### 2. **Training Curves** (4-plot grid)
- **Loss**: Train vs Valid loss over time
- **IoU Score**: Train vs Valid IoU
- **F1 Score**: Train vs Valid F1
- **Accuracy**: Train vs Valid accuracy

All with hover information and synchronized x-axis zoom.

### 3. **Learning Rate Schedule**
- Log-scale plot showing LR decay
- Helps verify scheduler is working correctly

### 4. **Per-Class IoU Visualization**
- **Heatmap**: See how each class improves over epochs
- **Bar Chart**: Latest per-class performance
- **Color-coded**: Red (bad) → Green (good)

### 5. **Prediction Samples**
- View actual predictions from your model
- Side-by-side: Input | Ground Truth | Prediction
- Updates every N epochs (configurable)

### 6. **Auto-Refresh Mode**
- Enable in sidebar
- Dashboard refreshes every 10 seconds
- Watch training progress live!

### 7. **Hyperparameters & Model Info**
- Expandable sections showing:
  - All hyperparameters used
  - Model architecture details
  - Training stage transitions

---

## 🎯 Example Dashboard View

```
╔══════════════════════════════════════════════════════════════╗
║   🏥 Wound Segmentation Training Dashboard                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   📈 Current Performance                                     ║
║   ┌──────────┬──────────┬──────────┬──────────┐            ║
║   │ Train IoU│ Valid IoU│ Valid F1 │  Epochs  │            ║
║   │  0.4523  │  0.4201  │  0.5234  │    45    │            ║
║   │Best:0.45 │Best:0.42 │Best:0.52 │Completed │            ║
║   └──────────┴──────────┴──────────┴──────────┘            ║
║                                                              ║
║   📊 Training Curves                                         ║
║   [4x2 Grid of Interactive Plotly Charts]                   ║
║                                                              ║
║   📉 Learning Rate Schedule                                  ║
║   [Log-scale LR decay visualization]                        ║
║                                                              ║
║   🔥 Per-Class IoU Heatmap                                   ║
║   [Heatmap showing class-wise performance]                  ║
║                                                              ║
║   🖼️ Prediction Samples                                      ║
║   [Sample 1] [Sample 2] [Sample 3]                          ║
║   Input | GT | Pred                                         ║
╚══════════════════════════════════════════════════════════════╝
```

---

## 🔍 What Gets Logged?

### Per Epoch:
- ✅ Train/Valid Loss
- ✅ Train/Valid Accuracy
- ✅ Train/Valid IoU
- ✅ Train/Valid F1
- ✅ Learning Rate (train only)
- ✅ Per-class IoU (11 classes)

### Per Training Run:
- ✅ All hyperparameters
- ✅ Model architecture info
- ✅ Total/trainable/frozen parameters
- ✅ Training stage transitions
- ✅ Best achieved metrics
- ✅ Prediction samples (optional)

### File Structure:
```
training_logs/
└── run_20250122_143052/
    ├── metrics.json              # All metrics
    ├── hyperparameters.json      # Hyperparameters
    ├── model_info.json           # Model details
    ├── stage_changes.json        # Training stages
    ├── summary.json              # Final summary
    └── predictions/              # Prediction samples
        ├── epoch_005/
        │   ├── valid_sample_00.npz
        │   └── valid_sample_01.npz
        └── epoch_010/
            └── ...
```

---

## 🎛️ Customization

### Change Refresh Rate
In `dashboard.py`, line ~215:
```python
time.sleep(10)  # Change to 5 for faster refresh
```

### Change Number of Samples Displayed
In `dashboard.py`, line ~279:
```python
for idx, sample_file in enumerate(sample_files[:6]):  # Change 6 to more/less
```

### Add Custom Metrics
In `training_logger.py`, `log_metrics()` method:
```python
def log_metrics(self, phase, loss, accuracy, iou, f1, lr=None, per_class_iou=None, custom_metric=None):
    # Add custom metric tracking
    self.metrics[phase]['custom'].append(custom_metric)
```

---

## 💡 Tips

1. **Start dashboard BEFORE training** for true real-time monitoring
2. **Use auto-refresh** to see live updates
3. **Compare multiple runs** by selecting different runs from dropdown
4. **Save important runs** - dashboard reads from `training_logs/` directory
5. **Check per-class IoU** to see which classes need more work

---

## 🐛 Troubleshooting

### Dashboard shows "No training runs found"
- Make sure `training_logs/` directory exists
- Check that logger was initialized in training code

### Plots not updating
- Click "🔄 Refresh Now" button
- Or enable "Auto-refresh"
- Verify metrics.json is being updated

### Import error for TrainingLogger
- Make sure `training_logger.py` is in `Code/training/` folder
- Check Python path includes training directory

### Prediction samples not showing
- Make sure you're calling `log_prediction_sample()`
- Check `predictions/` folder exists in run directory

---

## 🚀 Next Steps

1. **Install dependencies**: `pip install streamlit plotly pandas`
2. **Integrate logger** into Main_gridsearch.py (use code snippets above)
3. **Start training** with logging enabled
4. **Launch dashboard** in separate terminal
5. **Enable auto-refresh** and watch your model train!

---

## 📸 Example Outputs

The dashboard will show you:
- **Is your model learning?** (IoU/F1 going up)
- **Is LR decaying correctly?** (Smooth curve, not collapsed)
- **Which classes struggle?** (Per-class IoU heatmap)
- **Are predictions improving?** (Visual samples)
- **Any overfitting?** (Train vs Valid gap)

All in one beautiful, real-time interface! 🎨

---

**Ready to visualize your training? Follow the integration steps above!**
