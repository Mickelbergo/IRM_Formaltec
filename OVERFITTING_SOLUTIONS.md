# Overfitting Solutions Implemented

## Problem Statement

Training a 1.1B parameter DINOv2-giant model on ~2000 images resulted in:
- **Training loss**: Continuously decreasing
- **Training metrics**: Continuously improving
- **Validation IoU**: Plateauing at ~0.4
- **Validation F1**: Plateauing at ~0.5

This is a classic **overfitting problem** where the model memorizes training data but fails to generalize to unseen validation data.

---

## Solutions Implemented

### 1. Stochastic Depth (DropPath) - **IMPLEMENTED**

**What it does**: Randomly drops entire transformer blocks during training, forcing the model to learn more robust representations.

**Why it works**:
- Standard dropout only drops individual neurons
- Stochastic depth drops entire residual paths in transformers
- Proven effective in all modern ViT architectures (DeiT, Swin, DINOv2)
- Particularly important for very deep models like DINOv2-giant (40 layers)

**Implementation**:
- Added `stochastic_depth_rate` parameter to model initialization
- Configured via `training_config.json`: `"stochastic_depth_rate": 0.1`
- Automatically applied to transformer encoder during model loading
- Rate of 0.1 means 10% of blocks are randomly dropped during each forward pass

**Code Changes**:
- [model.py:92-113](Code/training/model.py#L92-L113) - Added stochastic depth to ViT initialization
- [Main_gridsearch.py:122-123](Code/training/Main_gridsearch.py#L122-L123) - Pass parameter to model
- [training_config.json:20](Code/configs/training_config.json#L20) - Configuration

**Expected Impact**: **+2-5% IoU** based on ViT papers

**References**:
- Deep Networks with Stochastic Depth: https://arxiv.org/abs/1603.09382
- DeiT: https://arxiv.org/abs/2012.12877

---

### 2. Early Stopping with Patience - **IMPLEMENTED**

**What it does**: Automatically stops training when validation performance stops improving for N consecutive epochs.

**Why it works**:
- Prevents wasted computation on epochs that don't improve the model
- Stops training at the optimal point before severe overfitting
- Standard best practice in deep learning

**Implementation**:
- Added patience-based early stopping to Stage 2 and Stage 3 of progressive training
- Configured via `training_config.json`: `"early_stopping_patience": 15`
- Tracks `epochs_without_improvement` counter
- Stops training if validation IoU doesn't improve for `patience` epochs

**Code Changes**:
- [Main_gridsearch.py:635-661](Code/training/Main_gridsearch.py#L635-L661) - Stage 2 early stopping
- [Main_gridsearch.py:714-740](Code/training/Main_gridsearch.py#L714-L740) - Stage 3 early stopping
- [training_config.json:23](Code/configs/training_config.json#L23) - Configuration

**Expected Impact**: **Saves ~20-30% compute time**, prevents overfitting

---

### 3. Model Configuration Updates - **IMPLEMENTED**

Updated [training_config.json](Code/configs/training_config.json) with new hyperparameters:

```json
"vit_config": {
    "dropout_rate": 0.5,                    # Existing - decoder dropout
    "stochastic_depth_rate": 0.1,           # NEW - encoder regularization
    "use_ema": true,                        # NEW - for future implementation
    "ema_decay": 0.9999,                    # NEW - EMA decay rate
    "early_stopping_patience": 15           # NEW - early stopping
}
```

---

## Additional Recommendations (Not Yet Implemented)

### 4. EMA (Exponential Moving Average) - **CODE READY**

**What it does**: Maintains a slow-moving average of model weights during training. Often outperforms the final trained model.

**Why recommended**:
- Key technique used in DINOv2 and DINO papers
- Typically provides +1-2% IoU improvement
- Free performance boost with minimal computational cost

**Implementation prepared**:
- [ema.py](Code/training/ema.py) - Complete EMA implementation ready to use
- Two versions: `ModelEMA` (simple) and `ModelEMAv2` (with warmup)
- Requires integration into training loop (update after each batch/epoch)

**Expected Impact**: **+1-2% IoU**

**To enable**:
1. Initialize EMA model after creating main model
2. Call `ema.update(model)` after each training step
3. Evaluate both standard and EMA models on validation set
4. Save whichever performs better

---

### 5. Label Smoothing

**What it does**: Softens hard labels (0 or 1) to softer targets (e.g., 0.1 or 0.9), preventing overconfident predictions.

**Why recommended**:
- Reduces overfitting by preventing the model from becoming too confident
- Improves calibration and generalization
- Simple one-line change to loss function

**How to implement**:
```python
# In create_loss_functions():
CE_Loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```

**Expected Impact**: **+1-3% IoU**

**References**:
- Rethinking the Inception Architecture: https://arxiv.org/abs/1512.00567

---

### 6. Test-Time Augmentation (TTA)

**What it does**: Makes multiple predictions on augmented versions of the same image and averages them.

**Why recommended**:
- Improves predictions at inference time without retraining
- Typically provides +2-4% IoU on medical imaging tasks
- No training cost - only inference cost

**How to implement**:
- Create horizontally flipped versions of test images
- Run model on original + flipped versions
- Average the predictions

**Expected Impact**: **+2-4% IoU** (inference only)

---

### 7. MixUp/CutMix Augmentation

**What it does**: Creates synthetic training samples by mixing images and their labels.

**Why recommended**:
- Effectively increases dataset size
- Proven effective for medical image segmentation
- Smooths decision boundaries

**Implementation complexity**: Medium (requires modifications to data loader and loss calculation)

**Expected Impact**: **+3-5% IoU** on small datasets

**References**:
- MixUp: https://arxiv.org/abs/1710.09412
- CutMix: https://arxiv.org/abs/1905.04899

---

## Summary of Changes

| Change | Status | Difficulty | Expected Impact | Code Modified |
|--------|--------|-----------|-----------------|---------------|
| Stochastic Depth | ✅ Implemented | Easy | +2-5% IoU | model.py, Main_gridsearch.py, training_config.json |
| Early Stopping | ✅ Implemented | Easy | Saves 20-30% compute | Main_gridsearch.py, training_config.json |
| EMA | 📦 Code Ready | Medium | +1-2% IoU | ema.py created, needs integration |
| Label Smoothing | 📋 Planned | Easy | +1-3% IoU | One line in create_loss_functions() |
| Test-Time Aug | 📋 Planned | Easy | +2-4% IoU (inference) | New inference script |
| MixUp/CutMix | 📋 Planned | Hard | +3-5% IoU | Data loader + loss modifications |

---

## Recommended Next Steps

### Immediate (Run these now):
1. **Train with current changes** - Test stochastic depth + early stopping
2. **Enable label smoothing** - Add `label_smoothing=0.1` to CrossEntropyLoss
3. **Monitor training carefully** - Use the dashboard to track train/valid gap

### Short-term (Next week):
4. **Integrate EMA** - Use the prepared ema.py code
5. **Implement TTA** - Add to inference/prediction scripts
6. **Experiment with dropout rates** - Try 0.3, 0.4, 0.5 for decoder

### Long-term (Future work):
7. **Add MixUp/CutMix** - If still overfitting after above changes
8. **Collect more data** - Ultimate solution for overfitting
9. **Try smaller model** - DINOv2-large (300M) instead of giant (1.1B)

---

## Expected Final Performance

With **stochastic depth + early stopping + EMA + label smoothing**:
- **Validation IoU**: 0.5-0.6 (up from 0.4)
- **Validation F1**: 0.6-0.7 (up from 0.5)
- **Training time**: Reduced by 20-30% due to early stopping

With **all techniques** (including TTA at inference):
- **Validation IoU**: 0.6-0.7
- **Validation F1**: 0.7-0.75

---

## Configuration File Updates

### Current training_config.json (vit_config section):
```json
{
    "model_name": "facebook/dinov2-giant",
    "stage1_epochs": 30,
    "stage2_epochs": 30,
    "unfreeze_layers_stage2": 8,
    "unfreeze_layers_stage3": 16,
    "stage2_lr_factor": 0.3,
    "stage3_lr_factor": 0.05,
    "dropout_rate": 0.5,
    "stochastic_depth_rate": 0.1,        // NEW
    "use_ema": true,                     // NEW
    "ema_decay": 0.9999,                 // NEW
    "early_stopping_patience": 15        // NEW
}
```

---

## Training Monitoring

When training with these changes, monitor:
1. **Train/valid IoU gap** - Should be smaller now (ideally < 0.1)
2. **Early stopping triggers** - Check if stopping happens in Stage 2/3
3. **Per-class IoU** - Ensure no class is being ignored
4. **Dashboard metrics** - Use Streamlit dashboard for real-time visualization

Launch dashboard:
```bash
streamlit run Code/training/dashboard.py
```

---

## References

1. **Stochastic Depth**: Huang et al., "Deep Networks with Stochastic Depth" (ECCV 2016)
2. **EMA**: Tarvainen & Valpola, "Mean teachers are better role models" (NeurIPS 2017)
3. **Label Smoothing**: Szegedy et al., "Rethinking the Inception Architecture" (CVPR 2016)
4. **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features" (2023)
5. **Medical Image Segmentation**: Isensee et al., "nnU-Net" (Nature Methods 2021)

---

**Generated**: 2025-10-22
**Model**: DINOv2-giant (1.1B parameters)
**Dataset**: ~2000 forensic wound images
**Problem**: Overfitting (validation plateaus at IoU 0.4, F1 0.5)
**Solution**: Stochastic depth + early stopping + (future: EMA, label smoothing, TTA)
