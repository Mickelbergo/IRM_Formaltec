import json
import os
from datetime import datetime
import numpy as np
import torch
from pathlib import Path


class TrainingLogger:
    """
    Comprehensive training logger for wound segmentation models.
    Logs all metrics, hyperparameters, and predictions for visualization.
    """

    def __init__(self, log_dir="training_logs", run_name=None):
        """
        Initialize the training logger.

        Args:
            log_dir: Directory to save logs
            run_name: Name for this training run (auto-generated if None)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Generate run name
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"

        self.run_name = run_name
        self.run_dir = self.log_dir / run_name
        self.run_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.run_dir / "predictions").mkdir(exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        # Initialize metrics storage
        self.metrics = {
            'train': {
                'loss': [],
                'accuracy': [],
                'iou': [],
                'f1': [],
                'lr': [],
                'per_class_iou': []
            },
            'valid': {
                'loss': [],
                'accuracy': [],
                'iou': [],
                'f1': [],
                'per_class_iou': []
            }
        }

        self.current_epoch = 0
        self.current_stage = "Stage 1"
        self.hyperparams = {}
        self.model_info = {}

        print(f"📊 Training logger initialized: {self.run_dir}")

    def log_hyperparameters(self, hyperparams):
        """Log hyperparameters"""
        self.hyperparams = hyperparams
        with open(self.run_dir / "hyperparameters.json", 'w') as f:
            json.dump(hyperparams, f, indent=4)
        print(f"✅ Hyperparameters logged")

    def log_model_info(self, model, model_name, total_params, trainable_params):
        """Log model architecture information"""
        self.model_info = {
            'model_name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
            'trainable_percentage': 100 * trainable_params / total_params
        }

        with open(self.run_dir / "model_info.json", 'w') as f:
            json.dump(self.model_info, f, indent=4)
        print(f"✅ Model info logged: {trainable_params:,} / {total_params:,} params trainable")

    def log_epoch_start(self, epoch, stage="Stage 1"):
        """Mark the start of an epoch"""
        self.current_epoch = epoch
        self.current_stage = stage

    def log_metrics(self, phase, loss, accuracy, iou, f1, lr=None, per_class_iou=None):
        """
        Log metrics for an epoch.

        Args:
            phase: 'train' or 'valid'
            loss: Loss value
            accuracy: Accuracy value
            iou: IoU score
            f1: F1 score
            lr: Learning rate (only for training)
            per_class_iou: Per-class IoU scores (numpy array)
        """
        self.metrics[phase]['loss'].append(float(loss))
        self.metrics[phase]['accuracy'].append(float(accuracy))
        self.metrics[phase]['iou'].append(float(iou))
        self.metrics[phase]['f1'].append(float(f1))

        if phase == 'train' and lr is not None:
            self.metrics[phase]['lr'].append(float(lr))

        if per_class_iou is not None:
            self.metrics[phase]['per_class_iou'].append(per_class_iou.tolist())

        # Save metrics after each epoch
        self._save_metrics()

    def log_prediction_sample(self, image, ground_truth, prediction, sample_idx, phase='valid'):
        """
        Save a prediction sample for visualization.

        Args:
            image: Input image (CHW tensor)
            ground_truth: Ground truth mask (HW tensor)
            prediction: Predicted mask (HW tensor)
            sample_idx: Sample index
            phase: 'train' or 'valid'
        """
        save_dir = self.run_dir / "predictions" / f"epoch_{self.current_epoch:03d}"
        save_dir.mkdir(exist_ok=True)

        # Convert to numpy and save
        sample_data = {
            'image': image.cpu().numpy() if torch.is_tensor(image) else image,
            'ground_truth': ground_truth.cpu().numpy() if torch.is_tensor(ground_truth) else ground_truth,
            'prediction': prediction.cpu().numpy() if torch.is_tensor(prediction) else prediction,
            'epoch': self.current_epoch,
            'stage': self.current_stage,
            'phase': phase
        }

        save_path = save_dir / f"{phase}_sample_{sample_idx:02d}.npz"
        np.savez_compressed(save_path, **sample_data)

    def log_stage_change(self, stage_name, unfrozen_layers=None):
        """Log when training stage changes"""
        self.current_stage = stage_name

        stage_info = {
            'epoch': self.current_epoch,
            'stage': stage_name,
            'unfrozen_layers': unfrozen_layers
        }

        # Append to stage log
        stage_log_path = self.run_dir / "stage_changes.json"
        if stage_log_path.exists():
            with open(stage_log_path, 'r') as f:
                stage_log = json.load(f)
        else:
            stage_log = []

        stage_log.append(stage_info)

        with open(stage_log_path, 'w') as f:
            json.dump(stage_log, f, indent=4)

        print(f"🎯 Stage changed: {stage_name}")

    def _save_metrics(self):
        """Save metrics to JSON file"""
        with open(self.run_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=4)

    def get_current_metrics_summary(self):
        """Get summary of current metrics"""
        if len(self.metrics['valid']['iou']) == 0:
            return "No metrics yet"

        latest_train_iou = self.metrics['train']['iou'][-1] if self.metrics['train']['iou'] else 0
        latest_valid_iou = self.metrics['valid']['iou'][-1]
        latest_train_f1 = self.metrics['train']['f1'][-1] if self.metrics['train']['f1'] else 0
        latest_valid_f1 = self.metrics['valid']['f1'][-1]
        best_valid_iou = max(self.metrics['valid']['iou'])
        best_valid_f1 = max(self.metrics['valid']['f1'])

        return (
            f"Epoch {self.current_epoch} | {self.current_stage}\n"
            f"Train IoU: {latest_train_iou:.4f} | Valid IoU: {latest_valid_iou:.4f} (best: {best_valid_iou:.4f})\n"
            f"Train F1: {latest_train_f1:.4f} | Valid F1: {latest_valid_f1:.4f} (best: {best_valid_f1:.4f})"
        )

    def finalize(self):
        """Finalize logging and create summary"""
        summary = {
            'run_name': self.run_name,
            'total_epochs': self.current_epoch,
            'best_valid_iou': max(self.metrics['valid']['iou']) if self.metrics['valid']['iou'] else 0,
            'best_valid_f1': max(self.metrics['valid']['f1']) if self.metrics['valid']['f1'] else 0,
            'final_train_iou': self.metrics['train']['iou'][-1] if self.metrics['train']['iou'] else 0,
            'final_valid_iou': self.metrics['valid']['iou'][-1] if self.metrics['valid']['iou'] else 0,
            'hyperparameters': self.hyperparams,
            'model_info': self.model_info
        }

        with open(self.run_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"✅ Training completed. Logs saved to: {self.run_dir}")
        return summary
