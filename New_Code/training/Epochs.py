import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm as tqdm
import sys
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torchmetrics

class Epoch:
    def __init__(self, model, CE_Loss, DICE_Loss, segmentation, stage_name, device=None, display_image=False, verbose=True, nr_classes=15, scheduler = None):
        self.model = model
        self.CE_Loss = CE_Loss
        self.DICE_Loss = DICE_Loss
        self.segmentation = segmentation
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.display_image = display_image
        self._to_device()
        self.nr_classes = nr_classes
        self.scheduler = scheduler

    def _to_device(self):
        self.model.to(self.device)
        self.CE_Loss.to(self.device)
        if self.DICE_Loss is not None:
            self.DICE_Loss.to(self.device)

    def display_images(self, image, ground_truth, prediction_mask):
        # Display the original image, ground truth mask, and prediction for the first image in the batch
        clipped_image = torch.clip(image[0].permute(1, 2, 0).cpu(), 0, 1)
        clipped_image_np = clipped_image.numpy()
        plt.subplot(1, 3, 1)
        plt.imshow(clipped_image_np)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(ground_truth[0].cpu().numpy(), cmap='gray')  # Ground truth mask
        plt.title('Ground Truth Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(prediction_mask[0].cpu().numpy(), cmap='gray')  # Predicted mask
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.draw()
        plt.pause(0.2)

    def run(self, dataloader):
        self.on_epoch_start()

        logs = {}
        losses = []
        accs = []
        iou_scores = []

        # Initialize IoU metric
        JaccardIndex = torchmetrics.JaccardIndex(task="multiclass", num_classes=self.nr_classes).to(self.device)

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
            for batch_idx, (x, binary_mask, multiclass_mask, _) in enumerate(iterator):
                x, binary_mask, multiclass_mask = x.to(self.device), binary_mask.to(self.device), multiclass_mask.to(self.device)

                if self.segmentation == "binary":
                    mask = binary_mask.squeeze(1)  # Now mask is [Batch, H, W]
                else:
                    mask = multiclass_mask.squeeze(1)

                loss, y_pred = self.batch_update(x, mask)

                # Convert predicted mask to single-channel by taking argmax
                pred_mask = y_pred.argmax(dim=1)  # Now pred_mask is [Batch, H, W]

                if self.display_image and batch_idx % 20 == 0:
                    self.display_images(x, mask, pred_mask)

                # Record loss, accuracy, and IoU for the batch
                losses.append(loss.item())
                acc = (pred_mask == mask).float().mean().item()  # Accuracy

                iou_score = JaccardIndex(pred_mask, mask)
                accs.append(acc)
                iou_scores.append(iou_score.item())

                # Update logs with current metrics
                logs.update({
                    'lr': self.scheduler.get_last_lr()[0],
                    'loss': np.mean(losses),
                    'accuracy': np.mean(accs),
                    'iou_score': np.mean(iou_scores)
                })
                
                self.scheduler.step()
                
                if self.verbose:
                    iterator.set_postfix_str(f"LR: {logs['lr']:.4f}, Loss: {logs['loss']:.4f}, Acc: {logs['accuracy']:.4f}, IoU: {logs['iou_score']:.4f}")

        JaccardIndex.reset()

        return logs

class TrainEpoch(Epoch):
    def __init__(self, model, CE_Loss, DICE_Loss=None, segmentation="binary", optimizer=None, device=None, grad_clip_value=1.0, display_image=False, verbose=True, nr_classes=15, scheduler = None):
        super().__init__(model, CE_Loss, DICE_Loss, segmentation, stage_name='train', device=device, display_image=display_image, verbose=verbose, nr_classes=nr_classes, scheduler=scheduler)
        self.optimizer = optimizer
        self.grad_clip_value = grad_clip_value
        self.scaler = torch.cuda.amp.GradScaler()  # Initialize GradScaler for mixed precision

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, mask):
        self.optimizer.zero_grad()

        with torch.amp.autocast(device_type = self.device):
            # Forward pass through the model
            y_pred = self.model(x)

            # Calculate segmentation loss
            loss = self.CE_Loss(y_pred, mask)

            # Incorporate Dice loss if provided
            if self.DICE_Loss is not None:
                loss += self.DICE_Loss(y_pred, mask)

        # Backward pass with scaled loss
        self.scaler.scale(loss).backward()

        # Optionally clip gradients
        if self.grad_clip_value is not None:
            self.scaler.unscale_(self.optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

        # Update the model's parameters
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss, y_pred

class ValidEpoch(Epoch):
    def __init__(self, model, CE_Loss, DICE_Loss=None, segmentation="binary", device=None, display_image=False, verbose=True, nr_classes=15, scheduler = None):
        super().__init__(model, CE_Loss, DICE_Loss, segmentation, stage_name='valid', device=device, display_image=display_image, verbose=verbose, nr_classes=nr_classes, scheduler=scheduler)

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, mask):
        with torch.no_grad():
            with torch.amp.autocast(device_type = self.device):
                # Forward pass through the model
                y_pred = self.model(x)

                # Calculate segmentation loss
                loss = self.CE_Loss(y_pred, mask)

                if self.DICE_Loss is not None:
                    loss += self.DICE_Loss(y_pred, mask)

        return loss, y_pred
