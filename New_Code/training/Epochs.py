import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm as tqdm
import sys
import matplotlib.pyplot as plt

class Epoch:
    def __init__(self, model, CE_Loss, DICE_Loss, segmentation, stage_name, device=None, display_image = False, verbose=True):
        self.model = model
        self.CE_Loss = CE_Loss
        self.DICE_Loss = DICE_Loss
        self.segmentation = segmentation
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.display_image = display_image
        self._to_device()

    # Move model and loss functions to the specified device (GPU or CPU)
    def _to_device(self):
        self.model.to(self.device)
        self.CE_Loss.to(self.device)
        self.DICE_Loss.to(self.device)


    def display_images(self, image, ground_truth, prediction_mask):

        # Display the original image, ground truth mask and prediction for the first image in the batch

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

        
        plt.draw()  # Draw the updated figure without blocking
        plt.pause(0.2)  # Small pause to allow the plot to update



    # Main method that runs through the dataset and processes batches
    def run(self, dataloader):
        self.on_epoch_start()

        logs = {}
        losses = []
        accs = []
        iou_scores = []
        

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
            for batch_idx, (x, binary_mask, multiclass_mask, _) in enumerate(iterator): 
                x, binary_mask, multiclass_mask = x.to(self.device), binary_mask.to(self.device), multiclass_mask.to(self.device)
                
                if self.segmentation == "binary":
                    # Squeeze the mask to remove the singleton dimension
                    mask = binary_mask.squeeze(1)  # Now mask is [Batch, 640, 640]

                    # Process the batch and calculate the loss
                    loss, y_pred = self.batch_update(x, mask)

                else:
                    mask = multiclass_mask.squeeze(1)
                    loss, y_pred = self.batch_update(x, mask)


                # Convert predicted mask to single-channel by taking argmax
                pred_mask = y_pred.argmax(dim=1)  # Now pred_mask is [Batch, 640, 640]

                if self.display_image == True and batch_idx % 10 == 0:
                    self.display_images(x, mask, pred_mask)


                # Record loss, accuracy, and IoU for the batch
                losses.append(loss.item())
                acc = (pred_mask == mask).float().mean().item()  # Accuracy
                iou_score = self.calculate_iou(pred_mask, mask)  # IoU calculation

                accs.append(acc)
                iou_scores.append(iou_score)

                # Update logs with current metrics
                logs.update({
                    'loss': np.mean(losses),
                    'accuracy': np.mean(accs),
                    'iou_score': np.mean(iou_scores)
                })

                if self.verbose:
                    iterator.set_postfix_str(logs)

        return logs

    # Method to calculate Intersection over Union (IoU)
    def calculate_iou(self, pred, target, n_classes=2):
        intersection = torch.logical_and(pred, target)
        union = torch.logical_or(pred, target)
        iou = torch.sum(intersection) / torch.sum(union)
        return iou.item()

# Training epoch class
class TrainEpoch(Epoch):
    def __init__(self, model, CE_Loss, DICE_Loss = None, segmentation = "binary", optimizer = None, device=None, grad_clip_value = 1.0, display_image = False, verbose=True):
        super().__init__(model, CE_Loss, DICE_Loss, segmentation, stage_name='train', device=device, display_image = display_image,  verbose=verbose)
        self.optimizer = optimizer
        self.grad_clip_value = grad_clip_value

    # Set the model to training mode at the start of each epoch
    def on_epoch_start(self):
        self.model.train()

    # Process a single batch and update the model's parameters
    def batch_update(self, x, mask):
        self.optimizer.zero_grad()

        # Forward pass through the model
        y_pred = self.model(x)

        # Calculate segmentation loss
        seg_loss = self.CE_Loss(y_pred, mask)

        #also incorporate dice loss
        if self.DICE_Loss != None:
            seg_loss2 = self.DICE_Loss(y_pred, mask)
        else: seg_loss2 = 0

        seg_loss = seg_loss + seg_loss2
        
        loss = seg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
        
        # Update the model's parameters
        self.optimizer.step()
        
        return loss, y_pred

# Validation epoch class
class ValidEpoch(Epoch):
    def __init__(self, model, CE_Loss, DICE_Loss = None, segmentation = "binary", device=None, display_image = False, verbose=True):
        super().__init__(model, CE_Loss, DICE_Loss, segmentation, stage_name='valid', device=device, display_image= display_image, verbose=verbose)

    # Set the model to evaluation mode at the start of each epoch
    def on_epoch_start(self):
        self.model.eval()

    # Process a single batch without updating the model's parameters
    def batch_update(self, x, mask):
        with torch.no_grad():
            # Forward pass through the model
            y_pred= self.model(x)

            # Calculate segmentation loss
            seg_loss = self.CE_Loss(y_pred, mask)

            if self.DICE_Loss != None:
                seg_loss2 = self.DICE_Loss(y_pred, mask)
            else: seg_loss2 = 0

            seg_loss = seg_loss + seg_loss2

            loss = seg_loss
        
        return loss, y_pred