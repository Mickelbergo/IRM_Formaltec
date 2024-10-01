import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm as tqdm
import sys
import matplotlib.pyplot as plt

class Epoch:
    def __init__(self, model, BCE_LOSS, DICE_Loss, classification_loss_fn, stage_name, device=None, verbose=True):
        self.model = model
        self.BCE_LOSS = BCE_LOSS
        self.DICE_Loss = DICE_Loss
        self.classification_loss_fn = classification_loss_fn
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self._to_device()

    # Move model and loss functions to the specified device (GPU or CPU)
    def _to_device(self):
        self.model.to(self.device)
        self.BCE_LOSS.to(self.device)
        self.DICE_Loss.to(self.device)
        self.classification_loss_fn.to(self.device)

    # Main method that runs through the dataset and processes batches
    def run(self, dataloader):
        self.on_epoch_start()

        logs = {}
        losses = []
        accs = []
        iou_scores = []
        
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
            for x, mask, mask_classes, _ in iterator:
                x, mask, mask_classes = x.to(self.device), mask.to(self.device), mask_classes.to(self.device)
                
                # Flatten the mask to remove any unnecessary dimensions
                
                mask = mask.squeeze(1)

                # Process the batch and calculate the loss
                loss, y_pred, class_pred = self.batch_update(x, mask, mask_classes)

                # Display the image
                maskk = mask
                plt.subplot(1, 2, 1)
                plt.imshow(maskk[0], cmap='gray')
                plt.title('mask')
                plt.axis('off')
                
                pred = y_pred.argmax(dim=1)
                # Display the corresponding mask
                plt.subplot(1, 2, 2)
                plt.imshow(pred[0], cmap='gray')
                plt.title('pred')
                plt.axis('off')
                
                plt.draw()
                plt.pause(0.001)

                # Record loss, accuracy, and IoU for the batch
                losses.append(loss.item())
                acc = (y_pred.argmax(dim=1) == mask).float().mean().item()
                iou_score = self.calculate_iou(y_pred.argmax(dim=1), mask)

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
    def __init__(self, model, BCE_LOSS, DICE_Loss = None,  classification_loss_fn = None , optimizer = None, device=None, verbose=True):
        super().__init__(model, BCE_LOSS, DICE_Loss, classification_loss_fn, stage_name='train', device=device, verbose=verbose)
        self.optimizer = optimizer

    # Set the model to training mode at the start of each epoch
    def on_epoch_start(self):
        self.model.train()

    # Process a single batch and update the model's parameters
    def batch_update(self, x, mask, mask_class):
        self.optimizer.zero_grad()

        # Forward pass through the model
        y_pred, class_pred = self.model(x)

        # Calculate segmentation loss
        seg_loss = self.BCE_LOSS(y_pred, mask)

        #also incorporate dice loss
        if self.DICE_Loss != None:
            seg_loss2 = self.DICE_Loss(y_pred, mask)
        else: seg_loss2 = 0

        seg_loss = seg_loss + seg_loss2
        

        ##not used at the moment
        #class_loss = self.classification_loss_fn(class_pred, mask_class)

        loss = seg_loss + 0 #class_loss
        loss.backward()

        # Update the model's parameters
        self.optimizer.step()
        
        return loss, y_pred, class_pred

# Validation epoch class
class ValidEpoch(Epoch):
    def __init__(self, model, BCE_LOSS, DICE_Loss = None,  classification_loss_fn = None, device=None, verbose=True):
        super().__init__(model, BCE_LOSS, DICE_Loss, classification_loss_fn, stage_name='valid', device=device, verbose=verbose)

    # Set the model to evaluation mode at the start of each epoch
    def on_epoch_start(self):
        self.model.eval()

    # Process a single batch without updating the model's parameters
    def batch_update(self, x, mask, mask_class):
        with torch.no_grad():
            # Forward pass through the model
            y_pred, class_pred = self.model(x)

            # Calculate segmentation loss
            seg_loss = self.BCE_LOSS(y_pred, mask)

            if self.DICE_Loss != None:
                seg_loss2 = self.DICE_Loss(y_pred, mask)
            else: seg_loss2 = 0

            seg_loss = seg_loss + seg_loss2


            #class_loss = self.classification_loss_fn(class_pred, mask_class)

            loss = seg_loss + 0 # class_loss
        
        return loss, y_pred, class_pred
