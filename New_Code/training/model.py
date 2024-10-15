import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetWithClassification(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', classes=2, activation='sigmoid'):
        super(UNetWithClassification, self).__init__()
        self.segmentation_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation
        )

    def forward(self, x):
        # Forward pass through UNet for segmentation
        y_pred = self.segmentation_model(x)
        
        return y_pred