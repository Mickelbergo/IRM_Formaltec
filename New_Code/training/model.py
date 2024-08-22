import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetWithClassification(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', classes=2, num_classes=14, activation='sigmoid'):
        super(UNetWithClassification, self).__init__()
        self.segmentation_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(self.segmentation_model.encoder.out_channels[-1], num_classes)  # num_classes corresponds to the number of wound classes
        )

    def forward(self, x):
        # Forward pass through UNet for segmentation
        y_pred = self.segmentation_model(x)
        
        # Forward pass through the encoder for classification
        features = self.segmentation_model.encoder(x)
        classification_output = self.classification_head(features[-1])
        
        return y_pred, classification_output
