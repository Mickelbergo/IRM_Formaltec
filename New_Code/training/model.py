import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from transformers import SwinModel
from torchvision.models import swin_b, swin_v2_t, swin_v2_s
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

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



class UNetDecoder(nn.Module):
    def __init__(self, decoder_channels):
        super(UNetDecoder, self).__init__()
        
        self.up1 = None
        self.up2 = nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2) #increse dimension (upsampling)
        self.up3 = nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(decoder_channels[3], decoder_channels[4], kernel_size=3, padding=1)

    def set_first_layer(self, in_channels, device):
        self.up1 = nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2).to(device)

    def forward(self, x, target_size):
        if self.up1 is None:
            raise ValueError("First upsampling layer is not set. Call set_first_layer() with the correct input channels.")

        # Upsampling steps
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        # Final convolution
        x = self.final_conv(x)

        # Crop or pad the output to match the target size
        output_size = x.shape[-2:]
        if output_size != target_size:
            x = nn.functional.interpolate(x, size=target_size, mode= "bilinear", align_corners=True)

        return x

class UNetWithSwinTransformer(nn.Module):
    def __init__(self, classes=2, activation='sigmoid'):
        super(UNetWithSwinTransformer, self).__init__()
        
        # Load Swin Transformer
        self.encoder = swin_v2_t()

        return_nodes = {
            "features.0": "stage1",
            "features.1": "stage2",
            "features.2": "stage3",
            "features.3": "stage4",
        }
        self.feature_extractor = create_feature_extractor(self.encoder, return_nodes=return_nodes)
        
        # U-Net decoder
        self.decoder = UNetDecoder(
            decoder_channels=[512, 256, 128, 64, 32]
        )
        
        # Final segmentation head
        self.segmentation_head = nn.Conv2d(32, classes, kernel_size=1)
        self.activation = nn.Sigmoid() if activation == 'sigmoid' else nn.Softmax(dim=1)
    
    def forward(self, x):
        target_size = x.shape[-2:]  # Store the original input size (height, width)
        features = self.feature_extractor(x)

        # Dynamically set the first decoder layer based on encoder output
        stage4_features = features["stage4"]
        self.decoder.set_first_layer(stage4_features.shape[1], device = x.device)
        
        # Decode and ensure the output size matches the input size
        decoder_output = self.decoder(stage4_features, target_size=target_size)
        
        # Final segmentation output
        segmentation_output = self.segmentation_head(decoder_output)
        
        return self.activation(segmentation_output)
    
class F_RCNN:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def get_faster_rcnn(self):
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = nn.Linear(in_features, self.num_classes)
        return model
    
# Example usage
if __name__ == "__main__":
    model = UNetWithSwinTransformer(classes=2, activation='sigmoid')
    input_image = torch.randn(8, 3, 400, 1024)  # Batch of 8 images, each 400x1024 pixels
    output = model(input_image)
    print(output.shape)  # Should match [batch_size, num_classes, height, width] = [8, 2, 400, 1024]
