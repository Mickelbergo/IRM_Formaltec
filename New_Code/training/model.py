import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from transformers import SwinModel
from torchvision.models import swin_b, swin_v2_t, swin_v2_s, swin_v2_b, Swin_V2_B_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#print(smp.encoders.get_encoder_names())

class UNetWithClassification(nn.Module):
    def __init__(self, encoder_name='resnet101', encoder_weights='imagenet', classes=2, activation='sigmoid'):
        super(UNetWithClassification, self).__init__()
        if encoder_name == "mit_b5":
            
            self.segmentation_model = smp.Unet(
                encoder_name=encoder_name, #resnet, mit_b5, etc -> segmentation_models.pytorch 
                encoder_weights=encoder_weights,
                classes=classes,
                activation=activation
            )
        else:
            self.segmentation_model = smp.UnetPlusPlus(
                encoder_name=encoder_name, #resnet, mit_b5, etc -> segmentation_models.pytorch 
                encoder_weights=encoder_weights,
                classes=classes,
                activation=activation
            )


    def forward(self, x):
        # Forward pass through UNet for segmentation
        y_pred = self.segmentation_model(x)
        
        return y_pred



########################
#transformer based

class UNetDecoder(nn.Module):
    def __init__(self, decoder_channels, num_classes):
        super(UNetDecoder, self).__init__()
        
        self.up1 = None
        self.up2 = nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2) #increse dimension (upsampling)
        self.up3 = nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2)
        self.up5 = nn.ConvTranspose2d(decoder_channels[3], decoder_channels[4], kernel_size=2, stride=2)

        #output number of classes
        self.final_conv = nn.Conv2d(decoder_channels[4], num_classes, kernel_size=1) #or kernel size = 3 + padding = 1 (we do not want to lose pixels)

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
        x = self.up5(x)

        # Final convolution
        x = self.final_conv(x)

        # Crop or pad the output to match the target size
        output_size = x.shape[-2:]
        # print(f'output size: {output_size}')
        if output_size != target_size:
            x = nn.functional.interpolate(x, size=target_size, mode= "bilinear", align_corners=True)

        return x

class UNetWithSwinTransformer(nn.Module):
    def __init__(self, classes=2, activation='sigmoid'):
        super(UNetWithSwinTransformer, self).__init__()
        
        # Load Swin Transformer encoder
        self.encoder = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)

        # Define the stages for skip connections
        return_nodes = {
            "features.0": "stage1",  
            "features.2": "stage2",  
            "features.4": "stage3",  
            "features.6": "stage4",  
        }
        self.feature_extractor = create_feature_extractor(self.encoder, return_nodes=return_nodes)

        # Decoder channels (must align with encoder outputs)
        self.decoder_channels = [1024, 512, 256, 128]
        self.num_classes = classes

        # U-Net decoder blocks
        self.up4 = self._decoder_block(self.decoder_channels[0], self.decoder_channels[1])
        self.up3 = self._decoder_block(self.decoder_channels[1], self.decoder_channels[2])
        self.up2 = self._decoder_block(self.decoder_channels[2], self.decoder_channels[3])
        self.up1 = self._decoder_block(self.decoder_channels[3], 64)  # 64 is the final spatial resolution

        # Final segmentation layer
        self.final_conv = nn.Conv2d(64, self.num_classes, kernel_size=1)

        # Activation function
        self.activation = nn.Sigmoid() if activation == 'sigmoid' else nn.Softmax(dim=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        # Store original input size for resizing output
        target_size = x.shape[-2:]

        # Extract features from encoder
        features = self.feature_extractor(x)
        features = {name: feature.permute(0, 3, 1, 2) for name, feature in features.items()}

        # Decoder with skip connections
        x = self.up4(features["stage4"]) + features["stage3"]
        x = self.up3(x) + features["stage2"]
        x = self.up2(x) + features["stage1"]
        x = self.up1(x)

        # Final segmentation layer
        x = self.final_conv(x)

        # Resize to match original input size
        if x.shape[-2:] != target_size:
            x = nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=True)

        # Apply activation
        return self.activation(x)


# Example usage
if __name__ == "__main__":
    model = UNetWithSwinTransformer(classes=2, activation='sigmoid')
    input_image = torch.randn(8, 3, 400, 1024)  # Batch of 8 images, each 400x1024 pixels
    output = model(input_image)
    print(output.shape)  # Should match [batch_size, num_classes, height, width] = [8, 2, 400, 1024]
