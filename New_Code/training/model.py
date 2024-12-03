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
    def __init__(self, encoder_name='resnet152', encoder_weights='imagenet', classes=2, activation='sigmoid'):
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
    
        # Load Swin Transformer
        self.encoder = swin_v2_b(weights=Swin_V2_B_Weights)


        return_nodes = { #this gives shapes: 
            "features.0": "stage1",
            "features.2": "stage2",
            "features.4": "stage3",
            "features.6": "stage4",
        }
        # this gives shapes: 
        # stage1: torch.Size([4, 64, 64, 128])
        # stage2: torch.Size([4, 32, 32, 256])
        # stage3: torch.Size([4, 16, 16, 512])
        # stage4: torch.Size([4, 8, 8, 1024])
        #for some reason the channel dimension is switched, so we need to switch them again in the forward method


        self.feature_extractor = create_feature_extractor(self.encoder, return_nodes=return_nodes)
        
        # U-Net decoder
        self.decoder = UNetDecoder(
            decoder_channels=[512, 256, 128, 64, 32], 
            num_classes=classes
        )
        
        # Final segmentation head

        self.activation = nn.Sigmoid() if activation == 'sigmoid' else nn.Softmax(dim=1)
    
    def forward(self, x):
        target_size = x.shape[-2:] # Store the original input size (height, width)

        features = self.feature_extractor(x) #we need to switch the channel dimension to the 2nd position again
        features = {name: feature.permute(0, 3, 1, 2) for name, feature in features.items()}
        # for name, feature in features.items():
        #     print(f'{name}: {feature.shape}')


        # Dynamically set the first decoder layer based on encoder output
        stage4_features = features["stage4"]
        self.decoder.set_first_layer(stage4_features.shape[1], device = x.device)
        
        # Decode and ensure the output size matches the input size
        decoder_output = self.decoder(stage4_features, target_size=target_size)
        
        # Final segmentation output
        return self.activation(decoder_output)

class Faster_RCNN:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
    

    def get_faster_rcnn(self):
        weights = FasterRCNN_ResNet50_FPN_Weights

        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        return model



# Example usage
if __name__ == "__main__":
    model = UNetWithSwinTransformer(classes=2, activation='sigmoid')
    input_image = torch.randn(8, 3, 400, 1024)  # Batch of 8 images, each 400x1024 pixels
    output = model(input_image)
    print(output.shape)  # Should match [batch_size, num_classes, height, width] = [8, 2, 400, 1024]
