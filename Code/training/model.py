import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from transformers import SwinModel, AutoModel, AutoConfig
from torchvision.models import swin_b, swin_v2_t, swin_v2_s, swin_v2_b, Swin_V2_B_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F

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
    

class UNetWithViT(nn.Module):
    """
    U-Net with Vision Transformer (DINOv2) encoder for state-of-the-art performance.
    
    Uses DINOv2 which achieves excellent results on various vision tasks:
    - Self-supervised learning with strong representations
    - Multi-scale feature extraction
    - Robust to domain shifts
    """
    def __init__(self, classes=2, activation=None, model_name="facebook/dinov2-base", dropout_rate=0.3, stochastic_depth_rate=0.1):
        super(UNetWithViT, self).__init__()

        # Available DINOv2 models (choose based on computational budget):
        # "facebook/dinov2-small" - 22M params, 384 dim
        # "facebook/dinov2-base" - 86M params, 768 dim
        # "facebook/dinov2-large" - 300M params, 1024 dim
        # "facebook/dinov2-giant" - 1.1B params, 1536 dim (best performance)

        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.stochastic_depth_rate = stochastic_depth_rate

        # Load pre-trained DINOv2 model
        print(f"Loading {model_name}...")
        config = AutoConfig.from_pretrained(model_name)
        # Enable stochastic depth (DropPath) in encoder
        if stochastic_depth_rate > 0:
            config.drop_path_rate = stochastic_depth_rate
            print(f"Stochastic Depth enabled: drop_path_rate={stochastic_depth_rate}")
        self.vit_encoder = AutoModel.from_pretrained(model_name, config=config)
        self.config = config
        
        # Get model dimensions
        self.hidden_size = self.config.hidden_size  # 768 for base, 1024 for large, etc.
        self.patch_size = self.config.patch_size    # Usually 14
        
        # Freeze encoder initially - all parameters frozen for Stage 1 training
        self.freeze_encoder()

        # Print parameter counts for verification
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Frozen params: {total_params - trainable_params:,}")
        
        # Feature projection layers for multi-scale features
        # We'll extract features from multiple transformer blocks
        self.feature_dims = {
            'early': self.hidden_size // 4,    # 192 for base
            'mid': self.hidden_size // 2,      # 384 for base  
            'late': self.hidden_size,          # 768 for base
            'final': self.hidden_size          # 768 for base
        }
        
        # Initialize projection layers with proper scaling for giant model
        def init_proj_layer(in_dim, out_dim):
            layer = nn.Conv2d(in_dim, out_dim, 1)
            if "giant" in model_name:
                # Use smaller initialization for giant model
                nn.init.xavier_normal_(layer.weight, gain=0.1)
                nn.init.constant_(layer.bias, 0)
            return layer
        
        self.early_proj = init_proj_layer(self.hidden_size, self.feature_dims['early'])
        self.mid_proj = init_proj_layer(self.hidden_size, self.feature_dims['mid'])
        self.late_proj = init_proj_layer(self.hidden_size, self.feature_dims['late'])
        self.final_proj = init_proj_layer(self.hidden_size, self.feature_dims['final'])

        #dropout
        self.encoder_dropout = nn.Dropout(dropout_rate)
        self.final_dropout = nn.Dropout(dropout_rate * 0.7)  # Slightly lower dropout before final layer

        # U-Net decoder with skip connections
        self.decoder4 = self._make_decoder_block(
            self.feature_dims['final'] + self.feature_dims['late'], 512
        )
        self.decoder3 = self._make_decoder_block(
            512 + self.feature_dims['mid'], 256
        )
        self.decoder2 = self._make_decoder_block(
            256 + self.feature_dims['early'], 128
        )
        self.decoder1 = self._make_decoder_block(128, 64)
        
        # Final segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, classes, kernel_size=1)
        )
        # Initialize final layer with xavier initialization
        nn.init.xavier_normal_(self.segmentation_head[-1].weight, gain=1.0)
        nn.init.constant_(self.segmentation_head[-1].bias, 0)

        # Activation function
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = None  # Return raw logits

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def extract_multiscale_features(self, x):
        """Extract features from multiple transformer layers"""
        batch_size, channels, height, width = x.shape
        
        # Calculate patch dimensions
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        
        # Get transformer outputs with intermediate hidden states
        outputs = self.vit_encoder(x, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states  # List of [B, N+1, D] tensors
        
        # Extract features from different layers
        num_layers = len(hidden_states)
        
        # Select layers for multi-scale features
        early_layer = hidden_states[num_layers // 4]      # Early features
        mid_layer = hidden_states[num_layers // 2]        # Mid-level features  
        late_layer = hidden_states[3 * num_layers // 4]  # Late features
        final_layer = hidden_states[-1]                  # Final features
        
        # Remove CLS tokens and reshape to spatial format
        def process_features(features, target_height, target_width):
            # Remove CLS token (first token)
            features = features[:, 1:, :]  # [B, H*W, D]
            B, N, D = features.shape
            
            # Reshape to spatial format
            features = features.transpose(1, 2).view(B, D, target_height, target_width)
            return features
        
        early_features = process_features(early_layer, patch_height, patch_width)
        mid_features = process_features(mid_layer, patch_height, patch_width)
        late_features = process_features(late_layer, patch_height, patch_width)
        final_features = process_features(final_layer, patch_height, patch_width)
        
        return {
            'early': early_features,
            'mid': mid_features, 
            'late': late_features,
            'final': final_features
        }

    def forward(self, x):
        original_size = x.shape[-2:]
        
        # Extract multi-scale features from ViT
        features = self.extract_multiscale_features(x)
        
        # Project features to desired dimensions
        early_feat = self.early_proj(features['early'])
        mid_feat = self.mid_proj(features['mid'])
        late_feat = self.late_proj(features['late'])
        final_feat = self.final_proj(features['final'])
        
        # Apply dropout to final features
        final_feat = self.encoder_dropout(final_feat)

        # U-Net decoder with skip connections
        # Start from deepest features
        x = final_feat
        
        # Decoder 4: Combine final + late features
        x = F.interpolate(x, size=late_feat.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, late_feat], dim=1)
        x = self.decoder4(x)
        
        # Decoder 3: Combine with mid features
        x = F.interpolate(x, size=mid_feat.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, mid_feat], dim=1)
        x = self.decoder3(x)
        
        # Decoder 2: Combine with early features
        x = F.interpolate(x, size=early_feat.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, early_feat], dim=1)
        x = self.decoder2(x)
        
        # Decoder 1: Final upsampling
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.decoder1(x)
        
        # Apply dropout before final segmentation head
        x = self.final_dropout(x)

        # Segmentation head
        x = self.segmentation_head(x)
        
        # Resize to original input size
        if x.shape[-2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=True)
        
        # Apply activation if specified
        if self.activation is not None:
            x = self.activation(x)
            
        return x


    def freeze_encoder(self):
        """Freeze all encoder parameters"""
        for param in self.vit_encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen")
    
    def unfreeze_encoder(self, num_layers=None):
        """
        Unfreeze encoder layers for fine-tuning
        Args:
            num_layers: Number of layers to unfreeze from the end. If None, unfreezes all layers
        """
        if num_layers is None:
            # Unfreeze ALL encoder parameters
            for param in self.vit_encoder.parameters():
                param.requires_grad = True
            print("Unfroze ALL encoder layers")
        else:
            # Unfreeze only last num_layers
            if hasattr(self.vit_encoder, 'encoder') and hasattr(self.vit_encoder.encoder, 'layer'):
                total_layers = len(self.vit_encoder.encoder.layer)
                layers_to_unfreeze = min(num_layers, total_layers)
                
                for layer in self.vit_encoder.encoder.layer[-layers_to_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                
                print(f"Unfroze last {layers_to_unfreeze} layers of ViT encoder")
            else:
                print("Warning: ViT encoder structure not recognized for partial unfreezing")
                print(f"Available attributes: {dir(self.vit_encoder)}")
        
        # Print updated parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"After unfreezing - Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    def unfreeze_all(self):
        """Convenience method to unfreeze all parameters"""
        self.unfreeze_encoder(num_layers=None)

        
    def get_trainable_params_info(self):
        """Get information about trainable parameters"""
        encoder_trainable = sum(p.numel() for p in self.vit_encoder.parameters() if p.requires_grad)
        decoder_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) - encoder_trainable
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'encoder_trainable': encoder_trainable,
            'decoder_trainable': decoder_trainable,
            'total_trainable': total_trainable,
            'total_params': total_params,
            'trainable_percentage': 100 * total_trainable / total_params
        }
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
