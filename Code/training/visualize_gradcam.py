import torch
from torchcam.methods import GradCAM
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
import sys
import json
import os
# Import models
from model import UNetWithClassification, UNetWithSwinTransformer



with open('New_Code/configs/training_config.json') as f:
    train_config = json.load(f)


def get_config(cfg, *keys, legacy_key=None):
    d = cfg
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return cfg.get(legacy_key or keys[-1])
    return d

    


def load_image(image_path, target_size=(384, 384)):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    return transform(img)  # Returns a tensor

# Function to visualize Grad-CAM
def visualize_gradcam(model, input_tensor, target_class, target_layer, save_path=None, show=True, config_path='New_Code/configs/preprocessing_config.json'):
    model.eval()
    input_tensor = input_tensor.to(next(model.parameters()).device)
    input_tensor.requires_grad_()
    # Load normalization from config
    with open(config_path, 'r') as f:
        preprocessing_config = json.load(f)
    mean = np.array(preprocessing_config['normalization']['mean'])
    std = np.array(preprocessing_config['normalization']['std'])
    # 1. Create the GradCAM extractor
    cam_extractor = GradCAM(model, target_layer=target_layer)
    # 2. Forward pass (registers hooks)
    output = model(input_tensor)
    # 3. Compute CAM for the chosen class
    cam = cam_extractor(target_class, output)[0].cpu().numpy()  # shape: (H, W) or (1, H, W)
    # Prepare the input image for overlay
    input_img_np = input_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    # De-normalize for visualization
    input_img_np = (input_img_np * std) + mean
    input_img_np = np.clip(input_img_np, 0, 1)
    # Squeeze and normalize the CAM
    cam = np.squeeze(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # Normalize to [0, 1]
    # Resize CAM to match input image size if needed
    if cam.shape != input_img_np.shape[:2]:
        cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)  # [1, 1, Hc, Wc]
        cam_resized = F.interpolate(cam_tensor, size=input_img_np.shape[:2], mode='bilinear', align_corners=False)
        cam = cam_resized.squeeze().numpy()
    # Plot: Grad-CAM heatmap, original image, and overlay
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # 1. Grad-CAM heatmap
    heatmap = axs[0].imshow(cam, cmap='jet')
    axs[0].set_title('Grad-CAM Heatmap', fontsize=14)
    axs[0].axis('off')
    cbar = plt.colorbar(heatmap, ax=axs[0], fraction=0.046, pad=0.04)
    cbar.set_label('Grad-CAM Intensity', fontsize=12)
    # 2. Original image
    axs[1].imshow(input_img_np)
    axs[1].set_title('Original Image', fontsize=14)
    axs[1].axis('off')
    # 3. Overlay
    axs[2].imshow(input_img_np, alpha=1.0)
    axs[2].imshow(cam, cmap='jet', alpha=0.3)
    axs[2].set_title('Overlay', fontsize=14)
    axs[2].axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    # Remove hooks after visualization
    try:
        cam_extractor.clear_hooks()
    except AttributeError:
        cam_extractor.reset_hooks()

# Standalone usage for testing
if __name__ == "__main__":
        
    model_type = get_config(train_config, "model", "encoder", legacy_key="encoder")

    model_path = "E:/projects/Wound_Segmentation_III/Data/best_models_multiclass/best_model_v1.5_epoch32_encoder_timm-efficientnet-l2_seg_multiclass_lambda5_optadamw_lr0.0003_dice+ce_wr50_200_samplerFalse_iou0.4858_f10.5855.pth"

    image_path = "E:/projects/Wound_Segmentation_III/Data/new_images_640_1280/2971800336_IMG_0566.JPG2039784.png"

    num_classes = get_config(train_config, "model", "segmentation_classes", legacy_key="segmentation_classes")

    target_class = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # For UNetWithClassification (using segmentation_models_pytorch FPN/Unet)
    UNET_TARGET_LAYER = "segmentation_model.decoder.seg_blocks.3.block.0.block.2"  # Last decoder conv block
    SWIN_TARGET_LAYER = "up4.0"

    if model_type != "transfomrmer":
        model = UNetWithClassification(
            encoder_name="timm-efficientnet-l2",
            encoder_weights="noisy-student-475",
            classes=11,
            activation="softmax"
        )
        target_layer = UNET_TARGET_LAYER
    else:
        model = UNetWithSwinTransformer(classes=num_classes)
        target_layer = SWIN_TARGET_LAYER

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)

        
    input_tensor = load_image(image_path).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_()  # Enable gradients for Grad-CAM

        
    #print("Available submodules:")
    #for name, _ in model.named_modules():
    #    print(name)

    visualize_gradcam(model, input_tensor, target_class, target_layer)