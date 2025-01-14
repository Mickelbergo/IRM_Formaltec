import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
import json
import matplotlib.pyplot as plt
import model
from model import UNetWithClassification, UNetWithSwinTransformer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load configurations from the config files
with open('New_Code/configs/training_config.json') as f:
    train_config = json.load(f)
with open('New_Code/configs/preprocessing_config.json') as f:
    preprocessing_config = json.load(f)

# Set device from config
device = torch.device(train_config["device"] if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set paths
model_path = os.path.join(train_config["path"], f"best_model_{train_config['model_version']}_58_transformer.pth") #this needs to be cahnged always
model_path = os.path.join(train_config["path"], "best_model_v1.4_epoch29_encoder_se_resnext101_32x4d_seg_multiclass_lambda1.0_optadamw_lr0.0001_dice+ce_wr50_200_samplerTrue_iou0.2669_f10.3336.pth")
image_dir = os.path.join(train_config["path"], "example_images")
output_dir = os.path.join(train_config["path"], "example_images_segmented")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Map 'Model' to 'model' in sys.modules if the module was named 'Model' during saving
sys.modules['Model'] = model  # Optional: Only if needed to resolve module name discrepancies


if train_config["encoder"] == "transformer": #using SWIN transformer from huggingface with pretrained weights
    model = UNetWithSwinTransformer(classes = train_config["segmentation_classes"], activation = train_config["activation"])
else:
    model = UNetWithClassification(
        encoder_name=train_config["encoder"],
        encoder_weights=train_config["encoder_weights"],
        classes= train_config["segmentation_classes"],  # Segmentation classes (e.g., wound vs. background),  # Replace with the actual number of wound classes
        activation=None #crossentropy loss expects raw logits
    )

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print("Model loaded successfully.")

# Get list of image files
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)])
image_files = image_files[:50]  # Process only the first 50 images

# Define preprocessing transformations
target_size = tuple(preprocessing_config["target_size"])  # Should be (H, W)
print(f"Target size from preprocessing config: {target_size}")

if len(target_size) == 2:
    height, width = target_size
else:
    raise ValueError("Invalid target_size in preprocessing_config")

preprocess = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=preprocessing_config["mean"], std=preprocessing_config["std"])
])

print(f"Preprocessing transformations: {preprocess}")

# Automatic color map generation based on the number of classes
def create_color_map(num_classes):
    cmap = plt.get_cmap('tab20', num_classes)  # Use a colormap with distinct colors
    color_map = {0: (0, 0, 0)}  # Ensure background (class 0) is always black
    for i in range(1, num_classes):
        rgba = cmap(i)
        rgb = tuple((np.array(rgba[:3]) * 255).astype(int))
        color_map[i] = rgb
    return color_map

color_map = create_color_map(train_config["segmentation_classes"])

# Loop over images and perform segmentation
for img_name in image_files:
    img_path = os.path.join(image_dir, img_name)
    img = Image.open(img_path).convert('RGB')

    print(f"\nProcessing {img_name}")
    print(f"Original image size: {img.size}")  # (Width, Height)

    # Keep a copy of the original image for overlay
    original_img = img.copy()

    # Preprocess the image
    input_img = preprocess(img)
    print(f"Input image shape after preprocessing: {input_img.shape}")  # (C, H, W)

    input_img = input_img.unsqueeze(0).to(device)  # Add batch dimension
    print(f"Model input shape: {input_img.shape}")  # (1, C, H, W)

    # Perform inference
    with torch.no_grad():
        output = model(input_img)
        if isinstance(output, (tuple, list)):
            segmentation_output = output[0]
        else:
            segmentation_output = output

        print(f"Raw model output shape: {segmentation_output.shape}")

        # Apply activation function from config
        activation = train_config.get("activation")
        if activation == "softmax":
            segmentation_output = torch.softmax(segmentation_output, dim=1)
        elif activation == "sigmoid":
            segmentation_output = torch.sigmoid(segmentation_output)
        # If no activation, proceed with raw outputs

        print(f"Segmentation output shape after activation: {segmentation_output.shape}")

        # For multi-class segmentation, use argmax over the channel dimension (dim=1)
        predicted_mask = torch.argmax(segmentation_output, dim=1)  # Shape: [batch_size, H, W]
        predicted_mask = predicted_mask.squeeze(0).cpu().numpy()    # Remove batch dimension



        #if you want to dispaly the predicted masks ############################

        # plt.imshow(predicted_mask, cmap='gray')
        # plt.title('Predicted Mask')
        # plt.show()
        # Ensure predicted_mask is of type uint8




        # Ensure predicted_mask is of type uint8
        predicted_mask = predicted_mask.astype(np.uint8)
        print(f"Predicted mask shape: {predicted_mask.shape}")
        print(f"Unique mask values: {np.unique(predicted_mask)}")

    # Create the color mask
    color_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        color_mask[predicted_mask == class_id] = color

    # Resize the color mask to match the original image size
    if original_img.size != (predicted_mask.shape[1], predicted_mask.shape[0]):  # PIL size is (Width, Height)
        color_mask_pil = Image.fromarray(color_mask)
        color_mask_pil = color_mask_pil.resize(original_img.size, resample=Image.NEAREST)
        color_mask = np.array(color_mask_pil)

        print(f"Resized color mask shape: {color_mask.shape}")
    else:
        color_mask_pil = Image.fromarray(color_mask)

    # Overlay the color mask on the original image
    original_img_np = np.array(original_img)
    if original_img_np.shape != color_mask.shape:
        print(f"Warning: Dimension mismatch between original image and color mask.")
        print(f"Original image shape: {original_img_np.shape}, Color mask shape: {color_mask.shape}")

    overlay = (0.7 * original_img_np + 0.3 * color_mask).astype(np.uint8)

    # Save the overlaid image
    overlay_img = Image.fromarray(overlay)
    output_path = os.path.join(output_dir, img_name)
    overlay_img.save(output_path)
    print(f"Saved segmented image: {output_path}")

print("\nProcessing completed.")
