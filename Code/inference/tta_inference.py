"""
Test-Time Augmentation (TTA) for improved inference accuracy.

TTA applies multiple augmentations to input images, runs inference on each,
and averages the predictions. This typically provides +2-4% IoU improvement
on medical image segmentation tasks.

Usage:
    python Code/inference/tta_inference.py --model_path path/to/model.pth --image_path path/to/image.png

Supported augmentations:
    - Original (no augmentation)
    - Horizontal flip
    - Vertical flip
    - 90° rotation
    - 180° rotation
    - 270° rotation
    - Combinations of above

References:
    - TTA for medical imaging: https://arxiv.org/abs/1807.07356
    - Original TTA paper: https://doi.org/10.1007/978-3-642-40994-3_27
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
import json
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import UNetWithViT, UNetWithClassification, UNetWithSwinTransformer
from torchvision import transforms


class TTAWrapper:
    """
    Test-Time Augmentation wrapper for segmentation models.

    Applies multiple augmentations to input, runs model on each,
    reverses augmentations on predictions, and averages results.
    """

    def __init__(self, model, device='cuda', augmentations=None):
        """
        Args:
            model: Trained segmentation model
            device: Device to run inference on
            augmentations: List of augmentation names to apply.
                          If None, uses default set: ['original', 'hflip', 'vflip', 'rot90']
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Default augmentation set (fast and effective)
        if augmentations is None:
            augmentations = ['original', 'hflip', 'vflip', 'rot90']

        self.augmentations = augmentations

        # Augmentation and reverse functions
        self.aug_funcs = {
            'original': (lambda x: x, lambda x: x),
            'hflip': (
                lambda x: torch.flip(x, [3]),  # Flip width dimension
                lambda x: torch.flip(x, [3])   # Flip back
            ),
            'vflip': (
                lambda x: torch.flip(x, [2]),  # Flip height dimension
                lambda x: torch.flip(x, [2])   # Flip back
            ),
            'rot90': (
                lambda x: torch.rot90(x, k=1, dims=[2, 3]),
                lambda x: torch.rot90(x, k=-1, dims=[2, 3])
            ),
            'rot180': (
                lambda x: torch.rot90(x, k=2, dims=[2, 3]),
                lambda x: torch.rot90(x, k=-2, dims=[2, 3])
            ),
            'rot270': (
                lambda x: torch.rot90(x, k=3, dims=[2, 3]),
                lambda x: torch.rot90(x, k=-3, dims=[2, 3])
            ),
        }

    def forward(self, x):
        """
        Apply TTA and return averaged predictions.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Averaged predictions [B, num_classes, H, W]
        """
        predictions = []

        with torch.no_grad():
            for aug_name in self.augmentations:
                if aug_name not in self.aug_funcs:
                    print(f"Warning: Unknown augmentation '{aug_name}', skipping")
                    continue

                aug_func, reverse_func = self.aug_funcs[aug_name]

                # Apply augmentation
                x_aug = aug_func(x)

                # Run model
                pred = self.model(x_aug)

                # Reverse augmentation on prediction
                pred_reversed = reverse_func(pred)

                predictions.append(pred_reversed)

        # Average all predictions
        avg_pred = torch.stack(predictions).mean(dim=0)

        return avg_pred

    def __call__(self, x):
        """Convenience method for forward pass."""
        return self.forward(x)


def load_model(model_path, config_path='Code/configs/training_config.json', device='cuda'):
    """
    Load trained model from checkpoint.

    Args:
        model_path: Path to model .pth file
        config_path: Path to training config JSON
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    encoder = config.get('model', {}).get('encoder', 'vit')
    num_classes = config.get('model', {}).get('segmentation_classes', 11)

    # Create model
    if encoder == 'vit':
        vit_config = config.get('model', {}).get('vit_config', {})
        model = UNetWithViT(
            classes=num_classes,
            activation=config.get('model', {}).get('activation'),
            model_name=vit_config.get('model_name', 'facebook/dinov2-large'),
            dropout_rate=vit_config.get('dropout_rate', 0.3),
            stochastic_depth_rate=vit_config.get('stochastic_depth_rate', 0.1)
        )
    elif encoder == 'transformer':
        model = UNetWithSwinTransformer(
            classes=num_classes,
            activation=config.get('model', {}).get('activation')
        )
    else:
        model = UNetWithClassification(
            encoder_name=encoder,
            encoder_weights=config.get('model', {}).get('encoder_weights', 'imagenet'),
            classes=num_classes,
            activation=None
        )

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    return model


def load_image(image_path, target_size=(384, 384)):
    """
    Load and preprocess image for inference.

    Args:
        image_path: Path to input image
        target_size: Target size (H, W)

    Returns:
        Preprocessed tensor [1, 3, H, W]
    """
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Resize
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)

    # Convert to tensor and normalize (ImageNet stats)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    return img_tensor, img


def save_prediction(pred_mask, output_path, original_size=None):
    """
    Save prediction mask as image.

    Args:
        pred_mask: Predicted class indices [H, W]
        output_path: Path to save output
        original_size: Optional (W, H) to resize output to original image size
    """
    # Convert to PIL image
    pred_mask = (pred_mask * 15).astype(np.uint8)  # Scale classes for visibility
    pred_img = Image.fromarray(pred_mask)

    # Resize if needed
    if original_size is not None:
        pred_img = pred_img.resize(original_size, Image.NEAREST)

    # Save
    pred_img.save(output_path)
    print(f"Prediction saved to {output_path}")


def visualize_prediction(image, pred_mask, output_path, class_names=None):
    """
    Create visualization with image and prediction side-by-side.

    Args:
        image: Original PIL image
        pred_mask: Predicted class indices [H, W]
        output_path: Path to save visualization
        class_names: Optional dict mapping class IDs to names
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Prediction
    im = axes[1].imshow(pred_mask, cmap='tab20', vmin=0, vmax=10)
    axes[1].set_title('TTA Prediction')
    axes[1].axis('off')

    # Add legend if class names provided
    if class_names:
        unique_classes = np.unique(pred_mask)
        patches = [mpatches.Patch(color=plt.cm.tab20(i/10), label=f"{i}: {class_names.get(str(i), 'Unknown')}")
                   for i in unique_classes if i in [int(k) for k in class_names.keys()]]
        plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"Visualization saved to {output_path}")


def run_tta_inference(model_path, image_path, output_dir='tta_outputs',
                       augmentations=None, visualize=True, config_path='Code/configs/training_config.json'):
    """
    Run TTA inference on a single image.

    Args:
        model_path: Path to trained model
        image_path: Path to input image
        output_dir: Directory to save outputs
        augmentations: List of augmentation names (None for default)
        visualize: Whether to create visualization
        config_path: Path to training config

    Returns:
        Predicted class mask as numpy array
    """
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    model = load_model(model_path, config_path, device)

    # Create TTA wrapper
    tta_model = TTAWrapper(model, device=device, augmentations=augmentations)
    print(f"TTA enabled with augmentations: {tta_model.augmentations}")

    # Load image
    img_tensor, original_img = load_image(image_path)
    img_tensor = img_tensor.to(device)

    # Run TTA inference
    print("Running TTA inference...")
    with torch.no_grad():
        pred_logits = tta_model(img_tensor)

    # Convert to class predictions
    pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()

    # Save outputs
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Save raw prediction mask
    mask_output_path = os.path.join(output_dir, f"{image_name}_tta_mask.png")
    save_prediction(pred_mask, mask_output_path, original_img.size)

    # Create visualization
    if visualize:
        # Load class names if available
        class_names = None
        try:
            preprocessing_config_path = 'Code/configs/preprocessing_config.json'
            if os.path.exists(preprocessing_config_path):
                with open(preprocessing_config_path, 'r') as f:
                    preproc_config = json.load(f)
                    class_names = preproc_config.get('class_names', {})
        except:
            pass

        viz_output_path = os.path.join(output_dir, f"{image_name}_tta_viz.png")
        visualize_prediction(original_img, pred_mask, viz_output_path, class_names)

    print(f"\nTTA inference complete!")
    print(f"  Prediction mask: {mask_output_path}")
    if visualize:
        print(f"  Visualization: {viz_output_path}")

    return pred_mask


def main():
    parser = argparse.ArgumentParser(description='Test-Time Augmentation Inference for Wound Segmentation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model .pth file')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='tta_outputs',
                        help='Directory to save outputs (default: tta_outputs)')
    parser.add_argument('--config_path', type=str, default='Code/configs/training_config.json',
                        help='Path to training config JSON')
    parser.add_argument('--augmentations', type=str, nargs='+',
                        choices=['original', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270'],
                        default=None,
                        help='Augmentations to use (default: original hflip vflip rot90)')
    parser.add_argument('--no_visualize', action='store_true',
                        help='Disable visualization output')

    args = parser.parse_args()

    # Run TTA inference
    run_tta_inference(
        model_path=args.model_path,
        image_path=args.image_path,
        output_dir=args.output_dir,
        augmentations=args.augmentations,
        visualize=not args.no_visualize,
        config_path=args.config_path
    )


if __name__ == '__main__':
    main()
