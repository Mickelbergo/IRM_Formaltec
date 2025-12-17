# generate_synthetic.py
# ------------------------------------------------------------
# Generate synthetic wound images from trained diffusion model
# Uses the enhanced mask-conditional diffusion model
# ------------------------------------------------------------

import os
import glob
import argparse
from typing import Tuple, Dict, Optional, List

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
# Add parent directory and training directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training'))

from diffusion_enhanced import (
    MaskConditionalDDPMPipeline,
    MaskEncoder,
    map_mask_pixels,
    one_hot_mask,
    resize_mask_nearest,
)

from diffusers import UNet2DConditionModel, DPMSolverMultistepScheduler


def load_model(checkpoint_dir: str, device: str = 'cuda', use_fp16: bool = False):
    """Load trained diffusion model from checkpoint"""
    checkpoint_path = os.path.join(checkpoint_dir, 'model.pt')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Determine dtype for inference
    dtype = torch.float16 if use_fp16 and device == 'cuda' else torch.float32
    if use_fp16:
        print(f"Using FP16 precision for faster inference")

    # Load configuration
    scheduler_config = checkpoint['scheduler_config']

    # Create UNet with same architecture as training
    # 6 blocks [128, 256, 512, 512, 768, 768] with cross-attention on blocks 0-4
    unet = UNet2DConditionModel(
        sample_size=384,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512, 768, 768),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=512,
        attention_head_dim=8,
    )

    # Load UNet weights
    unet.load_state_dict(checkpoint['unet'])

    # Load mask encoder (checkpoint was trained with 14 classes, but expects 0-14 = 15 values)
    mask_encoder = MaskEncoder(n_classes=14, embed_dim=512)
    mask_encoder.load_state_dict(checkpoint['mask_encoder'])

    # Create scheduler
    scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)

    # Move to device and set to eval mode for inference
    unet = unet.to(device=device, dtype=dtype).eval()
    mask_encoder = mask_encoder.to(device=device, dtype=dtype).eval()

    # Enable torch.compile for 2-3x speedup (PyTorch 2.0+, Linux/Mac only)
    # Triton is not available on Windows, so skip compilation
    import platform
    if hasattr(torch, 'compile') and platform.system() != 'Windows':
        print("Compiling models with torch.compile for faster inference...")
        try:
            unet = torch.compile(unet, mode='reduce-overhead')
            mask_encoder = torch.compile(mask_encoder, mode='reduce-overhead')
            print("Models compiled successfully!")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}. Continuing without compilation.")
    elif platform.system() == 'Windows':
        print("Note: torch.compile is not available on Windows (Triton required). Using eager mode.")

    # Create pipeline
    pipeline = MaskConditionalDDPMPipeline(
        unet=unet,
        mask_encoder=mask_encoder,
        scheduler=scheduler
    )

    return pipeline


def load_masks_from_directory(
    mask_dir: str,
    image_size: Tuple[int, int],
    palette_map: Optional[Dict[int, int]] = None,
    n_classes: int = 14,
    limit: Optional[int] = None,
) -> torch.Tensor:
    """Load and preprocess masks from directory"""
    # If no palette_map provided, create default that maps 15 classes to 14
    # According to CLAUDE.md: "Classes 11-14 were merged into class 6"
    if palette_map is None:
        palette_map = {
            0: 0,   # Background
            1: 1,   # Dermatorrhagia
            2: 2,   # Hematoma
            3: 3,   # Stab
            4: 4,   # Cut
            5: 5,   # Thermal
            6: 6,   # Skin abrasion
            7: 7,   # Puncture/gun shot
            8: 8,   # Contused-lacerated
            9: 9,   # Semisharp force
            10: 10, # Lacerations
            11: 6,  # Merged into class 6 (Skin abrasion)
            12: 6,  # Merged into class 6 (Skin abrasion)
            13: 6,  # Merged into class 6 (Skin abrasion)
            14: 6,  # Merged into class 6 (Skin abrasion)
        }

    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    if limit:
        mask_paths = mask_paths[:limit]

    masks = []
    for path in tqdm(mask_paths, desc="Loading masks"):
        # Load mask
        mask_img = Image.open(path).convert("L")
        mask_img = resize_mask_nearest(mask_img, image_size)

        # Convert to numpy
        np_mask = np.array(mask_img, dtype=np.int64)

        # Map pixels
        np_mask = map_mask_pixels(np_mask, palette_map, n_classes)

        # One-hot encode
        mask_tensor = one_hot_mask(np_mask, n_classes)
        masks.append(mask_tensor)

    return torch.stack(masks)


def generate_from_masks(
    pipeline: MaskConditionalDDPMPipeline,
    masks: torch.Tensor,
    output_dir: str,
    batch_size: int = 4,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.0,
    num_variants: int = 1,
    seed: Optional[int] = None,
):
    """Generate synthetic images from masks"""
    os.makedirs(output_dir, exist_ok=True)

    device = next(pipeline.unet.parameters()).device

    num_masks = len(masks)

    with torch.no_grad():
        for variant_idx in range(num_variants):
            print(f"\nGenerating variant {variant_idx + 1}/{num_variants}")

            for i in tqdm(range(0, num_masks, batch_size), desc="Generating images"):
                batch_masks = masks[i:i+batch_size].to(device)

                # Set seed for reproducibility if provided
                if seed is not None:
                    generator = torch.Generator(device=device).manual_seed(seed + i + variant_idx * num_masks)
                else:
                    generator = None

                # Generate images
                print(f"DEBUG: Starting generation for batch {i}, mask shape: {batch_masks.shape}")
                print(f"DEBUG: num_inference_steps={num_inference_steps}, guidance_scale={guidance_scale}")

                import time
                start_time = time.time()
                images = pipeline(
                    masks=batch_masks,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
                elapsed = time.time() - start_time
                print(f"DEBUG: Batch completed in {elapsed:.2f} seconds")

                # Save images
                for j, img in enumerate(images):
                    idx = i + j
                    if num_variants > 1:
                        filename = f"synthetic_{idx:04d}_v{variant_idx}.png"
                    else:
                        filename = f"synthetic_{idx:04d}.png"

                    img.save(os.path.join(output_dir, filename))


def generate_from_single_mask(
    pipeline: MaskConditionalDDPMPipeline,
    mask_path: str,
    output_dir: str,
    image_size: Tuple[int, int] = (512, 512),
    palette_map: Optional[Dict[int, int]] = None,
    n_classes: int = 14,
    num_samples: int = 10,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.0,
    seed: Optional[int] = None,
):
    """Generate multiple variations from a single mask"""
    os.makedirs(output_dir, exist_ok=True)

    # If no palette_map provided, create default that maps 15 classes to 14
    if palette_map is None:
        palette_map = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
            11: 6, 12: 6, 13: 6, 14: 6  # Classes 11-14 merged into class 6
        }

    device = next(pipeline.unet.parameters()).device

    # Load mask
    mask_img = Image.open(mask_path).convert("L")
    mask_img = resize_mask_nearest(mask_img, image_size)
    np_mask = np.array(mask_img, dtype=np.int64)
    np_mask = map_mask_pixels(np_mask, palette_map, n_classes)
    mask_tensor = one_hot_mask(np_mask, n_classes).unsqueeze(0)  # [1, C, H, W]

    # Repeat mask for batch generation
    mask_tensor = mask_tensor.to(device)

    print(f"Generating {num_samples} variations from mask: {mask_path}")

    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            if seed is not None:
                generator = torch.Generator(device=device).manual_seed(seed + i)
            else:
                generator = None

            images = pipeline(
                masks=mask_tensor,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            # Save image
            mask_name = os.path.splitext(os.path.basename(mask_path))[0]
            filename = f"{mask_name}_gen_{i:03d}.png"
            images[0].save(os.path.join(output_dir, filename))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic wound images from trained diffusion model")
    parser.add_argument("--checkpoint", type=str, default="diffusion_model_enhanced", help="Path to model checkpoint directory")
    parser.add_argument("--mask-dir", type=str, default="E:/projects/Wound_Segmentation_III/Data/new_masks_640_1280", help="Directory containing masks for generation")
    parser.add_argument("--single-mask", type=str, help="Path to single mask for multiple variations")
    parser.add_argument("--output-dir", type=str, default="E:/projects/Wound_Segmentation_III/Data/generated_samples", help="Output directory for generated images")
    parser.add_argument("--image-size", type=int, nargs=2, default=[384, 384], help="Image size (H W)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--num-inference-steps", type=int, default=1, help="Number of denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--num-variants", type=int, default=1, help="Number of variants per mask")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples for single mask mode")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of masks to process (for testing)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision for faster inference (CUDA only)")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU")
        args.device = "cpu"

    pipeline = load_model(args.checkpoint, device=args.device, use_fp16=args.fp16)
    print(f"Model loaded successfully on {args.device}!")

    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    image_size = tuple(args.image_size)

    if args.single_mask:
        # Generate multiple variations from single mask
        generate_from_single_mask(
            pipeline=pipeline,
            mask_path=args.single_mask,
            output_dir=args.output_dir,
            image_size=image_size,
            num_samples=args.num_samples,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
    elif args.mask_dir:
        # Generate from directory of masks
        print(f"Loading masks from {args.mask_dir}...")
        masks = load_masks_from_directory(
            mask_dir=args.mask_dir,
            image_size=image_size,
            limit=args.limit,
        )
        print(f"Loaded {len(masks)} masks")

        generate_from_masks(
            pipeline=pipeline,
            masks=masks,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_variants=args.num_variants,
            seed=args.seed,
        )
    else:
        raise ValueError("Either --mask-dir or --single-mask must be provided")

    print(f"\nGeneration complete! Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
