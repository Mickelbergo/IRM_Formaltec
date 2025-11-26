# compare_diffusion.py
# ------------------------------------------------------------
# Compare original vs enhanced diffusion model outputs
# Generates side-by-side comparisons and quality metrics
# ------------------------------------------------------------

import os
import glob
import argparse
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

# For FID score calculation
try:
    from pytorch_fid import fid_score
    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("Warning: pytorch-fid not installed. FID score will not be calculated.")
    print("Install with: pip install pytorch-fid")

# For LPIPS
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips not installed. LPIPS score will not be calculated.")
    print("Install with: pip install lpips")


def create_comparison_grid(
    real_paths: list,
    original_gen_paths: list,
    enhanced_gen_paths: list,
    mask_paths: list,
    output_path: str,
    num_samples: int = 8,
):
    """Create side-by-side comparison grid"""
    num_samples = min(num_samples, len(real_paths), len(original_gen_paths), len(enhanced_gen_paths))

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Load images
        real_img = Image.open(real_paths[i]).convert('RGB')
        mask_img = Image.open(mask_paths[i]).convert('L')
        original_img = Image.open(original_gen_paths[i]).convert('RGB') if i < len(original_gen_paths) else Image.new('RGB', real_img.size)
        enhanced_img = Image.open(enhanced_gen_paths[i]).convert('RGB') if i < len(enhanced_gen_paths) else Image.new('RGB', real_img.size)

        # Display
        axes[i, 0].imshow(mask_img, cmap='tab20')
        axes[i, 0].set_title('Mask (Ground Truth)', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(real_img)
        axes[i, 1].set_title('Real Image', fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(original_img)
        axes[i, 2].set_title('Original Diffusion', fontsize=10)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(enhanced_img)
        axes[i, 3].set_title('Enhanced Diffusion', fontsize=10)
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison grid saved to {output_path}")


def calculate_lpips_score(real_dir: str, gen_dir: str, device: str = 'cuda') -> float:
    """Calculate LPIPS (perceptual similarity) between real and generated images"""
    if not HAS_LPIPS:
        return -1.0

    loss_fn = lpips.LPIPS(net='alex').to(device)

    real_paths = sorted(glob.glob(os.path.join(real_dir, "*.png")))
    gen_paths = sorted(glob.glob(os.path.join(gen_dir, "*.png")))

    # Match by count
    min_len = min(len(real_paths), len(gen_paths))
    real_paths = real_paths[:min_len]
    gen_paths = gen_paths[:min_len]

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    scores = []
    with torch.no_grad():
        for real_path, gen_path in tqdm(zip(real_paths, gen_paths), total=len(real_paths), desc="Computing LPIPS"):
            real_img = Image.open(real_path).convert('RGB')
            gen_img = Image.open(gen_path).convert('RGB')

            real_tensor = transform(real_img).unsqueeze(0).to(device)
            gen_tensor = transform(gen_img).unsqueeze(0).to(device)

            score = loss_fn(real_tensor, gen_tensor).item()
            scores.append(score)

    return np.mean(scores)


def calculate_fid(real_dir: str, gen_dir: str, device: str = 'cuda') -> float:
    """Calculate FID score between real and generated images"""
    if not HAS_FID:
        return -1.0

    try:
        fid = fid_score.calculate_fid_given_paths(
            [real_dir, gen_dir],
            batch_size=8,
            device=device,
            dims=2048
        )
        return fid
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return -1.0


def calculate_image_statistics(image_dir: str) -> dict:
    """Calculate basic image statistics"""
    paths = glob.glob(os.path.join(image_dir, "*.png"))

    brightness_vals = []
    contrast_vals = []

    for path in tqdm(paths, desc="Computing statistics"):
        img = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0

        # Brightness (mean intensity)
        brightness = img.mean()
        brightness_vals.append(brightness)

        # Contrast (std of intensity)
        contrast = img.std()
        contrast_vals.append(contrast)

    return {
        'mean_brightness': np.mean(brightness_vals),
        'std_brightness': np.std(brightness_vals),
        'mean_contrast': np.mean(contrast_vals),
        'std_contrast': np.std(contrast_vals),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare original vs enhanced diffusion models")
    parser.add_argument("--real-dir", type=str, required=True, help="Directory with real images")
    parser.add_argument("--mask-dir", type=str, required=True, help="Directory with masks")
    parser.add_argument("--original-gen-dir", type=str, help="Directory with original diffusion outputs")
    parser.add_argument("--enhanced-gen-dir", type=str, help="Directory with enhanced diffusion outputs")
    parser.add_argument("--output-dir", type=str, default="comparison_results", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples in comparison grid")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get file paths
    real_paths = sorted(glob.glob(os.path.join(args.real_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(args.mask_dir, "*.png")))

    print(f"Found {len(real_paths)} real images")
    print(f"Found {len(mask_paths)} masks")

    # Create comparison grid if both model outputs are provided
    if args.original_gen_dir and args.enhanced_gen_dir:
        original_gen_paths = sorted(glob.glob(os.path.join(args.original_gen_dir, "*.png")))
        enhanced_gen_paths = sorted(glob.glob(os.path.join(args.enhanced_gen_dir, "*.png")))

        print(f"Found {len(original_gen_paths)} original generated images")
        print(f"Found {len(enhanced_gen_paths)} enhanced generated images")

        create_comparison_grid(
            real_paths=real_paths,
            original_gen_paths=original_gen_paths,
            enhanced_gen_paths=enhanced_gen_paths,
            mask_paths=mask_paths,
            output_path=os.path.join(args.output_dir, "comparison_grid.png"),
            num_samples=args.num_samples,
        )

    # Calculate metrics
    results = {}

    # Real image statistics
    print("\nCalculating real image statistics...")
    real_stats = calculate_image_statistics(args.real_dir)
    results['real'] = real_stats

    # Original diffusion statistics
    if args.original_gen_dir:
        print("\nCalculating original diffusion statistics...")
        orig_stats = calculate_image_statistics(args.original_gen_dir)
        results['original_gen'] = orig_stats

        print("\nCalculating metrics for original diffusion...")
        orig_lpips = calculate_lpips_score(args.real_dir, args.original_gen_dir, args.device)
        orig_fid = calculate_fid(args.real_dir, args.original_gen_dir, args.device)

        results['original_lpips'] = orig_lpips
        results['original_fid'] = orig_fid

    # Enhanced diffusion statistics
    if args.enhanced_gen_dir:
        print("\nCalculating enhanced diffusion statistics...")
        enh_stats = calculate_image_statistics(args.enhanced_gen_dir)
        results['enhanced_gen'] = enh_stats

        print("\nCalculating metrics for enhanced diffusion...")
        enh_lpips = calculate_lpips_score(args.real_dir, args.enhanced_gen_dir, args.device)
        enh_fid = calculate_fid(args.real_dir, args.enhanced_gen_dir, args.device)

        results['enhanced_lpips'] = enh_lpips
        results['enhanced_fid'] = enh_fid

    # Print results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    print("\nImage Statistics:")
    print("-" * 60)
    for key, stats in results.items():
        if isinstance(stats, dict):
            print(f"\n{key.upper()}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value:.4f}")

    print("\nPerceptual Metrics:")
    print("-" * 60)
    if 'original_lpips' in results:
        print(f"Original Diffusion LPIPS: {results['original_lpips']:.4f} (lower is better)")
    if 'enhanced_lpips' in results:
        print(f"Enhanced Diffusion LPIPS: {results['enhanced_lpips']:.4f} (lower is better)")

    if 'original_fid' in results and results['original_fid'] > 0:
        print(f"Original Diffusion FID: {results['original_fid']:.4f} (lower is better)")
    if 'enhanced_fid' in results and results['enhanced_fid'] > 0:
        print(f"Enhanced Diffusion FID: {results['enhanced_fid']:.4f} (lower is better)")

    # Save results to file
    results_file = os.path.join(args.output_dir, "metrics.txt")
    with open(results_file, 'w') as f:
        f.write("COMPARISON RESULTS\n")
        f.write("="*60 + "\n\n")

        f.write("Image Statistics:\n")
        f.write("-" * 60 + "\n")
        for key, stats in results.items():
            if isinstance(stats, dict):
                f.write(f"\n{key.upper()}:\n")
                for stat_name, value in stats.items():
                    f.write(f"  {stat_name}: {value:.4f}\n")

        f.write("\nPerceptual Metrics:\n")
        f.write("-" * 60 + "\n")
        if 'original_lpips' in results:
            f.write(f"Original Diffusion LPIPS: {results['original_lpips']:.4f}\n")
        if 'enhanced_lpips' in results:
            f.write(f"Enhanced Diffusion LPIPS: {results['enhanced_lpips']:.4f}\n")
        if 'original_fid' in results and results['original_fid'] > 0:
            f.write(f"Original Diffusion FID: {results['original_fid']:.4f}\n")
        if 'enhanced_fid' in results and results['enhanced_fid'] > 0:
            f.write(f"Enhanced Diffusion FID: {results['enhanced_fid']:.4f}\n")

    print(f"\nResults saved to {results_file}")
    print("="*60)


if __name__ == "__main__":
    main()
