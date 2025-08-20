from PIL import Image
import torch
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.diffusion import MaskConditionalDDPMPipeline
from diffusers import DDPMScheduler
from diffusers.utils import make_image_grid

from torchvision import transforms
import glob

# Same transforms as in training
mask_preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()  # keep 0-1 mask
])

def generate_images(model_dir="diffusion_model",
                    masks_dir="../Data/new_masks_640_1280",
                    num_inference_steps=10000,
                    seed=42,
                    output_dir="generated_samples"):
    """
    Loads a trained diffusion model and generates images conditioned on masks.
    """
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Load pipeline from saved model directory
    print(f"Loading model from: {model_dir}")
    pipeline = MaskConditionalDDPMPipeline.from_pretrained(model_dir)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)

    pipeline.scheduler.set_timesteps(num_inference_steps, device=device)



    generator = torch.Generator(device=device).manual_seed(seed)

    # Load masks
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    print(f"Found {len(mask_paths)} masks in {masks_dir}")

    for idx, mask_path in enumerate(mask_paths):
        mask = mask_preprocess(Image.open(mask_path).convert("L")).unsqueeze(0)  # shape [1,1,H,W]

        # Generate image conditioned on this mask
        print(f"Generating image for mask: {mask_path}")
        images = pipeline(
            masks=mask.to(device),
            num_inference_steps=num_inference_steps,
            generator=generator
        )

        # Save generated image
        image_out_path = os.path.join(output_dir, f"{os.path.basename(mask_path).replace('.png', '_gen.png')}")
        images[0].save(image_out_path)
        print(f"Saved: {image_out_path}")

        # OPTIONAL: Generate multiple variations per mask
        # Uncomment to generate different images for same mask
        # for variation_seed in [0, 1, 2]:
        #     generator = torch.Generator(device=device).manual_seed(variation_seed)
        #     images = pipeline(masks=mask.to(device), num_inference_steps=num_inference_steps, generator=generator)
        #     images[0].save(image_out_path.replace(".png", f"_v{variation_seed}.png"))

if __name__ == "__main__":
    generate_images(
        model_dir="diffusion_model",    # trained model folder
        masks_dir="../Data/new_masks_640_1280",    # directory of masks to condition on
        num_inference_steps=10000,                  # quality (more steps = better)
        seed=0,                                    # reproducibility (can vary)
        output_dir="../Data/generated_samples"          
    )
