import torch
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os

def generate_images(model_dir="pretrained_diffusion_model", 
                    batch_size=4, 
                    num_inference_steps=100, 
                    seed=42, 
                    output_dir="generated"):
    """
    Loads a trained diffusion model and generates images.
    """
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)

    # Load pipeline from saved model directory
    print(f"Loading model from: {model_dir}")
    pipeline = DDPMPipeline.from_pretrained(model_dir)

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)

    # Random seed for reproducibility
    generator = torch.manual_seed(seed)

    # Generate images
    print("Generating images...")
    images = pipeline(batch_size=batch_size, 
                      generator=generator, 
                      num_inference_steps=num_inference_steps).images

    # Save individual images
    for idx, img in enumerate(images):
        img_path = os.path.join(output_dir, f"generated_{idx+1}.png")
        img.save(img_path)
        print(f"Saved: {img_path}")

    # Save a grid for preview
    image_grid = make_image_grid(images, rows=2, cols=2)
    grid_path = os.path.join(output_dir, "grid.png")
    image_grid.save(grid_path)
    print(f"Saved grid image: {grid_path}")

if __name__ == "__main__":

    generate_images(
        model_dir='diffusion_model', # trained model folder
        batch_size=32,                           # number of images to generate
        num_inference_steps=100,                 # steps for better quality
        seed=0,                                  # reproducibility
        output_dir="generated_samples_diffusion" # output folder
    )
