import os
import torch
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from torchvision.utils import save_image
from training.Preprocessing import Dataset  # Use your custom Dataset
import numpy as np
import json

def get_config(cfg, *keys, legacy_key=None):
    d = cfg
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return cfg.get(legacy_key or keys[-1])
    return d

with open('configs/training_config.json') as f:
    train_config = json.load(f)
with open('configs/preprocessing_config.json') as f:
    preprocessing_config = json.load(f):
    

# =====================
# CONFIGURATION
# =====================
# TODO: Set these paths and parameters to match your project
DATA_ROOT = "<YOUR_DATA_ROOT>"  # e.g., '/path/to/data/root'
IMAGE_IDS = [...]  # List of image filenames (e.g., ['img1.png', ...])
MASK_IDS = [...]   # List of mask filenames (same order as IMAGE_IDS)
PREPROCESSING_CONFIG = {...}  # Your preprocessing config dict
BATCH_SIZE = 8
NUM_EPOCHS = 10
SAVE_DIR = "generated_samples"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

# =====================
# DATASET & DATALOADER
# =====================
dataset = Dataset(
    dir_path=DATA_ROOT,
    image_ids=IMAGE_IDS,
    mask_ids=MASK_IDS,
    augmentation="train",
    preprocessing_config=PREPROCESSING_CONFIG,
    device=DEVICE
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =====================
# MODEL & SCHEDULER
# =====================
# Assume images are RGB (3 channels) and masks are single-channel (1)
# We'll concatenate mask as a 4th channel
SAMPLE_SIZE = 128  # TODO: Set to your image size (height/width)
IN_CHANNELS = 4    # 3 (RGB) + 1 (mask)
OUT_CHANNELS = 3   # Output RGB image (or 1 for grayscale)

model = UNet2DModel(
    sample_size=SAMPLE_SIZE,
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    down_block_types=("DownBlock2D",) * 4,
    up_block_types=("UpBlock2D",) * 4,
)
model.to(DEVICE)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# =====================
# TRAINING LOOP (Simplified)
# =====================
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in dataloader:
        images, binary_masks, multiclass_masks, _ = batch
        # Use multiclass_masks as extra channel (or binary_masks if binary)
        images = images.to(DEVICE)
        masks = multiclass_masks.float().to(DEVICE)  # [B, 1, H, W]
        x = torch.cat([images, masks], dim=1)  # [B, 4, H, W]
        # Sample random timesteps for each image in the batch
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x.shape[0],), device=DEVICE).long()
        # Add noise
        noise = torch.randn_like(x)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
        # Predict the noise with the model
        noise_pred = model(noisy_x, timesteps).sample
        # Loss: MSE between predicted and true noise
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss.item():.4f}")
    # Optionally save checkpoint/model here

# =====================
# GENERATION/SAMPLING
# =====================
model.eval()
pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)

# Generate new samples (images + masks)
NUM_SAMPLES = 8  # How many to generate
with torch.no_grad():
    for i in range(NUM_SAMPLES):
        # Start from random noise
        sample = torch.randn(1, IN_CHANNELS, SAMPLE_SIZE, SAMPLE_SIZE).to(DEVICE)
        for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
            timesteps = torch.full((1,), t, device=DEVICE, dtype=torch.long)
            with torch.no_grad():
                noise_pred = model(sample, timesteps).sample
            sample = noise_scheduler.step(noise_pred, t, sample).prev_sample
        # Split generated sample into image and mask
        gen_img = sample[:, :3, :, :].clamp(0, 1)
        gen_mask = sample[:, 3:, :, :].clamp(0, 1)
        save_image(gen_img, os.path.join(SAVE_DIR, f"gen_image_{i}.png"))
        save_image(gen_mask, os.path.join(SAVE_DIR, f"gen_mask_{i}.png"))
        print(f"Saved generated image and mask {i}")

print("Generation complete. Samples saved in:", SAVE_DIR)

# =====================
# NOTES
# =====================
# - Fill in IMAGE_IDS, MASK_IDS, DATA_ROOT, and PREPROCESSING_CONFIG with your actual data/config.
# - You can use binary_masks instead of multiclass_masks if your task is binary segmentation.
# - For more advanced conditional generation, consider using ControlNet or Pix2Pix pipelines from diffusers.
# - This template is for educational/demo purposes. For production, add error handling, logging, and checkpointing. 