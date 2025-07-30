import torch 
import torch.nn as nn
import torch.nn.functional as f

from tqdm import tqdm

from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from torchvision.utils import save_image
from Preprocessing import Dataset  # Use your custom Dataset
import numpy as np
import os
import json
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from datasets import load_dataset
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
from dataclasses import dataclass
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from accelerate import notebook_launcher
import glob

@dataclass
class TrainingConfig:
    train_batch_size = 64
    eval_batch_size = 16
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 100
    mixed_precision = 'fp16'
    output_dir = 'pretrained_diffusion_model' 
    seed = 42
    num_train_timesteps = 1000

def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size = config.eval_batch_size,
        generator = torch.Generator(device='cpu').manual_seed(config.seed)
    ).images

    image_grid = make_image_grid(images, rows = 4, cols = 4)

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f'{test_dir}/{epoch:04d}.png')


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}



def train_diffusion_model(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision = config.mixed_precision,
        gradient_accumulation_steps= config.gradient_accumulation_steps,
        log_with = 'tensorboard'
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )



    global_step = 0


    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable = not accelerator.is_local_main_process)
        progress_bar.set_description(f'Epoch {epoch+1}')

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]

            noise = torch.randn(clean_images.shape, device = clean_images.device)

            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device= clean_images.device, dtype = torch.int64
            )


            #forward diffusion process
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            progress_bar.update(1)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}


            progress_bar.set_postfix(**logs)

            accelerator.log(logs, step = global_step)

            global_step += 1
    
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler = noise_scheduler)

            if(epoch+1) % config.save_image_epochs == 0 or epoch == config.num_epochs -1 :
                evaluate(config, epoch, pipeline)

            if(epoch+1) % config.save_model_epochs == 0 or epoch == config.num_epochs -1:
                pipeline.save_pretrained(config.output_dir)


model = UNet2DModel(
    sample_size=128,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

preprocess = transforms.Compose(
    [
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)

            
if __name__ == '__main__':

    config = TrainingConfig()
    dataset_name = "huggan/smithsonian_butterflies_subset"
    dataset = load_dataset(dataset_name, split="train")
    dataset.set_transform(transform)


    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps = (len(train_dataloader) * config.num_epochs)
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
   


    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

    notebook_launcher(train_diffusion_model, args, num_processes=1)

    sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
    Image.open(sample_images[-1]).show()