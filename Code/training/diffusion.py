import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass
from tqdm import tqdm

from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from accelerate import Accelerator, notebook_launcher

# ===============================
# === Config ====================
# ===============================
@dataclass
class TrainingConfig:
    train_batch_size = 16
    eval_batch_size = 16
    num_epochs = 1000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 50
    save_model_epochs = 50
    mixed_precision = "fp16"
    output_dir = "diffusion_model"
    seed = 42
    num_train_timesteps = 10000

config = TrainingConfig()

# ===============================
# === Dataset ===================
# ===============================
class WoundDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        assert len(self.image_paths) == len(self.mask_paths), "Images and masks must match"
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # return combined tensor (4 channels) and mask only
        combined = torch.cat((image, mask), dim=0)
        return {"images": combined, "mask_only": mask}

# ===============================
# === Transforms ================
# ===============================
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

mask_preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()  # keep mask 0-1
])

# ===============================
# === Pipeline ==================
# ===============================
from diffusers import DDPMPipeline

class MaskConditionalDDPMPipeline(DDPMPipeline):
    @torch.no_grad()
    def __call__(self, masks, num_inference_steps=1000, generator=None):
        """
        masks: torch.Tensor [B,1,H,W], conditioning masks
        """
        device = self.unet.device
        b, _, h, w = masks.shape

        image = torch.randn((b, 3, h, w), generator=generator, device=device)
        masks = masks.to(device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for t in self.scheduler.timesteps:
            noisy_input = torch.cat((image, masks), dim=1)
            model_output = self.unet(noisy_input, t).sample
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image.clamp(-1, 1) + 1) / 2.0
        images = [transforms.ToPILImage()(img.cpu()) for img in image]
        return images

# ===============================
# === Evaluate ==================
# ===============================
def evaluate(config, epoch, pipeline, eval_masks):
    device = pipeline.unet.device
    images = pipeline(
        masks=eval_masks,
        num_inference_steps=config.num_train_timesteps,
        generator=torch.Generator(device=device).manual_seed(config.seed)
    )

    image_grid = make_image_grid(images, rows=1, cols=len(images))
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

# ===============================
# === Train =====================
# ===============================
def train_diffusion_model(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, eval_masks):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard"
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    accelerator.init_trackers("diffusion_training")

    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{config.num_epochs}")

        for step, batch in enumerate(train_dataloader):
            combined = batch["images"]  # shape: [B,4,H,W]
            mask = combined[:, 3:, :, :]  # last channel is mask
            clean_images = combined[:, :3, :, :]  # RGB

            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,),
                device=clean_images.device, dtype=torch.int64
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noisy_with_mask = torch.cat((noisy_images, mask), dim=1)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_with_mask, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = MaskConditionalDDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline, eval_masks)
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)

# ===============================
# === Main ======================
# ===============================
if __name__ == "__main__":
    image_dir = "../Data/new_images_640_1280"
    mask_dir = "../Data/new_masks_640_1280"

    dataset = WoundDataset(image_dir=image_dir, mask_dir=mask_dir, image_transform=preprocess, mask_transform=mask_preprocess)

    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers = True)
    eval_masks = torch.stack([dataset[i]["mask_only"] for i in range(config.eval_batch_size)])

    # Model: in_channels = 4 (RGB + mask)
    model = UNet2DModel(
        sample_size=256,
        in_channels=4,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"
        )
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs)
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)

    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, eval_masks)
    notebook_launcher(train_diffusion_model, args, num_processes=1)

    sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
    if sample_images:
        Image.open(sample_images[-1]).show()
