# train_fast_diffusion.py
# ------------------------------------------------------------
# FAST VERSION of enhanced mask-conditional DDPM
# Optimizations for speed while maintaining quality:
# - Simpler U-Net architecture (fewer attention blocks)
# - No perceptual loss (LPIPS is very slow)
# - No adversarial training initially
# - Larger batch size
# - Reduced cross-attention overhead
# ------------------------------------------------------------
#
# === Paper Annotations =========================
# DDPM: Ho et al. 2020, https://arxiv.org/abs/2006.11239
# Improved DDPM: Nichol & Dhariwal 2021, https://arxiv.org/abs/2102.09672
# Cross-Attention: Rombach et al. 2022, https://arxiv.org/abs/2112.10752
# Classifier-Free Guidance: Ho & Salimans 2022, https://arxiv.org/abs/2207.12598
# v-prediction: Salimans & Ho 2022, https://arxiv.org/abs/2202.00512
# SNR weighting: Saharia et al. 2022, https://arxiv.org/abs/2205.11487
# DPMSolver: Lu et al. 2022, https://arxiv.org/abs/2206.00927
# Medical Augmentation: Tellez et al. 2018, https://arxiv.org/abs/1902.06543
# ==============================================

import os
import glob
import math
import random
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, List

import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torchvision.transforms.functional as TF

from accelerate import Accelerator
from tqdm import tqdm

from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from diffusers.training_utils import EMAModel

# --- AUTO-PALETTE HELPERS ---
def infer_palette_map_and_classes(mask_dir, ignore_values=None, limit=None):
    uniq = set()
    paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    for i, p in enumerate(paths):
        arr = np.array(Image.open(p).convert("L"))
        uniq.update(np.unique(arr).tolist())
        if limit is not None and (i + 1) >= limit:
            break
    if ignore_values:
        uniq = {int(v) for v in uniq if int(v) not in set(ignore_values)}
    vals = sorted(int(v) for v in uniq)
    if not vals:
        raise RuntimeError("No class values found in masks.")
    palette_map = {v: i for i, v in enumerate(vals)}
    n_classes = len(vals)
    return palette_map, n_classes, vals

def _maybe_infer_palette(cfg):
    if cfg.palette_map is None:
        palette_map, n_classes, vals = infer_palette_map_and_classes(cfg.mask_dir, ignore_values=[])
        cfg.palette_map = palette_map
        cfg.n_classes = n_classes
        print(f"[Auto palette] Found mask values: {vals}")
        print(f"[Auto palette] Using n_classes={n_classes}")
    else:
        print(f"[Palette] Using provided palette_map and n_classes={cfg.n_classes}")

# ===============================
# === Config ====================
# ===============================
@dataclass
class FastTrainingConfig:
    # --- Data ---
    image_dir: str = "E:/projects/Wound_Segmentation_III/Data/new_images_640_1280"
    mask_dir: str = "E:/projects/Wound_Segmentation_III/Data/new_masks_640_1280"
    image_size: Tuple[int, int] = (256, 256)  # MINIMUM 256 for medical images (was 128!)
    n_classes: int = 11
    palette_map: Optional[Dict[int, int]] = None

    # --- Training (OPTIMIZED FOR SPEED) ---
    train_batch_size: int = 8  # Increased from 4 (no discriminator = more memory)
    eval_batch_size: int = 8
    num_epochs: int = 400  # Reduced epochs since we're training faster
    gradient_accumulation_steps: int = 1  # No accumulation needed with larger batch
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.99)
    lr_warmup_steps: int = 500  # Reduced warmup
    mixed_precision: str = "fp16"
    clip_grad_norm: float = 1.0
    ema_decay: float = 0.9999

    # --- Diffusion ---
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "v_prediction"
    snr_gamma: float = 5.0

    # --- Conditioning ---
    cond_dropout_prob: float = 0.1
    guidance_scale: float = 3.0
    cross_attention_dim: int = 256  # REDUCED from 512 for speed

    # --- Model Architecture (SIMPLIFIED FOR SPEED) ---
    # Fewer blocks, less attention
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)  # 4 blocks instead of 6
    layers_per_block: int = 2
    down_block_types: Tuple[str, ...] = (
        "DownBlock2D",           # No attention on early layers (speed)
        "CrossAttnDownBlock2D",  # Attention on middle
        "CrossAttnDownBlock2D",  # Attention on middle
        "DownBlock2D"            # No attention on late layers (speed)
    )
    up_block_types: Tuple[str, ...] = (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D"
    )
    attention_head_dim: int = 8
    gradient_checkpointing: bool = True

    # --- Loss Weights (SPEED OPTIMIZED) ---
    use_perceptual_loss: bool = False  # DISABLED - LPIPS is VERY slow
    use_adversarial_loss: bool = False  # DISABLED - saves 50% training time

    # --- Medical Augmentations ---
    use_medical_aug: bool = True
    stain_brightness_range: Tuple[float, float] = (0.8, 1.2)
    stain_contrast_range: Tuple[float, float] = (0.8, 1.2)
    stain_saturation_range: Tuple[float, float] = (0.8, 1.2)

    # --- Eval / Saving ---
    output_dir: str = "diffusion_model_fast"
    save_image_epochs: int = 50
    save_model_epochs: int = 100
    seed: int = 42
    num_eval_inference_steps: int = 50
    num_vis: int = 8

    # --- Dataloader (OPTIMIZED) ---
    num_workers: int = 8  # Reduced to avoid CPU bottleneck
    pin_memory: bool = True
    persistent_workers: bool = True

config = FastTrainingConfig()

# ===============================
# === Utils =====================
# ===============================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_tensor_img(img: Image.Image, size: Tuple[int, int]) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tfm(img)

def resize_mask_nearest(mask: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return mask.resize(size, Image.NEAREST)

def map_mask_pixels(np_mask: np.ndarray, palette_map: Optional[Dict[int, int]], n_classes: int) -> np.ndarray:
    if palette_map is None:
        return np_mask
    lut = np.zeros(256, dtype=np.int64)
    for k, v in palette_map.items():
        lut[int(k)] = int(v)
    np_mask = lut[np_mask.astype(np.uint8)]
    np_mask = np.clip(np_mask, 0, n_classes - 1)
    return np_mask

def one_hot_mask(np_ids: np.ndarray, n_classes: int) -> torch.Tensor:
    t = torch.from_numpy(np_ids.astype(np.int64))
    oh = F.one_hot(t, num_classes=n_classes).permute(2, 0, 1).float()
    return oh

# ===============================
# === Medical Augmentation ======
# ===============================
class MedicalStainAugmentation:
    """Stain/color augmentation for medical images (Tellez et al. 2018)"""
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        hue_shift: float = 0.05,
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_shift = hue_shift

    def __call__(self, img: Image.Image) -> Image.Image:
        brightness_factor = random.uniform(*self.brightness_range)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        contrast_factor = random.uniform(*self.contrast_range)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

        saturation_factor = random.uniform(*self.saturation_range)
        img = ImageEnhance.Color(img).enhance(saturation_factor)

        if random.random() < 0.5:
            img_tensor = TF.to_tensor(img)
            img_tensor = TF.adjust_hue(img_tensor, random.uniform(-self.hue_shift, self.hue_shift))
            img = TF.to_pil_image(img_tensor)

        return img

class PairedAugment:
    def __init__(self, p_hflip=0.5, p_vflip=0.5, p_rot90=0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p_hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < self.p_vflip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < self.p_rot90:
            k = random.choice([1, 2, 3])
            img = img.rotate(90 * k, expand=True)
            mask = mask.rotate(90 * k, expand=True)
        return img, mask

# ===============================
# === Mask Encoder (SIMPLIFIED) =
# ===============================
class SimpleMaskEncoder(nn.Module):
    """
    Simplified mask encoder for cross-attention (Rombach et al. 2022).
    Fewer layers for speed.
    """
    def __init__(self, n_classes: int, embed_dim: int = 256):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = embed_dim

        # Simpler encoder (3 layers instead of 4)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_classes, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, embed_dim),
            nn.SiLU(),
        )

        self.to_seq = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(mask)  # [B, embed_dim, H/4, W/4]
        x = self.to_seq(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        return x

# ===============================
# === Dataset ===================
# ===============================
class WoundDatasetFast(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        size: Tuple[int, int],
        n_classes: int,
        palette_map: Optional[Dict[int, int]] = None,
        augment: bool = True,
        eval_subset: bool = False,
        num_vis: int = 8,
        use_medical_aug: bool = True,
        stain_aug_config: dict = None,
    ):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        assert len(self.image_paths) == len(self.mask_paths)

        self.size = size
        self.n_classes = n_classes
        self.palette_map = palette_map
        self.augment = augment
        self.paired_aug = PairedAugment()
        self.use_medical_aug = use_medical_aug

        if use_medical_aug and stain_aug_config:
            self.stain_aug = MedicalStainAugmentation(**stain_aug_config)
        else:
            self.stain_aug = None

        if eval_subset and len(self.image_paths) > num_vis:
            idx = np.linspace(0, len(self.image_paths) - 1, num=num_vis, dtype=int)
            self.image_paths = [self.image_paths[i] for i in idx]
            self.mask_paths = [self.mask_paths[i] for i in idx]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask_img = Image.open(self.mask_paths[idx]).convert("L")

        if self.augment:
            img, mask_img = self.paired_aug(img, mask_img)
            if self.stain_aug and random.random() < 0.5:
                img = self.stain_aug(img)

        img_t = to_tensor_img(img, self.size)
        mask_r = resize_mask_nearest(mask_img, self.size)

        np_ids = np.array(mask_r, dtype=np.int64)
        np_ids = map_mask_pixels(np_ids, self.palette_map, self.n_classes)
        one_hot = one_hot_mask(np_ids, self.n_classes)

        return {
            "images": img_t,
            "mask_onehot": one_hot,
        }

# ===============================
# === Pipeline ==================
# ===============================
class MaskConditionalDDPMPipeline:
    """Inference with CFG (Ho & Salimans 2022)"""
    def __init__(self, unet, mask_encoder, scheduler):
        self.unet = unet
        self.mask_encoder = mask_encoder
        self.scheduler = scheduler

    @torch.no_grad()
    def __call__(
        self,
        masks: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        generator: Optional[torch.Generator] = None,
    ) -> List[Image.Image]:
        device = self.unet.device
        b, c, h, w = masks.shape

        x = torch.randn((b, 3, h, w), generator=generator, device=device)
        masks = masks.to(device)

        mask_embeds = self.mask_encoder(masks)
        uncond_embeds = self.mask_encoder(torch.zeros_like(masks))

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for t in self.scheduler.timesteps:
            noise_pred_uncond = self.unet(x, t, encoder_hidden_states=uncond_embeds).sample
            noise_pred_cond = self.unet(x, t, encoder_hidden_states=mask_embeds).sample
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            x = self.scheduler.step(noise_pred, t, x, generator=generator).prev_sample

        x = (x.clamp(-1, 1) + 1) / 2.0
        images = [transforms.ToPILImage()(img.cpu()) for img in x]
        return images

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'unet': self.unet.state_dict(),
            'mask_encoder': self.mask_encoder.state_dict(),
            'scheduler_config': self.scheduler.config,
        }, os.path.join(path, 'model.pt'))

# ===============================
# === Loss helpers ==============
# ===============================
def snr_weight(scheduler: DDPMScheduler, timesteps: torch.LongTensor, gamma: float = 5.0) -> torch.Tensor:
    """p2 SNR reweighting (Saharia et al. 2022)"""
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)
    a_bar = alphas_cumprod.gather(0, timesteps)
    snr = a_bar / (1.0 - a_bar + 1e-8)
    w = torch.minimum(snr, torch.full_like(snr, gamma)) / (snr + 1e-8)
    return w

# ===============================
# === Evaluate ==================
# ===============================
@torch.no_grad()
def evaluate_and_save_samples(
    cfg: FastTrainingConfig,
    accelerator: Accelerator,
    model: UNet2DConditionModel,
    mask_encoder: SimpleMaskEncoder,
    base_scheduler: DDPMScheduler,
    eval_masks: torch.Tensor,
    epoch: int,
    ema_unet: Optional[EMAModel] = None,
    ema_encoder: Optional[EMAModel] = None,
):
    is_main = accelerator.is_main_process
    if not is_main:
        return

    if ema_unet is not None:
        ema_unet.store(model.parameters())
        ema_unet.copy_to(model.parameters())
    if ema_encoder is not None:
        ema_encoder.store(mask_encoder.parameters())
        ema_encoder.copy_to(mask_encoder.parameters())

    pipe = MaskConditionalDDPMPipeline(
        unet=accelerator.unwrap_model(model),
        mask_encoder=accelerator.unwrap_model(mask_encoder),
        scheduler=base_scheduler
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    masks = eval_masks.to(device)
    images = pipe(
        masks=masks,
        num_inference_steps=cfg.num_eval_inference_steps,
        guidance_scale=cfg.guidance_scale,
        generator=gen,
    )

    os.makedirs(os.path.join(cfg.output_dir, "samples"), exist_ok=True)
    grid = make_image_grid(images, rows=1, cols=len(images))
    out_path = os.path.join(cfg.output_dir, "samples", f"{epoch:04d}.png")
    grid.save(out_path)

    if ema_unet is not None:
        ema_unet.restore(model.parameters())
    if ema_encoder is not None:
        ema_encoder.restore(mask_encoder.parameters())

# ===============================
# === Train =====================
# ===============================
def train(cfg: FastTrainingConfig):
    set_seed(cfg.seed)

    # Create logging directory
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=log_dir,
    )

    is_main = accelerator.is_main_process

    _maybe_infer_palette(cfg)

    if is_main:
        print("="*60)
        print("FAST DIFFUSION CONFIG")
        print("="*60)
        print(f"Image size: {cfg.image_size}")
        print(f"Batch size: {cfg.train_batch_size}")
        print(f"Cross-attention dim: {cfg.cross_attention_dim}")
        print(f"Blocks: {len(cfg.block_out_channels)}")
        print(f"Perceptual loss: {cfg.use_perceptual_loss}")
        print(f"Adversarial loss: {cfg.use_adversarial_loss}")
        print(f"Medical augmentation: {cfg.use_medical_aug}")
        print("="*60)

    # Datasets
    stain_config = {
        'brightness_range': cfg.stain_brightness_range,
        'contrast_range': cfg.stain_contrast_range,
        'saturation_range': cfg.stain_saturation_range,
    }

    train_dataset = WoundDatasetFast(
        image_dir=cfg.image_dir,
        mask_dir=cfg.mask_dir,
        size=cfg.image_size,
        n_classes=cfg.n_classes,
        palette_map=cfg.palette_map,
        augment=True,
        use_medical_aug=cfg.use_medical_aug,
        stain_aug_config=stain_config,
    )

    eval_dataset = WoundDatasetFast(
        image_dir=cfg.image_dir,
        mask_dir=cfg.mask_dir,
        size=cfg.image_size,
        n_classes=cfg.n_classes,
        palette_map=cfg.palette_map,
        augment=False,
        eval_subset=True,
        num_vis=cfg.num_vis,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        drop_last=True,
    )

    eval_masks = torch.stack([eval_dataset[i]["mask_onehot"] for i in range(len(eval_dataset))])

    # Models
    if is_main:
        print(f"Creating simplified UNet2DConditionModel...")

    mask_encoder = SimpleMaskEncoder(
        n_classes=cfg.n_classes,
        embed_dim=cfg.cross_attention_dim
    )

    model = UNet2DConditionModel(
        sample_size=cfg.image_size[0],
        in_channels=3,
        out_channels=3,
        layers_per_block=cfg.layers_per_block,
        block_out_channels=cfg.block_out_channels,
        down_block_types=cfg.down_block_types,
        up_block_types=cfg.up_block_types,
        cross_attention_dim=cfg.cross_attention_dim,
        attention_head_dim=cfg.attention_head_dim,
    )

    if cfg.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(mask_encoder.parameters()),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    # LR scheduler
    num_training_steps = cfg.num_epochs * math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # EMA
    ema_unet = EMAModel(parameters=model.parameters(), power=cfg.ema_decay)
    ema_encoder = EMAModel(parameters=mask_encoder.parameters(), power=cfg.ema_decay)

    # Prepare
    model, mask_encoder, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, mask_encoder, optimizer, train_loader, lr_scheduler
    )

    ema_unet.to(accelerator.device)
    ema_encoder.to(accelerator.device)

    accelerator.init_trackers("fast_mask_diffusion")

    global_step = 0
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Training loop
    for epoch in range(cfg.num_epochs):
        model.train()
        mask_encoder.train()

        progress = tqdm(total=len(train_loader), disable=not is_main, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")

        for step, batch in enumerate(train_loader):
            clean_images: torch.Tensor = batch["images"]
            mask_onehot: torch.Tensor = batch["mask_onehot"]

            bs = clean_images.shape[0]

            with accelerator.accumulate(model):
                # Sample noise and timesteps
                noise = torch.randn_like(clean_images)
                noise = noise + 0.1 * torch.randn(bs, 1, 1, 1, device=noise.device)

                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bs,), device=clean_images.device, dtype=torch.int64
                )

                # Forward diffusion
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                # CFG dropout
                if random.random() < cfg.cond_dropout_prob:
                    cond_mask = torch.zeros_like(mask_onehot)
                else:
                    cond_mask = mask_onehot

                # Encode mask
                mask_embeds = mask_encoder(cond_mask)

                # Predict
                target_v = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                v_pred = model(noisy_images, timesteps, encoder_hidden_states=mask_embeds).sample

                # SNR weighted loss
                w = snr_weight(noise_scheduler, timesteps, gamma=cfg.snr_gamma)
                loss_diffusion = F.mse_loss(v_pred, target_v, reduction="none").mean(dim=[1,2,3])
                loss = (loss_diffusion * w).mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients and cfg.clip_grad_norm is not None:
                    accelerator.clip_grad_norm_(
                        list(model.parameters()) + list(mask_encoder.parameters()),
                        cfg.clip_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # EMA update
                ema_unet.step(accelerator.unwrap_model(model).parameters())
                ema_encoder.step(accelerator.unwrap_model(mask_encoder).parameters())

            # Logging
            logs = {
                "loss": float(loss.detach().item()),
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "step": global_step,
            }
            progress.set_postfix(loss=f"{logs['loss']:.4f}", lr=f"{logs['lr']:.6f}")
            accelerator.log(logs, step=global_step)
            global_step += 1
            progress.update(1)

        progress.close()

        # Evaluate
        if (epoch + 1) % cfg.save_image_epochs == 0 or epoch == cfg.num_epochs - 1:
            evaluate_and_save_samples(
                cfg, accelerator, model, mask_encoder, noise_scheduler,
                eval_masks, epoch, ema_unet, ema_encoder
            )

        # Save model
        if (epoch + 1) % cfg.save_model_epochs == 0 or epoch == cfg.num_epochs - 1:
            if is_main:
                ema_unet.store(accelerator.unwrap_model(model).parameters())
                ema_unet.copy_to(accelerator.unwrap_model(model).parameters())
                ema_encoder.store(accelerator.unwrap_model(mask_encoder).parameters())
                ema_encoder.copy_to(accelerator.unwrap_model(mask_encoder).parameters())

                pipe = MaskConditionalDDPMPipeline(
                    unet=accelerator.unwrap_model(model),
                    mask_encoder=accelerator.unwrap_model(mask_encoder),
                    scheduler=noise_scheduler
                )
                pipe.save_pretrained(cfg.output_dir)

                ema_unet.restore(accelerator.unwrap_model(model).parameters())
                ema_encoder.restore(accelerator.unwrap_model(mask_encoder).parameters())

    accelerator.end_training()
    if is_main:
        print("Fast training complete! Models saved to:", cfg.output_dir)

# ===============================
# === Main ======================
# ===============================
if __name__ == "__main__":
    train(config)

    # Preview latest sample
    latest = sorted(glob.glob(os.path.join(config.output_dir, "samples", "*.png")))
    if latest:
        try:
            Image.open(latest[-1]).show()
        except Exception:
            print("Latest sample:", latest[-1])
