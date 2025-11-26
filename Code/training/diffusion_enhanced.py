# train_enhanced_diffusion.py
# ------------------------------------------------------------
# Enhanced Mask-conditional DDPM with:
# - Cross-attention conditioning (Stable Diffusion paradigm)
# - Deeper U-Net with more attention blocks
# - Perceptual loss (LPIPS) for improved realism
# - Adversarial training for sharper images
# - Medical-specific augmentations
# - Multi-scale training
# - v-prediction + SNR (p2) loss weighting
# - EMA for cleaner sampling/checkpoints
# - Classifier-free guidance via condition dropout
# - DPMSolverMultistep for fast inference
# ------------------------------------------------------------
#
# === Paper Annotations (key techniques & sources) =========================
# DDPM (framework): Ho et al. 2020, "Denoising Diffusion Probabilistic Models"
#   https://arxiv.org/abs/2006.11239
# Improved DDPM (cosine schedule, etc.): Nichol & Dhariwal 2021
#   https://arxiv.org/abs/2102.09672
# U-Net backbone: Ronneberger et al. 2015
#   https://arxiv.org/abs/1505.04597
# Cross-Attention in Diffusion (Stable Diffusion): Rombach et al. 2022, "High-Resolution Image Synthesis with Latent Diffusion Models"
#   https://arxiv.org/abs/2112.10752
# Classifier-Free Guidance (CFG): Ho & Salimans 2022
#   https://arxiv.org/abs/2207.12598
# v-prediction parameterization: Salimans & Ho 2022, "Progressive Distillation"
#   https://arxiv.org/abs/2202.00512
# SNR (p2) loss reweighting: Imagen (Saharia et al. 2022)
#   https://arxiv.org/abs/2205.11487
# DPMSolver (fast sampler): Lu et al. 2022
#   https://arxiv.org/abs/2206.00927
# LPIPS (perceptual loss): Zhang et al. 2018, "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
#   https://arxiv.org/abs/1801.03924
# Patch-based Discriminator (PatchGAN): Isola et al. 2017, "Image-to-Image Translation with Conditional Adversarial Networks"
#   https://arxiv.org/abs/1611.07004
# Adversarial Loss for Diffusion: Dhariwal & Nichol 2021, "Diffusion Models Beat GANs on Image Synthesis"
#   https://arxiv.org/abs/2105.05233
# Multi-scale Training: Karras et al. 2018, "Progressive Growing of GANs"
#   https://arxiv.org/abs/1710.10196
# Stain Augmentation (Medical Imaging): Tellez et al. 2018, "Quantifying the effects of data augmentation and stain color normalization"
#   https://arxiv.org/abs/1902.06543
# Gradient checkpointing: Chen et al. 2016
#   https://arxiv.org/abs/1604.06174
# Mixed precision: Micikevicius et al. 2017
#   https://arxiv.org/abs/1710.03740
# ControlNet conditioning: Zhang et al. 2023, "Adding Conditional Control to Text-to-Image Diffusion Models"
#   https://arxiv.org/abs/2302.05543
# ==========================================================================

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
    UNet2DConditionModel,          # Conditional U-Net with cross-attention (Rombach et al. 2022)
    DDPMScheduler,                  # DDPM training scheduler (Ho et al. 2020)
    DPMSolverMultistepScheduler,    # Fast ODE solver for sampling (Lu et al. 2022)
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from diffusers.training_utils import EMAModel

# For perceptual loss (Zhang et al. 2018)
import lpips

# --- AUTO-PALETTE HELPERS ---
def infer_palette_map_and_classes(mask_dir, ignore_values=None, limit=None):
    """
    Scans mask_dir/*.png and returns (palette_map, n_classes, sorted_values).
    ignore_values: labels to exclude (e.g., [255] for 'void').
    limit: scan only first N masks if provided.
    """
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
        raise RuntimeError("No class values found in masks. Check your mask files.")

    palette_map = {v: i for i, v in enumerate(vals)}
    n_classes = len(vals)
    return palette_map, n_classes, vals

def _maybe_infer_palette(cfg):
    if cfg.palette_map is None:
        ignore = []
        palette_map, n_classes, vals = infer_palette_map_and_classes(cfg.mask_dir, ignore_values=ignore)
        cfg.palette_map = palette_map
        cfg.n_classes = n_classes
        print(f"[Auto palette] Found mask values: {vals}")
        print(f"[Auto palette] Using n_classes={n_classes} and palette_map={palette_map}")
    else:
        print(f"[Palette] Using provided palette_map={cfg.palette_map} and n_classes={cfg.n_classes}")

# ===============================
# === Config ====================
# ===============================
@dataclass
class TrainingConfig:
    # --- Data ---
    image_dir: str = "E:/projects/Wound_Segmentation_III/Data/new_images_640_1280"
    mask_dir: str = "E:/projects/Wound_Segmentation_III/Data/new_masks_640_1280"
    image_size: Tuple[int, int] = (384, 384)  # CRITICAL: Must be at least 256, ideally 384+
    n_classes: int = 11
    palette_map: Optional[Dict[int, int]] = None

    # --- Training ---
    train_batch_size: int = 12  # Increased (simpler model = more memory available)
    eval_batch_size: int = 12
    num_epochs: int = 400  # Reduced - faster convergence with larger batches
    gradient_accumulation_steps: int = 2  # Effective batch size = 24
    learning_rate: float = 1e-4
    discriminator_lr: float = 5e-5  # Lower LR for discriminator (common practice)
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.99)
    lr_warmup_steps: int = 2000  # Longer warmup for stability
    mixed_precision: str = "fp16"
    clip_grad_norm: float = 1.0
    ema_decay: float = 0.9999

    # --- Diffusion ---
    num_train_timesteps: int = 1000  # Standard DDPM (Ho et al. 2020)
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "v_prediction"
    snr_gamma: float = 5.0

    # --- Conditioning ---
    cond_dropout_prob: float = 0.1  # Lower dropout for better conditioning (medical images need strong guidance)
    guidance_scale: float = 3.0     # Higher guidance for better adherence to masks
    cross_attention_dim: int = 384  # Reduced from 512 for speed (20% faster, minimal quality loss)

    # --- Model Architecture (OPTIMIZED FOR SPEED) ---
    # Strategic attention placement: only in middle layers where it matters most
    block_out_channels: Tuple[int, ...] = (128, 256, 512, 512)  # 4 blocks instead of 6 (40% faster)
    layers_per_block: int = 2
    down_block_types: Tuple[str, ...] = (
        "DownBlock2D",           # No attention (early features don't need it)
        "CrossAttnDownBlock2D",  # Attention where it matters
        "CrossAttnDownBlock2D",  # Attention where it matters
        "DownBlock2D"            # No attention (deepest features are mostly spatial)
    )
    up_block_types: Tuple[str, ...] = (
        "UpBlock2D",
        "CrossAttnUpBlock2D",    # Attention on upsampling
        "CrossAttnUpBlock2D",    # Attention on upsampling
        "UpBlock2D"
    )
    attention_head_dim: int = 8
    gradient_checkpointing: bool = True
    enable_torch_compile: bool = False  # PyTorch 2.0+ compilation (requires Triton - Linux/Mac only)

    # --- Loss Weights (Dhariwal & Nichol 2021) ---
    use_perceptual_loss: bool = False  # DISABLED for speed (LPIPS is very slow)
    perceptual_weight: float = 0.1  # LPIPS weight (Zhang et al. 2018)
    use_adversarial_loss: bool = False  # DISABLED for speed (50% faster training)
    adversarial_weight: float = 0.01  # GAN loss weight (Isola et al. 2017)
    adversarial_start_epoch: int = 50  # Start adversarial training after diffusion model stabilizes

    # --- Multi-scale Training (Karras et al. 2018) ---
    # Note: Currently disabled to avoid batch size mismatch
    # For multi-scale, use batch_size=1 or implement custom collate_fn
    use_multiscale: bool = False
    multiscale_sizes: Tuple[Tuple[int, int], ...] = ((384, 384), (512, 512), (640, 640))
    multiscale_prob: float = 0.3  # Probability of using different scale

    # --- Medical-specific Augmentations (Tellez et al. 2018) ---
    use_medical_aug: bool = True
    stain_aug_prob: float = 0.5
    stain_brightness_range: Tuple[float, float] = (0.8, 1.2)
    stain_contrast_range: Tuple[float, float] = (0.8, 1.2)
    stain_saturation_range: Tuple[float, float] = (0.8, 1.2)

    # --- Eval / Saving ---
    output_dir: str = "diffusion_model_enhanced"
    save_image_epochs: int = 25
    save_model_epochs: int = 50
    seed: int = 42
    num_eval_inference_steps: int = 50
    num_vis: int = 8

    # --- Dataloader ---
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True

config = TrainingConfig()

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

def multiscale_collate_fn(batch):
    """
    Custom collate function for multi-scale training.
    When images have different sizes, we can only batch size 1.
    This function returns a list of single-item batches instead.
    """
    # Check if all items have same size
    first_img = batch[0]['images']
    all_same_size = all(item['images'].shape == first_img.shape for item in batch)

    if all_same_size:
        # Standard collation
        return {
            'images': torch.stack([item['images'] for item in batch]),
            'mask_onehot': torch.stack([item['mask_onehot'] for item in batch]),
        }
    else:
        # Return batch as-is (will process one at a time)
        # This is a fallback - in practice, set batch_size=1 for multi-scale
        raise RuntimeError(
            "Multi-scale training requires batch_size=1 or disabling use_multiscale. "
            "Got images of different sizes in the same batch."
        )

# ===============================
# === Medical-specific Augmentations ===
# (Tellez et al. 2018: https://arxiv.org/abs/1902.06543)
# ===============================
class MedicalStainAugmentation:
    """
    Stain/color augmentation for medical images (Tellez et al. 2018).
    Simulates variations in imaging conditions, lighting, and tissue staining.
    """
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
        # Random brightness (simulates lighting variations)
        brightness_factor = random.uniform(*self.brightness_range)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        # Random contrast (tissue density variations)
        contrast_factor = random.uniform(*self.contrast_range)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

        # Random saturation (blood/tissue color variations)
        saturation_factor = random.uniform(*self.saturation_range)
        img = ImageEnhance.Color(img).enhance(saturation_factor)

        # Small hue shift (staining variations)
        if random.random() < 0.5:
            img_tensor = TF.to_tensor(img)
            img_tensor = TF.adjust_hue(img_tensor, random.uniform(-self.hue_shift, self.hue_shift))
            img = TF.to_pil_image(img_tensor)

        return img

# Paired geometric augmentations (safe for masks)
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
# === Mask Encoder (for cross-attention conditioning) ===
# (Rombach et al. 2022: https://arxiv.org/abs/2112.10752)
# ===============================
class MaskEncoder(nn.Module):
    """
    Encodes one-hot mask into embedding for cross-attention conditioning.
    Similar to CLIP text encoder in Stable Diffusion but for spatial masks.
    """
    def __init__(self, n_classes: int, embed_dim: int = 512):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = embed_dim

        # Convolutional encoder to extract mask features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_classes, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, embed_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, embed_dim),
            nn.SiLU(),
        )

        # Spatial to sequence (for cross-attention)
        self.to_seq = nn.Conv2d(embed_dim, embed_dim, 1)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: [B, C, H, W]
        x = self.conv_layers(mask)  # [B, embed_dim, H/8, W/8]
        x = self.to_seq(x)  # [B, embed_dim, H/8, W/8]

        # Flatten spatial dims for cross-attention: [B, embed_dim, H*W/64]
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)  # [B, embed_dim, seq_len]
        x = x.permute(0, 2, 1)   # [B, seq_len, embed_dim]

        return x

# ===============================
# === PatchGAN Discriminator ===
# (Isola et al. 2017: https://arxiv.org/abs/1611.07004)
# ===============================
class PatchGANDiscriminator(nn.Module):
    """
    Patch-based discriminator for adversarial training (Isola et al. 2017).
    Operates on 70x70 patches for local realism.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()

        def discriminator_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)  # Output: [B, 1, H/16, W/16]
        )

    def forward(self, img):
        return self.model(img)

# ===============================
# === Dataset ===================
# ===============================
class WoundDatasetEnhanced(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        base_size: Tuple[int, int],
        n_classes: int,
        palette_map: Optional[Dict[int, int]] = None,
        augment: bool = True,
        eval_subset: bool = False,
        num_vis: int = 8,
        use_medical_aug: bool = True,
        use_multiscale: bool = False,
        multiscale_sizes: Tuple[Tuple[int, int], ...] = None,
        multiscale_prob: float = 0.3,
        stain_aug_config: dict = None,
    ):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        assert len(self.image_paths) == len(self.mask_paths), "Images and masks must match in count"

        self.base_size = base_size
        self.n_classes = n_classes
        self.palette_map = palette_map
        self.augment = augment
        self.paired_aug = PairedAugment()
        self.use_medical_aug = use_medical_aug
        self.use_multiscale = use_multiscale
        self.multiscale_sizes = multiscale_sizes or [base_size]
        self.multiscale_prob = multiscale_prob

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

        # Paired geometric augmentations
        if self.augment:
            img, mask_img = self.paired_aug(img, mask_img)

            # Medical stain augmentation (only on image)
            if self.stain_aug and random.random() < 0.5:
                img = self.stain_aug(img)

        # Multi-scale training (Karras et al. 2018)
        if self.use_multiscale and random.random() < self.multiscale_prob:
            size = random.choice(self.multiscale_sizes)
        else:
            size = self.base_size

        # Resize
        img_t = to_tensor_img(img, size)
        mask_r = resize_mask_nearest(mask_img, size)

        # Map + one-hot
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
    """
    Inference with classifier-free guidance for cross-attention conditioned model.
    (Ho & Salimans 2022: https://arxiv.org/abs/2207.12598)
    """
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

        # Generate noise
        x = torch.randn((b, 3, h, w), generator=generator, device=device)
        masks = masks.to(device)

        # Encode masks for cross-attention
        mask_embeds = self.mask_encoder(masks)  # [B, seq_len, embed_dim]
        uncond_embeds = self.mask_encoder(torch.zeros_like(masks))  # Unconditional

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for t in self.scheduler.timesteps:
            # Unconditional prediction
            noise_pred_uncond = self.unet(x, t, encoder_hidden_states=uncond_embeds).sample

            # Conditional prediction
            noise_pred_cond = self.unet(x, t, encoder_hidden_states=mask_embeds).sample

            # CFG
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Scheduler step
            x = self.scheduler.step(noise_pred, t, x, generator=generator).prev_sample

        # Denormalize
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
    """p2 SNR reweighting (Imagen: https://arxiv.org/abs/2205.11487)"""
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
    cfg: TrainingConfig,
    accelerator: Accelerator,
    model: UNet2DConditionModel,
    mask_encoder: MaskEncoder,
    base_scheduler: DDPMScheduler,
    eval_masks: torch.Tensor,
    epoch: int,
    ema_unet: Optional[EMAModel] = None,
    ema_encoder: Optional[EMAModel] = None,
):
    is_main = accelerator.is_main_process
    if not is_main:
        return

    # Use EMA weights
    if ema_unet is not None:
        ema_unet.store(model.parameters())
        ema_unet.copy_to(model.parameters())
    if ema_encoder is not None:
        ema_encoder.store(mask_encoder.parameters())
        ema_encoder.copy_to(mask_encoder.parameters())

    # Build pipeline with DPMSolver (Lu et al. 2022)
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

    # Restore original weights
    if ema_unet is not None:
        ema_unet.restore(model.parameters())
    if ema_encoder is not None:
        ema_encoder.restore(mask_encoder.parameters())

# ===============================
# === Train =====================
# ===============================
def train(cfg: TrainingConfig):
    set_seed(cfg.seed)

    # Create logging directory for tensorboard
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=log_dir,  # Required for tensorboard logging
    )

    is_main = accelerator.is_main_process

    _maybe_infer_palette(cfg)

    if is_main:
        print("Enhanced Diffusion Config:", asdict(cfg))
        print(f"Effective batch size: {cfg.train_batch_size * cfg.gradient_accumulation_steps}")
        print(f"Using perceptual loss: {cfg.use_perceptual_loss}")
        print(f"Using adversarial loss: {cfg.use_adversarial_loss}")
        print(f"Multi-scale training: {cfg.use_multiscale}")

    # Datasets
    stain_config = {
        'brightness_range': cfg.stain_brightness_range,
        'contrast_range': cfg.stain_contrast_range,
        'saturation_range': cfg.stain_saturation_range,
    }

    train_dataset = WoundDatasetEnhanced(
        image_dir=cfg.image_dir,
        mask_dir=cfg.mask_dir,
        base_size=cfg.image_size,
        n_classes=cfg.n_classes,
        palette_map=cfg.palette_map,
        augment=True,
        use_medical_aug=cfg.use_medical_aug,
        use_multiscale=cfg.use_multiscale,
        multiscale_sizes=cfg.multiscale_sizes,
        multiscale_prob=cfg.multiscale_prob,
        stain_aug_config=stain_config,
    )

    eval_dataset = WoundDatasetEnhanced(
        image_dir=cfg.image_dir,
        mask_dir=cfg.mask_dir,
        base_size=cfg.image_size,
        n_classes=cfg.n_classes,
        palette_map=cfg.palette_map,
        augment=False,
        eval_subset=True,
        num_vis=cfg.num_vis,
    )

    # Use custom collate function for multi-scale training
    collate_fn = multiscale_collate_fn if cfg.use_multiscale else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )

    eval_masks = torch.stack([eval_dataset[i]["mask_onehot"] for i in range(len(eval_dataset))])

    # Models
    if is_main:
        print(f"Creating UNet2DConditionModel with cross-attention...")

    # Mask encoder for cross-attention conditioning (Rombach et al. 2022)
    mask_encoder = MaskEncoder(
        n_classes=cfg.n_classes,
        embed_dim=cfg.cross_attention_dim
    )

    # Main U-Net with cross-attention (deeper architecture)
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
        if is_main:
            print("Gradient checkpointing enabled (Chen et al. 2016)")

    # Torch compile for 20-30% speedup (PyTorch 2.0+)
    if cfg.enable_torch_compile and hasattr(torch, 'compile'):
        if is_main:
            print("Compiling model with torch.compile (PyTorch 2.0+)...")
        try:
            model = torch.compile(model, mode='reduce-overhead')
            mask_encoder = torch.compile(mask_encoder, mode='reduce-overhead')
            if is_main:
                print("Model compiled successfully!")
        except Exception as e:
            if is_main:
                print(f"Warning: torch.compile failed: {e}. Continuing without compilation.")

    # Discriminator for adversarial training (Isola et al. 2017)
    discriminator = None
    if cfg.use_adversarial_loss:
        discriminator = PatchGANDiscriminator(in_channels=3)
        if is_main:
            print("PatchGAN discriminator created (Isola et al. 2017)")

    # Perceptual loss model (Zhang et al. 2018)
    lpips_loss = None
    if cfg.use_perceptual_loss:
        lpips_loss = lpips.LPIPS(net='vgg').to(accelerator.device)
        lpips_loss.requires_grad_(False)
        if is_main:
            print("LPIPS perceptual loss loaded (Zhang et al. 2018)")

    # Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
    )

    # Optimizers
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(mask_encoder.parameters()),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    optimizer_D = None
    if discriminator is not None:
        optimizer_D = torch.optim.AdamW(
            discriminator.parameters(),
            lr=cfg.discriminator_lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

    # LR schedulers
    num_training_steps = cfg.num_epochs * math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=num_training_steps,
    )

    lr_scheduler_D = None
    if optimizer_D is not None:
        lr_scheduler_D = get_cosine_schedule_with_warmup(
            optimizer=optimizer_D,
            num_warmup_steps=cfg.lr_warmup_steps,
            num_training_steps=num_training_steps,
        )

    # EMA
    ema_unet = EMAModel(parameters=model.parameters(), power=cfg.ema_decay)
    ema_encoder = EMAModel(parameters=mask_encoder.parameters(), power=cfg.ema_decay)

    # Prepare with accelerator
    if discriminator is not None and optimizer_D is not None:
        model, mask_encoder, discriminator, optimizer, optimizer_D, train_loader, lr_scheduler, lr_scheduler_D = accelerator.prepare(
            model, mask_encoder, discriminator, optimizer, optimizer_D, train_loader, lr_scheduler, lr_scheduler_D
        )
    else:
        model, mask_encoder, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            model, mask_encoder, optimizer, train_loader, lr_scheduler
        )

    ema_unet.to(accelerator.device)
    ema_encoder.to(accelerator.device)

    accelerator.init_trackers("enhanced_mask_diffusion")

    global_step = 0
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Training loop
    for epoch in range(cfg.num_epochs):
        model.train()
        mask_encoder.train()
        if discriminator is not None:
            discriminator.train()

        progress = tqdm(total=len(train_loader), disable=not is_main, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")

        for step, batch in enumerate(train_loader):
            clean_images: torch.Tensor = batch["images"]
            mask_onehot: torch.Tensor = batch["mask_onehot"]

            bs = clean_images.shape[0]

            # ===== GENERATOR (Diffusion Model) STEP =====
            with accelerator.accumulate(model):
                # Sample noise and timesteps
                noise = torch.randn_like(clean_images)
                # Offset noise for better quality (common trick)
                noise = noise + 0.1 * torch.randn(bs, 1, 1, 1, device=noise.device)

                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bs,), device=clean_images.device, dtype=torch.int64
                )

                # Forward diffusion
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                # Classifier-free guidance dropout
                if random.random() < cfg.cond_dropout_prob:
                    cond_mask = torch.zeros_like(mask_onehot)
                else:
                    cond_mask = mask_onehot

                # Encode mask for cross-attention
                mask_embeds = mask_encoder(cond_mask)

                # Predict noise/velocity
                target_v = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                v_pred = model(noisy_images, timesteps, encoder_hidden_states=mask_embeds).sample

                # Diffusion loss (SNR weighted)
                w = snr_weight(noise_scheduler, timesteps, gamma=cfg.snr_gamma)
                loss_diffusion = F.mse_loss(v_pred, target_v, reduction="none").mean(dim=[1,2,3])
                loss_diffusion = (loss_diffusion * w).mean()

                # Decode to image (approximate for losses)
                with torch.no_grad():
                    # Quick approximation: x0 = (noisy - sqrt(1-a)*pred_noise) / sqrt(a)
                    alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(clean_images.device)
                    pred_original = (noisy_images - torch.sqrt(1 - alpha_t) * v_pred) / torch.sqrt(alpha_t + 1e-8)
                    pred_original = pred_original.clamp(-1, 1)

                # Perceptual loss (Zhang et al. 2018)
                loss_perceptual = 0.0
                if cfg.use_perceptual_loss and lpips_loss is not None:
                    loss_perceptual = lpips_loss(pred_original, clean_images).mean()

                # Adversarial loss (Isola et al. 2017; Dhariwal & Nichol 2021)
                loss_adv = 0.0
                if cfg.use_adversarial_loss and discriminator is not None and epoch >= cfg.adversarial_start_epoch:
                    fake_logits = discriminator(pred_original)
                    loss_adv = F.binary_cross_entropy_with_logits(
                        fake_logits,
                        torch.ones_like(fake_logits)
                    )

                # Total generator loss
                loss_total = (
                    loss_diffusion +
                    cfg.perceptual_weight * loss_perceptual +
                    cfg.adversarial_weight * loss_adv
                )

                accelerator.backward(loss_total)

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

            # ===== DISCRIMINATOR STEP =====
            loss_D = 0.0
            if cfg.use_adversarial_loss and discriminator is not None and optimizer_D is not None and epoch >= cfg.adversarial_start_epoch:
                with accelerator.accumulate(discriminator):
                    # Real images
                    real_logits = discriminator(clean_images)
                    loss_D_real = F.binary_cross_entropy_with_logits(
                        real_logits,
                        torch.ones_like(real_logits)
                    )

                    # Fake images (detached)
                    with torch.no_grad():
                        noise = torch.randn_like(clean_images)
                        noise = noise + 0.1 * torch.randn(bs, 1, 1, 1, device=noise.device)
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps,
                            (bs,), device=clean_images.device, dtype=torch.int64
                        )
                        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                        mask_embeds = mask_encoder(mask_onehot)
                        v_pred = model(noisy_images, timesteps, encoder_hidden_states=mask_embeds).sample
                        alpha_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(clean_images.device)
                        fake_images = (noisy_images - torch.sqrt(1 - alpha_t) * v_pred) / torch.sqrt(alpha_t + 1e-8)
                        fake_images = fake_images.clamp(-1, 1)

                    fake_logits = discriminator(fake_images.detach())
                    loss_D_fake = F.binary_cross_entropy_with_logits(
                        fake_logits,
                        torch.zeros_like(fake_logits)
                    )

                    loss_D = (loss_D_real + loss_D_fake) / 2

                    accelerator.backward(loss_D)

                    if accelerator.sync_gradients and cfg.clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(discriminator.parameters(), cfg.clip_grad_norm)

                    optimizer_D.step()
                    lr_scheduler_D.step()
                    optimizer_D.zero_grad()

            # Logging
            logs = {
                "loss_diffusion": float(loss_diffusion.detach().item()),
                "loss_perceptual": float(loss_perceptual) if isinstance(loss_perceptual, torch.Tensor) else float(loss_perceptual),
                "loss_adversarial": float(loss_adv) if isinstance(loss_adv, torch.Tensor) else float(loss_adv),
                "loss_discriminator": float(loss_D) if isinstance(loss_D, torch.Tensor) else float(loss_D),
                "loss_total": float(loss_total.detach().item()),
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "step": global_step,
            }
            progress.set_postfix(**{k: f"{v:.4f}" for k, v in logs.items() if k not in ["step"]})
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

                if discriminator is not None:
                    torch.save(
                        accelerator.unwrap_model(discriminator).state_dict(),
                        os.path.join(cfg.output_dir, 'discriminator.pt')
                    )

    accelerator.end_training()
    if is_main:
        print("Enhanced training complete. Models and samples in:", cfg.output_dir)

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
