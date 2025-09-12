# train_mask_conditional_diffusion.py
# ------------------------------------------------------------
# Mask-conditional DDPM with:
# - One-hot mask conditioning
# - v-prediction + SNR (p2) loss weighting
# - EMA for cleaner sampling/checkpoints
# - Classifier-free guidance via condition dropout
# - DPMSolverMultistep for fast inference
# - Paired geometric augmentations safe for masks
# ------------------------------------------------------------
#
# === Paper Annotations (key techniques & sources) =========================
# DDPM (framework): Ho et al. 2020, "Denoising Diffusion Probabilistic Models"
#   https://arxiv.org/abs/2006.11239
# Improved DDPM (cosine schedule, etc.): Nichol & Dhariwal 2021
#   https://arxiv.org/abs/2102.09672
# U-Net backbone: Ronneberger et al. 2015
#   https://arxiv.org/abs/1505.04597
# Attention in diffusion U-Nets (context): Dhariwal & Nichol 2021
#   https://arxiv.org/abs/2105.05233
# Classifier-Free Guidance (CFG): Ho & Salimans 2022
#   https://arxiv.org/abs/2207.12598
# v-prediction parameterization: Salimans & Ho 2022, "Progressive Distillation"
#   https://arxiv.org/abs/2202.00512
# SNR (p2) loss reweighting: Imagen (Saharia et al. 2022)
#   https://arxiv.org/abs/2205.11487
# DPMSolver (fast sampler): Lu et al. 2022
#   https://arxiv.org/abs/2206.00927
# Gradient checkpointing: Chen et al. 2016
#   https://arxiv.org/abs/1604.06174
# Mixed precision: Micikevicius et al. 2017
#   https://arxiv.org/abs/1710.03740
# Image-to-image diffusion (conditioning paradigm akin to mask-conditional): Palette (Saharia et al. 2021)
#   https://arxiv.org/abs/2111.05826
# Paired geometric augmentations for segmentation: Shorten & Khoshgoftaar 2019 (survey)
#   https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0
# ==========================================================================

import os
import glob
import math
import random
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from accelerate import Accelerator
from tqdm import tqdm

from diffusers import (
    UNet2DModel,                  # U-Net backbone inside diffusion model (Ronneberger et al. 2015; attention + diffusion context: Dhariwal & Nichol 2021)
    DDPMScheduler,                # DDPM training scheduler (Ho et al. 2020)
    DPMSolverMultistepScheduler,  # Fast ODE solver for sampling (Lu et al. 2022)
    DDPMPipeline,
)
from diffusers.optimization import get_cosine_schedule_with_warmup  # Cosine schedule popularized in Improved DDPM (Nichol & Dhariwal 2021)
from diffusers.utils import make_image_grid
import diffusers
from diffusers.training_utils import EMAModel  # EMA of weights (Polyak & Juditsky 1992; widely used e.g., StyleGAN2)

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

    # Map raw labels (e.g., 0,15,30,...) -> contiguous ids (0,1,2,...)
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
    image_dir: str = "../Data/new_images_640_1280"
    mask_dir: str = "../Data/new_masks_640_1280"
    image_size: Tuple[int, int] = (384, 384)
    n_classes: int = 11  # <-- set to number of semantic classes
    # If mask pixels are raw values (e.g., {0, 15, 30, 45}),
    # map to contiguous ids {0..C-1} here; otherwise set to None.
    palette_map: Optional[Dict[int, int]] = None  # e.g., {0:0, 15:1, 30:2, 45:3}

    # --- Training ---
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_epochs: int = 1000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.99)
    lr_warmup_steps: int = 1000
    mixed_precision: str = "fp16"  # Mixed precision training (Micikevicius et al. 2017: https://arxiv.org/abs/1710.03740)
    clip_grad_norm: float = 1.0
    ema_decay: float = 0.9999  # EMA of weights (Polyak & Juditsky 1992)

    # --- Diffusion ---
    num_train_timesteps: int = 10000
    beta_schedule: str = "squaredcos_cap_v2"  # Cosine schedule (Improved DDPM: https://arxiv.org/abs/2102.09672)
    prediction_type: str = "v_prediction"  # v-prediction parameterization (Salimans & Ho 2022: https://arxiv.org/abs/2202.00512)
    snr_gamma: float = 5.0  # p2 reweighting cap (Imagen: https://arxiv.org/abs/2205.11487)

    # --- Conditioning ---
    cond_dropout_prob: float = 0.2  # Classifier-Free Guidance training (https://arxiv.org/abs/2207.12598)
    guidance_scale: float = 2.0     # CFG inference scale (https://arxiv.org/abs/2207.12598)

    # --- Model ---
    # block_out_channels: Tuple[int, ...] = (128, 256, 256, 512)
    layers_per_block: int = 2
    # down_block_types: Tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D")
    # up_block_types: Tuple[str, ...] = ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")
    block_out_channels: Tuple[int, ...] = (128, 128, 256, 256, 512, 512)
    down_block_types: Tuple[str, ...] = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
    up_block_types: Tuple[str, ...] = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    gradient_checkpointing: bool = True  # Gradient checkpointing (Chen et al. 2016: https://arxiv.org/abs/1604.06174)

    # --- Eval / Saving ---
    output_dir: str = "diffusion_model"
    save_image_epochs: int = 50
    save_model_epochs: int = 50
    seed: int = 42
    num_eval_inference_steps: int = 50  # Fast sampler steps with DPMSolver (https://arxiv.org/abs/2206.00927)
    num_vis: int = 8  # number of eval samples to visualize

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
    # [-1,1] normalized 3xHxW
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Common normalization practice in DDPM pipelines
    ])
    return tfm(img)

def resize_mask_nearest(mask: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return mask.resize(size, Image.NEAREST)  # Preserve discrete labels (augmentation survey: https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)

def map_mask_pixels(np_mask: np.ndarray, palette_map: Optional[Dict[int, int]], n_classes: int) -> np.ndarray:
    if palette_map is None:
        # assume already in [0..C-1]
        return np_mask
    # vectorized mapping; unknown values go to 0
    lut = np.zeros(256, dtype=np.int64)
    for k, v in palette_map.items():
        lut[int(k)] = int(v)
    np_mask = lut[np_mask.astype(np.uint8)]
    # clip to [0..C-1]
    np_mask = np.clip(np_mask, 0, n_classes - 1)
    return np_mask

def one_hot_mask(np_ids: np.ndarray, n_classes: int) -> torch.Tensor:
    # returns [C,H,W] float
    t = torch.from_numpy(np_ids.astype(np.int64))
    oh = F.one_hot(t, num_classes=n_classes).permute(2, 0, 1).float()
    return oh

# Safe paired geometric augs (no interpolation artifacts on labels)
class PairedAugment:
    def __init__(self, p_hflip=0.5, p_vflip=0.5, p_rot90=0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90

    def __call__(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Horizontal flip
        if random.random() < self.p_hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # Vertical flip
        if random.random() < self.p_vflip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        # Rotate by k * 90
        if random.random() < self.p_rot90:
            k = random.choice([1, 2, 3])
            img = img.rotate(90 * k, expand=True)
            mask = mask.rotate(90 * k, expand=True)
        return img, mask

# ===============================
# === Dataset ===================
# ===============================
class WoundDataset(Dataset):
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
    ):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_paths  = sorted(glob.glob(os.path.join(mask_dir,  "*.png")))
        assert len(self.image_paths) == len(self.mask_paths), "Images and masks must match in count"

        self.size = size
        self.n_classes = n_classes
        self.palette_map = palette_map
        self.augment = augment
        self.paired_aug = PairedAugment()

        # Optional small eval subset for quick visualization
        if eval_subset and len(self.image_paths) > num_vis:
            idx = np.linspace(0, len(self.image_paths) - 1, num=num_vis, dtype=int)
            self.image_paths = [self.image_paths[i] for i in idx]
            self.mask_paths  = [self.mask_paths[i]  for i in idx]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask_img = Image.open(self.mask_paths[idx]).convert("L")

        # paired augmentations before resizing
        if self.augment:
            img, mask_img = self.paired_aug(img, mask_img)

        # resize
        img_t  = to_tensor_img(img, self.size)  # [-1,1]
        mask_r = resize_mask_nearest(mask_img, self.size)

        # map + one-hot
        np_ids = np.array(mask_r, dtype=np.int64)
        np_ids = map_mask_pixels(np_ids, self.palette_map, self.n_classes)
        one_hot = one_hot_mask(np_ids, self.n_classes)  # [C,H,W]

        # combined input for UNet: [3 + C, H, W]
        combined = torch.cat([img_t, one_hot], dim=0)
        return {
            "images": img_t,            # [3,H,W] (clean)
            "mask_onehot": one_hot,     # [C,H,W]
            "combined": combined,       # [3+C,H,W] (for convenience)
        }
        # Note: Conditioning with masks follows the image-to-image diffusion paradigm (cf. Palette: https://arxiv.org/abs/2111.05826)

# ===============================
# === Pipeline ==================
# ===============================
class MaskConditionalDDPMPipeline(DDPMPipeline):
    """
    Inference with classifier-free guidance (CFG).
    Expects UNet trained with mask dropout (cond_dropout_prob>0).
    (Ho & Salimans 2022: https://arxiv.org/abs/2207.12598)
    """
    @torch.no_grad()
    def __call__(
        self,
        masks: torch.Tensor,                # [B,C,H,W] one-hot mask (float)
        num_inference_steps: int = 50,
        guidance_scale: float = 2.0,
        generator: Optional[torch.Generator] = None,
    ) -> List[Image.Image]:
        device = self.unet.device
        b, c, h, w = masks.shape
        x = torch.randn((b, 3, h, w), generator=generator, device=device)
        masks = masks.to(device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for t in self.scheduler.timesteps:
            # unconditional (mask=0) — unconditional branch for CFG
            uncond_in = torch.cat([x, torch.zeros_like(masks)], dim=1)
            eps_uncond = self.unet(uncond_in, t).sample

            # conditional — conditioned on mask
            cond_in = torch.cat([x, masks], dim=1)
            eps_cond = self.unet(cond_in, t).sample

            # CFG combine
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            step = self.scheduler.step(eps, t, x, generator=generator)
            x = step.prev_sample

        x = (x.clamp(-1, 1) + 1) / 2.0
        images = [transforms.ToPILImage()(img.cpu()) for img in x]
        return images

# ===============================
# === Loss helpers ==============
# ===============================
def snr_weight(scheduler: DDPMScheduler, timesteps: torch.LongTensor, gamma: float = 5.0) -> torch.Tensor:
    # p2 SNR reweighting (Imagen: https://arxiv.org/abs/2205.11487)
    alphas_cumprod = scheduler.alphas_cumprod.to(timesteps.device)  # [T]
    a_bar = alphas_cumprod.gather(0, timesteps)                      # [B]
    snr = a_bar / (1.0 - a_bar + 1e-8)
    # p2 reweighting
    w = torch.minimum(snr, torch.full_like(snr, gamma)) / (snr + 1e-8)
    return w

# ===============================
# === Evaluate ==================
# ===============================
@torch.no_grad()
def evaluate_and_save_samples(
    cfg: TrainingConfig,
    accelerator: Accelerator,
    model: UNet2DModel,
    base_scheduler: DDPMScheduler,
    eval_masks: torch.Tensor,
    epoch: int,
    ema: Optional[EMAModel] = None,
):
    is_main = accelerator.is_main_process
    if not is_main:
        return

    # Use EMA weights for sampling (Polyak averaging for smoother samples)
    if ema is not None:
        ema.store(model.parameters())
        ema.copy_to(model.parameters())

    # Build a pipeline with DPMSolver scheduler for speed (Lu et al. 2022: https://arxiv.org/abs/2206.00927)
    pipe = MaskConditionalDDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=base_scheduler)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    masks = eval_masks.to(device)  # [B,C,H,W]
    images = pipe(
        masks=masks,
        num_inference_steps=config.num_eval_inference_steps,
        guidance_scale=config.guidance_scale,
        generator=gen,
    )

    os.makedirs(os.path.join(cfg.output_dir, "samples"), exist_ok=True)
    grid = make_image_grid(images, rows=1, cols=len(images))
    out_path = os.path.join(cfg.output_dir, "samples", f"{epoch:04d}.png")
    grid.save(out_path)

    # Restore original (non-EMA) weights after eval
    if ema is not None:
        ema.restore(model.parameters())

# ===============================
# === Train =====================
# ===============================
def train(cfg: TrainingConfig):
    set_seed(cfg.seed)

    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,              # Mixed precision (Micikevicius et al. 2017)
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with="tensorboard",
    )

    is_main = accelerator.is_main_process
    
    _maybe_infer_palette(cfg)

    if is_main:
        print("Config:", asdict(cfg))

    
    # Datasets / loaders
    train_dataset = WoundDataset(
        image_dir=cfg.image_dir,
        mask_dir=cfg.mask_dir,
        size=cfg.image_size,
        n_classes=cfg.n_classes,
        palette_map=cfg.palette_map,
        augment=True,
        eval_subset=False,
    )
    eval_dataset = WoundDataset(
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

    eval_masks = torch.stack([eval_dataset[i]["mask_onehot"] for i in range(len(eval_dataset))])  # [B,C,H,W]

    # Model
    in_channels = 3 + cfg.n_classes
    model = UNet2DModel(
        sample_size=cfg.image_size[0],
        in_channels=in_channels,
        out_channels=3,
        layers_per_block=cfg.layers_per_block,
        block_out_channels=cfg.block_out_channels,
        down_block_types=cfg.down_block_types,
        up_block_types=cfg.up_block_types,
    )
    if cfg.gradient_checkpointing:
        model.enable_gradient_checkpointing()  # Gradient checkpointing (Chen et al. 2016)

    # Scheduler (training)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,          # cosine schedule (Improved DDPM)
        prediction_type=cfg.prediction_type,      # v-prediction (Salimans & Ho 2022)
    )

    # Optimizer / LR sched
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    num_training_steps = cfg.num_epochs * math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # EMA (Polyak averaging)
    ema = EMAModel(parameters=model.parameters(), power=cfg.ema_decay)

    # Prepare with accelerator
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    ema.to(accelerator.device)

    accelerator.init_trackers("mask_conditional_ddpm")

    global_step = 0
    os.makedirs(cfg.output_dir, exist_ok=True)

    for epoch in range(cfg.num_epochs):
        model.train()
        progress = tqdm(total=len(train_loader), disable=not is_main, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for step, batch in enumerate(train_loader):
            clean_images: torch.Tensor = batch["images"]          # [B,3,H,W] in [-1,1]
            mask_onehot: torch.Tensor = batch["mask_onehot"]      # [B,C,H,W]

            bs = clean_images.shape[0]
            # Sample noise and timesteps
            noise = torch.randn_like(clean_images)
            # Optional: offset noise for slightly improved stability
            noise = noise + 0.1 * torch.randn_like(clean_images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bs,), device=clean_images.device, dtype=torch.int64
            )

            # Forward diffusion (Ho et al. 2020)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Classifier-free guidance training: randomly drop mask (Ho & Salimans 2022)
            if random.random() < cfg.cond_dropout_prob:
                cond_mask = torch.zeros_like(mask_onehot)
            else:
                cond_mask = mask_onehot

            model_input = torch.cat([noisy_images, cond_mask], dim=1)  # [B,3+C,H,W]

            with accelerator.accumulate(model):
                # Target is velocity for v-pred (Salimans & Ho 2022)
                target_v = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                # Forward
                out = model(model_input, timesteps, return_dict=True)
                v_pred = out.sample

                # SNR(p2) weighted MSE (Imagen 2022)
                w = snr_weight(noise_scheduler, timesteps, gamma=cfg.snr_gamma)  # [B]
                loss_per = F.mse_loss(v_pred, target_v, reduction="none").mean(dim=[1,2,3])
                loss = (loss_per * w).mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients and cfg.clip_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # EMA update on unwrapped params
                ema.step(accelerator.unwrap_model(model).parameters())

            # Logging
            logs = {
                "loss": float(loss.detach().item()),
                "lr": float(lr_scheduler.get_last_lr()[0]),
                "step": global_step,
            }
            progress.set_postfix(loss=logs["loss"], lr=logs["lr"])
            accelerator.log(logs, step=global_step)
            global_step += 1
            progress.update(1)

        progress.close()

        # Evaluate + sample grid (use EMA + DPMSolver)
        if (epoch + 1) % cfg.save_image_epochs == 0 or epoch == cfg.num_epochs - 1:
            evaluate_and_save_samples(cfg, accelerator, model, noise_scheduler, eval_masks, epoch, ema=ema)

        # Save model (EMA weights)
        if (epoch + 1) % cfg.save_model_epochs == 0 or epoch == cfg.num_epochs - 1:
            if is_main:
                # temporarily copy EMA to model, then save pipeline
                ema.store(accelerator.unwrap_model(model).parameters())
                ema.copy_to(accelerator.unwrap_model(model).parameters())

                pipe = MaskConditionalDDPMPipeline(
                    unet=accelerator.unwrap_model(model),
                    scheduler=noise_scheduler
                )
                pipe.save_pretrained(cfg.output_dir)

                # restore original weights
                ema.restore(accelerator.unwrap_model(model).parameters())

    accelerator.end_training()
    if is_main:
        print("Training complete. Models and samples are in:", cfg.output_dir)

# ===============================
# === Main ======================
# ===============================
if __name__ == "__main__":
    # Optional: tweak a few config fields here, then run.
    # Example for palette:
    # config.palette_map = {0:0, 15:1, 30:2, 45:3}
    # config.n_classes = 4

    train(config)
    
    # After training, to quickly preview the latest grid:
    latest = sorted(glob.glob(os.path.join(config.output_dir, "samples", "*.png")))
    if latest:
        try:
            Image.open(latest[-1]).show()
        except Exception:
            print("Latest sample:", latest[-1])
