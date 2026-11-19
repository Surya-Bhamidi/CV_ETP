"""Configuration, synthetic camera degradation and reference metrics for the UI."""

import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .network import SRNet, n_params


# %% Checkpoint configuration
@dataclass
class Config:
    work_dir: str = (
        "/content/work" if os.path.exists("/content") else "./work"
    )  # data + checkpoints live here
    scale: int = 4  # super-resolution factor (28x28 -> 112x112)
    hr_size: int = 112  # HR crop size = ArcFace input size
    n_val: int = 1000  # fixed validation images (pick the best epoch with these)
    n_test: int = 2000  # fixed test split (historically evaluated multiple times)
    feat: int = 48  # feature channels of the network
    blocks: int = 6  # number of residual blocks
    keep_ratio: float = 0.5  # fraction of pixels kept as attention keys (the "sparse" part)
    batch: int = 64
    epochs: int = 30
    lr: float = 2e-4
    num_workers: int = 0 if os.name == "nt" else 4  # Windows notebooks need in-process loading
    seed: int = 42


# %% Construct the low-resolution observation
class Degrader:
    def __init__(self, scale=4, blur_p=0.7, noise=(2, 15), jpeg=(30, 70)):
        self.scale, self.blur_p, self.noise, self.jpeg = scale, blur_p, noise, jpeg

    def __call__(self, hr, rng=None):  # hr: uint8 HxWx3  ->  lr: uint8 (H/s)x(W/s)x3
        rng = rng or np.random.default_rng()  # seeded rng => reproducible val/test images
        img = hr
        if rng.random() < self.blur_p:  # 1) random Gaussian blur (lens / motion blur)
            img = cv2.GaussianBlur(
                img, (int(rng.choice([3, 5, 7])),) * 2, float(rng.uniform(0.2, 2.0))
            )
        h, w = hr.shape[:2]
        lr = np.asarray(
            Image.fromarray(img).resize((w // self.scale, h // self.scale), Image.BICUBIC)
        )  # 2) bicubic downsample (antialiased)
        lr = lr + rng.normal(0, rng.uniform(*self.noise), lr.shape)  # 3) sensor noise
        lr = np.clip(lr, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(lr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(*self.jpeg))]
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("JPEG decoding failed")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


# %% Score a reconstruction against a clean reference
class Metrics:
    @staticmethod
    def rgb_to_y(x):  # x: BCHW in [0,1] -> luminance Y in [0,1]
        return (16 + 65.481 * x[:, 0] + 128.553 * x[:, 1] + 24.966 * x[:, 2]) / 255.0

    @staticmethod
    def psnr_y(sr, hr, shave):  # per-image PSNR, border pixels cropped
        a, b = (
            Metrics.rgb_to_y(sr)[:, shave:-shave, shave:-shave],
            Metrics.rgb_to_y(hr)[:, shave:-shave, shave:-shave],
        )
        return 10 * torch.log10(1 / ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10))

    @staticmethod
    def ssim_y(sr, hr, shave):
        from skimage.metrics import structural_similarity as ssim

        a, b = (
            Metrics.rgb_to_y(sr)[:, shave:-shave, shave:-shave].numpy(),
            Metrics.rgb_to_y(hr)[:, shave:-shave, shave:-shave].numpy(),
        )
        return torch.tensor([ssim(p, q, data_range=1.0) for p, q in zip(a, b)])
