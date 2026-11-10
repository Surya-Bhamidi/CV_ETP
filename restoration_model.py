"""Optional official pretrained GFPGAN baseline, separate from the custom SRNet.

Operates on already cropped, approximately aligned faces. Does not detect or
identify people. Returns the same x4 dimensions for a fair UI comparison.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'work/restoration_assets/GFPGANv1.4.pth'


class FaceRestoration(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        from restoration_vendor.gfpganv1_clean_arch import GFPGANv1Clean
        self.net = GFPGANv1Clean(out_size=512, num_style_feat=512,
            channel_multiplier=2, decoder_load_path=None, fix_decoder=True,
            num_mlp=8, input_is_latent=True, different_w=True, narrow=1, sft_half=True)
        checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
        self.net.load_state_dict(checkpoint['params_ema'], strict=True)
        self.to(device).eval().requires_grad_(False)

    def forward(self, low):
        target = (low.shape[-2] * 4, low.shape[-1] * 4)
        # Batches of two bound memory for 512-pixel StyleGAN synthesis.
        outputs = []
        for chunk in low.split(2):
            up = F.interpolate(chunk, size=(512, 512), mode='bicubic', align_corners=False).clamp(0, 1)
            result, _ = self.net(up * 2 - 1, return_rgb=False, randomize_noise=False)
            restored = ((result + 1) / 2).clamp(0, 1)
            outputs.append(F.interpolate(restored, size=target, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1))
        return torch.cat(outputs)
