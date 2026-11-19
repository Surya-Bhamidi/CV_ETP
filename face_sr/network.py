"""Architecture shared by the training runner and comparison server."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d


# %% Learn local sampling positions
class SparseSamplingConv(nn.Module):
    """9 learnable sample points per pixel (deformable-style), kept inside a 7x7 window."""

    def __init__(self, c, max_offset=2.0):
        super().__init__()
        self.max_offset = max_offset
        self.offset = nn.Conv2d(c, 18, 3, padding=1)  # predicts (dy, dx) for each of the 9 points
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)  # start exactly as a normal 3x3 conv
        self.weight = nn.Parameter(torch.empty(c, c, 3, 3))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        off = (
            torch.tanh(self.offset(x)) * self.max_offset
        )  # 1 (grid) + 2 (offset) = 3 px -> 7x7 window
        return deform_conv2d(x, off, self.weight, self.bias, padding=(1, 1))


# %% Select spatial keys and aggregate global context
class AdaptiveSparseLinearAttention(nn.Module):
    """Linear attention O(n) where only the top-k most important pixels act as keys (content-adaptive sparsity)."""

    def __init__(self, c, keep_ratio=0.5):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.score = nn.Sequential(
            nn.Conv2d(c, c // 4, 1), nn.GELU(), nn.Conv2d(c // 4, 1, 1)
        )  # per-pixel importance
        self.proj = nn.Conv2d(c, c, 1)

    def key_mask(self, x):
        s = torch.sigmoid(self.score(x)).flatten(1)  # (N, HW) importance in (0,1)
        k = max(1, int(self.keep_ratio * s.shape[1]))
        hard = torch.zeros_like(s).scatter_(
            1, s.topk(k, dim=1).indices, 1.0
        )  # 1 for the top-k pixels, 0 elsewhere
        return (
            hard + s - s.detach()
        )  # forward = hard mask, backward = gradient of s (straight-through)

    def forward(self, x):
        n, c, h, w = x.shape
        q, k, v = self.qkv(x).flatten(2).chunk(3, dim=1)  # each (N, C, HW)
        q = F.elu(q) + 1  # positive feature map phi(q)
        k = (F.elu(k) + 1) * self.key_mask(x).unsqueeze(1)  # phi(k), zeroed for non-selected pixels
        kv = torch.einsum(
            "ncl,ndl->ncd", k, v
        )  # (N,C,C) global summary, computed once -> linear cost
        z = 1.0 / (torch.einsum("ncl,nc->nl", q, k.sum(2)) + 1e-6)  # normalisation per query pixel
        out = torch.einsum("ncd,ncl,nl->ndl", kv, q, z).reshape(n, c, h, w)
        return self.proj(out)


# %% Combine local information, attention and channel mixing
class Block(nn.Module):
    def __init__(self, c, use_ssc=True, use_attn=True, keep_ratio=0.5):
        super().__init__()
        self.n1 = nn.GroupNorm(1, c)  # LayerNorm-style (no BatchNorm in SR)
        self.spatial = (
            SparseSamplingConv(c) if use_ssc else nn.Conv2d(c, c, 3, padding=1)
        )  # SSC or plain conv (ablation)
        self.attn = AdaptiveSparseLinearAttention(c, keep_ratio) if use_attn else None
        self.n2 = nn.GroupNorm(1, c) if use_attn else None
        self.n3 = nn.GroupNorm(1, c)
        self.ffn = nn.Sequential(
            nn.Conv2d(c, 2 * c, 1), nn.GELU(), nn.Conv2d(2 * c, c, 1)
        )  # adds non-linearity

    def forward(self, x):
        x = x + self.spatial(self.n1(x))  # local branch (residual)
        if self.attn is not None:
            x = x + self.attn(self.n2(x))  # global branch (residual)
        return x + self.ffn(self.n3(x))


# %% Reconstruct RGB at four times the input size
class SRNet(nn.Module):
    def __init__(self, cfg, use_ssc=True, use_attn=True):
        super().__init__()
        c, self.scale = cfg.feat, cfg.scale
        self.head = nn.Conv2d(3, c, 3, padding=1)  # RGB -> features
        self.body = nn.Sequential(
            *[Block(c, use_ssc, use_attn, cfg.keep_ratio) for _ in range(cfg.blocks)]
        )
        self.tail = nn.Conv2d(c, c, 3, padding=1)
        up = []
        for _ in range(int(math.log2(cfg.scale))):
            up += [nn.Conv2d(c, 4 * c, 3, padding=1), nn.PixelShuffle(2)]  # 2x, twice
        self.up = nn.Sequential(*up, nn.Conv2d(c, 3, 3, padding=1))

    def forward(self, x):
        f = self.head(x)
        f = self.tail(self.body(f)) + f  # long skip connection
        return self.up(f) + F.interpolate(
            x, scale_factor=self.scale, mode="bicubic", align_corners=False
        )  # network learns only the residual over bicubic


def n_params(m):
    return sum(p.numel() for p in m.parameters())
