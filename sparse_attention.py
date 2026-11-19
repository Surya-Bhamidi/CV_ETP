"""Auditable spatial key selection and gathered linear attention.

Global 50% selection is checkpoint-compatible with the notebook attention.
Quadrant selection is an experimental policy, not a paper reproduction.
Training retains the dense straight-through mask so scorer gradients are kept.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SpatialSparseLinearAttention(nn.Module):
    def __init__(self, channels, keep_ratio=.5, policy='global', backend='gather', strength=1., spatial_context=False):
        super().__init__()
        if not 0 < keep_ratio <= 1: raise ValueError('keep_ratio must be in (0, 1]')
        if policy not in {'global', 'quadrant'}: raise ValueError('Unknown key-selection policy')
        if backend not in {'dense', 'gather'}: raise ValueError('Unknown attention backend')
        self.keep_ratio, self.policy, self.backend, self.strength = keep_ratio, policy, backend, strength
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.score = nn.Sequential(nn.Conv2d(channels, channels // 4, 1), nn.GELU(), nn.Conv2d(channels // 4, 1, 1))
        self.proj = nn.Conv2d(channels, channels, 1)
        self.score_context = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False) if spatial_context else None
        if self.score_context is not None: nn.init.zeros_(self.score_context.weight)

    def importance(self, x):
        value = x if self.score_context is None else x + self.score_context(x)
        return self.score(value).sigmoid().flatten(1)

    def selected_indices(self, scores, height, width):
        count = max(1, int(self.keep_ratio * height * width))
        if self.policy == 'global': return scores.topk(count, dim=1).indices
        # Guarantee spatial coverage while retaining an exact global key budget.
        grid = torch.arange(height * width, device=scores.device).view(height, width)
        regions = [grid[y0:y1, x0:x1].flatten() for y0, y1 in [(0, height // 2), (height // 2, height)]
                   for x0, x1 in [(0, width // 2), (width // 2, width)] if y1 > y0 and x1 > x0]
        sizes = [region.numel() for region in regions]
        raw = [count * size / (height * width) for size in sizes]
        budgets = [math.floor(value) for value in raw]
        order = sorted(range(len(regions)), key=lambda i: raw[i] - budgets[i], reverse=True)
        for i in order[:count - sum(budgets)]: budgets[i] += 1
        chosen = []
        for region, budget in zip(regions, budgets):
            if budget:
                local = scores[:, region].topk(budget, dim=1).indices
                chosen.append(region[local])
        return torch.cat(chosen, dim=1)

    def key_mask(self, x):
        scores = self.importance(x)
        indices = self.selected_indices(scores, x.shape[-2], x.shape[-1])
        hard = torch.zeros_like(scores).scatter_(1, indices, 1.)
        return hard + scores - scores.detach()

    def forward(self, x):
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(x).flatten(2).chunk(3, dim=1)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        if self.training or self.backend == 'dense':
            k = k * self.key_mask(x).unsqueeze(1)
        else:
            scores = self.importance(x)
            indices = self.selected_indices(scores, height, width)
            selected = indices[:, None].expand(-1, channels, -1)
            weights = scores.gather(1, indices)
            # Match the notebook's finite-precision straight-through mask value.
            weights = 1 + weights - weights.detach()
            k = k.gather(2, selected) * weights[:, None]
            v = v.gather(2, selected)
        kv = torch.einsum('ncl,ndl->ncd', k, v)
        normalizer = 1 / (torch.einsum('ncl,nc->nl', q, k.sum(2)) + 1e-6)
        result = torch.einsum('ncd,ncl,nl->ndl', kv, q, normalizer).reshape(batch, channels, height, width)
        return self.proj(result) * self.strength


def install_attention(model, policy='global', keep_ratio=.5, backend='gather', strength=1., spatial_context=False):
    """Replace attention while preserving every learned tensor and state-dict name."""
    for block in model.body:
        old = block.attn
        if old is None: continue
        sample = next(old.parameters())
        new = SpatialSparseLinearAttention(old.qkv.in_channels, keep_ratio, policy, backend, strength, spatial_context).to(device=sample.device, dtype=sample.dtype)
        missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
        if unexpected or missing not in [[], ['score_context.weight']]:
            raise ValueError(f'Attention weights do not match: missing={missing}, unexpected={unexpected}')
        new.train(old.training).requires_grad_(sample.requires_grad)
        block.attn = new
    return model


class SelfEnsemble(nn.Module):
    """Average inverse-transformed predictions; same parameters, 2x/8x passes."""
    def __init__(self, model, passes=2):
        super().__init__()
        if passes not in {2, 8}: raise ValueError('Choose two or eight passes')
        self.model, self.passes = model, passes

    def forward(self, x):
        outputs = []
        for rotation in (range(4) if self.passes == 8 else [0]):
            for flip in [False, True]:
                value = x.rot90(rotation, (-2, -1))
                if flip: value = value.flip(-1)
                prediction = self.model(value).clamp(0, 1)
                if flip: prediction = prediction.flip(-1)
                outputs.append(prediction.rot90(-rotation, (-2, -1)))
        return torch.stack(outputs).mean(0)
