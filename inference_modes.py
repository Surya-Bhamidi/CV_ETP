"""Small input-only inference options; no recognition or training dependencies."""
from torch import nn
from torch.nn import functional as F


class InputFidelityBlend(nn.Module):
    def __init__(self, model, weight=.5):
        super().__init__(); self.model, self.weight = model, weight

    def forward(self, low):
        restored = self.model(low).clamp(0, 1)
        observed = F.interpolate(low, scale_factor=4, mode='bicubic', align_corners=False).clamp(0, 1)
        return self.weight * restored + (1 - self.weight) * observed
