"""Official LPIPS metric with workspace-local, verified pretrained weights."""
from pathlib import Path
import warnings
import torch
from torchvision.models import squeezenet1_1


def load_lpips(device):
    import lpips
    # Disable automatic weight downloads, then install the complete official
    # ImageNet backbone into the metric's identical sliced feature modules.
    # LPIPS 0.1.4 calls torchvision's legacy pretrained=False argument. These
    # two known deprecations are harmless here (we load the backbone ourselves)
    # but Windows PowerShell can misclassify their stderr as a failed command.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="The parameter 'pretrained' is deprecated.*", category=UserWarning)
        warnings.filterwarnings('ignore', message='Arguments other than a weight enum or `None`.*', category=UserWarning)
        metric = lpips.LPIPS(net='squeeze', version='0.1', pnet_rand=True, verbose=False)
    path = Path(__file__).resolve().parent / 'work/quality_assets/squeezenet1_1-b8a52dc0.pth'
    backbone = squeezenet1_1(weights=None)
    backbone.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    intervals = [(0, 2), (2, 5), (5, 8), (8, 10), (10, 11), (11, 12), (12, 13)]
    for index, (start, stop) in enumerate(intervals, 1):
        block = getattr(metric.net, f'slice{index}')
        for layer in range(start, stop):
            block._modules[str(layer)].load_state_dict(backbone.features[layer].state_dict(), strict=True)
    metric.pnet_rand = False  # All backbone weights above are now pretrained.
    return metric.to(device).eval().requires_grad_(False)
