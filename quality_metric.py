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


def load_lpips_alex(device):
    """Separate evaluation backbone, never used by the restoration training loss."""
    import hashlib
    import lpips
    path = Path(__file__).resolve().parent / 'work/quality_assets/alexnet-owt-7be5be79.pth'
    with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if not digest.startswith('7be5be79'): raise ValueError('Official AlexNet checksum mismatch')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message="The parameter 'pretrained' is deprecated.*", category=UserWarning)
        warnings.filterwarnings('ignore', message='Arguments other than a weight enum or `None`.*', category=UserWarning)
        metric = lpips.LPIPS(net='alex', version='0.1', pnet_rand=True, verbose=False)
    state = torch.load(path, map_location='cpu', weights_only=True)
    for index, (start, stop) in enumerate([(0,2),(2,5),(5,8),(8,10),(10,12)], 1):
        block = getattr(metric.net, f'slice{index}')
        for layer in range(start, stop):
            module = block._modules[str(layer)]
            values = {key: state[f'features.{layer}.{key}'] for key in module.state_dict()}
            module.load_state_dict(values, strict=True)
    metric.pnet_rand = False
    return metric.to(device).eval().requires_grad_(False)
