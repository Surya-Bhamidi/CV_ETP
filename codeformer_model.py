"""External CodeFormer research baseline, not the custom SR architecture."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'work/codeformer_assets/codeformer.pth'


class CodeFormerRestoration(nn.Module):
    def __init__(self, device='cuda', fidelity=1., chunk_size=1):
        super().__init__()
        from codeformer_vendor.codeformer_arch import CodeFormer
        self.net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                              connect_list=['32','64','128','256'])
        checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
        self.net.load_state_dict(checkpoint['params_ema'], strict=True)
        self.fidelity = fidelity
        self.chunk_size = int(chunk_size)
        if not 1 <= self.chunk_size <= 4: raise ValueError('CodeFormer chunk size must be 1 to 4')
        self.to(device).eval().requires_grad_(False)

    def forward(self, low):
        low = low.contiguous()
        shape = (low.shape[-2]*4, low.shape[-1]*4)
        outputs=[]
        for chunk in low.split(self.chunk_size):
            up = F.interpolate(chunk, size=(512,512), mode='bicubic', align_corners=False).clamp(0,1).contiguous()
            result, _, _ = self.net(up*2-1, w=self.fidelity, adain=True)
            result = ((result+1)/2).clamp(0,1)
            outputs.append(F.interpolate(result,size=shape,mode='bicubic',align_corners=False,antialias=True).clamp(0,1))
        return torch.cat(outputs)
