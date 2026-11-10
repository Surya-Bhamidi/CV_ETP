"""External generic Real-ESRGAN compact x4 baseline; no face generator."""
from pathlib import Path
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent


class GenericRestoration(nn.Module):
    def __init__(self,device='cuda',denoise=.5):
        super().__init__()
        from restoration_vendor.srvgg_arch import SRVGGNetCompact
        self.net=SRVGGNetCompact(num_in_ch=3,num_out_ch=3,num_feat=64,num_conv=32,upscale=4,act_type='prelu')
        folder=ROOT/'work/realesrgan_assets'
        def state(name):
            value=torch.load(folder/name,map_location='cpu',weights_only=True)
            return value['params_ema'] if 'params_ema' in value else value['params']
        strong=state('realesr-general-x4v3.pth');weak=state('realesr-general-wdn-x4v3.pth')
        assert strong.keys()==weak.keys()
        self.net.load_state_dict({key:denoise*strong[key]+(1-denoise)*weak[key] for key in strong},strict=True)
        self.to(device).eval().requires_grad_(False)

    def forward(self,low):
        return self.net(low).clamp(0,1)
