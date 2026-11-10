"""Validation-only comparison with a pretrained general SR model."""
import json,numpy as np,torch
from run_project import ROOT,runtime
from real_esrgan_model import GenericRestoration
from train_visual_sr import grid


if __name__=='__main__':
    n,cfg=runtime();n.device='cpu';torch.set_num_threads(3)
    hr=np.load(ROOT/'work/data/ffhq128_u8.npy',mmap_mode='r')
    with np.load(ROOT/'work/full/split_ids.npz') as z:ids=z['validation'][:256]
    data=n.FixedEvalSet(hr,ids,n.Degrader(),112,1000)
    original=n.SRNet(cfg).eval().requires_grad_(False)
    original.load_state_dict(torch.load(ROOT/'work/full/ckpt/ours_ssc_attn_best.pt',map_location='cpu',weights_only=True)['model'])
    models={'Original custom SR':original,'Generic SR denoise .5':GenericRestoration('cpu',.5),'Generic SR denoise 1':GenericRestoration('cpu',1.)}
    results={}
    for label,model in models.items():
        results[label]=n.evaluate(model,data,'cpu',bs=16,ssim=True)
        print(label,json.dumps(results[label]),flush=True)
    out=ROOT/'work/realesrgan_assets'
    grid(n,models,data,out/'validation_comparison.png')
    (out/'validation_metrics.json').write_text(json.dumps(results,indent=2))
