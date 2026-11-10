"""Compare official CodeFormer fidelity settings on validation images only."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
from codeformer_model import CodeFormerRestoration
from run_project import ROOT,runtime


if __name__=='__main__':
    n,cfg=runtime();torch.set_num_threads(3)
    hr=np.load(ROOT/'work/data/ffhq128_u8.npy',mmap_mode='r')
    with np.load(ROOT/'work/full/split_ids.npz') as z:ids=z['validation'][:8]
    data=n.FixedEvalSet(hr,ids,n.Degrader(),112,1000)
    low,target=next(data.batches(8));model=CodeFormerRestoration('cpu')
    rows=[('Clean reference',target),('Low-res input',F.interpolate(low,scale_factor=4,mode='nearest'))]
    results={}
    with torch.inference_mode():
        for fidelity in [0.,.5,1.]:
            model.fidelity=fidelity;value=model(low)
            results[str(fidelity)]={'psnr':n.Metrics.psnr_y(value,target,4).mean().item(),
                                    'ssim':n.Metrics.ssim_y(value,target,4).mean().item()}
            rows.append((f'CodeFormer w={fidelity}',value))
            print(fidelity,json.dumps(results[str(fidelity)]),flush=True)
    canvas=Image.new('RGB',(150+8*140,len(rows)*160),'#e8e9e6');draw=ImageDraw.Draw(canvas)
    for r,(label,images) in enumerate(rows):
        draw.text((5,r*160+65),label,fill='#161c16')
        for c,image in enumerate(images):
            array=(image.permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(array).resize((140,140),Image.Resampling.NEAREST),(150+c*140,r*160+10))
    out=ROOT/'work/codeformer_assets'
    canvas.save(out/'fidelity_comparison.png')
    (out/'fidelity_metrics.json').write_text(json.dumps(results,indent=2))
