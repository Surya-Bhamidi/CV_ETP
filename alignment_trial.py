"""Validation-only trial of landmark alignment for the external baselines.

No clean image is used to estimate the transform. This is an audit experiment,
not automatically enabled in the application.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw
from restoration_model import FaceRestoration
from codeformer_model import CodeFormerRestoration
from run_project import ROOT,runtime

TEMPLATE=np.array([[192.98138,239.94708],[318.90277,240.1936],[256.63416,314.01935],
                   [201.26117,371.41043],[313.08905,371.15118]],dtype=np.float32)


def transform(points):
    source=points.astype(np.float64)
    matrix=np.zeros((10,4));values=TEMPLATE.astype(np.float64).reshape(-1)
    matrix[0::2,0]=source[:,0];matrix[0::2,1]=-source[:,1];matrix[0::2,2]=1
    matrix[1::2,0]=source[:,1];matrix[1::2,1]=source[:,0];matrix[1::2,3]=1
    a,b,tx,ty=np.linalg.lstsq(matrix,values,rcond=None)[0]
    return np.array([[a,-b,tx],[b,a,ty]],dtype=np.float32)


def main():
    n,cfg=runtime();torch.set_num_threads(3);cv2.setNumThreads(2)
    out=ROOT/'work/alignment_assets'
    detector=cv2.FaceDetectorYN_create(str(out/'face_detection_yunet_2023mar.onnx'),'',(320,320),.6,.3,1000)
    hr=np.load(ROOT/'work/data/ffhq128_u8.npy',mmap_mode='r')
    with np.load(ROOT/'work/full/split_ids.npz') as z:ids=z['validation'][:8]
    data=n.FixedEvalSet(hr,ids,n.Degrader(),112,1000)
    low,targets=next(data.batches(8))
    matrices=[];aligned=[];records=[]
    up=F.interpolate(low,size=(512,512),mode='bicubic',align_corners=False).clamp(0,1)
    for image in up:
        rgb=image.permute(1,2,0).numpy()
        detection=cv2.cvtColor(cv2.resize((rgb*255).round().astype(np.uint8),(320,320)),cv2.COLOR_RGB2BGR)
        _,faces=detector.detect(detection)
        matrix=None
        if faces is not None:
            face=max(faces,key=lambda f:f[2]*f[3]);points=face[4:14].reshape(5,2)*512/320
            # Enforce image-left to image-right ordering for eyes and mouth.
            if points[0,0]>points[1,0]:points[[0,1]]=points[[1,0]]
            if points[3,0]>points[4,0]:points[[3,4]]=points[[4,3]]
            candidate=transform(points)
            projected=points@candidate[:,:2].T+candidate[:,2]
            error=float(np.linalg.norm(projected-TEMPLATE,axis=1).mean())
            scale=float(np.linalg.norm(candidate[0,:2]))
            if error<20 and .5<scale<2.2:matrix=candidate
            records.append({'detected':True,'fit_error':error,'scale':scale,'used':matrix is not None})
        else:records.append({'detected':False,'used':False})
        matrices.append(matrix)
        warped=rgb if matrix is None else cv2.warpAffine(rgb,matrix,(512,512),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101)
        aligned.append(torch.from_numpy(np.ascontiguousarray(warped)).permute(2,0,1).clamp(0,1))
    aligned=torch.stack(aligned)
    rows=[('Clean reference',targets),('Low-res input',F.interpolate(low,scale_factor=4,mode='nearest'))]
    results={}
    with torch.inference_mode():
        for name,model in [('GFPGAN',FaceRestoration('cpu')),('CodeFormer',CodeFormerRestoration('cpu',fidelity=1.))]:
            raw=model(low)
            restored=[]
            for i,image in enumerate(aligned):
                if name=='GFPGAN': value,_=model.net(image[None]*2-1,return_rgb=False,randomize_noise=False)
                else:value,_,_=model.net(image[None]*2-1,w=1.,adain=True)
                rgb=((value[0]+1)/2).clamp(0,1).permute(1,2,0).numpy()
                matrix=matrices[i]
                if matrix is not None:
                    inverse=cv2.invertAffineTransform(matrix)
                    restored_rgb=cv2.warpAffine(rgb,inverse,(512,512),flags=cv2.INTER_CUBIC)
                    mask=cv2.warpAffine(np.ones((512,512),np.float32),inverse,(512,512),flags=cv2.INTER_LINEAR)
                    mask=cv2.GaussianBlur(cv2.erode(mask,np.ones((9,9),np.uint8)),(15,15),3).clip(0,1)
                    original=up[i].permute(1,2,0).numpy()
                    rgb=restored_rgb*mask[:,:,None]+original*(1-mask[:,:,None])
                tensor=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None]
                restored.append(F.interpolate(tensor,size=(112,112),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)[0])
            restored=torch.stack(restored)
            rows += [(name+' direct',raw),(name+' aligned',restored)]
            results[name]={}
            for label,value in [('direct',raw),('aligned',restored)]:
                results[name][label]={'psnr':n.Metrics.psnr_y(value,targets,4).mean().item(),
                                     'ssim':n.Metrics.ssim_y(value,targets,4).mean().item()}
            print(name,json.dumps(results[name]),flush=True)
    canvas=Image.new('RGB',(150+8*140,len(rows)*160),'#e8e9e6');draw=ImageDraw.Draw(canvas)
    for r,(label,images) in enumerate(rows):
        draw.text((5,r*160+65),label,fill='#161c16')
        for c,image in enumerate(images):
            array=(image.permute(1,2,0).numpy().clip(0,1)*255).round().astype(np.uint8)
            canvas.paste(Image.fromarray(array).resize((140,140),Image.Resampling.NEAREST),(150+c*140,r*160+10))
    canvas.save(out/'validation_alignment.png')
    (out/'trial.json').write_text(json.dumps({'validation_images':8,'selection_on_test':False,'detections':records,'metrics':results},indent=2))
    print('Alignment trial complete',flush=True)


if __name__=='__main__':main()
