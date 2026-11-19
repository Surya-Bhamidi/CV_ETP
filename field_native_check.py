"""Deployment-style replay of native camera crops, with explicit coverage.

This is not live video, full-frame detection/tracking, or an untouched dataset.
The existing app's min8/max256 constraints and exported8-bit x4 RGB are used.
No training, epoch selection or test-based calibration occurs here.
"""
import collections
import io
import json
import time
import zipfile

import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from audit_restoration_leakage import exact_key
from evaluate_restoration_v3 import extra_pairs
from investigate_identity import load
from refine_model import FrozenFace,sha,write_json
from run_project import ROOT,notebook_definitions,runtime

OUT=ROOT/'work/field_check_v2'
CHECKPOINTS={'app_default':ROOT/'work/visual_sr/full_reconstruction/best.pt',
             'camera_v3':ROOT/'work/restoration_v3/final.pt',
             'identity_v4':ROOT/'work/recognition_v4/best.pt'}


def prepare_image(image):
    image=image.convert('RGB')
    if min(image.size)<8:return None
    if max(image.size)>256:
        ratio=256/max(image.size);size=tuple(max(1,round(d*ratio)) for d in image.size)
        if min(size)<8:return None
        image=image.resize(size,Image.Resampling.LANCZOS)
    return np.array(image,copy=True)


def decode(zipped,paths):
    images=[];sizes=[];keys=[]
    for path in paths:
        with Image.open(io.BytesIO(zipped.read(path))) as image:
            sizes.append(list(image.size));keys.append(exact_key(np.asarray(image.convert('RGB'))));images.append(prepare_image(image))
    return images,sizes,keys


@torch.inference_mode()
def native_bank(n,model,images,face,example_indices=()):
    groups=collections.defaultdict(list)
    for index,image in enumerate(images):
        if image is not None:groups[image.shape[:2]].append(index)
    values={}
    examples={};start=last=time.monotonic();completed=0
    for shape,ids in sorted(groups.items()):
        for offset in range(0,len(ids),32):
            chosen=ids[offset:offset+32]
            low=torch.from_numpy(np.stack([images[i] for i in chosen])).permute(0,3,1,2).float().to(n.device)/255
            sr=(F.interpolate(low,scale_factor=4,mode='bicubic',align_corners=False) if model is None else model(low)).clamp(0,1)
            # Quantization matches downloadable PNG pixels; no HR target enters.
            exported=(sr*255).round().clamp(0,255)/255
            normalized=F.interpolate(exported,size=(112,112),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            normal=face(normalized);mirrored=F.normalize(normal+face(normalized.flip(-1)),dim=1)
            assert torch.isfinite(normal).all() and torch.isfinite(mirrored).all()
            if not values:
                values={key:np.full((len(images),normal.shape[1]),np.nan,dtype='float32') for key in ['single','mirror']}
            values['single'][chosen]=normal.cpu().numpy();values['mirror'][chosen]=mirrored.cpu().numpy()
            for local,index in enumerate(chosen):
                if index in example_indices:examples[index]=(sr[local].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
            completed+=len(chosen)
            if time.monotonic()-last>35:
                print('NATIVE REPLAY',completed,'/',len(images),'seconds',round(time.monotonic()-start,1),flush=True);last=time.monotonic()
    return values,examples


def score(bank,pairs):
    supported=np.isfinite(bank['single']).all(1)
    eligible=supported[pairs['ia']]&supported[pairs['ib']]
    values={key:np.full(len(eligible),np.nan,dtype='float32') for key in bank}
    for key,embedding in bank.items():values[key][eligible]=(embedding[pairs['ia'][eligible]]*embedding[pairs['ib'][eligible]]).sum(1)
    return values,eligible


def summarize(values,labels,supported,threshold):
    correct=np.zeros(len(labels),dtype=bool);correct[supported]=(values[supported]>=threshold)==labels[supported]
    return {'supported_pair_accuracy':float(correct[supported].mean()),'all_pair_success_with_abstentions_counted_unsuccessful':float(correct.mean()),
        'supported_pair_auc':float(roc_auc_score(labels[supported],values[supported])),'threshold_from_native_training_validation':float(threshold),
        'pairs':len(labels),'supported_pairs':int(supported.sum()),'unsupported_pairs':int((~supported).sum()),
        'coverage':float(supported.mean()),'correct_pairs':int(correct.sum())}


def save_examples(images,examples,names,indices):
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',14);tile=170;label=220;height=200
    rows=[('Native input',None),*[(name,name) for name in examples]]
    canvas=Image.new('RGB',(label+len(indices)*tile,75+height*len(rows)),'#f1f2ed');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),'Native QMUL crops | same input per method | no clean HR original',font=font,fill='#18271e')
    draw.text((12,34),'Aspect ratio preserved. Nearest-neighbour display enlargement. Detail changes are not verified recovery.',font=font,fill='#455149')
    for row,(name,method) in enumerate(rows):
        top=75+row*height;draw.text((10,top+65),name,font=font,fill='#18271e')
        for col,index in enumerate(indices):
            value=images[index] if method is None else examples[method][index]
            image=Image.fromarray(value);ratio=min((tile-8)/image.width,(tile-8)/image.height)
            resized=image.resize((round(image.width*ratio),round(image.height*ratio)),Image.Resampling.NEAREST)
            canvas.paste(resized,(label+col*tile+(tile-resized.width)//2,top+(tile-resized.height)//2))
            draw.text((label+col*tile+3,top+tile),names[index][:22],font=font,fill='#455149')
            if method is None:draw.text((label+col*tile+3,top+tile+18),f'{value.shape[1]} x {value.shape[0]} native',font=font,fill='#455149')
    canvas.save(OUT/'native_comparison.png')


@torch.inference_mode()
def main():
    if OUT.exists():raise ValueError('Preserve recorded field check')
    OUT.mkdir(parents=True);(OUT/'emb').mkdir()
    protocol={'source_sha256':sha(__file__),'checkpoints':{key:sha(path) for key,path in CHECKPOINTS.items()},
        'pipeline':'Native crop -> app min8/max256 handling -> x4 SR or bicubic -> round to8-bit RGB -> bicubic antialias resize112 -> same frozen MobileFace single/mirrored features.',
        'unsupported':'Reject min side<8, or resizing max256 makes min side<8, exactly as app. Retain every pair; report coverage and count abstentions as unsuccessful in all-pair success.',
        'calibration':'Original512 native crops from256 held-out official-training identities, original balanced512 pairs. Fit using supported calibration pairs only; report count. No test threshold fitting.',
        'batching':'Exact native shapes grouped for throughput; no padding/resizing to a batch shape. Recognition112 normalization only after exported SR.',
        'scope':'Recorded still-crop replay; not live camera/video or full-frame detector/tracker. Repeatedly evaluated cohorts; unknown upstream/FFHQ person overlap remains.'}
    write_json(OUT/'protocol.json',protocol)
    n,_=runtime();torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    r=notebook_definitions('sr_part2a_recognition.ipynb',[1,6,7,8,9,11],'native_field_protocol')
    face=FrozenFace(ROOT/'work/refinement_assets/w600k_mbf.onnx').to(n.device).eval().requires_grad_(False)
    domain=json.loads((ROOT/'work/qmul_domain/data_manifest.json').read_text())
    main_manifest=json.loads((ROOT/'work/qmul_study/cache/manifest.json').read_text())
    additional=json.loads((ROOT/'work/leakage_review_v2/additional_cohort.json').read_text())
    with zipfile.ZipFile(ROOT/domain['source_archive']) as zipped:
        by_name={path.rsplit('/',1)[-1]:path for path in zipped.namelist() if '__MACOSX' not in path and '/Face_' in path and path.endswith('.jpg')}
        validation,validation_sizes,_=decode(zipped,domain['validation_files'])
        original,original_sizes,_=decode(zipped,[by_name[name] for name in main_manifest['files']])
        extra,extra_sizes,extra_hashes=decode(zipped,additional['files'])
    assert original_sizes==main_manifest['native_sizes']
    ia=np.arange(0,512,2);ib=ia+1
    validation_pairs={'ia':np.r_[ia,ia],'ib':np.r_[ib,np.roll(ib,-1)],'labels':np.r_[np.ones(256),np.zeros(256)]}
    extra_pair=extra_pairs(additional['files'],extra_hashes)
    with np.load(ROOT/'work/qmul_study/cache/pairs.npz') as data:original_pairs={key:data[key] for key in ['ia','ib','labels']}
    image_checks={name:{'images':len(images),'supported_images':sum(image is not None for image in images),'unsupported_images':sum(image is None for image in images)}
                  for name,images in [('validation',validation),('main',original),('additional',extra)]}
    write_json(OUT/'input_coverage.json',image_checks)
    indices=np.linspace(0,len(original)-1,6,dtype=int).tolist();assert all(original[i] is not None for i in indices)
    # All method settings were fixed above. Native calibration for every method
    # is completed and frozen before any test recognition scores are computed.
    calibration={}
    for name,path in [('bicubic',None),*CHECKPOINTS.items()]:
        model=None if path is None else load(n,path)
        features,_=native_bank(n,model,validation,face);values,supported=score(features,validation_pairs)
        calibration[name]={key:{'threshold':float(r.VerificationEval.best_threshold(vector[supported],validation_pairs['labels'][supported])),
              'supported_validation_pairs':int(supported.sum()),'all_validation_pairs':512} for key,vector in values.items()}
        np.savez(OUT/'emb'/f'{name}_validation.npz',labels=validation_pairs['labels'],supported=supported,**{key+'_scores':v for key,v in values.items()})
        del model;torch.cuda.empty_cache();print('NATIVE CALIBRATION',name,json.dumps(calibration[name]),flush=True)
    write_json(OUT/'calibration.json',calibration)
    result={'protocol':protocol,'calibration_sha256':sha(OUT/'calibration.json'),'input_coverage':image_checks,'recognition':{}}
    example_bank={}
    for name,path in [('bicubic',None),*CHECKPOINTS.items()]:
        model=None if path is None else load(n,path);result['recognition'][name]={}
        for dataset,images,pairs in [('main',original,original_pairs),('additional',extra,extra_pair)]:
            features,examples=native_bank(n,model,images,face,indices if dataset=='main' else ())
            values,supported=score(features,pairs)
            result['recognition'][name][dataset]={key:summarize(vector,pairs['labels'],supported,calibration[name][key]['threshold']) for key,vector in values.items()}
            np.savez(OUT/'emb'/f'{name}_{dataset}.npz',labels=pairs['labels'],supported=supported,**{key+'_scores':v for key,v in values.items()})
            if dataset=='main':example_bank[name]=examples
            print('NATIVE RESULT',name,dataset,json.dumps(result['recognition'][name][dataset]),flush=True)
        del model;torch.cuda.empty_cache()
    save_examples(original,example_bank,main_manifest['files'],indices)
    write_json(OUT/'native_results.json',result);print('NATIVE FIELD CHECK COMPLETE',flush=True)


if __name__=='__main__':main()
